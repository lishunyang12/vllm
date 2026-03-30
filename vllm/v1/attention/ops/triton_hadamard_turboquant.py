# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused Triton kernels for TurboQuant with Hadamard rotation.

Single-kernel encode: normalize → sign_flip → Hadamard → quantize
Single-kernel decode: codebook lookup → Hadamard → sign_flip → scale
Packed 4-bit decode: nibble unpack → codebook → Hadamard → scale
Direct-from-cache decode: reads packed slots from paged cache layout
Rotated-domain decode: skips Hadamard entirely for pre-rotated queries

The Hadamard butterfly uses XOR-based partner indexing with a small
scratch buffer in global memory (stays in L1 cache, ~512 bytes per
thread block).

Attribution:
  - Algorithm: Zandieh, Daliri, Hadian, Mirrokni (arxiv 2504.19874)
  - Pre-rotated query technique inspired by 0xSero/turboquant
"""

import math

import torch

from vllm.triton_utils import tl, triton


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------


def _next_power_of_2(n: int) -> int:
    if n <= 0:
        return 1
    return 1 << (n - 1).bit_length()


def _hadamard_transform_torch(x: torch.Tensor) -> torch.Tensor:
    """PyTorch FWHT fallback on the last dimension.

    Used for query rotation (called once per decode step), where launching
    a Triton kernel would add unnecessary overhead.
    """
    d = x.shape[-1]
    h = 1
    while h < d:
        groups = x.reshape(*x.shape[:-1], d // (2 * h), 2, h)
        x_even = groups[..., 0, :].clone()
        x_odd = groups[..., 1, :].clone()
        groups[..., 0, :] = x_even + x_odd
        groups[..., 1, :] = x_even - x_odd
        h *= 2
    return x / math.sqrt(d)


# ---------------------------------------------------------------------------
# Triton kernels
# ---------------------------------------------------------------------------


@triton.jit
def _fused_hadamard_encode_kernel(
    # Input: [num_tokens, num_kv_heads, head_size]
    x_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Boundaries: [num_centroids - 1] float32
    boundaries_ptr,
    # Scratch buffer: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Output indices: [num_tokens, num_kv_heads, head_size] uint8
    indices_ptr,
    # Output norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    num_boundaries: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides
    x_stride_token: tl.int64,
    x_stride_head: tl.int64,
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    BLOCK_D: tl.constexpr,
):
    """Fused encode: norm → sign_flip → Hadamard → quantize."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load and normalize
    x_base = token_idx * x_stride_token + head_idx * x_stride_head
    x = tl.load(x_ptr + x_base + dim_offs, mask=mask, other=0.0).to(tl.float32)
    norm_sq = tl.sum(x * x, axis=0)
    norm = tl.sqrt(norm_sq + 1e-16)
    x = x / norm

    # Sign flips
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # Store to scratch for Hadamard butterfly
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, x)

    # Hadamard butterfly: log2(BLOCK_D) passes.
    # Barrier after each store to prevent inter-warp races
    # (element i in warp 0 may partner with element j in warp 1).
    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.debug_barrier()
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        tl.debug_barrier()
        h = h * 2

    # Load result, scale
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    # Quantize: vectorized bucketize
    # For each element, count boundaries exceeded
    idx = tl.zeros([BLOCK_D], dtype=tl.int32)
    for b in range(num_boundaries):
        bnd = tl.load(boundaries_ptr + b)
        idx = idx + (x > bnd).to(tl.int32)

    # Store indices and norm
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    tl.store(indices_ptr + idx_base + dim_offs, idx.to(tl.uint8), mask=mask)
    tl.store(norms_ptr + token_idx * norm_stride_token + head_idx, norm.to(tl.float16))


@triton.jit
def _fused_hadamard_decode_kernel(
    # Input indices: [num_tokens, num_kv_heads, head_size] uint8
    indices_ptr,
    # Input norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Codebook: [num_centroids] float32
    codebook_ptr,
    # Scratch buffer: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Output: [num_tokens, num_kv_heads, head_size]
    out_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides
    idx_stride_token: tl.int64,
    idx_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Fused decode: codebook → Hadamard → sign_flip → scale."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Load indices and codebook lookup
    idx_base = token_idx * idx_stride_token + head_idx * idx_stride_head
    indices = tl.load(indices_ptr + idx_base + dim_offs, mask=mask, other=0).to(
        tl.int32
    )
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # Store to scratch for Hadamard butterfly
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, reconstructed)

    # Hadamard butterfly (inverse = same as forward, just scale)
    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(is_lower, val_self + val_partner, val_partner - val_self)
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    # Load, scale, sign flip
    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    # Sign flips (inverse = same signs)
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # Scale by norm
    norm = tl.load(norms_ptr + token_idx * norm_stride_token + head_idx).to(tl.float32)
    x = x * norm

    # Store output
    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    if OUTPUT_BF16:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Python wrappers
# ---------------------------------------------------------------------------


def hadamard_turboquant_encode(
    x: torch.Tensor,  # [num_tokens, num_kv_heads, head_size]
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids] (unused, kept for API compat)
    boundaries: torch.Tensor,  # [num_centroids - 1]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Fused Hadamard encode: normalize → sign_flip → FWHT → quantize."""
    num_tokens, num_kv_heads, head_size = x.shape
    BLOCK_D = sign_flips.shape[0]  # padded to power of 2
    LOG2_D = int(math.log2(BLOCK_D))
    num_boundaries = boundaries.shape[0]

    indices = torch.empty(
        (num_tokens, num_kv_heads, head_size), dtype=torch.uint8, device=x.device
    )
    norms = torch.empty(
        (num_tokens, num_kv_heads), dtype=torch.float16, device=x.device
    )
    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D), dtype=torch.float32, device=x.device
    )

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_encode_kernel[grid](
        x_ptr=x,
        signs_ptr=sign_flips,
        boundaries_ptr=boundaries,
        scratch_ptr=scratch,
        indices_ptr=indices,
        norms_ptr=norms,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        num_boundaries=num_boundaries,
        LOG2_D=LOG2_D,
        x_stride_token=x.stride(0),
        x_stride_head=x.stride(1),
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        BLOCK_D=BLOCK_D,
        # num_warps=1: required for Hadamard butterfly correctness.
        # Multi-warp causes inter-warp races on scratch buffer.
        num_warps=4,
        num_stages=1,
    )

    return indices, norms


def hadamard_turboquant_decode(
    indices: torch.Tensor,  # [num_tokens, num_kv_heads, head_size] uint8
    norms: torch.Tensor,  # [num_tokens, num_kv_heads] float16
    sign_flips: torch.Tensor,  # [BLOCK_D] float32
    codebook: torch.Tensor,  # [num_centroids]
    output_dtype: torch.dtype = torch.bfloat16,
) -> torch.Tensor:
    """Fused Hadamard decode: codebook → FWHT → sign_flip → scale."""
    num_tokens, num_kv_heads, head_size = indices.shape
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size), dtype=output_dtype, device=indices.device
    )
    scratch = torch.empty(
        (num_tokens * num_kv_heads, BLOCK_D), dtype=torch.float32, device=indices.device
    )

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_decode_kernel[grid](
        indices_ptr=indices,
        norms_ptr=norms,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        out_ptr=out,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        LOG2_D=LOG2_D,
        idx_stride_token=indices.stride(0),
        idx_stride_head=indices.stride(1),
        norm_stride_token=norms.stride(0),
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    return out


# ---------------------------------------------------------------------------
# Packed 4-bit decode kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_hadamard_decode_packed4_kernel(
    # Input packed indices: [num_tokens, num_kv_heads, packed_bytes] uint8
    packed_ptr,
    # Input norms: [num_tokens, num_kv_heads] float16
    norms_ptr,
    # Sign flips: [BLOCK_D] float32
    signs_ptr,
    # Codebook: [num_centroids] float32
    codebook_ptr,
    # Scratch buffer: [num_tokens * num_kv_heads, BLOCK_D] float32
    scratch_ptr,
    # Output: [num_tokens, num_kv_heads, head_size]
    out_ptr,
    # Shapes
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides
    packed_stride_token: tl.int64,
    packed_stride_head: tl.int64,
    norm_stride_token: tl.int64,
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Fused decode for packed 4-bit indices: unpack -> codebook -> Hadamard."""
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Unpack nibbles: even dims get low nibble, odd dims get high nibble
    packed_base = (
        token_idx * packed_stride_token + head_idx * packed_stride_head
    )
    packed_vals = tl.load(
        packed_ptr + packed_base + (dim_offs // 2), mask=mask, other=0
    ).to(tl.int32)
    is_lower = (dim_offs & 1) == 0
    indices = tl.where(
        is_lower, packed_vals & 0x0F, (packed_vals >> 4) & 0x0F
    )
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, reconstructed)

    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(
            is_lower, val_self + val_partner, val_partner - val_self
        )
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    norm = tl.load(
        norms_ptr + token_idx * norm_stride_token + head_idx
    ).to(tl.float32)
    x = x * norm

    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    if OUTPUT_BF16:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Direct-from-cache packed 4-bit decode kernel
# ---------------------------------------------------------------------------


@triton.jit
def _fused_hadamard_decode_packed4_cache_kernel(
    # Input full cache: [num_blocks, block_size, num_kv_heads, slot_bytes]
    cache_ptr,
    # Referenced block ids: [num_entries]
    block_ids_ptr,
    # Sign flips / codebook / scratch / output
    signs_ptr,
    codebook_ptr,
    scratch_ptr,
    out_ptr,
    # Shapes
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    LOG2_D: tl.constexpr,
    # Strides (cache layout)
    cache_stride_block: tl.int64,
    cache_stride_token: tl.int64,
    cache_stride_head: tl.int64,
    packed_offset: tl.constexpr,
    norm_offset: tl.constexpr,
    # Output strides
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Fused packed 4-bit decode reading directly from paged cache.

    Reads packed nibbles and the two-byte float16 norm directly from the
    paged KV cache layout, avoiding an intermediate gather step.
    The norm is reconstructed via ``norm_lo | (norm_hi << 8)`` then
    bitcast to float16.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)
    scratch_row = token_idx * num_kv_heads + head_idx

    # Map flat token index to (block_entry, slot_within_block)
    entry_idx = token_idx // block_size
    block_offset = token_idx % block_size
    block_id = tl.load(block_ids_ptr + entry_idx).to(tl.int64)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Read packed nibbles from cache
    packed_base = (
        block_id * cache_stride_block
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
        + packed_offset
    )
    packed_vals = tl.load(
        cache_ptr + packed_base + (dim_offs // 2), mask=mask, other=0
    ).to(tl.int32)
    is_lower = (dim_offs & 1) == 0
    indices = tl.where(
        is_lower, packed_vals & 0x0F, (packed_vals >> 4) & 0x0F
    )
    reconstructed = tl.load(codebook_ptr + indices)
    reconstructed = tl.where(mask, reconstructed, 0.0)

    # Hadamard butterfly via scratch
    scratch_base = scratch_row * BLOCK_D
    tl.store(scratch_ptr + scratch_base + dim_offs, reconstructed)

    h = 1
    for _level in range(LOG2_D):
        partner = dim_offs ^ h
        val_self = tl.load(scratch_ptr + scratch_base + dim_offs)
        val_partner = tl.load(scratch_ptr + scratch_base + partner)
        is_lower = (dim_offs & h) == 0
        result = tl.where(
            is_lower, val_self + val_partner, val_partner - val_self
        )
        tl.store(scratch_ptr + scratch_base + dim_offs, result)
        h = h * 2

    tl.debug_barrier()
    x = tl.load(scratch_ptr + scratch_base + dim_offs)
    scale = 1.0 / tl.sqrt(float(BLOCK_D))
    x = x * scale

    # Sign flips
    signs = tl.load(signs_ptr + dim_offs)
    x = x * signs

    # Reconstruct float16 norm from two cache bytes (in-kernel bitcast)
    norm_base = (
        cache_ptr
        + block_id * cache_stride_block
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
        + norm_offset
    )
    norm_lo = tl.load(norm_base).to(tl.uint16)
    norm_hi = tl.load(norm_base + 1).to(tl.uint16)
    norm_bits = norm_lo | (norm_hi << 8)
    norm = norm_bits.to(tl.float16, bitcast=True).to(tl.float32)
    x = x * norm

    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    if OUTPUT_BF16:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.bfloat16), mask=mask)
    else:
        tl.store(out_ptr + out_base + dim_offs, x.to(tl.float16), mask=mask)


# ---------------------------------------------------------------------------
# Rotated-domain decode kernel (no Hadamard in inner loop)
# ---------------------------------------------------------------------------


@triton.jit
def _decode_packed4_cache_rotated_kernel(
    # Paged cache and block table
    cache_ptr,
    block_ids_ptr,
    # Codebook and output
    codebook_ptr,
    out_ptr,
    # Shapes
    block_size: tl.constexpr,
    head_size: tl.constexpr,
    num_kv_heads: tl.constexpr,
    # Cache strides
    cache_stride_block: tl.int64,
    cache_stride_token: tl.int64,
    cache_stride_head: tl.int64,
    packed_offset: tl.constexpr,
    norm_offset: tl.constexpr,
    # Output strides
    out_stride_token: tl.int64,
    out_stride_head: tl.int64,
    BLOCK_D: tl.constexpr,
    OUTPUT_BF16: tl.constexpr,
):
    """Decode packed 4-bit cache slots into the rotated domain only.

    This kernel skips the Hadamard butterfly and sign flips entirely,
    outputting codebook values scaled by the stored norm. When the query
    has been pre-rotated into the same domain, attention dot products
    are equivalent to the standard-domain computation.

    This is the key performance optimization: eliminating log2(d)
    butterfly passes from the inner decode loop.
    """
    token_idx = tl.program_id(0)
    head_idx = tl.program_id(1)

    entry_idx = token_idx // block_size
    block_offset = token_idx % block_size
    block_id = tl.load(block_ids_ptr + entry_idx).to(tl.int64)

    dim_offs = tl.arange(0, BLOCK_D)
    mask = dim_offs < head_size

    # Read packed nibbles
    packed_base = (
        block_id * cache_stride_block
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
        + packed_offset
    )
    packed_vals = tl.load(
        cache_ptr + packed_base + (dim_offs // 2), mask=mask, other=0
    ).to(tl.int32)
    is_lower = (dim_offs & 1) == 0
    indices = tl.where(
        is_lower, packed_vals & 0x0F, (packed_vals >> 4) & 0x0F
    )
    decoded = tl.load(codebook_ptr + indices)
    decoded = tl.where(mask, decoded, 0.0)

    # Reconstruct float16 norm from two cache bytes
    norm_base = (
        cache_ptr
        + block_id * cache_stride_block
        + block_offset * cache_stride_token
        + head_idx * cache_stride_head
        + norm_offset
    )
    norm_lo = tl.load(norm_base).to(tl.uint16)
    norm_hi = tl.load(norm_base + 1).to(tl.uint16)
    norm_bits = norm_lo | (norm_hi << 8)
    norm = norm_bits.to(tl.float16, bitcast=True).to(tl.float32)
    decoded = decoded * norm

    out_base = token_idx * out_stride_token + head_idx * out_stride_head
    if OUTPUT_BF16:
        tl.store(
            out_ptr + out_base + dim_offs, decoded.to(tl.bfloat16), mask=mask
        )
    else:
        tl.store(
            out_ptr + out_base + dim_offs, decoded.to(tl.float16), mask=mask
        )


# ---------------------------------------------------------------------------
# Python wrappers for new kernels
# ---------------------------------------------------------------------------


def hadamard_turboquant_decode_packed4(
    packed_indices: torch.Tensor,
    norms: torch.Tensor,
    sign_flips: torch.Tensor,
    codebook: torch.Tensor,
    head_size: int,
    output_dtype: torch.dtype = torch.bfloat16,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Hadamard decode for packed 4-bit indices."""
    num_tokens, num_kv_heads, _ = packed_indices.shape
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=output_dtype,
        device=packed_indices.device,
    )

    required_rows = num_tokens * num_kv_heads
    if (
        scratch is None
        or scratch.device != packed_indices.device
        or scratch.dtype != torch.float32
        or scratch.shape[0] < required_rows
        or scratch.shape[1] < BLOCK_D
    ):
        scratch = torch.empty(
            (required_rows, BLOCK_D),
            dtype=torch.float32,
            device=packed_indices.device,
        )
    else:
        scratch = scratch[:required_rows, :BLOCK_D]

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_decode_packed4_kernel[grid](
        packed_ptr=packed_indices,
        norms_ptr=norms,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        out_ptr=out,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        LOG2_D=LOG2_D,
        packed_stride_token=packed_indices.stride(0),
        packed_stride_head=packed_indices.stride(1),
        norm_stride_token=norms.stride(0),
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    return out


def hadamard_turboquant_decode_packed4_cache(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    sign_flips: torch.Tensor,
    codebook: torch.Tensor,
    head_size: int,
    packed_offset: int,
    norm_offset: int,
    output_dtype: torch.dtype = torch.bfloat16,
    scratch: torch.Tensor | None = None,
) -> torch.Tensor:
    """Fused Hadamard decode reading directly from paged cache.

    Avoids a separate gather step by reading packed nibbles and norms
    directly from the paged KV cache layout.
    """
    _, block_size, num_kv_heads, _ = cache.shape
    num_entries = block_ids.shape[0]
    BLOCK_D = sign_flips.shape[0]
    LOG2_D = int(math.log2(BLOCK_D))
    num_tokens = num_entries * block_size

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=output_dtype,
        device=cache.device,
    )

    required_rows = num_tokens * num_kv_heads
    if (
        scratch is None
        or scratch.device != cache.device
        or scratch.dtype != torch.float32
        or scratch.shape[0] < required_rows
        or scratch.shape[1] < BLOCK_D
    ):
        scratch = torch.empty(
            (required_rows, BLOCK_D),
            dtype=torch.float32,
            device=cache.device,
        )
    else:
        scratch = scratch[:required_rows, :BLOCK_D]

    grid = (num_tokens, num_kv_heads)

    _fused_hadamard_decode_packed4_cache_kernel[grid](
        cache_ptr=cache,
        block_ids_ptr=block_ids,
        signs_ptr=sign_flips,
        codebook_ptr=codebook,
        scratch_ptr=scratch,
        out_ptr=out,
        block_size=block_size,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        LOG2_D=LOG2_D,
        cache_stride_block=cache.stride(0),
        cache_stride_token=cache.stride(1),
        cache_stride_head=cache.stride(2),
        packed_offset=packed_offset,
        norm_offset=norm_offset,
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=BLOCK_D,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=4,
        num_stages=1,
    )

    return out


def hadamard_turboquant_rotate_query(
    query: torch.Tensor,
    sign_flips: torch.Tensor,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Rotate query vectors into the TurboQuant rotated domain.

    Pre-rotating Q allows attention to be computed in the rotated domain,
    eliminating the per-position Hadamard butterfly from the decode
    kernel. Technique inspired by 0xSero/turboquant.
    """
    head_size = query.shape[-1]
    block_d = sign_flips.shape[0]
    if head_size > block_d:
        raise ValueError(
            f"query head_size {head_size} exceeds "
            f"TurboQuant block size {block_d}"
        )

    q = query.to(torch.float32)
    if head_size < block_d:
        q = torch.nn.functional.pad(q, (0, block_d - head_size))
    q = q * sign_flips[:block_d]
    q = _hadamard_transform_torch(q)
    q = q[..., :head_size]
    return q.to(output_dtype)


def hadamard_turboquant_unrotate_output(
    output: torch.Tensor,
    sign_flips: torch.Tensor,
    output_dtype: torch.dtype | None = None,
) -> torch.Tensor:
    """Map rotated-domain attention output back to the standard domain.

    Applies the inverse rotation: Hadamard -> sign_flip (both are
    self-inverse operations).
    """
    head_size = output.shape[-1]
    block_d = sign_flips.shape[0]
    if head_size > block_d:
        raise ValueError(
            f"output head_size {head_size} exceeds "
            f"TurboQuant block size {block_d}"
        )

    x = output.to(torch.float32)
    if head_size < block_d:
        x = torch.nn.functional.pad(x, (0, block_d - head_size))
    x = _hadamard_transform_torch(x)
    x = x * sign_flips[:block_d]
    x = x[..., :head_size]
    if output_dtype is None:
        output_dtype = output.dtype
    return x.to(output_dtype)


def hadamard_turboquant_decode_packed4_cache_rotated(
    cache: torch.Tensor,
    block_ids: torch.Tensor,
    codebook: torch.Tensor,
    head_size: int,
    packed_offset: int,
    norm_offset: int,
    output_dtype: torch.dtype = torch.float16,
) -> torch.Tensor:
    """Decode packed 4-bit cache slots into the rotated domain only.

    Skips the Hadamard butterfly and sign flips entirely. The output
    is in the rotated domain and must be paired with a pre-rotated
    query for correct attention scores.
    """
    _, block_size, num_kv_heads, _ = cache.shape
    num_entries = block_ids.shape[0]
    block_d = _next_power_of_2(head_size)
    num_tokens = num_entries * block_size

    out = torch.empty(
        (num_tokens, num_kv_heads, head_size),
        dtype=output_dtype,
        device=cache.device,
    )

    grid = (num_tokens, num_kv_heads)

    _decode_packed4_cache_rotated_kernel[grid](
        cache_ptr=cache,
        block_ids_ptr=block_ids,
        codebook_ptr=codebook,
        out_ptr=out,
        block_size=block_size,
        head_size=head_size,
        num_kv_heads=num_kv_heads,
        cache_stride_block=cache.stride(0),
        cache_stride_token=cache.stride(1),
        cache_stride_head=cache.stride(2),
        packed_offset=packed_offset,
        norm_offset=norm_offset,
        out_stride_token=out.stride(0),
        out_stride_head=out.stride(1),
        BLOCK_D=block_d,
        OUTPUT_BF16=(output_dtype == torch.bfloat16),
        num_warps=2,
        num_stages=1,
    )

    return out
