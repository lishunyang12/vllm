# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""TurboQuant attention backend for compressed KV cache.

Separate backend for TurboQuant (ICLR 2026) KV cache quantization.
Stores K/V in packed uint8 with outlier-aware layout, decodes to bf16
before running standard Triton attention kernels.
"""

import math
from typing import ClassVar

import torch

from vllm.config.cache import CacheDType
from vllm.logger import init_logger
from vllm.platforms.interface import DeviceCapability
from vllm.v1.attention.backend import (
    AttentionBackend,
    AttentionImpl,
    AttentionLayer,
    AttentionType,
    MultipleOf,
)
from vllm.v1.attention.backends.triton_attn import (
    TritonAttentionMetadata,
    TritonAttentionMetadataBuilder,
)
from vllm.v1.attention.ops.triton_unified_attention import unified_attention

logger = init_logger(__name__)


# ---------------------------------------------------------------------------
# Bit-packing helpers
# ---------------------------------------------------------------------------


def _pack_3bit_vectorized(
    indices: torch.Tensor,  # [N, head_size] uint8
    head_size: int,
    packed_bytes: int,
) -> torch.Tensor:
    """Pack 3-bit indices into bytes: 10 values per 30 bits (4 bytes)."""
    N = indices.shape[0]
    device = indices.device

    padded = ((head_size + 9) // 10) * 10
    if head_size < padded:
        indices = torch.nn.functional.pad(indices, (0, padded - head_size), value=0)

    num_groups = padded // 10
    grouped = indices.reshape(N, num_groups, 10).to(torch.int32)

    packed_u32 = torch.zeros(N, num_groups, dtype=torch.int32, device=device)
    for shift_idx in range(10):
        packed_u32 |= (grouped[:, :, shift_idx] & 0x7) << (shift_idx * 3)

    packed_u8 = packed_u32.view(torch.uint8).reshape(N, -1)
    return packed_u8[:, :packed_bytes]


def _unpack_3bit_vectorized(
    packed: torch.Tensor,  # [N, packed_bytes] uint8
    head_size: int,
    device: torch.device,
) -> torch.Tensor:
    """Unpack 3-bit indices from bytes: 10 values per 4 bytes."""
    N = packed.shape[0]
    num_groups = (head_size + 9) // 10

    needed_bytes = num_groups * 4
    if packed.shape[1] < needed_bytes:
        packed = torch.nn.functional.pad(
            packed, (0, needed_bytes - packed.shape[1]), value=0
        )

    packed_u32 = packed[:, :needed_bytes].reshape(N, num_groups, 4)
    words = packed_u32.to(torch.int32)
    words = (
        words[:, :, 0]
        | (words[:, :, 1] << 8)
        | (words[:, :, 2] << 16)
        | (words[:, :, 3] << 24)
    )

    indices_list = []
    for shift_idx in range(10):
        indices_list.append((words >> (shift_idx * 3)) & 0x7)
    indices = torch.stack(indices_list, dim=-1).reshape(N, -1).to(torch.uint8)
    return indices[:, :head_size]


# ---------------------------------------------------------------------------
# Backend
# ---------------------------------------------------------------------------


class TurboQuantAttentionBackend(AttentionBackend):
    """Attention backend for TurboQuant compressed KV cache."""

    accept_output_buffer: bool = True
    supported_dtypes: ClassVar[list[torch.dtype]] = [
        torch.float16,
        torch.bfloat16,
    ]
    supported_kv_cache_dtypes: ClassVar[list[CacheDType]] = ["turboquant"]
    forward_includes_kv_cache_update: bool = False

    @staticmethod
    def get_supported_kernel_block_sizes() -> list[int | MultipleOf]:
        return [MultipleOf(16)]

    @staticmethod
    def get_name() -> str:
        return "TURBOQUANT"

    @staticmethod
    def get_impl_cls() -> type["TurboQuantAttentionImpl"]:
        return TurboQuantAttentionImpl

    @staticmethod
    def get_builder_cls() -> type[TritonAttentionMetadataBuilder]:
        # Reuse Triton metadata builder — same metadata format
        return TritonAttentionMetadataBuilder

    @staticmethod
    def get_kv_cache_shape(
        num_blocks: int,
        block_size: int,
        num_kv_heads: int,
        head_size: int,
        cache_dtype_str: str = "auto",
    ) -> tuple[int, ...]:
        if block_size % 16 != 0:
            raise ValueError("Block size must be a multiple of 16.")
        # head_size here is actually slot_bytes (set by get_kv_cache_spec)
        return (num_blocks, 2, block_size, num_kv_heads, head_size)

    @staticmethod
    def get_kv_cache_stride_order(
        include_num_layers_dimension: bool = False,
    ) -> tuple[int, ...]:
        if include_num_layers_dimension:
            return (1, 0, 2, 3, 4, 5)
        return (0, 1, 2, 3, 4)

    @staticmethod
    def use_cascade_attention(*args, **kwargs) -> bool:
        return False

    @classmethod
    def supports_head_size(cls, head_size: int) -> bool:
        return head_size >= 32

    @classmethod
    def supports_attn_type(cls, attn_type: str) -> bool:
        return attn_type == AttentionType.DECODER

    @classmethod
    def supports_compute_capability(cls, capability: DeviceCapability) -> bool:
        return capability.major >= 8

    @classmethod
    def supports_kv_cache_dtype(cls, kv_cache_dtype: CacheDType | None) -> bool:
        return kv_cache_dtype == "turboquant"


# ---------------------------------------------------------------------------
# Impl
# ---------------------------------------------------------------------------


class TurboQuantAttentionImpl(AttentionImpl):
    """TurboQuant attention: decode compressed KV → bf16 → Triton attention."""

    def __init__(
        self,
        num_heads: int,
        head_size: int,
        scale: float,
        num_kv_heads: int,
        alibi_slopes: list[float] | None,
        sliding_window: int | None,
        kv_cache_dtype: str,
        logits_soft_cap: float | None = None,
        attn_type: AttentionType = AttentionType.DECODER,
        kv_sharing_target_layer_name: int | None = None,
        **kwargs,
    ) -> None:
        self.num_heads = num_heads
        self.head_size = head_size
        self.scale = float(scale)
        self.num_kv_heads = num_kv_heads
        if alibi_slopes is not None:
            alibi_slopes = torch.tensor(alibi_slopes, dtype=torch.float32)
        self.alibi_slopes = alibi_slopes
        if sliding_window is None:
            self.sliding_window = (-1, -1)
        else:
            self.sliding_window = (sliding_window - 1, 0)
        self.kv_cache_dtype = kv_cache_dtype
        self.logits_soft_cap = logits_soft_cap or 0
        self.kv_sharing_target_layer_name = kv_sharing_target_layer_name
        self.num_queries_per_kv = self.num_heads // self.num_kv_heads
        self.attn_type = attn_type

    def forward(
        self,
        layer: torch.nn.Module,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        output: torch.Tensor | None = None,
        output_scale: torch.Tensor | None = None,
        output_block_scale: torch.Tensor | None = None,
    ) -> torch.Tensor:
        assert output is not None, "Output tensor must be provided."

        if attn_metadata is None:
            return output.fill_(0)

        assert attn_metadata.use_cascade is False

        num_actual_tokens = attn_metadata.num_actual_tokens
        key_cache, value_cache = kv_cache.unbind(1)

        # Rotated-domain fast path: for single-token decode with 4-bit
        # no-outlier config, skip the Hadamard butterfly entirely by
        # pre-rotating Q and decoding K/V in the rotated domain.
        if hasattr(layer, "_tq_k_state"):
            k_state = layer._tq_k_state
            use_rotated_fastpath = (
                attn_metadata.max_query_len == 1
                and int(k_state.config.bit_width) == 4
                and int(layer._tq_v_state.config.bit_width) == 4
                and (k_state.head_size - k_state.normal_size) == 0
                and not k_state.config.lite_mode
                and self.alibi_slopes is None
                and self.sliding_window == (-1, -1)
            )
            if use_rotated_fastpath:
                return self._forward_turboquant_rotated(
                    query[:num_actual_tokens],
                    key_cache, value_cache,
                    output, attn_metadata, layer,
                )

        # General decode path: decompress all referenced blocks to bf16.
        block_table = attn_metadata.block_table
        if hasattr(layer, "_tq_k_state"):
            block_size = key_cache.shape[1]
            max_blocks_needed = (
                attn_metadata.max_seq_len + block_size - 1
            ) // block_size
            trimmed_bt = block_table[:, :max_blocks_needed]
            key_cache, value_cache, block_table = self._decode_turboquant_cache(
                key_cache, value_cache, layer, trimmed_bt,
                seq_lens=attn_metadata.seq_lens,
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        seq_threshold_3D = attn_metadata.seq_threshold_3D
        num_par_softmax_segments = attn_metadata.num_par_softmax_segments
        softmax_segm_output = attn_metadata.softmax_segm_output
        softmax_segm_max = attn_metadata.softmax_segm_max
        softmax_segm_expsum = attn_metadata.softmax_segm_expsum

        descale_shape = (cu_seqlens_q.shape[0] - 1, key_cache.shape[2])
        mm_prefix_range_tensor = attn_metadata.mm_prefix_range_tensor

        unified_attention(
            q=query[:num_actual_tokens],
            k=key_cache,
            v=value_cache,
            out=output[:num_actual_tokens],
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.scale,
            causal=True,
            alibi_slopes=self.alibi_slopes,
            window_size=self.sliding_window,
            block_table=block_table,
            softcap=self.logits_soft_cap,
            q_descale=None,
            k_descale=layer._k_scale.expand(descale_shape),
            v_descale=layer._v_scale.expand(descale_shape),
            seq_threshold_3D=seq_threshold_3D,
            num_par_softmax_segments=num_par_softmax_segments,
            softmax_segm_output=softmax_segm_output,
            softmax_segm_max=softmax_segm_max,
            softmax_segm_expsum=softmax_segm_expsum,
            output_scale=output_scale,
            mm_prefix_range=mm_prefix_range_tensor,
        )

        return output

    @torch.compiler.disable
    def _decode_turboquant_cache(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Decode only referenced blocks from packed uint8 to bf16.

        Returns compact bf16 caches and remapped block_table.
        Uses block deduplication via torch.unique to avoid decoding
        duplicate/padded blocks. Uses fused Triton kernel for 4-bit.
        """
        k_bits = int(layer._tq_k_state.config.bit_width)
        v_bits = int(layer._tq_v_state.config.bit_width)

        # Deduplicate blocks: prefix caching can make many block ids
        # repeat across requests. Decode each unique block only once.
        # Pass seq_lens to filter out stale/padded block table entries.
        unique_block_ids, new_block_table = self._get_live_turboquant_blocks(
            block_table, seq_lens, key_cache.shape[1],
        )
        num_entries = unique_block_ids.shape[0]

        if layer._tq_k_state.config.lite_mode:
            return self._decode_lite(
                key_cache,
                value_cache,
                layer,
                unique_block_ids,
                num_entries,
                new_block_table,
            )

        if k_bits == 4 and v_bits == 4:
            return self._decode_fused_4bit(
                key_cache,
                value_cache,
                layer,
                unique_block_ids,
                num_entries,
                new_block_table,
            )

        return self._decode_unfused(
            key_cache,
            value_cache,
            layer,
            unique_block_ids,
            num_entries,
            new_block_table,
        )

    def _decode_fused_4bit(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Fused 4-bit decode. Uses CUDA WPH kernel if available."""
        import os

        use_cuda_wph = os.environ.get("TQ_CUDA_WPH", "0") in ("1", "true")
        if use_cuda_wph:
            try:
                from vllm.v1.attention.ops.cuda_turboquant_decode import (
                    cuda_wph_decode_from_slots,
                )

                return self._decode_cuda_wph(
                    key_cache,
                    value_cache,
                    layer,
                    flat_bt,
                    num_entries,
                    new_block_table,
                    cuda_wph_decode_from_slots,
                )
            except Exception:
                pass  # Fall through to Triton

        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_paged_decode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        packed_bytes = math.ceil(normal_size * 4 / 8)

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            decoded = fused_paged_decode(
                cache=cache,
                flat_bt=flat_bt,
                sign_flips=state.sign_flips,
                codebook=state.codebook,
                normal_idx=state.normal_idx,
                outlier_idx=state.outlier_idx,
                head_size=head_size,
                normal_size=normal_size,
                n_outliers=n_outliers,
                packed_bytes=packed_bytes,
            )
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_cuda_wph(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
        decode_fn,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """CUDA warp-per-head decode (no shared memory races)."""
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        packed_bytes = math.ceil(normal_size * 4 / 8)

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            decoded_flat = decode_fn(
                flat_slots=flat,
                sign_flips=state.sign_flips,
                codebook=state.codebook,
                normal_idx=state.normal_idx,
                outlier_idx=state.outlier_idx,
                head_size=head_size,
                normal_size=normal_size,
                n_outliers=n_outliers,
                packed_bytes=packed_bytes,
            )
            decoded_caches.append(
                decoded_flat.reshape(num_entries, block_size, num_kv_heads, head_size)
            )

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_unfused(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Non-fused decode for 2-bit/3-bit/8-bit."""
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_decode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        outlier_byte_count = n_outliers * 2

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            pos = 0
            outlier_vals = None
            if n_outliers > 0:
                outlier_vals = (
                    flat[:, pos : pos + outlier_byte_count]
                    .clone()
                    .view(torch.bfloat16)
                    .reshape(N, n_outliers)
                )
                pos += outlier_byte_count
            flat_packed = flat[:, pos : pos + packed_bytes]
            pos += packed_bytes

            norms = flat[:, pos : pos + 2].clone().view(torch.float16).reshape(N)

            if bits == 4:
                low = flat_packed & 0x0F
                high = (flat_packed >> 4) & 0x0F
                indices = torch.stack([low, high], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 2:
                b0 = flat_packed & 0x03
                b1 = (flat_packed >> 2) & 0x03
                b2 = (flat_packed >> 4) & 0x03
                b3 = (flat_packed >> 6) & 0x03
                indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 3:
                indices = _unpack_3bit_vectorized(
                    flat_packed, normal_size, cache.device
                )
                indices = indices[:N, :normal_size]
            else:
                indices = flat_packed[:, :normal_size]

            indices_3d = indices.reshape(N, 1, normal_size)
            norms_2d = norms.reshape(N, 1)
            normal_decoded = hadamard_turboquant_decode(
                indices_3d,
                norms_2d,
                state.sign_flips,
                state.codebook,
                output_dtype=torch.bfloat16,
            ).reshape(N, normal_size)

            full = torch.empty(N, head_size, dtype=torch.bfloat16, device=cache.device)
            if state.normal_idx is not None and outlier_vals is not None:
                full[:, state.normal_idx] = normal_decoded
                full[:, state.outlier_idx] = outlier_vals
            else:
                full = normal_decoded

            decoded = full.reshape(num_entries, block_size, num_kv_heads, head_size)
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def _decode_lite(
        self,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        layer: torch.nn.Module,
        flat_bt: torch.Tensor,
        num_entries: int,
        new_block_table: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Lite decode: unpack, codebook lookup, scale by norm. No Hadamard."""
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        outlier_byte_count = n_outliers * 2

        decoded_caches = []
        for cache, state in [(key_cache, k_state), (value_cache, v_state)]:
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            _, block_size, num_kv_heads, slot_bytes = cache.shape
            used = cache[flat_bt]
            N = num_entries * block_size * num_kv_heads
            flat = used.reshape(N, slot_bytes)

            # Parse slot layout: [outlier_bytes | packed | norm(2B)]
            pos = 0
            outlier_vals = None
            if n_outliers > 0:
                outlier_vals = (
                    flat[:, pos : pos + outlier_byte_count]
                    .clone()
                    .view(torch.bfloat16)
                    .reshape(N, n_outliers)
                )
                pos += outlier_byte_count
            flat_packed = flat[:, pos : pos + packed_bytes]
            pos += packed_bytes
            norms = flat[:, pos : pos + 2].clone().view(torch.float16).reshape(N)

            # Unpack indices
            if bits == 4:
                low = flat_packed & 0x0F
                high = (flat_packed >> 4) & 0x0F
                indices = torch.stack([low, high], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 2:
                b0 = flat_packed & 0x03
                b1 = (flat_packed >> 2) & 0x03
                b2 = (flat_packed >> 4) & 0x03
                b3 = (flat_packed >> 6) & 0x03
                indices = torch.stack([b0, b1, b2, b3], dim=-1).reshape(N, -1)[
                    :, :normal_size
                ]
            elif bits == 3:
                indices = _unpack_3bit_vectorized(
                    flat_packed, normal_size, cache.device
                )
                indices = indices[:N, :normal_size]
            else:
                indices = flat_packed[:, :normal_size]

            # Codebook lookup + scale by norm (no inverse Hadamard)
            reconstructed = state.codebook[indices.long()]
            normal_decoded = (reconstructed * norms.unsqueeze(-1).float()).to(
                torch.bfloat16
            )

            # Reassemble outlier + normal channels
            full = torch.empty(N, head_size, dtype=torch.bfloat16, device=cache.device)
            if state.normal_idx is not None and outlier_vals is not None:
                full[:, state.normal_idx] = normal_decoded
                full[:, state.outlier_idx] = outlier_vals
            else:
                full = normal_decoded

            decoded = full.reshape(num_entries, block_size, num_kv_heads, head_size)
            decoded_caches.append(decoded)

        return decoded_caches[0], decoded_caches[1], new_block_table

    def do_kv_cache_update(
        self,
        layer: AttentionLayer,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
    ):
        if not hasattr(layer, "_tq_k_state"):
            return

        # Calibrate outlier channels on first REAL batch (not profiling).
        # During profiling/warmup, attn_metadata is None and K/V values
        # come from dummy inputs — calibrating on those produces wrong
        # outlier channels. We detect profiling via the forward context.
        if getattr(layer, "_tq_needs_calibration", False):
            from vllm.forward_context import get_forward_context

            ctx = get_forward_context()
            is_profile = ctx.attn_metadata is None
            if not is_profile:
                num_actual = slot_mapping.shape[0]
                k_flat = key[:num_actual].reshape(-1, key.shape[-1])
                v_flat = value[:num_actual].reshape(-1, value.shape[-1])
                layer._tq_k_state.calibrate_outliers(k_flat)
                layer._tq_v_state.calibrate_outliers(v_flat)
                layer._tq_needs_calibration = False

        self._encode_turboquant_cache(key, value, kv_cache, slot_mapping, layer)

    @torch.compiler.disable
    def _encode_turboquant_cache(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        slot_mapping: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Encode K/V with outlier-aware layout into paged uint8 cache."""
        # Clamp slot_mapping: padding tokens (-1) go to slot 0.
        # This is safe because calibration is skipped during profiling
        # (checked in do_kv_cache_update), and real requests have no
        # -1 entries. Using clamp avoids GPU→CPU sync that would break
        # CUDA graph capture.
        num_actual = slot_mapping.shape[0]
        clamped_slots = slot_mapping.clamp(min=0)

        k_bits = int(layer._tq_k_state.config.bit_width)
        v_bits = int(layer._tq_v_state.config.bit_width)
        block_size = kv_cache.shape[2]
        block_indices = clamped_slots // block_size
        block_offsets = clamped_slots % block_size

        if layer._tq_k_state.config.lite_mode:
            self._encode_lite(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )
        elif k_bits == 4 and v_bits == 4:
            self._encode_fused_4bit(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )
        else:
            self._encode_unfused(
                key[:num_actual],
                value[:num_actual],
                kv_cache,
                block_indices,
                block_offsets,
                layer,
            )

    def _encode_fused_4bit(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Fused encode path for 4-bit: single Triton kernel per K/V."""
        from vllm.v1.attention.ops.triton_fused_turboquant import (
            fused_hadamard_encode_and_store,
        )

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, layer._tq_k_state),
                (value, layer._tq_v_state),
            ]
        ):
            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            cache = kv_cache[:, kv_idx]
            fused_hadamard_encode_and_store(
                normal_x=normal_x,
                outlier_x=outlier_x,
                sign_flips=state.sign_flips,
                boundaries=state.boundaries,
                cache=cache,
                block_indices=block_indices,
                block_offsets=block_offsets,
                bit_width=4,
            )

    def _encode_unfused(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Non-fused encode path for 2-bit/3-bit/8-bit."""
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_encode,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        slot_bytes = kv_cache.shape[-1]
        num_actual = key.shape[0]

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, k_state),
                (value, v_state),
            ]
        ):
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            indices, norms = hadamard_turboquant_encode(
                normal_x, state.sign_flips, state.codebook, state.boundaries
            )

            flat_indices = indices.reshape(-1, normal_size)
            N = flat_indices.shape[0]
            # Pad for 4-bit/2-bit interleaving (3-bit packs internally)
            if bits in (4, 2):
                align = {4: 2, 2: 4}[bits]
                if normal_size % align != 0:
                    pad = align - (normal_size % align)
                    flat_indices = torch.nn.functional.pad(
                        flat_indices, (0, pad), value=0
                    )
            if bits == 4:
                packed = flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)
            elif bits == 2:
                packed = (
                    flat_indices[:, 0::4]
                    | (flat_indices[:, 1::4] << 2)
                    | (flat_indices[:, 2::4] << 4)
                    | (flat_indices[:, 3::4] << 6)
                )
            elif bits == 3:
                packed = _pack_3bit_vectorized(flat_indices, normal_size, packed_bytes)
            else:
                packed = flat_indices[:, :packed_bytes]
            packed = packed[:, :packed_bytes]

            parts = []
            if outlier_x is not None:
                ob = (
                    outlier_x.reshape(N, n_outliers)
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(N, n_outliers * 2)
                )
                parts.append(ob)
            parts.append(packed)
            norm_bytes_data = (
                norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
            )
            parts.append(norm_bytes_data)
            slot_data = torch.cat(parts, dim=-1)

            # Pad to slot_bytes when asymmetric V uses fewer bits than K,
            # leaving unused trailing bytes in the slot.
            actual_bytes = slot_data.shape[-1]
            if actual_bytes < slot_bytes:
                slot_data = torch.nn.functional.pad(
                    slot_data, (0, slot_bytes - actual_bytes), value=0
                )

            num_kv_heads = tensor.shape[1]
            slot_3d = slot_data.reshape(num_actual, num_kv_heads, slot_bytes)
            cache = kv_cache[:, kv_idx]
            cache[block_indices, block_offsets] = slot_3d

    def _encode_lite(
        self,
        key: torch.Tensor,
        value: torch.Tensor,
        kv_cache: torch.Tensor,
        block_indices: torch.Tensor,
        block_offsets: torch.Tensor,
        layer: torch.nn.Module,
    ) -> None:
        """Lite encode path: no rotation, pure scalar quantization.

        Layout per slot (same as standard): [outlier_bytes | packed | norm(2B)]
        No Hadamard, no sign flips — just normalize, quantize, pack, write.
        """
        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        head_size = k_state.head_size
        normal_size = k_state.normal_size
        n_outliers = head_size - normal_size
        slot_bytes = kv_cache.shape[-1]
        num_actual = key.shape[0]

        for kv_idx, (tensor, state) in enumerate(
            [
                (key, k_state),
                (value, v_state),
            ]
        ):
            bits = int(state.config.bit_width)
            packed_bytes = math.ceil(normal_size * bits / 8)

            # Split outlier / normal channels
            if state.outlier_idx is not None:
                normal_x = tensor[..., state.normal_idx].contiguous()
                outlier_x = tensor[..., state.outlier_idx]
            else:
                normal_x = tensor
                outlier_x = None

            # Flatten to (N, normal_size)
            flat = normal_x.reshape(-1, normal_size).float()
            N = flat.shape[0]

            # Compute norms and normalize
            norms = torch.norm(flat, dim=-1, keepdim=True)
            flat_normed = flat / (norms + 1e-16)

            # Scalar quantize (no rotation)
            indices = torch.bucketize(flat_normed.contiguous(), state.boundaries).to(
                torch.uint8
            )

            # Pack indices
            flat_indices = indices.reshape(N, normal_size)
            align = {4: 2, 2: 4, 3: 10}.get(bits, 1)
            if normal_size % align != 0:
                pad = align - (normal_size % align)
                flat_indices = torch.nn.functional.pad(flat_indices, (0, pad), value=0)
            if bits == 4:
                packed = flat_indices[:, 0::2] | (flat_indices[:, 1::2] << 4)
            elif bits == 2:
                packed = (
                    flat_indices[:, 0::4]
                    | (flat_indices[:, 1::4] << 2)
                    | (flat_indices[:, 2::4] << 4)
                    | (flat_indices[:, 3::4] << 6)
                )
            elif bits == 3:
                packed = _pack_3bit_vectorized(flat_indices, normal_size, packed_bytes)
            else:
                packed = flat_indices[:, :packed_bytes]
            packed = packed[:, :packed_bytes]

            # Assemble slot data: [outlier_bytes | packed | norm_bytes]
            parts: list[torch.Tensor] = []
            if outlier_x is not None:
                ob = (
                    outlier_x.reshape(N, n_outliers)
                    .to(torch.bfloat16)
                    .view(torch.uint8)
                    .reshape(N, n_outliers * 2)
                )
                parts.append(ob)
            parts.append(packed)
            norm_bytes_data = (
                norms.reshape(N).to(torch.float16).view(torch.uint8).reshape(N, 2)
            )
            parts.append(norm_bytes_data)
            slot_data = torch.cat(parts, dim=-1)

            # Pad to slot_bytes when asymmetric V uses fewer bits than K
            actual_bytes = slot_data.shape[-1]
            if actual_bytes < slot_bytes:
                slot_data = torch.nn.functional.pad(
                    slot_data, (0, slot_bytes - actual_bytes), value=0
                )

            num_kv_heads = tensor.shape[1]
            slot_3d = slot_data.reshape(num_actual, num_kv_heads, slot_bytes)
            cache = kv_cache[:, kv_idx]
            cache[block_indices, block_offsets] = slot_3d

    # ------------------------------------------------------------------
    # Decode optimizations
    # ------------------------------------------------------------------

    @torch.compiler.disable
    def _get_live_turboquant_blocks(
        self,
        block_table: torch.Tensor,
        seq_lens: torch.Tensor | None,
        block_size: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return unique live block ids and a compact remapped block table.

        Prefix caching can make many block ids repeat across requests,
        and hybrid layouts can leave trailing block slots unused.
        This deduplicates to minimize the number of blocks decoded.
        """
        if seq_lens is None:
            flat_bt = block_table.reshape(-1)
            unique_block_ids, remapped = torch.unique(
                flat_bt, sorted=True, return_inverse=True,
            )
            return unique_block_ids, remapped.reshape(block_table.shape)

        blocks_per_seq = torch.div(
            seq_lens + block_size - 1, block_size, rounding_mode="floor",
        )
        block_positions = torch.arange(
            block_table.shape[1], device=block_table.device,
        )
        valid_mask = block_positions.unsqueeze(0) < blocks_per_seq.unsqueeze(1)
        flat_bt = block_table[valid_mask]
        unique_block_ids, remapped = torch.unique(
            flat_bt, sorted=True, return_inverse=True,
        )
        new_block_table = torch.zeros_like(block_table)
        new_block_table[valid_mask] = remapped.to(block_table.dtype)
        return unique_block_ids, new_block_table

    @torch.compiler.disable
    def _forward_turboquant_rotated(
        self,
        attn_query: torch.Tensor,
        key_cache: torch.Tensor,
        value_cache: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: TritonAttentionMetadata,
        layer: torch.nn.Module,
    ) -> torch.Tensor:
        """Rotated-domain fast path for TurboQuant decode.

        Pre-rotates Q into the TurboQuant domain and decodes K/V
        without the Hadamard butterfly (codebook + norm only). This
        eliminates log2(d) butterfly passes per cache position from
        the inner decode loop.

        Guard conditions (checked by caller):
          - 4-bit K+V, no outliers
          - max_query_len == 1 (pure decode step)
          - No ALiBi, sliding window, sinks, or multimodal prefix
        """
        from vllm.v1.attention.ops.triton_decode_attention import (
            decode_attention_fwd,
        )
        from vllm.v1.attention.ops.triton_hadamard_turboquant import (
            hadamard_turboquant_decode_packed4_cache_rotated,
            hadamard_turboquant_rotate_query,
            hadamard_turboquant_unrotate_output,
        )

        k_state = layer._tq_k_state
        v_state = layer._tq_v_state
        block_table = attn_metadata.block_table
        seq_lens = attn_metadata.seq_lens
        num_actual_tokens = attn_metadata.num_actual_tokens

        packed_bytes = (
            k_state.normal_size * int(k_state.config.bit_width) + 7
        ) // 8
        batch_size, num_query_heads, head_size = attn_query.shape

        # Decode K/V cache in rotated domain (no Hadamard).
        unique_k_ids, new_bt = self._get_live_turboquant_blocks(
            block_table, seq_lens, key_cache.shape[1],
        )
        k_rotated = hadamard_turboquant_decode_packed4_cache_rotated(
            key_cache, unique_k_ids, k_state.codebook,
            head_size=k_state.normal_size,
            packed_offset=0, norm_offset=packed_bytes,
            output_dtype=torch.float16,
        )
        v_rotated = hadamard_turboquant_decode_packed4_cache_rotated(
            value_cache, unique_k_ids, v_state.codebook,
            head_size=v_state.normal_size,
            packed_offset=0, norm_offset=packed_bytes,
            output_dtype=torch.float16,
        )

        # Reshape to paged layout for decode attention kernel.
        block_size = key_cache.shape[1]
        num_entries = unique_k_ids.shape[0]
        num_kv_heads = key_cache.shape[2]
        k_paged = k_rotated.reshape(
            num_entries, block_size, num_kv_heads, head_size,
        )
        v_paged = v_rotated.reshape(
            num_entries, block_size, num_kv_heads, head_size,
        )

        # Pre-rotate Q into TurboQuant domain.
        q_rotated = hadamard_turboquant_rotate_query(
            attn_query, k_state.sign_flips, output_dtype=torch.float16,
        )

        # Decode-only attention in rotated domain.
        num_kv_splits = 4
        decode_output, lse, attn_logits = (
            self._get_turboquant_decode_buffers(
                layer, batch_size, num_query_heads, head_size,
                num_kv_splits, head_size + 1,
                torch.float16, attn_query.device,
            )
        )
        decode_attention_fwd(
            q_rotated, k_paged, v_paged,
            decode_output, lse,
            new_bt, seq_lens, attn_logits,
            num_kv_splits, self.scale,
            page_size=block_size,
            logit_cap=self.logits_soft_cap,
        )

        # Unrotate output back to standard domain.
        decode_output = hadamard_turboquant_unrotate_output(
            decode_output, v_state.sign_flips,
            output_dtype=decode_output.dtype,
        )
        output[:num_actual_tokens].copy_(decode_output.to(output.dtype))
        return output

    def _get_turboquant_decode_buffers(
        self,
        layer: torch.nn.Module,
        batch_size: int,
        num_query_heads: int,
        head_size: int,
        num_kv_splits: int,
        logits_width: int,
        output_dtype: torch.dtype,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get or create persistent decode buffers for TurboQuant.

        Reuses layer-owned buffers across decode steps to avoid
        repeated allocation overhead.
        """
        output_shape = (batch_size, num_query_heads, head_size)
        lse_shape = (batch_size, num_query_heads)
        logits_shape = (
            batch_size, num_query_heads, num_kv_splits, logits_width,
        )

        decode_output = getattr(layer, "_tq_decode_output", None)
        if (
            decode_output is None
            or decode_output.device != device
            or decode_output.dtype != output_dtype
            or any(
                a < b
                for a, b in zip(decode_output.shape, output_shape)
            )
        ):
            decode_output = torch.empty(
                output_shape, dtype=output_dtype, device=device,
            )
            lse = torch.empty(
                lse_shape, dtype=torch.float32, device=device,
            )
            attn_logits = torch.empty(
                logits_shape, dtype=torch.float32, device=device,
            )
            layer._tq_decode_output = decode_output
            layer._tq_lse = lse
            layer._tq_attn_logits = attn_logits
        else:
            lse = layer._tq_lse
            attn_logits = layer._tq_attn_logits

        return (
            decode_output[:batch_size],
            lse[:batch_size],
            attn_logits[:batch_size],
        )
