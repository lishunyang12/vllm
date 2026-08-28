# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Behavior checks for FlashInfer SM120 sparse MLA backend selection."""

from types import SimpleNamespace

import torch

from vllm.config import set_current_vllm_config
from vllm.models.deepseek_v4.nvidia.flashinfer_sparse import (
    _required_sm120_sparse_topk,
)
from vllm.platforms.interface import DeviceCapability
from vllm.utils import flashinfer as fi_utils
from vllm.v1.attention.backends.mla import flashinfer_mla_sparse_sm120 as sm120_impl
from vllm.v1.attention.backends.mla.flashinfer_mla_sparse import (
    FlashInferMLASparseSM120Backend,
)
from vllm.v1.attention.backends.registry import AttentionBackendEnum


def _fake_vllm_config(model_type: str, **text_config) -> SimpleNamespace:
    text_config.setdefault("index_topk", 2048)
    return SimpleNamespace(
        model_config=SimpleNamespace(
            hf_text_config=SimpleNamespace(model_type=model_type, **text_config),
        ),
    )


def test_sm120_backend_uses_dedicated_backend_name() -> None:
    assert FlashInferMLASparseSM120Backend.get_name() == "FLASHINFER_MLA_SPARSE_SM120"
    assert (
        AttentionBackendEnum.FLASHINFER_MLA_SPARSE_SM120.get_class()
        is FlashInferMLASparseSM120Backend
    )


def test_sm120_backend_uses_sparse_mqa_for_prefill() -> None:
    impl_cls = FlashInferMLASparseSM120Backend.get_impl_cls()

    assert impl_cls.is_sparse
    assert not impl_cls.supports_dense_mha_prefill


def test_v32_glm_sm120_backend_accepts_glm_block_size(
    monkeypatch,
) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    with set_current_vllm_config(_fake_vllm_config("glm4_moe")):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=576,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8",
            block_size=256,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_glm53_nope_sm120_backend_accepts_native_shape(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    config = _fake_vllm_config(
        "glm5_next_text",
        index_kpool=4,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    with set_current_vllm_config(config):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert invalid_reasons == []


def test_glm53_nope_sm120_backend_rejects_wrong_topk_capacity(monkeypatch) -> None:
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)

    config = _fake_vllm_config(
        "glm5_next_text",
        index_topk=2044,
        index_kpool=4,
        kv_lora_rank=512,
        qk_rope_head_dim=0,
    )
    with set_current_vllm_config(config):
        invalid_reasons = FlashInferMLASparseSM120Backend.validate_configuration(
            head_size=512,
            dtype=torch.bfloat16,
            kv_cache_dtype="fp8_ds_mla",
            block_size=64,
            use_mla=True,
            has_sink=False,
            use_sparse=True,
            use_mm_prefix=False,
            use_per_head_quant_scales=False,
            device_capability=DeviceCapability(12, 0),
            attn_type="decoder",
        )

    assert any(
        "topk buffer width 2176; got 2048" in reason for reason in invalid_reasons
    )


def test_sm120_dsv4_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_DSV4_DISPATCH=frozenset({(32, 128), (32, 192)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 128)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 192)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(32, 256)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_config(16, 192)

    fi_utils.has_flashinfer_sparse_mla_sm120_config.cache_clear()


def test_sm120_nope_capability_checks_exact_dispatch_shape(monkeypatch) -> None:
    fake_module = SimpleNamespace(
        _DECODE_GLM53_NOPE_DISPATCH=frozenset({(16, 2176), (32, 2176)})
    )
    monkeypatch.setattr(fi_utils, "has_flashinfer_sparse_mla_sm120", lambda: True)
    monkeypatch.setattr(fi_utils, "_get_submodule", lambda _name: fake_module)
    fi_utils.has_flashinfer_sparse_mla_sm120_nope_config.cache_clear()

    assert fi_utils.has_flashinfer_sparse_mla_sm120_nope_config(16, 2176)
    assert fi_utils.has_flashinfer_sparse_mla_sm120_nope_config(32, 2176)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_nope_config(8, 2176)
    assert not fi_utils.has_flashinfer_sparse_mla_sm120_nope_config(16, 2048)

    fi_utils.has_flashinfer_sparse_mla_sm120_nope_config.cache_clear()


def test_sm120_nope_forward_keeps_native_query_and_uses_buffer_width(
    monkeypatch,
) -> None:
    impl = sm120_impl.FlashInferMLASparseSM120Impl.__new__(
        sm120_impl.FlashInferMLASparseSM120Impl
    )
    impl.num_heads = 16
    impl.kv_lora_rank = 512
    impl.qk_nope_head_dim = 256
    impl.qk_rope_head_dim = 0
    impl.is_nope_mla = True
    impl.scale = 0.125
    impl.kv_scale_format = "arbitrary_fp32"
    impl.topk_indices_buffer = torch.zeros((1, 2176), dtype=torch.int32)
    impl._workspace_buffer = torch.empty(1, dtype=torch.uint8)

    monkeypatch.setattr(
        sm120_impl,
        "triton_convert_req_index_to_global_index",
        lambda _req_ids, _block_table, indices, **_kwargs: indices,
    )
    call: dict[str, object] = {}

    def fake_decode(**kwargs):
        call.update(kwargs)
        return kwargs["out"]

    monkeypatch.setattr(
        fi_utils,
        "flashinfer_trtllm_batch_decode_with_kv_cache_mla",
        fake_decode,
    )

    q_nope = torch.zeros((1, 16, 512), dtype=torch.bfloat16)
    q_rope = torch.empty((1, 16, 0), dtype=torch.bfloat16)
    metadata = SimpleNamespace(
        req_id_per_token=torch.zeros(1, dtype=torch.int32),
        block_table=torch.zeros((1, 1), dtype=torch.int32),
        block_size=64,
        topk_tokens=2048,
    )
    output, _ = impl.forward_mqa(
        (q_nope, q_rope),
        torch.zeros((1, 64, 656), dtype=torch.uint8),
        metadata,
        SimpleNamespace(),
    )

    assert output.shape == (1, 16, 512)
    query = call["query"]
    assert isinstance(query, torch.Tensor)
    assert query.shape == (1, 1, 16, 512)
    assert call["qk_rope_head_dim"] == 0
    assert call["sparse_mla_top_k"] == 2176
    assert call["max_seq_len"] == 2176


def test_sm120_dsv4_required_topk_tracks_dspark_width() -> None:
    causal = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=False),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )
    dspark = SimpleNamespace(
        attention_config=SimpleNamespace(use_non_causal=True),
        speculative_config=SimpleNamespace(num_speculative_tokens=5),
    )

    assert _required_sm120_sparse_topk(causal, 128) == 128
    assert _required_sm120_sparse_topk(dspark, 128) == 192
