"""Unit tests for native-Qwen MTP draft KV accounting (pool_configurator form).

Covers the deploy-branch form of the correctness fix: when EAGLE/STANDALONE
runs with a native in-checkpoint MTP draft (no --speculative-draft-model-path),
the target worker must budget the checkpoint-declared ``mtp_num_hidden_layers``
into the KV pool cell size. Without it the draft KV pool allocation overruns
the static memory envelope on 32 GB cards.

This tests ``pool_configurator._resolve_eagle_draft_num_layers`` directly (the
local deploy form; upstream PR #35566 implements the same fallback in
``spec_aux_hidden_state.py``). The regression cases assert:

1. native MTP (no external draft path) budgets mtp_num_hidden_layers;
2. external draft path is EXPLICITLY excluded before the native-MTP fallback,
   even when the target declares mtp_num_hidden_layers and the external draft's
   inferred layer field is missing/None — the fallback must NOT fire;
3. non-EAGLE/STANDALONE and draft workers never budget via the fallback;
4. architectures=None is tolerated (getattr default);
5. multi-layer native MTP budgets every declared layer.
"""

import sys
from types import SimpleNamespace

import pytest

from sglang.srt.model_executor.pool_configurator import (
    _resolve_eagle_draft_num_layers,
)
from sglang.test.ci.ci_register import register_cpu_ci

register_cpu_ci(est_time=1, suite="base-a-test-cpu")


class _FakeEagleAlgorithm:
    def __init__(self, eagle: bool = True, standalone: bool = False):
        self._eagle = eagle
        self._standalone = standalone

    def is_eagle(self) -> bool:
        return self._eagle

    def is_standalone(self) -> bool:
        return self._standalone

    def is_eagle3(self) -> bool:
        return False


def _kvc(
    *,
    is_draft_worker: bool = False,
    mtp_num_hidden_layers=None,
    draft_path=None,
    algorithm=None,
    architectures=None,
    spec_aux_layers=None,
    full_attention_layer_ids=(0, 4, 8),
) -> SimpleNamespace:
    """Minimal KVCacheConfigurator stand-in for _resolve_eagle_draft_num_layers."""
    return SimpleNamespace(
        is_draft_worker=is_draft_worker,
        spec_algorithm=algorithm or _FakeEagleAlgorithm(),
        spec_aux_config=SimpleNamespace(eagle_draft_num_layers=spec_aux_layers),
        server_args=SimpleNamespace(speculative_draft_model_path=draft_path),
        model_config=SimpleNamespace(
            hf_config=SimpleNamespace(architectures=architectures),
            hf_text_config=SimpleNamespace(mtp_num_hidden_layers=mtp_num_hidden_layers),
        ),
        mambaish_config=SimpleNamespace(
            full_attention_layer_ids=full_attention_layer_ids
        ),
    )


def test_native_mtp_budgets_declared_layers():
    """Native MTP (no external draft): mtp_num_hidden_layers is budgeted."""
    kvc = _kvc(
        mtp_num_hidden_layers=1,
        draft_path=None,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) == 1


def test_native_mtp_multiple_layers():
    """Multi-layer native MTP budgets every declared layer."""
    kvc = _kvc(
        mtp_num_hidden_layers=3,
        draft_path=None,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) == 3


def test_external_draft_path_excludes_native_fallback():
    """REGRESSION: external draft + mtp_num_hidden_layers + None inferred field.

    A Qwen target with mtp_num_hidden_layers AND an explicit
    --speculative-draft-model-path must NOT take the native-MTP fallback, even
    when the external draft's inferred layer field is missing/None. The external
    draft config (loaded elsewhere) owns the layer count; the target's MTP field
    is ignored here.
    """
    kvc = _kvc(
        mtp_num_hidden_layers=7,
        draft_path="/some/external/draft",
        architectures=["Qwen3_5ForConditionalGeneration"],
        spec_aux_layers=None,  # external draft inferred layer field is None
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


def test_external_draft_path_with_populated_spec_aux():
    """External draft with spec_aux already populated: that value wins."""
    kvc = _kvc(
        mtp_num_hidden_layers=7,
        draft_path="/some/external/draft",
        architectures=["Qwen3_5ForConditionalGeneration"],
        spec_aux_layers=12,  # external draft layer count resolved elsewhere
    )
    assert _resolve_eagle_draft_num_layers(kvc) == 12


def test_architectures_none_tolerated():
    """architectures=None must not crash the fallback (getattr default)."""
    kvc = _kvc(
        mtp_num_hidden_layers=1,
        draft_path=None,
        architectures=None,
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


def test_non_qwen_arch_never_falls_back():
    """Non-Qwen3.5 target with mtp field must not take the fallback."""
    kvc = _kvc(
        mtp_num_hidden_layers=1,
        draft_path=None,
        architectures=["SomeOtherForCausalLM"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


def test_no_mtp_field_stays_none():
    """Non-MTP target (no mtp_num_hidden_layers) must not fabricate a draft."""
    kvc = _kvc(
        mtp_num_hidden_layers=None,
        draft_path=None,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


def test_draft_worker_never_budgets_target_mtp():
    """The draft worker itself must not double-count via the target fallback."""
    kvc = _kvc(
        is_draft_worker=True,
        mtp_num_hidden_layers=1,
        draft_path=None,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


def test_standalone_algorithm_also_falls_back():
    """STANDALONE (not just EAGLE) uses the native-MTP fallback."""
    kvc = _kvc(
        mtp_num_hidden_layers=1,
        draft_path=None,
        algorithm=_FakeEagleAlgorithm(eagle=False, standalone=True),
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) == 1


def test_non_speculative_algorithm_never_falls_back():
    """A non-EAGLE/STANDALONE algorithm must not trigger the fallback."""
    algo = _FakeEagleAlgorithm(eagle=False, standalone=False)
    kvc = _kvc(
        mtp_num_hidden_layers=1,
        draft_path=None,
        algorithm=algo,
        architectures=["Qwen3_5ForConditionalGeneration"],
    )
    assert _resolve_eagle_draft_num_layers(kvc) is None


if __name__ == "__main__":
    sys.exit(pytest.main([__file__, "-v"]))
