# SPDX-License-Identifier: MIT
"""Token identity shared by forced-acceptance target and draft forwards."""


def resolve_synthetic_token_id(config) -> int | None:
    """Pick one ordinary vocabulary ID for the entire benchmark run.

    Avoid configured special/stop tokens so a constant stream does not end
    immediately. Resolution is host-only and deterministic across ranks.
    """
    spec = config.speculative_config
    if spec is None or spec.synthetic_acceptance_rates is None:
        return None

    excluded = set()
    configs = (
        config,
        config.hf_config,
        getattr(config, "generation_config", None),
        spec.draft_model_hf_config,
    )
    for source in configs:
        for name in ("bos_token_id", "eos_token_id", "pad_token_id", "stop_token_ids"):
            ids = getattr(source, name, None)
            if ids is not None:
                excluded.update(ids if isinstance(ids, (list, tuple, set)) else [ids])

    vocab_size = min(
        config.hf_config.vocab_size,
        getattr(spec.draft_model_hf_config, "vocab_size", config.hf_config.vocab_size),
    )
    for token_id in range(vocab_size):
        if token_id not in excluded:
            return token_id
    raise ValueError("Forced speculative acceptance needs a non-special token ID.")
