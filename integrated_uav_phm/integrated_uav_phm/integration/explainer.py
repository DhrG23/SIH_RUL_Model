"""
explainer.py
=============
Turns one live-monitor step's numbers (health index, OOD score, trend
features, fault-classifier probabilities, RUL estimate) into a
human-readable "what happened and why" explanation.

Two modes, chosen automatically:

  1. LOCAL (default, always available, no API key, no network). Builds the
     explanation directly from the pipeline's own outputs: which raw
     signals are driving the trend/OOD score the most (ranked by
     deviation from the fitted healthy baseline), which fault family the
     classifier's family-level sub-models agree on vs. disagree on, and
     the RUL estimate's spread. This is genuinely explainable -- every
     sentence is traceable to a specific number the pipeline computed --
     not a black box description of a black box.

  2. LLM-ASSISTED (optional). If the `anthropic` package is installed AND
     an `ANTHROPIC_API_KEY` environment variable is set, `explain()` will
     additionally ask Claude to turn the SAME structured facts (not raw
     sensor dumps -- see `_build_facts()`) into a more fluent narrative,
     with the local explanation kept as `facts["local_explanation"]` so
     you can always see the ground truth the LLM was given and compare.
     If the key/package isn't available, this step is silently skipped --
     the local explanation is a complete, correct answer on its own.
"""
from __future__ import annotations

import os
import numpy as np

from feature_adapter import RAW_FEATURE_NAMES


def _rank_contributors(raw_vector, healthy_mean, healthy_std, top_k=3):
    """Which raw signals are furthest (in std units) from the healthy
    baseline right now -- the actual evidence behind a health-index rise."""
    z = (raw_vector - healthy_mean) / (healthy_std + 1e-9)
    order = np.argsort(-np.abs(z))[:top_k]
    return [(RAW_FEATURE_NAMES[i], float(raw_vector[i]), float(z[i])) for i in order]


def _build_facts(unit_id, step_idx, raw_vector, health_val, ood_score,
                  onset_detected, onset_step, healthy_mean, healthy_std,
                  fault_proba=None, fault_classes=None, rul_estimate=None,
                  rul_spread=None):
    facts = {
        "unit_id": unit_id,
        "step_idx": step_idx,
        "health_index": round(health_val, 3),
        "ood_score": round(ood_score, 3),
        "onset_detected": onset_detected,
        "onset_step": onset_step,
        "top_contributors": _rank_contributors(raw_vector, healthy_mean, healthy_std),
    }
    if fault_proba is not None and fault_classes is not None:
        order = np.argsort(-fault_proba)
        facts["fault_ranking"] = [(str(fault_classes[i]), round(float(fault_proba[i]), 3))
                                   for i in order]
    if rul_estimate is not None:
        facts["rul_estimate"] = round(float(rul_estimate), 1)
        facts["rul_spread"] = round(float(rul_spread), 1) if rul_spread is not None else None
    return facts


def _local_explanation(facts) -> str:
    lines = []
    if not facts["onset_detected"]:
        lines.append(
            f"Unit {facts['unit_id']}, step {facts['step_idx']}: health index "
            f"{facts['health_index']:.2f}, OOD score {facts['ood_score']:.2f} -- "
            f"within the persistence-gated healthy band, no onset flagged yet.")
    else:
        lines.append(
            f"Unit {facts['unit_id']}, step {facts['step_idx']}: fault onset was "
            f"flagged at step {facts['onset_step']} (health index now "
            f"{facts['health_index']:.2f}, OOD score {facts['ood_score']:.2f}).")

    top = facts["top_contributors"]
    contrib_str = "; ".join(f"{name}={val:.3g} ({z:+.1f}sigma from healthy baseline)"
                             for name, val, z in top)
    lines.append(f"Signals driving this reading the most: {contrib_str}.")

    if "fault_ranking" in facts:
        best_name, best_p = facts["fault_ranking"][0]
        second = facts["fault_ranking"][1] if len(facts["fault_ranking"]) > 1 else None
        if second and (best_p - second[1]) < 0.15:
            lines.append(
                f"Fault classifier is not confident: leading guess '{best_name}' "
                f"({best_p:.0%}) is close to '{second[0]}' ({second[1]:.0%}) -- "
                f"treat the classification as provisional until more readings arrive.")
        else:
            lines.append(f"Fault classifier's leading hypothesis: '{best_name}' ({best_p:.0%}).")

    if facts.get("rul_estimate") is not None:
        spread = facts.get("rul_spread")
        spread_str = f" +/- {spread:.1f}" if spread else ""
        lines.append(f"Estimated remaining time before failure: {facts['rul_estimate']:.1f}"
                      f"{spread_str} steps (wider spread = less certain; this is an "
                      f"ensemble-of-families estimate, not a single model's guess).")

    return " ".join(lines)


def _try_llm_explanation(facts) -> str | None:
    if not os.environ.get("ANTHROPIC_API_KEY"):
        return None
    try:
        import anthropic
    except ImportError:
        return None
    try:
        client = anthropic.Anthropic()
        prompt = (
            "You are explaining a UAV prognostics-and-health-management alert to a "
            "flight operator. Given ONLY these structured facts computed by the "
            "monitoring pipeline (do not invent any numbers not present here), write "
            "a 2-4 sentence plain-language explanation of what is happening and why, "
            "and what it implies operationally. Facts:\n" + str(facts)
        )
        resp = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=300,
            messages=[{"role": "user", "content": prompt}],
        )
        return "".join(b.text for b in resp.content if getattr(b, "type", "") == "text").strip()
    except Exception as e:
        return f"[LLM explanation unavailable: {e}]"


class AIExplainer:
    """Stateful only in that it remembers each unit's healthy-baseline
    mean/std (needed to say WHICH signals moved); everything else is
    computed fresh per call. `use_llm=True` (default) will opportunistically
    use the Anthropic API when a key is configured, and silently fall back
    to the local explanation otherwise -- see module docstring."""

    def __init__(self, healthy_mean: np.ndarray, healthy_std: np.ndarray, use_llm: bool = True):
        self.healthy_mean = np.asarray(healthy_mean, dtype=float)
        self.healthy_std = np.asarray(healthy_std, dtype=float)
        self.use_llm = use_llm

    def explain(self, unit_id, step_idx, raw_vector, health_val, ood_score,
                onset_detected, onset_step=None, fault_proba=None, fault_classes=None,
                rul_estimate=None, rul_spread=None) -> dict:
        facts = _build_facts(unit_id, step_idx, raw_vector, health_val, ood_score,
                              onset_detected, onset_step, self.healthy_mean, self.healthy_std,
                              fault_proba, fault_classes, rul_estimate, rul_spread)
        local = _local_explanation(facts)
        result = {"facts": facts, "local_explanation": local, "explanation": local}
        if self.use_llm:
            llm_text = _try_llm_explanation(facts)
            if llm_text:
                result["llm_explanation"] = llm_text
                result["explanation"] = llm_text
        return result
