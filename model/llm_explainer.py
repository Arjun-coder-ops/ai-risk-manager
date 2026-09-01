"""
OPTIONAL LLM explanation layer (Phase 15).

Role: take the risk engine's ALREADY-COMPUTED score + signals (verified
model output) and rephrase them into a merchant-friendly paragraph. The LLM
is a summarizer of verified evidence, never a second opinion on whether the
transaction is fraudulent, and it never sees or infers anything the risk
engine didn't already compute.

Hard constraints enforced in code, not just in the prompt:
  - The prompt includes ONLY the structured signals/score/level already
    produced by RiskEngine -- no raw transaction PII beyond what's already
    in `signals`.
  - The LLM is instructed to restate the given evidence, not add new claims.
  - If the API key is missing, the request times out, or the call errors,
    this falls back to a deterministic templated explanation built directly
    from the structured signals (never blocks the pipeline, never silently
    fabricates).

This module is optional: the core risk engine and dashboard work fully
without it. Set ANTHROPIC_API_KEY to enable it.
"""

import os
import json

TIMEOUT_SECONDS = 8


def _fallback_explanation(result_dict: dict) -> str:
    level = result_dict.get("risk_level")
    signals = result_dict.get("signals") or []
    if not signals or signals[0].get("feature") is None:
        signal_text = "no single signal stood out sharply"
    else:
        signal_text = "; ".join(s["description"] for s in signals)
    return (
        f"Risk level: {level}. Based on verified transaction signals: {signal_text} "
        f"Recommended action: {result_dict.get('recommended_action')}"
    )


def explain_with_llm(result_dict: dict) -> dict:
    """
    Returns {"explanation": str, "source": "llm" | "fallback_template", "error": str|None}
    """
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key or result_dict.get("status") != "ok":
        return {"explanation": _fallback_explanation(result_dict), "source": "fallback_template", "error": None}

    try:
        import anthropic  # imported lazily so the dependency is optional

        client = anthropic.Anthropic(api_key=api_key, timeout=TIMEOUT_SECONDS)
        evidence = json.dumps({
            "risk_score": result_dict.get("risk_score"),
            "risk_level": result_dict.get("risk_level"),
            "signals": result_dict.get("signals"),
            "recommended_action": result_dict.get("recommended_action"),
        })
        system_prompt = (
            "You summarize an already-computed fraud-risk assessment for a merchant. "
            "You must restate ONLY the evidence given to you -- do not invent additional "
            "reasons, do not speculate about intent, do not accuse anyone of fraud beyond "
            "what the evidence states, and do not give instructions related to bypassing "
            "fraud detection or payment security. Keep it to 2-3 plain-language sentences."
        )
        msg = client.messages.create(
            model="claude-sonnet-4-6",
            max_tokens=200,
            system=system_prompt,
            messages=[{"role": "user", "content": f"Verified risk evidence:\n{evidence}"}],
        )
        text = "".join(block.text for block in msg.content if getattr(block, "type", None) == "text")
        return {"explanation": text.strip(), "source": "llm", "error": None}
    except Exception as e:
        return {"explanation": _fallback_explanation(result_dict), "source": "fallback_template", "error": str(e)}
