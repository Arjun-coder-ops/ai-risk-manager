"""
False-positive / false-negative cost model.

These are DISCLOSED, CONFIGURABLE assumptions for a prototype, not real
Razorpay financial figures (which we do not have access to). Defaults are
picked to be directionally reasonable for an INR merchant-payments context
and are documented here so a reader can override them.

FALSE POSITIVE cost = a legitimate merchant-day incorrectly flagged HIGH:
  - manual_review_cost: analyst time to investigate a flagged merchant-day
  - merchant_friction_cost: expected revenue/trust impact if flagging leads
    to a hold, extra verification step, or delayed settlement for a
    legitimate merchant

FALSE NEGATIVE cost = a real fraud-spike day missed entirely:
  - undetected_fraud_loss: expected chargeback/fraud loss left unmitigated
    for that merchant-day if nothing is flagged at all

These are per-merchant-day estimates, deliberately conservative and
labeled as assumptions everywhere they are surfaced (README, API output).
"""

from dataclasses import dataclass


@dataclass
class CostAssumptions:
    manual_review_cost_inr: float = 150.0        # analyst time per flagged merchant-day
    merchant_friction_cost_inr: float = 300.0     # est. friction/trust cost per false alarm
    undetected_fraud_loss_inr: float = 8000.0     # est. unmitigated loss per missed spike-day

    @property
    def false_positive_cost(self) -> float:
        return self.manual_review_cost_inr + self.merchant_friction_cost_inr

    @property
    def false_negative_cost(self) -> float:
        return self.undetected_fraud_loss_inr


def evaluate_cost(tp: int, fp: int, tn: int, fn: int, assumptions: CostAssumptions = None) -> dict:
    assumptions = assumptions or CostAssumptions()
    fp_cost_total = fp * assumptions.false_positive_cost
    fn_cost_total = fn * assumptions.false_negative_cost
    return {
        "assumptions": {
            "manual_review_cost_inr": assumptions.manual_review_cost_inr,
            "merchant_friction_cost_inr": assumptions.merchant_friction_cost_inr,
            "undetected_fraud_loss_inr": assumptions.undetected_fraud_loss_inr,
            "false_positive_cost_per_event_inr": assumptions.false_positive_cost,
            "false_negative_cost_per_event_inr": assumptions.false_negative_cost,
            "disclosure": "These are configurable illustrative assumptions for a prototype, not real Razorpay financial figures.",
        },
        "confusion_matrix": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "total_false_positive_cost_inr": round(fp_cost_total, 2),
        "total_false_negative_cost_inr": round(fn_cost_total, 2),
        "total_estimated_cost_inr": round(fp_cost_total + fn_cost_total, 2),
    }
