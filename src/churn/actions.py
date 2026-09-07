"""Map per-customer churn drivers to retention actions."""

ACTION_RULES = {
    "Contract=Month-to-month": "Offer discounted 1-year contract",
    "tenure": "Enroll in 90-day onboarding check-in program",
    "InternetService=Fiber optic": "Price/quality review call for fiber plan",
    "PaymentMethod=Electronic check": "Offer autopay discount",
    "TechSupport=No": "Offer free 3-month support/security bundle",
    "OnlineSecurity=No": "Offer free 3-month support/security bundle",
    "MonthlyCharges": "Loyalty discount review",
}
DEFAULT_ACTION = "Personal retention call"


def recommend_action(drivers):
    """Return the action for the first driver (in contribution order) that has a
    rule; fall back to the default when no driver matches."""
    for driver in drivers:
        if driver in ACTION_RULES:
            return ACTION_RULES[driver]
    return DEFAULT_ACTION
