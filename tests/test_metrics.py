from credit_risk_platform.metrics import default_rate


def test_default_rate():
    assert default_rate([0, 0, 1, 0, 1]) == 0.4