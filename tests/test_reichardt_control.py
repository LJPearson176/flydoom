"""Unit tests for pure synthetic Reichardt motion detector control."""

import pytest
from fly_doom.dynamics.reichardt_control import ReichardtParameters, SyntheticReichardtCorrelator


def test_synthetic_reichardt_motion_dsi():
    params = ReichardtParameters(tau_delay_ms=30.0, dt_ms=1.0)
    emd = SyntheticReichardtCorrelator(params)

    # Test motion at 20 px/s (transit time = 50ms, matching the ~30-50ms delay filter)
    r_pref, r_null, dsi = emd.test_motion(velocity_px_s=20.0, pixel_spacing=1.0, duration_ms=250.0)

    print(f"\nSynthetic Reichardt Control: Preferred={r_pref:.3f}, Null={r_null:.3f}, DSI={dsi:.3f}")

    # The mathematical coincidence mechanism must produce strong directional preference
    assert r_pref > r_null, "Preferred motion response must be strictly greater than null motion response"
    assert dsi > 0.70, f"Synthetic Reichardt detector DSI must exceed 0.70, got {dsi:.3f}"
