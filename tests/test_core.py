import torch
import pytest
from cruxy.core.variance import VarianceTracker
from cruxy.core.curvature import CurvatureEstimator
from cruxy.core.phase import PhaseDetector

def test_variance_tracker_v1():
    tracker = VarianceTracker(gamma=2.0)
    grads = [torch.tensor([1.0, 1.0]), torch.tensor([2.0])]
    # L2 norm sq = 1+1 + 4 = 6
    
    v_fast, v_slow = tracker.update(grads, use_gamma_norm=False)
    
    # fast = 0.9*0 + 0.1*6 = 0.6
    assert v_fast == pytest.approx(0.6)
    assert v_slow == pytest.approx(0.006) # 0.999*0 + 0.001*6

def test_variance_tracker_gamma():
    tracker = VarianceTracker(gamma=1.5)
    grads = [torch.tensor([1.0, 1.0])]
    # Gamma norm = (1^1.5 + 1^1.5)^(1/1.5) = 2^(2/3) approx 1.587
    
    v_fast, v_slow = tracker.update(grads, use_gamma_norm=True)
    expected_norm = 2**(1/1.5)
    assert v_fast == pytest.approx(0.1 * expected_norm)

def test_curvature_estimator():
    est = CurvatureEstimator()
    # First step returns 0
    c = est.update(loss=1.0, grad_norm=1.0)
    assert c == 0.0
    
    # Second step
    # ct = (0.5 - 1.0) / (1.0 + 1e-8) = -0.5
    # ema = 0.95*0 + 0.05*(-0.5) = -0.025
    c = est.update(loss=0.5, grad_norm=0.5)
    assert c == pytest.approx(-0.025)

def test_phase_detector():
    # Equilibrium
    phi = PhaseDetector.compute_phase_v1(0.5, 0.5)
    assert phi == pytest.approx(0.5)
    
    # Volatility
    phi = PhaseDetector.compute_phase_v1(0.8, 0.2)
    assert phi == pytest.approx(0.8)
