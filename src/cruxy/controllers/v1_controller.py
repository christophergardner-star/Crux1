import math
from cruxy.core.variance import VarianceTracker
from cruxy.core.phase import PhaseDetector

class CruxyV1Controller:
    """
    Cruxy v1.0: Baseline Architecture
    """
    def __init__(self, lr_base, beta2_base, kp=0.05, kd=0.02, tau=0.1):
        self.lr_base = lr_base
        self.beta2_base = beta2_base
        self.kp = kp
        self.kd = kd
        self.tau = tau
        
        self.variance_tracker = VarianceTracker(gamma=2.0) # v1 uses L2 norm (gamma=2 effectively)
        self.prev_error = 0.0

    def step(self, grads):
        # Update variance
        # v1 uses L2 norm, so we pass use_gamma_norm=False to our tracker which defaults to L2 sum
        var_fast, var_slow = self.variance_tracker.update(grads, use_gamma_norm=False)
        
        # Phase indicator Eq (3)
        phi_t = PhaseDetector.compute_phase_v1(var_fast, var_slow)
        
        # Variance Residual Eq (4)
        rt = var_fast - var_slow
        
        # PD Control Eq (5)-(8)
        et = rt
        delta_et = et - self.prev_error
        ut = self.kp * et + self.kd * delta_et
        u_bounded = math.tanh(ut / self.tau)
        
        self.prev_error = et
        
        # Learning Rate Modulation Eq (9)
        if phi_t > 0.55:
            lr_factor = (1 - 0.3 * phi_t)
        elif phi_t < 0.45:
            lr_factor = (1 + 0.2 * (1 - phi_t))
        else:
            lr_factor = 1.0
        
        lr_t = self.lr_base * lr_factor
        
        # Momentum Modulation Eq (10)
        beta2_t = self.beta2_base * (1 + 0.15 * u_bounded)
        
        # Clamp beta2 to safe range
        beta2_t = max(0.9, min(beta2_t, 0.9999))
        
        return {
            "lr": lr_t,
            "beta2": beta2_t,
            "beta1": 0.9, # Fixed in v1
            "phase": phi_t,
            "var_fast": var_fast,
            "var_slow": var_slow
        }
