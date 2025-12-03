import math
import torch
from cruxy.core.variance import VarianceTracker
from cruxy.core.curvature import CurvatureEstimator
from cruxy.core.phase import PhaseDetector

class CruxyV2Controller:
    """
    Cruxy v2.0: Curvature-Adaptive Enhancement
    """
    def __init__(
        self, 
        lr_base, 
        beta1_base=0.9,
        beta2_base=0.999,
        kp=0.05, 
        kd=0.02,
        gamma=1.5
    ):
        self.lr_base = lr_base
        self.beta1_base = beta1_base
        self.beta2_base = beta2_base
        self.kp = kp
        self.kd = kd
        
        self.variance_tracker = VarianceTracker(gamma=gamma, memory_depth=14)
        self.curvature_estimator = CurvatureEstimator()
        
        self.prev_error = 0.0
        self.prev_lr = lr_base
        
        # Early warning state
        self.warning_history = []

    def step(self, grads, loss_val, grad_norm):
        # 1. Curvature Estimation
        curv_ema = self.curvature_estimator.update(loss_val, grad_norm)
        
        # 2. Variance Update (Gamma Norm)
        var_fast, var_slow = self.variance_tracker.update(grads, use_gamma_norm=True)
        
        # 3. Historical Bias Correction Eq (33)
        var_fast = self.variance_tracker.apply_historical_correction()
        
        # 4. Phase Detection Eq (26)
        phi_t = PhaseDetector.compute_phase_v2(var_fast, var_slow, curv_ema)
        
        # 5. Control Signal
        rt = var_fast - var_slow
        et = rt
        delta_et = et - self.prev_error
        
        # Temperature Modulation Eq (34)
        if var_fast > var_slow:
            at = 0.15 # a_up
        elif var_fast < var_slow:
            at = 0.08 # a_down
        else:
            at = 0.1
            
        ut = math.tanh((self.kp * et + self.kd * delta_et) / at)
        self.prev_error = et
        
        # 6. Hyperparameter Modulation
        
        # Beta1 Eq (17)
        # beta1 = base * (1 + 0.15 * tanh((c - 0.01)/0.05))
        beta1_t = self.beta1_base * (1 + 0.15 * math.tanh((curv_ema - 0.01) / 0.05))
        beta1_t = max(0.85, min(beta1_t, 0.95)) # Bounds Eq (18)
        
        # Beta2 Eq (32 in algo summary, Eq 10 in text)
        beta2_t = self.beta2_base * (1 + 0.15 * ut)
        beta2_t = max(0.9, min(beta2_t, 0.9999))
        
        # LR Scheduling Eq (27)-(29)
        phi_thresh = 0.55
        lambda_decay = 0.15
        lambda_recovery = 0.08
        
        if phi_t > phi_thresh:
            # Decay
            lr_t = self.prev_lr * math.exp(-lambda_decay * max(0, phi_t - phi_thresh))
        elif phi_t < (phi_thresh - 0.05):
            # Recovery
            lr_t = self.prev_lr * (1 + lambda_recovery * max(0, phi_thresh - phi_t))
        else:
            lr_t = self.prev_lr
            
        self.prev_lr = lr_t
        
        # 7. Predictive Gradient Clipping Eq (24)
        theta_base = 1.0
        xi = 0.5
        theta_clip = theta_base * (1 + xi * math.sqrt(var_fast / (var_slow + 1e-8)))
        
        # 8. Early Warning Signal (Simplified)
        # Eq (40) Isurge
        i_surge = 1.0 if (var_fast / (var_slow + 1e-8)) > 2.5 else 0.0
        # Eq (42) Icurvature
        i_curv = 1.0 if abs(curv_ema) > 0.1 else 0.0
        
        # Composite Wt
        wt = 0.5 * i_surge + 0.2 * i_curv # Ignoring trend for simplicity
        
        # Emergency Intervention Eq (44)-(45)
        if wt > 0.7:
            lr_t *= 0.5
            beta2_t = min(beta2_t + 0.01, 0.9999)
            self.prev_lr = lr_t # Update state
            
        return {
            "lr": lr_t,
            "beta1": beta1_t,
            "beta2": beta2_t,
            "clip_threshold": theta_clip,
            "phase": phi_t,
            "curvature": curv_ema,
            "warning": wt,
            "var_fast": var_fast,
            "var_slow": var_slow
        }
