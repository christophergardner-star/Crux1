import math
from cruxy.controllers.v2_controller import CruxyV2Controller
from cruxy.utils.safety import SafetyGuard

class MetaCruxyController:
    """
    Meta-Cruxy 3.0: Meta-Optimization Framework
    Wraps v2.0 controller but overrides LR and Beta2 scheduling with meta-dynamics.
    """
    def __init__(
        self, 
        inner_controller: CruxyV2Controller,
        meta_lr_eta=0.05,
        meta_lr_beta2=0.03,
        meta_interval=10,
        n_ref=50
    ):
        self.inner = inner_controller
        self.meta_lr_eta = meta_lr_eta
        self.meta_lr_beta2 = meta_lr_beta2
        self.meta_interval = meta_interval
        self.n_ref = n_ref
        
        self.step_count = 0
        self.current_lr = inner_controller.lr_base
        self.current_beta2 = inner_controller.beta2_base
        
        # Reference variance buffer
        self.var_history = [] # Stores var_fast

    def step(self, grads, loss_val, grad_norm):
        self.step_count += 1
        
        # 1. Run Inner Controller (for curvature, variance, beta1, clipping)
        # We ignore the LR and Beta2 returned by v2, as we manage them.
        v2_out = self.inner.step(grads, loss_val, grad_norm)
        
        var_fast = v2_out["var_fast"]
        var_slow = v2_out["var_slow"]
        
        # Update history
        self.var_history.append(var_fast)
        if len(self.var_history) > self.n_ref:
            self.var_history.pop(0)
            
        # 2. Meta-Update Loop (Slow Loop)
        if self.step_count % self.meta_interval == 0 and len(self.var_history) >= 2:
            # Eq (48) Reference Variance
            sigma_ref = sum(self.var_history) / len(self.var_history)
            
            # Eq (49) Normalized Meta-Signal
            # zt = (var_fast - var_ref) / (var_ref + eps)
            zt = (var_fast - sigma_ref) / (sigma_ref + 1e-6)
            
            # Eq (51) LR Meta-Optimization
            # eta_{t+1} = eta_t * exp(-kappa_eta * tanh(zt))
            self.current_lr = self.current_lr * math.exp(-self.meta_lr_eta * math.tanh(zt))
            
            # Eq (52) Momentum Meta-Optimization
            # beta2_{t+1} = beta2_t + kappa_beta2 * tanh(zt) * (1 - beta2_t)
            self.current_beta2 = self.current_beta2 + self.meta_lr_beta2 * math.tanh(zt) * (1 - self.current_beta2)
            
        # 3. Phase-Aware Modulation (Fast Loop modulation on top of meta-params)
        # Eq (53) Phase
        phase_t = var_fast / (var_fast + var_slow + 1e-8)
        
        # Eq (54) Phase-Modulated LR
        lr_final = self.current_lr * (1 + 0.2 * phase_t)
        
        # Eq (55) Phase-Modulated Momentum
        beta2_final = self.current_beta2 * (1 - 0.3 * phase_t)
        
        # 4. Safety Bounds
        lr_final, _, beta2_final = SafetyGuard.clamp_hyperparameters(lr_final, 0.9, beta2_final)
        
        return {
            "lr": lr_final,
            "beta1": v2_out["beta1"], # Use v2's curvature-adaptive beta1
            "beta2": beta2_final,
            "clip_threshold": v2_out["clip_threshold"],
            "phase": phase_t,
            "curvature": v2_out["curvature"],
            "warning": v2_out["warning"],
            "meta_signal_z": (var_fast - (sum(self.var_history)/len(self.var_history))) if self.var_history else 0.0
        }
