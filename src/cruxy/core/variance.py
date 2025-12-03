import torch
import math

class VarianceTracker:
    """
    Implements dual-window variance estimation with gamma-norm aggregation.
    Handles both v1.0 (L2 norm) and v2.0 (Gamma norm + History) logic.
    """
    def __init__(
        self, 
        alpha_fast=0.9, 
        alpha_slow=0.999, 
        gamma=1.5, 
        memory_depth=14,
        tau_mem=5.0,
        correction_delta=0.2
    ):
        self.alpha_fast = alpha_fast
        self.alpha_slow = alpha_slow
        self.gamma = gamma
        self.memory_depth = memory_depth
        self.tau_mem = tau_mem
        self.correction_delta = correction_delta
        
        self.var_fast = 0.0
        self.var_slow = 0.0
        self.history = [] # List of past var_fast values

    def update(self, grads, use_gamma_norm=True):
        """
        Update variance estimates based on current gradients.
        grads: List of parameter gradients (tensors)
        """
        # Calculate norm
        if use_gamma_norm:
            # Eq (19): Gamma-norm aggregation
            # sum(|g|^gamma)^(1/gamma)
            total_norm_sq = 0.0
            # For efficiency, we might approximate or do per-tensor
            # To match paper exactly: sum over all parameters d
            # Implementation detail: flattening all grads can be expensive.
            # We'll do a per-tensor aggregation sum.
            sum_pow_gamma = 0.0
            for g in grads:
                if g is not None:
                    sum_pow_gamma += torch.sum(torch.abs(g) ** self.gamma).item()
            
            current_var = sum_pow_gamma ** (1.0 / self.gamma)
            # Note: Paper defines sigma^2_gamma,t. 
            # If the norm is the "variance proxy", we treat it as the signal.
        else:
            # v1.0 L2 norm
            total_norm_sq = 0.0
            for g in grads:
                if g is not None:
                    total_norm_sq += torch.sum(g ** 2).item()
            current_var = total_norm_sq # L2 norm squared is often used as variance proxy

        # Eq (1) & (2) / (20) & (21)
        self.var_fast = self.alpha_fast * self.var_fast + (1 - self.alpha_fast) * current_var
        self.var_slow = self.alpha_slow * self.var_slow + (1 - self.alpha_slow) * current_var

        # Update history for v2.0 memory kernels
        self.history.append(self.var_fast)
        if len(self.history) > self.memory_depth:
            self.history.pop(0)

        return self.var_fast, self.var_slow

    def apply_historical_correction(self):
        """
        Eq (31)-(33): Apply historical bias correction to var_fast.
        """
        if not self.history:
            return self.var_fast

        # Eq (31) Weighted historical average
        numerator = 0.0
        denominator = 0.0
        K = len(self.history)
        
        # History is stored oldest to newest. 
        # Paper says t-k. k=0 is current.
        # history[-1] is t, history[-2] is t-1...
        
        for k in range(K):
            # k goes 0 to K-1
            val = self.history[-(k+1)]
            wk = math.exp(-k / self.tau_mem) # Eq (32)
            numerator += wk * val
            denominator += wk
            
        sigma_hist = numerator / (denominator + 1e-8)
        
        # Eq (33)
        self.var_fast = self.var_fast + self.correction_delta * (sigma_hist - self.var_fast)
        return self.var_fast
