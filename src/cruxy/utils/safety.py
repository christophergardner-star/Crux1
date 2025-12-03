import torch
import math

class SafetyGuard:
    """
    Handles NaN/Inf detection and hyperparameter clamping.
    """
    @staticmethod
    def check_numerical_safety(params, grads):
        """
        Section 5.5.1: Check for NaN/Inf in parameters and gradients.
        """
        for p, g in zip(params, grads):
            if p is not None:
                if torch.isnan(p).any() or torch.isinf(p).any():
                    raise ValueError("NaN/Inf detected in parameters")
            if g is not None:
                if torch.isnan(g).any() or torch.isinf(g).any():
                    raise ValueError("NaN/Inf detected in gradients")

    @staticmethod
    def clamp_hyperparameters(lr, beta1, beta2):
        """
        Section 5.5.2: Hard bounds for hyperparameters.
        """
        # Bounds from white paper
        lr_min, lr_max = 1e-7, 1e-2
        b1_min, b1_max = 0.8, 0.95
        b2_min, b2_max = 0.9, 0.99999

        lr = max(lr_min, min(lr, lr_max))
        beta1 = max(b1_min, min(beta1, b1_max))
        beta2 = max(b2_min, min(beta2, b2_max))
        
        return lr, beta1, beta2
