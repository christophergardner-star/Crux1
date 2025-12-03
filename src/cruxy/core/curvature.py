import torch

class CurvatureEstimator:
    """
    Implements curvature estimation via loss-gradient correlation.
    """
    def __init__(self, alpha_c=0.95):
        self.alpha_c = alpha_c
        self.curv_ema = 0.0
        self.prev_loss = None
        self.prev_grad_norm = None

    def update(self, loss, grad_norm):
        """
        Update curvature estimate.
        Eq (15): ct = (Lt - Lt-1) / (||gt-1|| + eps)
        Eq (16): EMA update
        """
        if self.prev_loss is None:
            self.prev_loss = loss
            self.prev_grad_norm = grad_norm
            return 0.0

        # Eq (15)
        eps = 1e-8
        ct = (loss - self.prev_loss) / (self.prev_grad_norm + eps)

        # Eq (16)
        self.curv_ema = self.alpha_c * self.curv_ema + (1 - self.alpha_c) * ct

        # Update state
        self.prev_loss = loss
        self.prev_grad_norm = grad_norm

        return self.curv_ema
