import math

class PhaseDetector:
    """
    Calculates phase indicator based on variance estimates.
    """
    @staticmethod
    def compute_phase_v1(var_fast, var_slow):
        """
        Eq (3): phi_t = var_fast / (var_fast + var_slow)
        """
        denom = var_fast + var_slow + 1e-8
        return var_fast / denom

    @staticmethod
    def compute_phase_v2(var_fast, var_slow, curvature_ema):
        """
        Eq (26): phi_t = (var_fast / (var_fast + var_slow)) * (1 + 0.1 * tanh(curv))
        """
        base_phi = PhaseDetector.compute_phase_v1(var_fast, var_slow)
        mod = 1.0 + 0.1 * math.tanh(curvature_ema)
        return base_phi * mod
