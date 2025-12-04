import torch
import math
from torch.optim import Optimizer
from cruxy.controllers import CruxyV1Controller, CruxyV2Controller, MetaCruxyController
from cruxy.utils.metrics import MetricsLogger

# Standalone update functions for torch.compile support
def _newton_schulz(G, steps=5, eps=1e-7):
    """
    Newton-Schulz iteration to compute the zeroth power of a matrix G.
    Used for orthogonalization in Muon optimizer.
    """
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G.bfloat16()
    X /= (X.norm() + eps)
    if G.size(0) > G.size(1):
        X = X.T
    
    for _ in range(steps):
        A = X @ X.T
        B = b * A + c * A @ A
        X = a * X + B @ X
        
    if G.size(0) > G.size(1):
        X = X.T
    return X

def _muon_update_fn(p, grad, exp_avg, lr, momentum, weight_decay, step):
    """
    Muon update function for >=2D parameters.
    """
    # 1. Momentum Update
    exp_avg.mul_(momentum).add_(grad)
    
    # 2. Orthogonalization (Newton-Schulz)
    if step > 0: # Skip first step or handle carefully? Usually fine.
        update = _newton_schulz(exp_avg)
    else:
        update = exp_avg.clone()

    # 3. Weight Decay (Decoupled)
    if weight_decay != 0:
        p.mul_(1 - lr * weight_decay)

    # 4. Apply Update
    # Scale by root-mean-square of parameters for consistent scaling? 
    # Standard Muon scales by max(1, p.size(0)/p.size(1))**0.5 sometimes, 
    # but basic version is just p -= lr * update * scale
    
    # For simplicity in this integration, we use standard LR scaling
    # But Muon often uses a specific scaling factor based on dimensions.
    # We will stick to the core orthogonalization update.
    
    p.add_(update.to(p.dtype), alpha=-lr)

def _lion_update_fn(p, grad, exp_avg, lr, beta1, beta2, weight_decay):
    if weight_decay != 0:
        p.mul_(1 - lr * weight_decay)
        
    update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1).sign()
    exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
    p.add_(update, alpha=-lr)

def _adam_update_fn(p, grad, exp_avg, exp_avg_sq, lr, beta1, beta2, eps, weight_decay, decoupled_wd, use_nesterov, step):
    if weight_decay != 0:
        if decoupled_wd:
            p.mul_(1 - lr * weight_decay)
        else:
            grad = grad.add(p, alpha=weight_decay)
    
    exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
    exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
    
    denom = exp_avg_sq.sqrt().add_(eps)
    
    bias_correction1 = 1 - beta1 ** step
    bias_correction2 = 1 - beta2 ** step
    
    step_size = lr * math.sqrt(bias_correction2) / bias_correction1
    
    if use_nesterov:
        numerator = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
        p.addcdiv_(numerator, denom, value=-step_size)
    else:
        p.addcdiv_(exp_avg, denom, value=-step_size)

class CruxyOptimizer(Optimizer):
    """
    Cruxy Stability Engine Optimizer.
    Drop-in replacement for Adam/AdamW with adaptive stability control.
    """
    def __init__(
        self,
        params,
        mode="meta3",
        lr=1e-3,
        betas=(0.9, 0.999),
        eps=1e-8,
        weight_decay=0.01,
        decoupled_weight_decay=True, # Enable AdamW style by default
        use_nesterov=True, # Enable Nesterov momentum by default
        use_lion=False,    # Enable Lion-style sign update
        use_gc=False,      # Enable Gradient Centralization
        meta_lr_eta=0.05,
        meta_lr_beta2=0.03,
        meta_interval=10,
        gamma_norm=1.5,
        log_metrics=False,
        metrics_path="./cruxy_metrics.jsonl",
        compile=False
    ):
        defaults = dict(
            lr=lr, 
            betas=betas, 
            eps=eps, 
            weight_decay=weight_decay,
            decoupled_weight_decay=decoupled_weight_decay,
            use_nesterov=use_nesterov,
            use_lion=use_lion,
            use_gc=use_gc
        )
        super().__init__(params, defaults)
        
        self.mode = mode
        self.log_metrics = log_metrics
        self.metrics_logger = MetricsLogger(metrics_path) if log_metrics else None
        
        # Setup Update Functions (Compiled or Standard)
        self.compile = compile
        if compile:
            # Advisory: This feature is experimental and untested on clusters
            try:
                self.lion_update = torch.compile(_lion_update_fn)
                self.adam_update = torch.compile(_adam_update_fn)
                self.muon_update = torch.compile(_muon_update_fn)
            except Exception as e:
                print(f"Warning: torch.compile failed (falling back to standard): {e}")
                self.lion_update = _lion_update_fn
                self.adam_update = _adam_update_fn
                self.muon_update = _muon_update_fn
        else:
            self.lion_update = _lion_update_fn
            self.adam_update = _adam_update_fn
            self.muon_update = _muon_update_fn
        
        # Initialize Controller
        if mode == "stability_v1":
            self.controller = CruxyV1Controller(
                lr_base=lr,
                beta2_base=betas[1]
            )
        elif mode == "stability_v2":
            self.controller = CruxyV2Controller(
                lr_base=lr,
                beta1_base=betas[0],
                beta2_base=betas[1],
                gamma=gamma_norm
            )
        elif mode == "meta3":
            inner = CruxyV2Controller(
                lr_base=lr,
                beta1_base=betas[0],
                beta2_base=betas[1],
                gamma=gamma_norm
            )
            self.controller = MetaCruxyController(
                inner_controller=inner,
                meta_lr_eta=meta_lr_eta,
                meta_lr_beta2=meta_lr_beta2,
                meta_interval=meta_interval
            )
        elif mode == "muon":
            # Muon uses a simple controller or just fixed params
            # We reuse V2 controller for basic LR scheduling if needed, 
            # but Muon is often fixed. Let's use V2 for consistency.
            self.controller = CruxyV2Controller(
                lr_base=lr,
                beta1_base=betas[0], # Momentum
                beta2_base=betas[1], # Not used for Muon update but kept for interface
                gamma=gamma_norm
            )
        else:
            raise ValueError(f"Unknown mode: {mode}")

    def state_dict(self):
        """
        Returns the state of the optimizer as a :class:`dict`.
        Includes controller state for Meta3.
        """
        state = super().state_dict()
        # Add controller state
        if hasattr(self, 'controller'):
            # We need to serialize the controller manually or via pickle
            # For simplicity, we'll assume controller has a state_dict or we just pickle it
            # But better to be explicit.
            # Let's implement a simple state dict for controller if it doesn't exist
            # Or just save the attributes we care about.
            
            # For Meta3, we care about: step_count, current_lr, current_beta2, var_history
            if self.mode == "meta3":
                controller_state = {
                    "step_count": self.controller.step_count,
                    "current_lr": self.controller.current_lr,
                    "current_beta2": self.controller.current_beta2,
                    "var_history": self.controller.var_history,
                    # Inner controller state (v2)
                    "inner_var_fast": self.controller.inner.variance_tracker.var_fast,
                    "inner_var_slow": self.controller.inner.variance_tracker.var_slow,
                    "inner_history": self.controller.inner.variance_tracker.history,
                    "inner_curv_ema": self.controller.inner.curvature_estimator.curv_ema,
                    "inner_prev_loss": self.controller.inner.curvature_estimator.prev_loss,
                    "inner_prev_grad_norm": self.controller.inner.curvature_estimator.prev_grad_norm
                }
                state['cruxy_controller'] = controller_state
        
        return state

    def load_state_dict(self, state_dict):
        """
        Loads the optimizer state.
        """
        # Pop controller state before loading standard state
        controller_state = state_dict.pop('cruxy_controller', None)
        
        super().load_state_dict(state_dict)
        
        if controller_state and hasattr(self, 'controller') and self.mode == "meta3":
            self.controller.step_count = controller_state["step_count"]
            self.controller.current_lr = controller_state["current_lr"]
            self.controller.current_beta2 = controller_state["current_beta2"]
            self.controller.var_history = controller_state["var_history"]
            
            # Restore inner components
            self.controller.inner.variance_tracker.var_fast = controller_state["inner_var_fast"]
            self.controller.inner.variance_tracker.var_slow = controller_state["inner_var_slow"]
            self.controller.inner.variance_tracker.history = controller_state["inner_history"]
            
            self.controller.inner.curvature_estimator.curv_ema = controller_state["inner_curv_ema"]
            self.controller.inner.curvature_estimator.prev_loss = controller_state["inner_prev_loss"]
            self.controller.inner.curvature_estimator.prev_grad_norm = controller_state["inner_prev_grad_norm"]

    @torch.no_grad()
    def step(self, closure=None, loss=None):
        """
        Performs a single optimization step.
        
        Args:
            closure (callable, optional): A closure that reevaluates the model and returns the loss.
            loss (float, optional): Current loss value. Required for curvature estimation in v2/meta3.
        """
        loss_val = None
        if closure is not None:
            with torch.enable_grad():
                loss_val = closure()
        
        if loss is not None:
            loss_val = float(loss)
            
        # If loss is still None, we might default to 0.0 but curvature estimation will be broken.
        # We'll use 0.0 and warn if needed, or just proceed.
        current_loss = loss_val if loss_val is not None else 0.0

        # 1. Collect Gradients & Compute Global Norm efficiently
        all_grads = []
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is not None:
                    all_grads.append(p.grad)
        
        if not all_grads:
            return loss_val

        # Vectorized Norm Calculation (Avoids CPU sync per parameter)
        # We stack the norms of individual gradients and compute the norm of that vector.
        # This keeps the calculation on the device (GPU/CPU) until the very end.
        grad_norms = torch.stack([g.detach().norm(2) for g in all_grads])
        grad_norm = torch.norm(grad_norms, 2)
        
        # 3. Controller Step
        # Pass gradients to controller for variance estimation
        # Pass loss and grad_norm for curvature
        if self.mode == "stability_v1":
            # v1 doesn't use loss/grad_norm for curvature
            control_out = self.controller.step(all_grads)
        else:
            # .item() is called inside controller only when needed, or we pass tensor
            # The controller expects float for some logic, but let's see.
            # Passing tensor is better if controller supports it, but for now we sync once here.
            control_out = self.controller.step(all_grads, current_loss, grad_norm.item())
            
        # Extract dynamic hyperparameters
        lr_t = control_out["lr"]
        beta1_t = control_out["beta1"]
        beta2_t = control_out["beta2"]
        
        # 4. Apply Gradient Clipping (if provided by controller)
        if "clip_threshold" in control_out:
            clip_val = control_out["clip_threshold"]
            torch.nn.utils.clip_grad_norm_(
                [p for group in self.param_groups for p in group['params']], 
                clip_val
            )
            
        # 5. Optimizer Update (AdamW-style with Nesterov)
        for group in self.param_groups:
            # Note: We override group['lr'] and betas with controller values
            # But we respect weight_decay
            
            eps = group['eps']
            weight_decay = group['weight_decay']
            decoupled_wd = group.get('decoupled_weight_decay', False)
            use_nesterov = group.get('use_nesterov', False)
            use_lion = group.get('use_lion', False)
            use_gc = group.get('use_gc', False)
            
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                
                # Gradient Centralization (GC)
                # Operates on Conv2D/Conv3D weights (dim > 1)
                if use_gc and grad.dim() > 1:
                    grad.add_(-grad.mean(dim=tuple(range(1, grad.dim())), keepdim=True))

                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    # Only allocate variance buffer if NOT using Lion
                    if not use_lion:
                        state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                # exp_avg_sq might not exist in Lion mode
                exp_avg_sq = state.get('exp_avg_sq') 
                
                state['step'] += 1
                
                # --- MUON MODE ---
                if self.mode == "muon" and p.ndim >= 2:
                    self.muon_update(p, grad, exp_avg, lr_t, beta1_t, beta2_t, weight_decay)
                    continue
                
                # --- LION MODE ---
                if use_lion:
                    self.lion_update(p, grad, exp_avg, lr_t, beta1_t, beta2_t, weight_decay)
                    continue
                
                # --- ADAM/CRUXY MODE ---
                
                # Ensure exp_avg_sq exists for Adam mode
                if exp_avg_sq is None:
                     exp_avg_sq = torch.zeros_like(p, memory_format=torch.preserve_format)
                     state['exp_avg_sq'] = exp_avg_sq

                self.adam_update(p, grad, exp_avg, exp_avg_sq, lr_t, beta1_t, beta2_t, eps, weight_decay, decoupled_wd, use_nesterov, state['step'])

        # 6. Logging
        if self.log_metrics and self.metrics_logger:
            metrics = {
                "step": self.state[list(self.state.keys())[0]]['step'], # Global step
                "loss": current_loss,
                "grad_norm": grad_norm.item(),
                **control_out
            }
            self.metrics_logger.log(metrics)
            
        return loss_val
