# ELBO Loss function class
import torch

class ELBO(torch.autograd.Function):
    @staticmethod
    def forward(q_mean, q_var, p_mean, p_var, z):
        det_covQ = torch.product(q_var)
        det_covP = torch.product(p_var)

        # z in N(0, I)
        





    @staticmethod
    def backward(ctx, grad_output):
        input, target = ctx.saved_tensors
        

        q_gradient = 
        grad_input = grad_output * 2 * (input - target) / input.numel()
        return grad_input, None # None for the target gradient