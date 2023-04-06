from torch.optim import SGD
import torch


class OffloadOptimizer(SGD):
    def __init__(self, params, lr, **kwargs):
        super().__init__(params, lr, **kwargs)

    def step(self):
        with torch.no_grad():
            for group in self.param_groups:
                for i in group["params"]:
                    i.add_(i.grad.to(torch.float32), alpha=-group["lr"]).to(
                        torch.float16
                    )
