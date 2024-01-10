import torch.nn as nn
import torch


class EMAHelper(object):
    def __init__(self, mu=0.999):
        self.mu = mu
        # self.model.eval()

    def register(self, module):
        self.model = self.ema_copy(module)
        self.model.requires_grad_(False)

    def update(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        with torch.no_grad():
            for param, ema_param in zip(module.parameters(), self.model.parameters()):
                if param.requires_grad:
                    ema_param.data = (1. - self.mu) * param.data + self.mu * ema_param.data
            for buffer, ema_buffer in zip(module.buffers(), self.model.buffers()):
                ema_buffer.data.copy_(buffer.data)
            
    def ema(self, module):
        if isinstance(module, nn.DataParallel):
            module = module.module
        # we don't update BN's moving mean and var
        for param, ema_param in zip(module.parameters(), self.model.parameters()):
            if param.requires_grad:
                param.data.copy_(ema_param.data)

    def ema_copy(self, module):
        if isinstance(module, nn.DataParallel):
            inner_module = module.module
            module_copy = type(inner_module)(
                inner_module.config).to(inner_module.config.device)
            module_copy.load_state_dict(inner_module.state_dict())
            module_copy = nn.DataParallel(module_copy)
        else:
            module_copy = type(module)(module.config).to(module.config.device)
            module_copy.load_state_dict(module.state_dict())
        # module_copy = copy.deepcopy(module)
        return module_copy

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state_dict, strict=True):
        self.model.load_state_dict(state_dict, strict=strict)
