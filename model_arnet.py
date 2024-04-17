import os
import numpy as np
import torch
import MinkowskiEngine as ME
from models.y_module import Enhancer_y
from models.u_module import Enhancer_u
from models.v_module import Enhancer_v

class ARNet(torch.nn.Module):
    def __init__(self, ckpt_y=None, ckpt_u=None, ckpt_v=None):
        super().__init__()
        if ckpt_y is not None:
            self.module_y = Enhancer_y()
            self.module_y = self.load_model(ckpt_y, self.module_y)
        if ckpt_u is not None:
            self.module_u = Enhancer_u()
            self.module_u = self.load_model(ckpt_u, self.module_u)
        if ckpt_v is not None:
            self.module_v = Enhancer_v()
            self.module_v = self.load_model(ckpt_v, self.module_v)

    def load_model(self, ckpt, model):
        ckpt_static = torch.load(ckpt)
        model_dict = model.state_dict()
        pretrained_static_dict = {k: v for k, v in ckpt_static['model'].items() if k in model_dict}
        model_dict.update(pretrained_static_dict)
        model.load_state_dict(model_dict)
        return model

    def forward(self, x, gt):
        if 'module_y' in self._modules: out_set = self.module_y(x, gt)
        if 'module_u' in self._modules: out_set = self.module_u(out_set['out'], gt)
        if 'module_v' in self._modules: out_set = self.module_v(out_set['out'], gt)

        return {'out': out_set['out'],
                'x': x}



if __name__ == '__main__':
    model = ARNet().to('cuda')
    print('params:',sum(param.numel() for param in model.parameters()))
