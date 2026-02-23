import torch
import torch.nn as nn
import torch.nn.functional as F


class AtttentionProbe(nn.Module):

    def __init__(self, layer_idx: int=-1):
        super().__init__()
        
        self.layer_idx = layer_idx
        
        self.enabled = False
        self._value = None


    def get(self):
        return self._value


    def enable(self, turn_on=True, idx=None):
        if isinstance(idx, int):
            if self.layer_idx == idx:
                self.enabled = turn_on
        elif idx is not None:
            if self.layer_idx in idx:
                self.enabled = turn_on
        else:
            self.enabled = turn_on

    def disable(self, idx=None):
        self.enable(turn_on=False, idx=idx)


    def clear(self, idx=None):
        if isinstance(idx, int):
            if self.layer_idx == idx:
                self._value = None
        elif idx is not None:
            if self.layer_idx in idx:
                self._value = None
        else:
            self._value = None
            

    def forward(self, x, keep_grad=False):
        if self.enabled:
            if keep_grad:
                self._value = x
            else:
                self._value = x.detach()

        return x
    

    @staticmethod
    def call_fn(
        module: nn.Module,
        fn_name: str,
        *args,
        **kwargs
    ):
        returns = []

        for m in module.modules():
            if isinstance(m, AtttentionProbe):

                try:
                    fn = getattr(m, fn_name)
                except:
                    raise ValueError(f"Function {fn_name} not found in AtttentionProbe")

                o = fn(*args, **kwargs)
                returns.append(o)

        out = []
        for o in returns:
            if o is not None:
                out.append(o)

        if len(out) == 0:
            return None
        if len(out) == 1:
            return out[0]
        return out
    