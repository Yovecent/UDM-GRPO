# ------------------------------------------------------------------------
# Copyright (c) 2024-present, BAAI. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ------------------------------------------------------------------------
"""Exponential Moving Average (EMA) of model updates."""

import copy
import torch


class DummyPeftModel(object):
    """Dummy model class with state dict for Peft."""

    def __init__(self):
        self._state_dict = {}

    def forward(self, *args, **kwargs):
        raise RuntimeError("Peft model does not support forward.")

    def state_dict(self):
        return self._state_dict

    def load_state_dict(self, state_dict, strict=True):
        for k, v in state_dict.items():
            self._state_dict[k].copy_(v) if k in self._state_dict else None


class ModelEMA(torch.nn.Module):
    """Model Exponential Moving Average."""

    def __init__(self, model, decay=0.99, update_every=100, device="gpu"):
        super().__init__()
        self.decay = decay
        self.update_every = update_every
        self.model = DummyPeftModel()
        if hasattr(model, "disable_adapter"):
            from peft import get_peft_model_state_dict
            for k, v in get_peft_model_state_dict(model).items():  # TODO: non-default adapter
                self.model._state_dict[k] = v = v.data.clone().float()
                self.model._state_dict[k] = v.cpu() if device == "cpu" else v
        else:
            self.model = copy.deepcopy(model).eval()
            self.model._apply(lambda t: t.float() if t.requires_grad else t) if decay < 1 else None
            [setattr(p, "requires_grad", False) for p in self.model.parameters()]
            self.model.cpu() if device == "cpu" else None

    def forward(self, *args, **kwargs):
        return self.model(*args, **kwargs)

    @torch.no_grad()
    def update(self, model):
        if isinstance(self.model, DummyPeftModel):
            from peft import get_peft_model_state_dict
            for name, model_v in get_peft_model_state_dict(model).items():
                name = name[len("module."):] if name.startswith("module.") else name
                new_value, ema_v = model_v.data.float(), self.model._state_dict[name]
                value = ema_v.to(device=new_value.device)
                ema_v.copy_(value.mul_(self.decay).add_(new_value, alpha=1 - self.decay))
        else:
            for ema_v, model_v in zip(self.model.parameters(), model.parameters()):
                if not model_v.requires_grad:
                    continue
                new_value = model_v.data.float()
                value = ema_v.to(device=new_value.device)
                ema_v.copy_(value.mul_(self.decay).add_(new_value, alpha=1 - self.decay))
