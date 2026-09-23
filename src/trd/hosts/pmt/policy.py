"""Parameter membership policies for methods using PMT gate components."""

from __future__ import annotations

from typing import Any

from torch import nn

from trd.hosts.pmt.components import GateModule, SkipGate


class PMTParameterPolicy(nn.Module):
    """Declare PMT optimizer and clipping membership without registering model state.

    Optimizer gates include named state-update/read modules; clipping gates only
    include SkipGate/GateModule. Keep that distinction until a separate policy change.
    """

    def optimizer_parameter_groups(self, optimizer_cfg: Any) -> list[dict[str, Any]]:
        """Return ordered PyTorch groups with the existing PMT LR/decay assignments."""
        max_lr = float(optimizer_cfg.max_lr)
        weight_decay = float(optimizer_cfg.weight_decay)
        gate_lr_mult = float(getattr(optimizer_cfg, "gate_lr_mult", 1.0))
        gate_weight_decay = float(getattr(optimizer_cfg, "gate_weight_decay", 0.0))
        vertical_lr_mult = float(getattr(optimizer_cfg, "vertical_lr_mult", 1.0))

        gate_param_names = set()
        vertical_param_names = set()
        for module_name, module in self.named_modules():
            if isinstance(module, (SkipGate, GateModule)) or module_name.endswith(
                ("state_update_gate", "state_read_gate")
            ):
                for param_name, _ in module.named_parameters(recurse=True):
                    gate_param_names.add(f"{module_name}.{param_name}" if module_name else param_name)
            # Preserve the legacy vertical detector during extraction. Dotted child
            # names do not match recurse=False; changing this would change LR policy.
            for pname, _ in module.named_parameters(recurse=False):
                if pname in {
                    "collapse_proj.weight",
                    "collapse_proj.bias",
                    "expand_proj.weight",
                    "expand_proj.bias",
                    "seed_cond_proj.weight",
                    "seed_cond_proj.bias",
                    "init_head_bank",
                }:
                    vertical_param_names.add(f"{module_name}.{pname}" if module_name else pname)

        base_params, gate_params, vertical_params = [], [], []
        for name, param in self.named_parameters():
            if not param.requires_grad:
                continue
            if name in gate_param_names:
                gate_params.append(param)
            elif name in vertical_param_names:
                vertical_params.append(param)
            else:
                base_params.append(param)

        groups = [{"params": base_params, "lr": max_lr, "weight_decay": weight_decay}]
        if gate_params:
            groups.append({"params": gate_params, "lr": max_lr * gate_lr_mult, "weight_decay": gate_weight_decay})
        if vertical_params:
            groups.append({"params": vertical_params, "lr": max_lr * vertical_lr_mult, "weight_decay": weight_decay})
        return groups

    def gradient_clip_parameters(self) -> tuple[list[nn.Parameter], list[nn.Parameter]]:
        """Return legacy gate/base clipping lists, preserving traversal and frozen entries."""
        gate_params = []
        for module in self.modules():
            if isinstance(module, (SkipGate, GateModule)):
                gate_params.extend(module.parameters(recurse=True))
        gate_param_ids = {id(param) for param in gate_params}
        base_params = [param for param in self.parameters() if id(param) not in gate_param_ids]
        return gate_params, base_params
