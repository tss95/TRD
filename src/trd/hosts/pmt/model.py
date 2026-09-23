"""The principal PMT recipe: ICL, memory-state PCL and optional dense TRD."""

from __future__ import annotations

import inspect
from typing import Any

import torch
from torch import nn

from trd.config import namespace
from trd.corruption import corrupt_with_reversed_segment
from trd.features import BackboneOutput
from trd.loss import TemporalReversalLoss
from trd.targets import token_labels_from_spans
from trd.hosts.pmt.common import PositionalEncoding
from trd.hosts.pmt.input_block import InputBlock
from trd.hosts.pmt.transformer import TransformerEncoderNeighborhood, TransformerDecoderCausal
from trd.hosts.pmt.memory import StackedPMA
from trd.hosts.pmt.projector import ProjectionHead
from trd.hosts.pmt.icl import InstanceContrastiveLoss
from trd.hosts.pmt.pcl import PMAContrastiveLoss
from trd.hosts.pmt.policy import PMTParameterPolicy


class PMT(PMTParameterPolicy):
    """Keep only the encoder and objectives used in the principal experiments."""

    def __init__(self, config: dict[str, Any]) -> None:
        super().__init__()
        self.config = config
        p = config['encoder']
        s = p['pma']['shared']
        self.embedding_dim = p['d_model']
        self.stride, self.patch = p['tokenizer_stride'], p['tokenizer_patch_size']
        self.trd_weight = config['trd']['weight']
        self.span, self.fade = config['trd']['span_samples'], config['trd']['fade_samples']
        self.positional_encoding = PositionalEncoding(p['d_model'])
        self.conv_encoder1 = InputBlock(
            in_channels=config['channels'],
            d_model=p['d_model'],
            conv_num_kernels=p['tokenizer_num_kernels'],
            conv_kernel_size=self.patch,
            conv_stride=self.stride,
            conv_dropout=p['tokenizer_dropout1'],
            activation=p['input_block_activation'],
            use_conv=p['input_block_use_conv'],
            n_conv_layers=p['input_block_n_conv_layers'],
            auto_transpose=False,
            cfg=namespace({'model': p, 'data_parameters': {'seq_len_inferred': config['data']['window_samples']}}),
            apply_token_pe=p['input_block_apply_pe'],
        )
        self.encoder1 = TransformerEncoderNeighborhood(
            d_model=p['d_model'],
            num_heads=p['nhead'],
            num_layers=p['num_encoder_blocks'],
            d_ff=p['transformer_hidden_dim_factor'] * p['d_model'],
            activation=p['transformer_activation'],
            dropout=p['transformer_dropout1'],
            neighborhood_size=p['transformer_neighborhood_size'],
            norm_type=s['norm_type'],
            norm_position=s['norm_position'],
            dropout_mode='legacy',
            apply_terminal_norm=p['transformer_apply_terminal_norm'],
        )
        accepted = inspect.signature(StackedPMA).parameters
        memory = {key: value for key, value in s.items() if key in accepted}
        memory.update(p['pma']['offsets'])
        memory.update(p['pma']['v5_4'])
        memory.update(
            d_model=p['d_model'],
            num_heads=s['num_heads'] or p['nhead'],
            d_ff=p['transformer_hidden_dim_factor'] * p['d_model'],
            dropout=s['dropout1'],
            window_stride=s['window_size'],
            gate_temperature=s['gate_activation_temperature'],
            gate_init_logit=s['skip_gate_init_logit'],
            pma_aggregator_use_conv_q=s['aggregator_use_conv_q'],
            rope_chunk_offset_on_states=p['pma']['rope_chunk_offset_on_states'],
        )
        self.pma_stack_cnn_1 = StackedPMA(**memory)

        def projector() -> ProjectionHead:
            return ProjectionHead(
                p['d_model'],
                p['proj_hidden_dim'],
                p['proj_output_dim'],
                num_layers=p['num_proj_layers'],
                dropout=p['proj_dropout'],
                activation=p['proj_activation'],
            )

        # These three native projections also preserve construction and dropout order.
        self.proj_head1_cls = projector()
        self.proj_head1_no_cls = projector()
        self.proj_head1_pma_pre = projector()
        self.pcl_proj_head1_pma_pre = nn.ModuleList([projector() for _ in range(s['n_states'])])
        # The source constructs this inactive predictor before initializing TRD.
        # Keep its RNG consumption, without retaining unused parameters.
        TransformerDecoderCausal(
            d_model=p['d_model'],
            num_heads=p['nhead'],
            num_layers=p['num_predictor_blocks'],
            d_ff=p['transformer_hidden_dim_factor'] * p['d_model'],
            activation=p['transformer_activation'],
            dropout=p['transformer_dropout1'],
            dropout_mode='legacy',
        )
        if config['native']['state_predictor_enabled']:
            raise ValueError('State prediction is outside the principal PMT recipe')
        contrast = config['native']['contrast']
        c = contrast['cls_contrastive_loss']
        self.cls_contrastive_loss = InstanceContrastiveLoss(
            temperature=c['temperature'],
            end_temperature=c['end_temperature'],
            schedule_epochs=config['training']['temperature_schedule_epochs'] if c['enable_temp_schedule'] else 0,
            temp_schedule_mode=c['temp_schedule_mode'],
            use_all_gather=c['use_all_gather'],
            use_local_anchors=c.get('use_local_anchors', False),
        )
        pcl = contrast['pma_contrastive_loss']
        self.pcl_cnn = PMAContrastiveLoss(
            **{key: value for key, value in pcl.items() if key in inspect.signature(PMAContrastiveLoss).parameters}
        )
        self.pcl_cnn.set_epoch(
            0, config['training']['temperature_schedule_epochs'] if pcl['enable_temp_schedule'] else 0
        )
        self.trd_loss = TemporalReversalLoss(self.embedding_dim) if self.trd_weight > 0 else None

    def set_epoch(self, epoch: int) -> None:
        """Advance native temperatures at the paper runner's reporting boundaries."""
        self.cls_contrastive_loss.set_epoch(epoch)
        self.pcl_cnn.set_epoch(epoch)

    def _encode_views(self, views: list[torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        stems = [self.conv_encoder1(view.transpose(1, 2)) for view in views]
        tokens = torch.cat([value[0] for value in stems])
        masks = torch.cat([value[1] for value in stems]) if stems[0][1] is not None else None
        context, final, states, keep, _ = self.pma_stack_cnn_1(
            tokens[:, 1:], patch_mask=None if masks is None else masks[:, 1:]
        )
        tokens = torch.cat((tokens[:, :1], context), dim=1)
        if masks is not None:
            masks = torch.cat((masks[:, :1], keep), dim=1)
        tokens = self.encoder1(self.positional_encoding(tokens), pad_mask=masks)
        return tokens, final, states, masks

    def forward(self, x: torch.Tensor, second: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        """Compute losses on weak/strong native views and a reversed weak view."""
        if second is None or second.shape != x.shape:
            raise ValueError('PMT needs two equally shaped BTC native views')
        views = [x, second]
        starts = None
        if self.trd_loss is not None:
            corrupted, starts = corrupt_with_reversed_segment(x, self.span, self.fade)
            views.append(corrupted)
        tokens, final, states, masks = self._encode_views(views)
        first, other = tokens[: x.size(0)], tokens[x.size(0) : 2 * x.size(0)]
        # Preserve the native projector forward order, including inactive heads.
        self.proj_head1_no_cls(first[:, 1:])
        self.proj_head1_no_cls(other[:, 1:])
        c1, c2 = self.proj_head1_cls(first[:, 0]), self.proj_head1_cls(other[:, 0])
        self.proj_head1_pma_pre(final[: x.size(0)].mean(dim=1))
        self.proj_head1_pma_pre(final[x.size(0) : 2 * x.size(0)].mean(dim=1))
        first_states, second_states = states[: x.size(0)], states[x.size(0) : 2 * x.size(0)]
        p1 = torch.stack([head(first_states[:, :, i]) for i, head in enumerate(self.pcl_proj_head1_pma_pre)], dim=2)
        p2 = torch.stack([head(second_states[:, :, i]) for i, head in enumerate(self.pcl_proj_head1_pma_pre)], dim=2)
        icl = self.cls_contrastive_loss(c1, c2)
        pcl = self.pcl_cnn.compute_loss(p1, p2)
        native = self.config['native']['icl_weight'] * icl + self.config['native']['pcl_weight'] * pcl
        auxiliary = native.new_zeros(())
        if self.trd_loss is not None:
            reversed_tokens = tokens[2 * x.size(0) :, 1:]
            labels = token_labels_from_spans(starts, self.span, reversed_tokens.size(1), self.stride, self.patch)
            auxiliary = self.trd_loss(reversed_tokens, labels)
        return {
            'total_loss': native + self.trd_weight * auxiliary,
            'icl_loss': icl,
            'pcl_loss': pcl,
            'trd_loss': auxiliary,
        }

    def encode(self, x: torch.Tensor) -> BackboneOutput:
        """Return clean contextual tokens without corruption or an auxiliary head."""
        tokens, _, states, masks = self._encode_views([x])
        return BackboneOutput(
            tokens=tokens[:, 1:],
            states=states.flatten(1, 2),
            cls=tokens[:, 0],
            mask=None if masks is None else masks[:, 1:],
            stride=self.stride,
        )
