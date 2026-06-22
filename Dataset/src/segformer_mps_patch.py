"""
Runtime patch for SegFormer MPS backward() crash.

Root cause: SegformerAttention and SegformerDecodeHead use .view()/.reshape() after
.transpose()/.permute(), producing non-contiguous tensors. PyTorch MPS backend
cannot run view_as on non-contiguous gradient tensors during backward, raising:
  RuntimeError: view size is not compatible with input tensor's size and stride

Fix: insert .contiguous() before every .reshape() that follows a .transpose() or
.permute(), in both the attention layers AND the decode head.  .contiguous() before
reshape means the backward of reshape is view_as(contiguous tensor), which always works.

Call apply_segformer_mps_patch() BEFORE loading the model.
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def apply_segformer_mps_patch():
    from transformers.models.segformer.modeling_segformer import (
        SegformerAttention,
        SegformerDepthWiseConv,
        SegformerDecodeHead,
        SegformerMLP,
    )

    # ------------------------------------------------------------------ #
    # 1. SegformerAttention: view → contiguous().view() for qkv projections
    # ------------------------------------------------------------------ #
    def patched_attention_forward(
        self,
        hidden_states: torch.Tensor,
        height: int,
        width: int,
        attention_mask=None,
        **kwargs,
    ):
        from transformers.models.segformer.modeling_segformer import (
            ALL_ATTENTION_FUNCTIONS,
            eager_attention_forward,
        )
        from typing import Callable

        input_shape = hidden_states.shape[:-1]
        hidden_shape = (*input_shape, self.num_attention_heads, self.head_dim)

        # Linear outputs are contiguous; reshape then transpose is safe
        query_states = self.q_proj(hidden_states).reshape(hidden_shape).transpose(1, 2)

        kv_hidden_states = hidden_states
        if self.sequence_reduction_ratio > 1:
            kv_hidden_states = self.sequence_reduction(hidden_states, height, width)

        kv_shape = (*kv_hidden_states.shape[:-1], self.num_attention_heads, self.head_dim)
        key_states   = self.k_proj(kv_hidden_states).reshape(kv_shape).transpose(1, 2)
        value_states = self.v_proj(kv_hidden_states).reshape(kv_shape).transpose(1, 2)

        attention_interface: Callable = ALL_ATTENTION_FUNCTIONS.get_interface(
            self.config._attn_implementation, eager_attention_forward
        )

        attn_output, attn_weights = attention_interface(
            self,
            query_states,
            key_states,
            value_states,
            attention_mask,
            dropout=0.0 if not self.training else self.attention_dropout,
            scaling=self.scaling,
            **kwargs,
        )

        # attn_output is already .contiguous() from eager_attention_forward
        flat_shape = (*input_shape, self.num_attention_heads * self.head_dim)
        attn_output = attn_output.reshape(flat_shape).contiguous()
        attn_output = self.o_proj(attn_output)
        return attn_output, attn_weights

    # ------------------------------------------------------------------ #
    # 2. SegformerDepthWiseConv: transpose.view → contiguous().reshape()
    # ------------------------------------------------------------------ #
    def patched_dwconv_forward(self, hidden_states, height, width):
        batch_size, seq_len, num_channels = hidden_states.shape
        # transpose → non-contiguous; must call .contiguous() before reshape
        hidden_states = hidden_states.transpose(1, 2).contiguous()
        hidden_states = hidden_states.reshape(batch_size, num_channels, height, width)
        hidden_states = self.dwconv(hidden_states)
        hidden_states = hidden_states.flatten(2).transpose(1, 2)
        return hidden_states

    # ------------------------------------------------------------------ #
    # 3. SegformerMLP: flatten.transpose → add .contiguous()
    # ------------------------------------------------------------------ #
    def patched_mlp_forward(self, hidden_states: torch.Tensor):
        # flatten(2) is contiguous; transpose makes it non-contiguous
        # linear can handle non-contiguous but the backward view on MPS cannot
        hidden_states = hidden_states.flatten(2).transpose(1, 2).contiguous()
        hidden_states = self.proj(hidden_states)
        return hidden_states

    # ------------------------------------------------------------------ #
    # 4. SegformerDecodeHead: transpose.reshape → add .contiguous()
    # ------------------------------------------------------------------ #
    def patched_decode_head_forward(self, encoder_hidden_states, **kwargs):
        batch_size = encoder_hidden_states[-1].shape[0]

        all_hidden_states = ()
        for encoder_hidden_state, linear_proj in zip(encoder_hidden_states, self.linear_projections):
            if self.config.reshape_last_stage is False and encoder_hidden_state.ndim == 3:
                height = width = int(math.sqrt(encoder_hidden_state.shape[-1]))
                encoder_hidden_state = (
                    encoder_hidden_state.reshape(batch_size, height, width, -1)
                    .permute(0, 3, 1, 2).contiguous()
                )

            height, width = encoder_hidden_state.shape[2], encoder_hidden_state.shape[3]
            encoder_hidden_state = linear_proj(encoder_hidden_state)
            # transpose → non-contiguous; add .contiguous() before reshape
            encoder_hidden_state = encoder_hidden_state.transpose(1, 2).contiguous()
            encoder_hidden_state = encoder_hidden_state.reshape(batch_size, -1, height, width)
            encoder_hidden_state = F.interpolate(
                encoder_hidden_state,
                size=encoder_hidden_states[0].size()[2:],
                mode="bilinear",
                align_corners=False,
            )
            all_hidden_states += (encoder_hidden_state,)

        hidden_states = self.linear_fuse(torch.cat(all_hidden_states[::-1], dim=1))
        hidden_states = self.batch_norm(hidden_states)
        hidden_states = self.activation(hidden_states)
        hidden_states = self.dropout(hidden_states)
        logits = self.classifier(hidden_states)
        return logits

    SegformerAttention.forward       = patched_attention_forward
    SegformerDepthWiseConv.forward   = patched_dwconv_forward
    SegformerMLP.forward             = patched_mlp_forward
    SegformerDecodeHead.forward      = patched_decode_head_forward

    print("[segformer_mps_patch] Patched: SegformerAttention, DepthWiseConv, MLP, DecodeHead "
          "(transpose → .contiguous().reshape() to fix MPS backward view crash)")
