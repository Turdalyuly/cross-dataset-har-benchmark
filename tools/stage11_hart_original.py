"""Original corrected HART classes from Stage 5E.3 session 194 line 14.

Class declarations are copied verbatim; training cell side effects removed.
Requires the original tsfm_public installation when imported.
"""
from dataclasses import dataclass
from typing import Optional
import torch
from torch import nn
from tsfm_public.models.tspulse import TSPulseForClassification

@dataclass
class HARTOutput:
    loss: Optional[torch.Tensor] = None
    logits: Optional[torch.Tensor] = None
    prediction_outputs: Optional[torch.Tensor] = None

    def __iter__(self):
        yield self.loss
        yield self.logits

    def __getitem__(self, idx):
        return (
            self.loss,
            self.logits,
        )[idx]

class HARTForHAR(nn.Module):

    def __init__(
        self,
        checkpoint_path,
        num_classes,
        class_weights,
    ):

        super().__init__()


        overrides = {

            "head_gated_attention_activation":
                "softmax",

            "channel_virtual_expand_scale":
                1,

            "mask_ratio":
                0.3,

            "head_reduce_d_model":
                1,

            "disable_mask_in_classification_eval":
                True,

            "fft_time_consistent_masking":
                True,

            "decoder_mode":
                "common_channel",

            "head_aggregation_dim":
                "patch",

            "head_aggregation":
                None,

            "num_input_channels":
                3,

            "num_targets":
                num_classes,

            "loss":
                "cross_entropy",

            "ignore_mismatched_sizes":
                True,
        }


        self._model = (
            TSPulseForClassification
            .from_pretrained(
                str(
                    checkpoint_path
                ),
                **overrides
            )
        )


        # ----------------------------------------------------
        # Freeze entire backbone
        # ----------------------------------------------------

        for p in (
            self._model
            .backbone
            .parameters()
        ):

            p.requires_grad = False


        # ----------------------------------------------------
        # Official HART adaptation:
        # unfreeze time-domain encoding
        # ----------------------------------------------------

        for p in (
            self._model
            .backbone
            .time_encoding
            .parameters()
        ):

            p.requires_grad = True


        # ----------------------------------------------------
        # Official HART adaptation:
        # unfreeze FFT encoding
        # ----------------------------------------------------

        for p in (
            self._model
            .backbone
            .fft_encoding
            .parameters()
        ):

            p.requires_grad = True


        # Classification/decoder head was never frozen.


        self.register_buffer(
            "class_weights",
            class_weights.clone()
        )


        cfg = (
            self._model
            .config
        )


        print(
            "\n[HART]"
        )

        print(
            " context_length:",
            cfg.context_length
        )

        print(
            " d_model:",
            cfg.d_model
        )

        print(
            " patch_length:",
            cfg.patch_length
        )

        print(
            " patch_stride:",
            cfg.patch_stride
        )

        print(
            " channels:",
            cfg.num_input_channels
        )

        print(
            " targets:",
            cfg.num_targets
        )


    def forward(
        self,
        past_values,
        target_values=None,
        past_observed_mask=None,
        **kwargs,
    ):

        enable_masking = True


        if (
            self
            ._model
            .config
            .disable_mask_in_classification_eval
        ):

            enable_masking = (
                self.training
            )


        model_output = (
            self._model
            .backbone(
                past_values,

                past_observed_mask=
                    past_observed_mask,

                output_hidden_states=
                    False,

                return_dict=
                    True,

                enable_masking=
                    enable_masking,
            )
        )


        encoder_output = (
            model_output
            .last_hidden_state
        )


        decoder_input = (
            encoder_output
            .flatten(
                start_dim=2
            )
        )


        decoder_output = (
            self._model
            .decoder_with_head(

                decoder_input=
                    decoder_input,

                loc=
                    model_output.loc,

                scale=
                    model_output.scale,

                output_hidden_states=
                    False,

                return_dict=
                    True,
            )
        )


        logits = (
            decoder_output
            .prediction_outputs
        )


        loss = None


        if target_values is not None:

            loss = nn.functional.cross_entropy(
                logits,
                target_values.long(),
                weight=self.class_weights,
            )


        return HARTOutput(
            loss=loss,
            logits=logits,
            prediction_outputs=
                logits,
        )
