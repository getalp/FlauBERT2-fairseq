# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
import logging
from typing import Optional

import torch
import torch.nn as nn
from torch import Tensor

from fairseq import utils
from fairseq.modules import LayerNorm, MultiheadAttention
from fairseq.modules.fairseq_dropout import FairseqDropout
from fairseq.modules.quant_noise import quant_noise
from fairseq.models import (
    register_model,
    register_model_architecture,
)
from fairseq.models.roberta import (
    RobertaModel,
    RobertaEncoder
)
from fairseq.models.transformer import DEFAULT_MIN_PARAMS_TO_WRAP, TransformerEncoder
from fairseq.modules.transformer_sentence_encoder import init_bert_params
from fairseq.modules.checkpoint_activations import checkpoint_wrapper
from fairseq.distributed import fsdp_wrap

logger = logging.getLogger(__name__)


class FourierTransformerEncoderLayer(nn.Module):

    def __init__(self, args, use_fft):
        super().__init__()
        self.args = args
        self.embed_dim = args.encoder_embed_dim
        self.quant_noise = getattr(args, "quant_noise_pq", 0)
        self.quant_noise_block_size = getattr(args, "quant_noise_pq_block_size", 8) or 8

        self.use_fft = use_fft
        if not self.use_fft:
            self.self_attn = self.build_self_attention(self.embed_dim, args)

        export = getattr(args, "export", False)
        self.self_attn_layer_norm = LayerNorm(self.embed_dim, export=export)
        self.dropout_module = FairseqDropout(
            args.dropout, module_name=self.__class__.__name__
        )
        self.activation_fn = utils.get_activation_fn(
            activation=getattr(args, "activation_fn", "relu") or "relu"
        )
        activation_dropout_p = getattr(args, "activation_dropout", 0) or 0
        if activation_dropout_p == 0:
            # for backwards compatibility with models that use args.relu_dropout
            activation_dropout_p = getattr(args, "relu_dropout", 0) or 0
        self.activation_dropout_module = FairseqDropout(
            float(activation_dropout_p), module_name=self.__class__.__name__
        )
        self.normalize_before = args.encoder_normalize_before
        self.fc1 = self.build_fc(
            self.embed_dim,
            args.encoder_ffn_embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )
        self.fc2 = self.build_fc(
            args.encoder_ffn_embed_dim,
            self.embed_dim,
            self.quant_noise,
            self.quant_noise_block_size,
        )

        self.final_layer_norm = LayerNorm(self.embed_dim, export=export)

    def build_fc(self, input_dim, output_dim, q_noise, qn_block_size):
        return quant_noise(
            nn.Linear(input_dim, output_dim), p=q_noise, block_size=qn_block_size
        )

    def build_self_attention(self, embed_dim, args):
        return MultiheadAttention(
            embed_dim,
            args.encoder_attention_heads,
            dropout=args.attention_dropout,
            self_attention=True,
            q_noise=self.quant_noise,
            qn_block_size=self.quant_noise_block_size,
        )

    def fourier_transform(self, x):
        # Fast fourier transform algorithm from PyTorch does not support half precision.
        mixed_precision = x.dtype is torch.float16
        if mixed_precision:
            x = x.float()
        x = torch.fft.fft2(x, dim=(-1, -2)).real
        return x.half() if mixed_precision else x

    def upgrade_state_dict_named(self, state_dict, name):
        """
        Rename layer norm states from `...layer_norms.0.weight` to
        `...self_attn_layer_norm.weight` and `...layer_norms.1.weight` to
        `...final_layer_norm.weight`
        """
        layer_norm_map = {"0": "self_attn_layer_norm", "1": "final_layer_norm"}
        for old, new in layer_norm_map.items():
            for m in ("weight", "bias"):
                k = "{}.layer_norms.{}.{}".format(name, old, m)
                if k in state_dict:
                    state_dict["{}.{}.{}".format(name, new, m)] = state_dict[k]
                    del state_dict[k]

    def forward(
        self,
        x,
        encoder_padding_mask: Optional[Tensor],
        attn_mask: Optional[Tensor] = None,
    ):
        """
        Args:
            x (Tensor): input to the layer of shape `(seq_len, batch, embed_dim)`
            encoder_padding_mask (ByteTensor): binary ByteTensor of shape
                `(batch, seq_len)` where padding elements are indicated by ``1``.
            attn_mask (ByteTensor): binary tensor of shape `(tgt_len, src_len)`,
                where `tgt_len` is the length of output and `src_len` is the
                length of input, though here both are equal to `seq_len`.
                `attn_mask[tgt_i, src_j] = 1` means that when calculating the
                embedding for `tgt_i`, we exclude (mask out) `src_j`. This is
                useful for strided self-attention.
        Returns:
            encoded output of shape `(seq_len, batch, embed_dim)`
        """
        # anything in original attn_mask = 1, becomes -1e8
        # anything in original attn_mask = 0, becomes 0
        # Note that we cannot use -inf here, because at some edge cases,
        # the attention weight (before softmax) for some padded element in query
        # will become -inf, which results in NaN in model parameters
        if attn_mask is not None:
            attn_mask = attn_mask.masked_fill(attn_mask.to(torch.bool), -1e8)

        residual = x
        if self.normalize_before:
            x = self.self_attn_layer_norm(x)
        if self.use_fft:
            x = self.fourier_transform(x)
        else:
            x, _ = self.self_attn(
                query=x,
                key=x,
                value=x,
                key_padding_mask=encoder_padding_mask,
                need_weights=False,
                attn_mask=attn_mask,
            )
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.self_attn_layer_norm(x)

        residual = x
        if self.normalize_before:
            x = self.final_layer_norm(x)
        x = self.activation_fn(self.fc1(x))
        x = self.activation_dropout_module(x)
        x = self.fc2(x)
        x = self.dropout_module(x)
        x = residual + x
        if not self.normalize_before:
            x = self.final_layer_norm(x)
        return x


class FourierTransformerEncoder(TransformerEncoder):

    def __init__(self, args, dictionary, embed_tokens):
        super().__init__(args, dictionary, embed_tokens)
        assert len(args.fft_layers) == len(self.layers)
        for i in range(len(self.layers)):
            self.layers[i] = self.build_encoder_layer(args, args.fft_layers[i])

    def build_encoder_layer(self, args, use_fft=False):
        layer = FourierTransformerEncoderLayer(args, use_fft)
        checkpoint = getattr(args, "checkpoint_activations", False)
        if checkpoint:
            offload_to_cpu = getattr(args, "offload_activations", False)
            layer = checkpoint_wrapper(layer, offload_to_cpu=offload_to_cpu)
        # if we are checkpointing, enforce that FSDP always wraps the
        # checkpointed layer, regardless of layer size
        min_params_to_wrap = (
            getattr(args, "min_params_to_wrap", DEFAULT_MIN_PARAMS_TO_WRAP)
            if not checkpoint
            else 0
        )
        layer = fsdp_wrap(layer, min_num_params=min_params_to_wrap)
        return layer


class SBOLayer(nn.Module):

    def __init__(self, input_size, hidden_size, activation, export):
        super().__init__()
        self.layer = nn.Linear(input_size, hidden_size)
        self.activ = utils.get_activation_fn(activation)
        self.norm = LayerNorm(hidden_size, export)

    def forward(self, x):
        return self.norm(self.activ(self.layer(x)))


class SBONetwork(nn.Module):

    def __init__(self, input_size, hidden_size, activation, export):
        super().__init__()
        self.layers = nn.ModuleList([
            self.build_sbo_layer(input_size, hidden_size, activation, export),
            self.build_sbo_layer(hidden_size, hidden_size, activation, export)
        ])
        self.layers = nn.Sequential(*self.layers)

    def build_sbo_layer(self, input_size, output_size, activation, export):
        return SBOLayer(input_size, output_size, activation, export)

    def forward(self, x):
        return self.layers(x)


class SBOHead(nn.Module):

    def __init__(self, args, embedding_weights, max_targets=20, position_embedding_size=200):
        super().__init__()

        self.position_embeddings = nn.Embedding(max_targets, position_embedding_size)

        export = getattr(args, "export", False)
        hidden_size = args.encoder_embed_dim
        input_size = hidden_size * 2 + position_embedding_size
        activation = getattr(args, "activation_fn", "relu") or "relu"

        self.mlp_layer_norm = self.build_sbo_network(input_size, hidden_size, activation, export)

        # The output weights are the same as the input embeddings, but there is
        # an output-only bias for each token.
        self.decoder = nn.Linear(
            embedding_weights.size(1),
            embedding_weights.size(0),
            bias=False
        )
        if embedding_weights is not None:
            self.decoder.weight = embedding_weights

        self.bias = nn.Parameter(torch.zeros(embedding_weights.size(0)))
        self.max_targets = max_targets

    def build_sbo_network(self, input_size, hidden_size, activation, export):
        return SBONetwork(input_size, hidden_size, activation, export)

    def forward(self, hidden_states, pairs):
        bs, num_pairs, _ = pairs.size()
        bs, seq_len, dim = hidden_states.size()
        # pair indices: (bs, num_pairs)
        left, right = pairs[:,:, 0], pairs[:, :, 1]
        # (bs, num_pairs, dim)
        left_hidden = torch.gather(hidden_states, 1, left.unsqueeze(2).repeat(1, 1, dim))
        # pair states: bs * num_pairs, max_targets, dim
        left_hidden = left_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)
        right_hidden = torch.gather(hidden_states, 1, right.unsqueeze(2).repeat(1, 1, dim))
        # bs * num_pairs, max_targets, dim
        right_hidden = right_hidden.contiguous().view(bs * num_pairs, dim).unsqueeze(1).repeat(1, self.max_targets, 1)

        # (max_targets, dim)
        position_embeddings = self.position_embeddings.weight
        hidden_states = self.mlp_layer_norm(torch.cat((left_hidden, right_hidden, position_embeddings.unsqueeze(0).repeat(bs * num_pairs, 1, 1)), -1))
        # target scores : bs * num_pairs, max_targets, vocab_size
        target_scores = self.decoder(hidden_states) + self.bias
        return target_scores


class FlaubertEncoder(RobertaEncoder):

    def __init__(self, args, dictionary):
        super().__init__(args, dictionary)

        self.sbo_head = self.build_sbo_head(args)

    def build_sbo_head(self, args):
        return SBOHead(
            args,
            embedding_weights=(
                self.sentence_encoder.embed_tokens.weight
                if not args.untie_weights_roberta
                else None
            )
        )

    def build_encoder(self, args, dictionary, embed_tokens):
        encoder = FourierTransformerEncoder(args, dictionary, embed_tokens)
        encoder.apply(init_bert_params)
        return encoder

    def forward(
        self,
        src_tokens,
        features_only=False,
        return_all_hiddens=False,
        masked_tokens=None,
        pairs=None,
        **unused,
    ):
        """
        Args:
            src_tokens (LongTensor): input tokens of shape `(batch, src_len)`
            features_only (bool, optional): skip LM head and just return
                features. If True, the output will be of shape
                `(batch, src_len, embed_dim)`.
            return_all_hiddens (bool, optional): also return all of the
                intermediate hidden states (default: False).
        Returns:
            tuple:
                - the LM output of shape `(batch, src_len, vocab)`
                - a dictionary of additional data, where 'inner_states'
                  is a list of hidden states. Note that the hidden
                  states have shape `(src_len, batch, vocab)`.
        """
        x, extra = self.extract_features(
            src_tokens, return_all_hiddens=return_all_hiddens
        )
        if not features_only:
            x = self.output_layer(x, masked_tokens=masked_tokens, pairs=pairs)
        return x, extra

    def output_layer(self, features, masked_tokens=None, pairs=None, **unused):
        lm_out = self.lm_head(features, masked_tokens)
        if pairs is not None:
            sbo_out = self.sbo_head(features, pairs)
            return lm_out, sbo_out
        else:
            return lm_out


@register_model("flaubert")
class FlaubertModel(RobertaModel):

    @staticmethod
    def add_args(parser):
        RobertaModel.add_args(parser)

        # Arguments related to FFT and multi-head self-attention
        parser.add_argument(
            "--use-fft",
            action="store_true",
            help="replaces attention layers with fourier transforms"
        )
        parser.add_argument(
            "--fft-layers",
            nargs="*",
            type=int,
            help="indices of fourier layers in case of a mixed Fourier-Attention network",
        )
        parser.add_argument(
            "--attention-layers",
            nargs="*",
            type=int,
            help="indices of multi-head attention layer in case of network hybridized with FFT layers"
        )

    @classmethod
    def build_model(cls, args, task):
        """Build a new model instance."""

        # make sure all arguments are present
        base_architecture(args)

        if not hasattr(args, "max_positions"):
            args.max_positions = args.tokens_per_sample

        encoder = FlaubertEncoder(args, task.source_dictionary)
        return cls(args, encoder)


def fft_layers_parse(use_fft, fft_layers, attention_layers, num_encoder_layers):
    if not use_fft:
        # User should not to use FFT layers, all layers default to Multi Head Attention
        if isinstance(fft_layers, list) and len(fft_layers) > 0:
            # User does not want FFT layers, yet gave indices
            logger.warning("fft_layers is not empty but use_fft is set to False. Multi Head Attention chosen")
        return [False] * num_encoder_layers

    def filter_indices(given_indices, num_encoder_layers, name):
        correct_indices = []
        for index in given_indices:
            if -num_encoder_layers <= index < num_encoder_layers:
                correct_indices.append(index % num_encoder_layers)
            else:
                logger.warning(f"{name}: out of bound index given (index {index} out of {num_encoder_layers} layers) -> index ignored")
        return correct_indices

    # Filter indices so they are in the correct range (negative indices allowed like a regular list)
    fft_layers = filter_indices(fft_layers, num_encoder_layers, "fft-layers")
    attention_layers = filter_indices(attention_layers, num_encoder_layers, "attention-layers")

    if len(fft_layers) == 0 and len(attention_layers) == 0:
        # If neither is given, use Fourier transform by default in every layer
        return [True] * num_encoder_layers

    if ((len(fft_layers) == 0) ^ (len(attention_layers) == 0)): # XOR operation
        # Only one of them is given, should be the normal case for a FNet or Hybrid network

        # The default is FFT if attention_layers is given and vice versa
        default = len(fft_layers) == 0
        use_fft_layers = [default] * num_encoder_layers

        for i in (attention_layers if default else fft_layers):
            # Iterating through the non-default layer list 
            use_fft_layers[i] = not default
        return use_fft_layers

    # Both of them are given: must check compatibility
    if list(sorted(fft_layers + attention_layers)) != list(range(num_encoder_layers)):
        # Incompatibility: either a duplicate, or some layers are not defined either as a FFT or Multi Head Attention.
        raise AssertionError("fft-layers and attention-layers were both given but are not considered compatible: do not form a partition of [0:num_encoder_layers-1]")

    use_fft_layers = [None] * num_encoder_layers
    for i in fft_layers:
        use_fft_layers[i] = True
    for i in attention_layers:
        use_fft_layers[i] = False
    return use_fft_layers


@register_model_architecture("flaubert", "flaubert")
def base_architecture(args):
    args.encoder_layers = getattr(args, "encoder_layers", 24)
    args.encoder_embed_dim = getattr(args, "encoder_embed_dim", 1024)
    args.encoder_ffn_embed_dim = getattr(args, "encoder_ffn_embed_dim", 4096)
    args.encoder_attention_heads = getattr(args, "encoder_attention_heads", 16)

    args.dropout = getattr(args, "dropout", 0.1)
    args.attention_dropout = getattr(args, "attention_dropout", 0.1)
    args.activation_dropout = getattr(args, "activation_dropout", 0.0)
    args.pooler_dropout = getattr(args, "pooler_dropout", 0.0)

    args.max_source_positions = getattr(args, "max_positions", 512)
    args.no_token_positional_embeddings = getattr(
        args, "no_token_positional_embeddings", False
    )

    # BERT has a few structural differences compared to the original Transformer
    args.encoder_learned_pos = getattr(args, "encoder_learned_pos", True)
    args.layernorm_embedding = getattr(args, "layernorm_embedding", False)
    args.encoder_normalize_before = getattr(args, "encoder_normalize_before", True)
    args.no_scale_embedding = getattr(args, "no_scale_embedding", True)
    args.activation_fn = getattr(args, "activation_fn", "gelu")
    args.pooler_activation_fn = getattr(args, "pooler_activation_fn", "tanh")
    args.untie_weights_roberta = getattr(args, "untie_weights_roberta", False)

    # Adaptive input config
    args.adaptive_input = getattr(args, "adaptive_input", False)

    # LayerDrop config
    args.encoder_layerdrop = getattr(args, "encoder_layerdrop", 0.0)
    args.encoder_layers_to_keep = getattr(args, "encoder_layers_to_keep", None)

    # Quantization noise config
    args.quant_noise_pq = getattr(args, "quant_noise_pq", 0)
    args.quant_noise_pq_block_size = getattr(args, "quant_noise_pq_block_size", 8)
    args.quant_noise_scalar = getattr(args, "quant_noise_scalar", 0)

    # R4F config
    args.spectral_norm_classification_head = getattr(
        args, "spectral_norm_classification_head", False
    )

    # FFT config
    args.use_fft = getattr(args, "use_fft", False)
    args.fft_layers = getattr(args, "fft_layers", [])
    args.attention_layers = getattr(args, "attention_layers", [])

    if len(args.fft_layers) != args.encoder_layers or not all(map(lambda x : isinstance(x, bool), args.fft_layers)):
        # If it has not been parsed yet, it shall be parsed now
        args.fft_layers = fft_layers_parse(args.use_fft, args.fft_layers, args.attention_layers, args.encoder_layers)




