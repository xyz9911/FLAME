from typing import Optional

import torch
import torch.nn as nn
from einops import rearrange, repeat
from transformers import LlamaTokenizer
from llm_nav.model.clip_xformer import CLIPVisionModel
from llm_nav.model.llama_xformer import LlamaForCausalLM
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.modeling_utils import PreTrainedModel
from transformers.generation.configuration_utils import GenerationConfig
from transformers.generation.logits_process import LogitsProcessorList


from llm_nav.config import FlamingoConfig

__KNOWN_DECODER_LAYERS_ATTR_NAMES = {
    "opt": "model.decoder.layers",
    "gptneo": "transformer.h",
    "gptj": "transformer.h",
    "gpt-j": "transformer.h",
    "pythia": "gpt_neox.layers",
    "llama": "model.layers",
    "RWForCausalLM": "transformer.h",
    "MptForCausalLM": "transformer.blocks",
    "MosaicGPT": "transformer.blocks",
}

MODEL_CLASSES = {
    "LlamaForCausalLM": "llama",
    "OPTForCausalLM": "opt",
    "GPTJForCausalLM": "gptj",
    "GPTNeoXForCausalLM": "gpt_neox",
    "MPTForCausalLM": "mpt",
    "MosaicGPT": "mpt",
}


def _infer_decoder_layers_attr_name(model: nn.Module):
    for k in __KNOWN_DECODER_LAYERS_ATTR_NAMES:
        if k.lower() in model.__class__.__name__.lower():
            return __KNOWN_DECODER_LAYERS_ATTR_NAMES[k]

    raise ValueError(
        f"We require the attribute name for the nn.ModuleList in the decoder storing the transformer block layers. Please supply this string manually."
    )


def extend_instance(obj, mixin):
    """Apply mixins to a class instance after creation"""
    base_cls = obj.__class__
    base_cls_name = obj.__class__.__name__
    obj.__class__ = type(base_cls_name, (mixin, base_cls),
                         {})  # mixin needs to go first for our forward() logic to work


def getattr_recursive(obj, att):
    """
    Return nested attribute of obj
    Example: getattr_recursive(obj, 'a.b.c') is equivalent to obj.a.b.c
    """
    if att == "":
        return obj
    i = att.find(".")
    if i < 0:
        return getattr(obj, att)
    else:
        return getattr_recursive(getattr(obj, att[:i]), att[i + 1:])


def setattr_recursive(obj, att, val):
    """
    Set nested attribute of obj
    Example: setattr_recursive(obj, 'a.b.c', val) is equivalent to obj.a.b.c = val
    """
    if "." in att:
        obj = getattr_recursive(obj, ".".join(att.split(".")[:-1]))
    setattr(obj, att.split(".")[-1], val)


def exists(val):
    return val is not None


class FlamingoPerceiverBlock(nn.Module):
    def __init__(self, *, dim: int, dim_head: int = 64, heads: int = 8, mult: int = 4):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads
        ff_dim = dim * mult
        self.norm_media = nn.LayerNorm(dim)
        self.norm_latents = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, ff_dim, bias=False),
                nn.GELU(),
                nn.Linear(ff_dim, dim, bias=False),
            ]
        )

    def forward(self, x: torch.Tensor, latents: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, n1, D)
            latent (torch.Tensor): latent features
                shape (b, T, n2, D)
        """
        x = self.norm_media(x)
        residual_latents = latents
        latents = self.norm_latents(latents)

        h = self.heads

        q = self.to_q(latents)
        kv_input = torch.cat((x, latents), dim=-2)
        k, v = self.to_kv(kv_input).chunk(2, dim=-1)
        q = rearrange(q, "b t n (h d) -> b h t n d", h=h)
        k = rearrange(k, "b t n (h d) -> b h t n d", h=h)
        v = rearrange(v, "b t n (h d) -> b h t n d", h=h)
        q = q * self.scale

        # attention
        sim = torch.einsum("... i d, ... j d  -> ... i j", q, k)
        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h t n d -> b t n (h d)", h=h)
        out = self.to_out(out) + residual_latents
        residual_out = out
        for layer in self.feed_forward:
            out = layer(out)
        return out + residual_out


class FlamingoPerceiverResampler(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            depth: int = 6,
            dim_head: int = 64,
            heads: int = 8,
            num_latents: int = 64,
            max_num_media: Optional[int] = None,
            max_num_frames: Optional[int] = 128,
            ff_mult: int = 4,
            use_frame_embs: bool = True,
    ):
        super().__init__()
        self.latents = nn.Parameter(torch.randn(num_latents, dim))
        self.frame_embs = nn.Parameter(torch.randn(max_num_frames, dim))
        self.use_frame_embs = use_frame_embs

        self.media_time_embs = nn.Parameter(torch.randn(max_num_media, 1, dim)) if exists(max_num_media) else None

        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(FlamingoPerceiverBlock(dim=dim, dim_head=dim_head, heads=heads, mult=ff_mult))

        self.norm = nn.LayerNorm(dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): image features
                shape (b, T, F, v, D)
        Returns:
            shape (b, T, n, D) where n is self.num_latents
        """
        b, T, F, v = x.shape[:4]

        # frame and media time embeddings
        if self.use_frame_embs:
            frame_embs = repeat(self.frame_embs[:T], "T d -> b T F v d", b=b, F=F, v=v)
            x = x + frame_embs
        x = rearrange(x, "b T F v d -> b T (F v) d")  # flatten the frame and spatial dimensions
        if exists(self.media_time_embs):
            x = x + self.media_time_embs[:T]

        # blocks
        latents = repeat(self.latents, "n d -> b T n d", b=b, T=T)
        for block in self.layers:
            latents = block(x, latents)
        return self.norm(latents)


class FlamingoMaskedCrossAttention(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            dim_visual: int,
            dim_head: int = 64,
            heads: int = 8,
            only_attend_immediate_media: bool = True,
            stride: int = 1,
    ):
        super().__init__()
        self.scale = dim_head ** -0.5
        self.heads = heads
        inner_dim = dim_head * heads

        self.norm = nn.LayerNorm(dim)

        self.to_q = nn.Linear(dim, inner_dim, bias=False)
        self.to_kv = nn.Linear(dim_visual, inner_dim * 2, bias=False)
        self.to_out = nn.Linear(inner_dim, dim, bias=False)

        # whether for text to only attend to immediate preceding image, or all previous images
        self.only_attend_immediate_media = only_attend_immediate_media
        self.stride = stride

    def forward(
            self,
            x: torch.Tensor,
            media: torch.Tensor,
            media_locations: Optional[torch.BoolTensor] = None,
            attend_previous: bool = True,
            trunc_locations: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        """
        Args:
            x (torch.Tensor): text features
                shape (B, T_txt, D_txt)
            media (torch.Tensor): image features
                shape (B, T_img, n, D_img) where n is the dim of the latents
            media_locations: boolean mask identifying the media tokens in x
                shape (B, T_txt)
            attend_previous: bool
                If false, ignores immediately preceding image and starts attending when following image
            trunc_locations: boolean mask identifying the media sessions in x
                shape (B, T_txt)
        """
        _, T_img, n = media.shape[:3]
        h = self.heads

        x = self.norm(x)

        q = self.to_q(x)
        media = rearrange(media, "b t n d -> b (t n) d")

        k, v = self.to_kv(media).chunk(2, dim=-1)
        q = rearrange(q, "b n (h d) -> b h n d", h=h)
        k = rearrange(k, "b n (h d) -> b h n d", h=h)
        v = rearrange(v, "b n (h d) -> b h n d", h=h)

        q = q * self.scale

        sim = torch.einsum("... i d, ... j d -> ... i j", q, k)

        if exists(media_locations):
            # at each boolean of True, increment the time counter (relative to media time)
            text_time = media_locations.cumsum(dim=-1)
            media_time = torch.arange(T_img, device=x.device) + 1

            # text time must equal media time if only attending to most immediate image
            # otherwise, as long as text time is greater than media time (if attending to all previous images / media)
            mask_op = torch.eq if self.only_attend_immediate_media else torch.ge

            text_to_media_mask = mask_op(
                rearrange(text_time, "b i -> b 1 i 1"),
                repeat(media_time, "j -> 1 1 1 (j n)", n=n),
            )
            for i in range(self.stride - 1):
                media_time_tmp = media_time + i + 1
                text_to_media_mask_tmp = mask_op(
                    rearrange(text_time, "b i -> b 1 i 1"),
                    repeat(media_time_tmp, "j -> 1 1 1 (j n)", n=n),
                )
                text_to_media_mask = torch.logical_or(text_to_media_mask, text_to_media_mask_tmp)
            if exists(trunc_locations):
                trunc_time = trunc_locations * text_time
                trunc_time, _ = trunc_time.cummax(dim=-1)
                trunc_mask = torch.ge(repeat(media_time, "j -> 1 1 1 (j n)", n=n),
                                      rearrange(trunc_time, "b i -> b 1 i 1"))
                text_to_media_mask = torch.logical_and(text_to_media_mask, trunc_mask)

            text_to_media_mask = text_to_media_mask[:, :, -x.shape[-2]:]
            sim = sim.masked_fill(~text_to_media_mask, -torch.finfo(sim.dtype).max)

        sim = sim - sim.amax(dim=-1, keepdim=True).detach()
        attn = sim.softmax(dim=-1)

        if exists(media_locations) and self.only_attend_immediate_media:
            # any text without a preceding media needs to have attention zeroed out
            text_without_media_mask = text_time == 0
            text_without_media_mask = rearrange(text_without_media_mask, "b i -> b 1 i 1")
            text_without_media_mask = text_without_media_mask[:, :, -x.shape[-2]:]
            attn = attn.masked_fill(text_without_media_mask, 0.0)

        out = torch.einsum("... i j, ... j d -> ... i d", attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class FlamingoGatedCrossAttentionBlock(nn.Module):
    def __init__(
            self,
            *,
            dim: int,
            dim_visual: int,
            dim_head: int = 64,
            heads: int = 8,
            ff_mult: int = 4,
            only_attend_immediate_media: bool = True,
    ):
        super().__init__()
        self.attn = FlamingoMaskedCrossAttention(
            dim=dim,
            dim_visual=dim_visual,
            dim_head=dim_head,
            heads=heads,
            only_attend_immediate_media=only_attend_immediate_media,
        )
        self.attn_gate = nn.Parameter(torch.tensor([0.0]))
        self.feed_forward = nn.ModuleList(
            [
                nn.LayerNorm(dim),
                nn.Linear(dim, dim * ff_mult, bias=False),
                nn.GELU(),
                nn.Linear(dim * ff_mult, dim, bias=False),
            ]
        )
        self.ff_gate = nn.Parameter(torch.tensor([0.0]))

    def forward(
            self,
            x: torch.Tensor,
            media: torch.Tensor,
            media_locations: Optional[torch.BoolTensor] = None,
            attend_previous: bool = True,
            trunc_locations: Optional[torch.BoolTensor] = None,
    ) -> torch.Tensor:
        x = (
                self.attn(
                    x,
                    media,
                    media_locations=media_locations,
                    attend_previous=attend_previous,
                    trunc_locations=trunc_locations
                )
                * self.attn_gate.tanh()
                + x
        )
        residual_x = x
        for ff in self.feed_forward:
            x = ff(x)
        x = x * self.ff_gate.tanh() + residual_x

        return x


class FlamingoLayer(nn.Module):
    def __init__(self, gated_cross_attn_layer: nn.Module, decoder_layer: nn.Module):
        super().__init__()
        self.gated_cross_attn_layer = gated_cross_attn_layer
        self.decoder_layer = decoder_layer
        self.vis_x = None
        self.media_locations = None
        self.trunc_locations = None

    def is_conditioned(self) -> bool:
        """Check whether the layer is conditioned."""
        return self.vis_x is not None

    # Used this great idea from this implementation of Flamingo (https://github.com/dhansmair/flamingo-mini/)
    def condition_vis_x(self, vis_x) -> None:
        self.vis_x = vis_x

    def condition_trunc_locations(self, trunc_locations) -> None:
        self.trunc_locations = trunc_locations

    def condition_media_locations(self, media_locations) -> None:
        self.media_locations = media_locations

    def condition_attend_previous(self, attend_previous) -> None:
        self.attend_previous = attend_previous

    def forward(
            self,
            lang_x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            **decoder_layer_kwargs,
    ):
        if self.gated_cross_attn_layer is None:
            return self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)

        if self.vis_x is None:
            raise ValueError("vis_x must be conditioned before forward pass")

        if self.media_locations is None:
            raise ValueError("media_locations must be conditioned before forward pass")

        lang_x = self.gated_cross_attn_layer(
            lang_x,
            self.vis_x,
            media_locations=self.media_locations,
            attend_previous=self.attend_previous,
            trunc_locations=self.trunc_locations,
        )
        lang_x = self.decoder_layer(lang_x, attention_mask=attention_mask, **decoder_layer_kwargs)
        return lang_x


class FlamingoLMMixin(nn.Module):
    """
    Mixin to add cross-attention layers to a language model.
    """

    def set_decoder_layers_attr_name(self, decoder_layers_attr_name):
        self.decoder_layers_attr_name = decoder_layers_attr_name

    def _get_decoder_layers(self):
        return getattr_recursive(self, self.decoder_layers_attr_name)

    def _set_decoder_layers(self, value):
        setattr_recursive(self, self.decoder_layers_attr_name, value)

    def init_flamingo(
            self,
            media_token_id: int,
            vis_hidden_size: int,
            cross_attn_every_n_layers: int,
            only_attend_immediate_media: bool,
    ):
        """
        Initialize Flamingo by adding a new gated cross attn to the decoder. Store the media token id for computing the media locations.
        """

        gated_cross_attn_layers = nn.ModuleList(
            [
                FlamingoGatedCrossAttentionBlock(
                    dim=self.config.hidden_size,
                    dim_visual=vis_hidden_size,
                    only_attend_immediate_media=only_attend_immediate_media,
                )
                if (layer_idx + 1) % cross_attn_every_n_layers == 0
                else None
                for layer_idx, _ in enumerate(self._get_decoder_layers())
            ]
        )
        self._set_decoder_layers(
            nn.ModuleList(
                [
                    FlamingoLayer(gated_cross_attn_layer, decoder_layer)
                    for gated_cross_attn_layer, decoder_layer in
                    zip(gated_cross_attn_layers, self._get_decoder_layers())
                ]
            )
        )
        self.media_token_id = media_token_id
        self.use_media_placement_augmentation = False
        self.initialized_flamingo = True

    def forward(self,
                input_ids: torch.Tensor,
                attention_mask: Optional[torch.Tensor] = None,
                labels: Optional[torch.Tensor] = None,
                past_key_values: Optional[torch.Tensor] = None,
                use_cache: Optional[bool] = None,
                trunc_counts: Optional[list] = None,
                **kwargs):
        """Condition the Flamingo layers on the media locations before forward()"""
        if not self.initialized_flamingo:
            raise ValueError("Flamingo layers are not initialized. Please call `init_flamingo` first.")

        media_locations = input_ids == self.media_token_id
        if past_key_values is not None:
            input_ids = input_ids[:, past_key_values[0][0].shape[-2]:]

        attend_previous = True

        if trunc_counts is not None:
            trunc_locations = torch.zeros_like(media_locations)
            for b in range(media_locations.size(0)):
                indices = trunc_counts[b]
                ones_pos = (media_locations[b] == 1).nonzero(as_tuple=True)[0]
                for idx in indices:
                    if idx < len(ones_pos):
                        trunc_locations[b, ones_pos[idx]] = 1

        if self.__class__.__name__ == "LlamaForCausalLM":
            for layer in self.get_decoder().layers:
                layer.condition_media_locations(media_locations)
                if trunc_counts is not None:
                    layer.condition_trunc_locations(trunc_locations)
                layer.condition_attend_previous(attend_previous)
        else:
            print("inavaliable text encoder")

        return super().forward(input_ids=input_ids,
                               attention_mask=attention_mask,
                               labels=labels,
                               past_key_values=past_key_values,
                               use_cache=use_cache,
                               **kwargs)  # Call the other parent's forward method

    def is_conditioned(self) -> bool:
        """Check whether all decoder layers are already conditioned."""
        return all(l.is_conditioned() for l in self._get_decoder_layers())

    def clear_conditioned_layers(self) -> None:
        for layer in self._get_decoder_layers():
            layer.condition_vis_x(None)
            layer.condition_media_locations(None)
            layer.condition_trunc_locations(None)
            layer.condition_attend_previous(None)


class FlamingoPreTrainedModel(PreTrainedModel):
    """
    An abstract class to handle weights initialization and a simple interface for downloading and loading pretrained
    models.
    """

    config_class = FlamingoConfig
    base_model_prefix = "flamingo"
    supports_gradient_checkpointing = True
    _no_split_modules = ["FlamingoPerceiverBlock", "CLIPEncoderLayer", "FlamingoLayer"]

    def _init_weights(self, module):
        """Flamingo requires no specific initialization"""
        return super()._init_weights(module)

    def _set_gradient_checkpointing(self, module, value=False):
        module.gradient_checkpointing = value


class FlamingoForConditionalGeneration(FlamingoPreTrainedModel):
    config_class = FlamingoConfig

    def __init__(
            self,
            config: FlamingoConfig,
    ):
        super().__init__(config)

        if "llama" in config.text_config._name_or_path:
            text_tokenizer = LlamaTokenizer.from_pretrained('/data0/models/luodian-llama-7b-hf')
            # text_tokenizer = LlamaTokenizer.from_pretrained(config.text_config._name_or_path)
            lang_encoder = LlamaForCausalLM(config=config.text_config)
        else:
            raise ValueError("Only LlamaForCausalLM is supported for now.")

        text_tokenizer.add_special_tokens({"additional_special_tokens": ["<|endofchunk|>", "<image>", "<answer>"]})
        if text_tokenizer.pad_token is None:
            text_tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.text_tokenizer = text_tokenizer
        self.eoc_token_id = text_tokenizer.encode("<|endofchunk|>")[-1]
        self.media_token_id = text_tokenizer.encode("<image>")[-1]
        self.answer_token_id = text_tokenizer.encode("<answer>")[-1]

        extend_instance(lang_encoder, FlamingoLMMixin)
        decoder_layers_attr_name = _infer_decoder_layers_attr_name(lang_encoder)
        lang_encoder.set_decoder_layers_attr_name(decoder_layers_attr_name)
        if "LlamaForCausalLM" in lang_encoder.__class__.__name__:
            lang_encoder.resize_token_embeddings(len(text_tokenizer))
        self.lang_encoder = lang_encoder

        self.cross_attn_every_n_layers = config.cross_attn_every_n_layers if hasattr(config,
                                                                                     "cross_attn_every_n_layers") else 4
        self.only_attend_immediate_media = config.only_attend_immediate_media
        if not config.feature_as_input:
            vision_encoder = CLIPVisionModel(config=config.vision_config)
            vision_encoder.output_tokens = True
            self.vision_encoder = vision_encoder
        else:
            self.vision_encoder = None
        self.lm_head = None

        self.vis_dim = 1024
        self.perceiver = FlamingoPerceiverResampler(dim=self.vis_dim,
                                                    use_frame_embs=not config.only_attend_immediate_media)

        self.lang_encoder.init_flamingo(
            media_token_id=self.media_token_id,
            vis_hidden_size=self.vis_dim,
            cross_attn_every_n_layers=self.cross_attn_every_n_layers,
            only_attend_immediate_media=self.only_attend_immediate_media,
        )

        self.post_init()

    def get_input_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_input_embeddings()

    def set_input_embeddings(self, new_embeddings):
        self.lang_encoder.set_input_embeddings(new_embeddings)

    def get_output_embeddings(self) -> nn.Module:
        return self.lang_encoder.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.lang_encoder.set_output_embeddings(new_embeddings)

    def get_image_encoder(self) -> nn.Module:
        return self.vision_encoder

    def get_lang_encoder(self) -> nn.Module:
        return self.lang_encoder

    def init_weights(self):
        # Freeze all parameters in vision encoder
        if self.vision_encoder is not None:
            for param in self.vision_encoder.parameters():
                param.requires_grad = False

        for name, param in self.lang_encoder.named_parameters():
            if "gated_cross_attn_layer" not in name:
                param.requires_grad = False

        decoder_layers = self.lang_encoder._get_decoder_layers()
        # last_two_layers_indices = []
        last_two_layers_indices = [-1, -2]
        for index in last_two_layers_indices:
            layer = decoder_layers[index]
            for param in layer.parameters():
                param.requires_grad = True
        if "LlamaForCausalLM" in self.lang_encoder.__class__.__name__:
            self.lang_encoder.lm_head.requires_grad_(True)

        print("====================Model Grad Part====================")
        total_params = 0
        for name, param in self.named_parameters():
            if param.requires_grad:
                total_params += param.numel()
                # print(f"Parameter: {name}, Size: {param.numel() / 1e6:.6f} M")
        print(f"Total Trainable param: {total_params / 1e9:.4f} B")
        print(f"Total Trainable param: {(sum(p.numel() for p in self.parameters() if p.requires_grad)) / 1e9:.3f} B")

    def forward(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            labels: Optional[torch.Tensor] = None,
            use_cached_vision_x: bool = False,
            clear_conditioned_layers: bool = True,
            past_key_values: Optional[torch.Tensor] = None,
            use_cache: bool = False,
            **kwargs,
    ) -> CausalLMOutputWithPast:
        """
        Forward pass of Flamingo.

        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W) with F=1
            lang_x (torch.Tensor): Language input ids
                shape (B, T_txt)
            attention_mask (torch.Tensor, optional): Attention mask. Defaults to None.
            labels (torch.Tensor, optional): Labels. Defaults to None.
            clear_conditioned_layers: if True, clear the conditioned layers
                once the foward pass is completed. Set this to false if the
                same set of images will be reused in another subsequent
                forward pass.
            past_key_values: pre-computed values to pass to language model.
                See past_key_values documentation in Hugging Face
                CausalLM models.
            use_cache: whether to use cached key values. See use_cache
                documentation in Hugging Face CausalLM models.
        """
        assert (
                       vision_x is not None) or use_cached_vision_x, "Must provide either vision_x or use_cached_vision_x to True."

        if use_cached_vision_x:
            # Case: use cached; vision_x should be cached and other
            # vision-related inputs should not be provided.
            assert vision_x is None, "Expect vision_x to be None when use_cached_vision_x is True."
            assert self.lang_encoder.is_conditioned()

        else:
            # Case: do not use caching (i.e. this is a standard forward pass);
            self._encode_vision_x(vision_x=vision_x)

        if self.only_attend_immediate_media and "trunc_counts" in kwargs:
            kwargs.pop('trunc_counts')

        output = self.lang_encoder(
            input_ids=lang_x,
            attention_mask=attention_mask,
            labels=labels,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        if clear_conditioned_layers:
            self.lang_encoder.clear_conditioned_layers()

        return output

    def _encode_vision_x(self, vision_x: torch.Tensor):
        """
        Compute media tokens from vision input by passing it through vision encoder and conditioning language model.
        Args:
            vision_x (torch.Tensor): Vision input
                shape (B, T_img, F, C, H, W)
                Images in the same chunk are collated along T_img, and frames are collated along F
                Currently only F=1 is supported (single-frame videos)

        rearrange code based on https://github.com/dhansmair/flamingo-mini
        """
        if self.vision_encoder is None:
            assert vision_x.ndim == 5, "vision_x should be of shape (b, T_img, F, v, d)"
            vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        else:
            assert vision_x.ndim == 6, "vision_x should be of shape (b, T_img, F, C, H, W)"
            b, T, F = vision_x.shape[:3]
            # assert F == 1, "Only single frame supported"

            vision_x = rearrange(vision_x, "b T F c h w -> (b T F) c h w")
            with torch.no_grad():
                vision_x = self.vision_encoder(vision_x)[0][:, 1:, :]
            vision_x = rearrange(vision_x, "(b T F) v d -> b T F v d", b=b, T=T, F=F)
            vision_x = self.perceiver(vision_x)  # reshapes to (b, T, n, d)

        for layer in self.lang_encoder._get_decoder_layers():
            layer.condition_vis_x(vision_x)

    @torch.inference_mode()
    def generate_lightning_greedy_search(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: Optional[int] = None,
            past_key_values: Optional[torch.Tensor] = None,
            ended: Optional[int] = None,
            **kwargs):

        pad_token_id = self.text_tokenizer.pad_token_id
        eos_token_id = [self.eoc_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(lang_x.device) if eos_token_id is not None else None
        unfinished_sequences = 1 - torch.tensor(ended, dtype=torch.long, device=lang_x.device)
        self._encode_vision_x(vision_x=vision_x)
        for i in range(max_new_tokens):
            model_kwargs = self.lang_encoder.prepare_inputs_for_generation(
                input_ids=lang_x,
                past_key_values=past_key_values,
                attention_mask=attention_mask,
                use_cache=True,
                **kwargs,
            )
            output = self.lang_encoder.forward(**model_kwargs)
            past_key_values = output.past_key_values
            next_token_logits = output.logits[:, -1, :]
            next_tokens = torch.argmax(next_token_logits, dim=-1).to(lang_x.device)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            lang_x = torch.cat([lang_x, next_tokens[:, None]], dim=-1)
            attention_mask = torch.cat(
                [attention_mask, attention_mask.new_ones((attention_mask.shape[0], 1))], dim=-1
            )
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        return {"sequences": lang_x.cpu(), "past_key_values": past_key_values}

    @torch.inference_mode()
    def generate_lightning(
            self,
            vision_x: torch.Tensor,
            lang_x: torch.Tensor,
            attention_mask: Optional[torch.Tensor] = None,
            max_new_tokens: Optional[int] = None,
            past_key_values: Optional[torch.Tensor] = None,
            ended: Optional[int] = None,
            num_return_sequences: Optional[int] = 1,
            temperature: Optional[float] = 0.7,
            **kwargs):

        if temperature == 0.0:
            return self.generate_lightning_greedy_search(
                vision_x=vision_x,
                lang_x=lang_x,
                attention_mask=attention_mask,
                max_new_tokens=max_new_tokens,
                past_key_values=past_key_values,
                ended=ended,
                **kwargs,
            )

        if num_return_sequences > 1:
            vision_x = vision_x.repeat_interleave(num_return_sequences, dim=0)

        generation_config = GenerationConfig.from_model_config(self.config)
        model_kwargs = generation_config.update(
            **{"past_key_values": past_key_values, "attention_mask": attention_mask, "temperature": temperature,
               "num_return_sequences": num_return_sequences, "max_new_tokens": max_new_tokens})
        logits_warper = self._get_logits_warper(generation_config)
        input_ids_length = lang_x.shape[-1]
        logits_processor = self._get_logits_processor(
            generation_config=generation_config,
            input_ids_seq_length=input_ids_length,
            encoder_input_ids=lang_x,
            prefix_allowed_tokens_fn=None,
            logits_processor=LogitsProcessorList(),
            model_kwargs=model_kwargs,
        )
        if past_key_values is not None and num_return_sequences > 1:
            past_key_values_new = [(past_key_value[0].repeat_interleave(num_return_sequences, dim=0),
                                    past_key_value[1].repeat_interleave(num_return_sequences, dim=0)) for past_key_value
                                   in
                                   past_key_values]
            past_key_values = past_key_values_new
        input_ids, model_kwargs = self._expand_inputs_for_generation(
            input_ids=lang_x,
            expand_size=num_return_sequences,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
        )
        scores = []
        pad_token_id = self.text_tokenizer.pad_token_id
        eos_token_id = [self.eoc_token_id]
        eos_token_id_tensor = torch.tensor(eos_token_id).to(input_ids.device) if eos_token_id is not None else None
        unfinished_sequences = 1 - torch.tensor(ended, dtype=torch.long, device=input_ids.device)
        self._encode_vision_x(vision_x=vision_x)
        for i in range(max_new_tokens):
            model_inputs = self.lang_encoder.prepare_inputs_for_generation(
                input_ids=input_ids,
                use_cache=True,
                **model_kwargs,
            )
            outputs = self.lang_encoder.forward(**model_inputs)
            past_key_values = outputs.past_key_values
            next_token_logits = outputs.logits[:, -1, :]
            next_token_scores = logits_processor(input_ids, next_token_logits)
            next_token_scores = logits_warper(input_ids, next_token_scores)
            probs = nn.functional.softmax(next_token_scores, dim=-1)
            scores.append(probs)
            next_tokens = torch.multinomial(probs, num_samples=1).squeeze(1).to(input_ids.device)
            next_tokens = next_tokens * unfinished_sequences + pad_token_id * (1 - unfinished_sequences)
            input_ids = torch.cat([input_ids, next_tokens[:, None]], dim=-1)
            model_kwargs = self._update_model_kwargs_for_generation(
                outputs, model_kwargs, is_encoder_decoder=self.config.is_encoder_decoder
            )
            unfinished_sequences = unfinished_sequences.mul(
                next_tokens.tile(eos_token_id_tensor.shape[0], 1).ne(eos_token_id_tensor.unsqueeze(1)).prod(dim=0)
            )

            # stop when each sentence is finished
            if unfinished_sequences.max() == 0:
                break

        return {"sequences": input_ids.cpu(), "logits": outputs.logits, "scores": torch.stack(scores, dim=1),
                "past_key_values": past_key_values}

