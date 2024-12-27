import copy

from transformers.configuration_utils import PretrainedConfig
from transformers.utils import logging
from transformers.models.auto import CONFIG_MAPPING
from transformers.models.clip import CLIPVisionConfig

logger = logging.get_logger(__name__)


class FlamingoConfig(PretrainedConfig):
    model_type = "flamingo"
    is_composition = True

    def __init__(self, vision_config=None, text_config=None, cross_attn_every_n_layers: int = 4,
                 only_attend_immediate_media: bool = True, **kwargs):
        super().__init__(**kwargs)

        if vision_config is None:
            vision_config = {}
            logger.info("vision_config is None. initializing the vision config with default values.")

        self.vision_config = CLIPVisionConfig(**vision_config)
        if text_config is None:
            text_config = {}
            logger.info(
                "text_config is None. Initializing the text config with default values."
            )

        self.hidden_size = 4096
        self.text_config = CONFIG_MAPPING[text_config.pop("model_type")](**text_config)
        self.cross_attn_every_n_layers = cross_attn_every_n_layers
        self.only_attend_immediate_media = only_attend_immediate_media
        self.feature_as_input = True

    def to_dict(self):
        """
        Serializes this instance to a Python dictionary. Override the default [`~PretrainedConfig.to_dict`].

        Returns:
            `Dict[str, any]`: Dictionary of all the attributes that make up this configuration instance,
        """
        output = copy.deepcopy(self.__dict__)
        output["vision_config"] = self.vision_config.to_dict()
        output["text_config"] = self.text_config.to_dict()
        output["model_type"] = self.__class__.model_type
        output["cross_attn_every_n_layers"] = self.cross_attn_every_n_layers
        output[
            "only_attend_immediate_media"
        ] = self.only_attend_immediate_media
        return output
