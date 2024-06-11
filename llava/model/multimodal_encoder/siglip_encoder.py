import torch
from llava.model.multimodal_encoder.vision_encoder import VisionTower, VisionTowerS2

from transformers import AutoConfig, PretrainedConfig, AutoModel
from transformers import SiglipVisionConfig, SiglipVisionModel, SiglipImageProcessor
# from .siglip import (
#     SiglipVisionConfig,
#     SiglipVisionModel,
#     SiglipImageProcessor,
# )

def interpolate_pos_embed_siglip(model, new_size):
    # 577 1024
    pos_emb = model.vision_model.embeddings.position_embedding.weight.float()
    ori_size = int((pos_emb.shape[0])**0.5)
    dim = pos_emb.shape[1]
    print("Position interpolate from %dx%d to %dx%d" % (ori_size, ori_size, new_size, new_size))
    # 1, 1024
    # 576 1024
    pos_tokens = pos_emb
    # 1 24 24 1024 -> 1 1024 24 24
    pos_tokens = pos_tokens.reshape(-1, ori_size, ori_size, dim).permute(0, 3, 1, 2)
    # 1 1024 32 32
    pos_tokens = torch.nn.functional.interpolate(
            pos_tokens, size=(new_size, new_size), mode='bicubic', align_corners=False)
    # 1 32 32 1024, -> 1 1024 1024
    pos_tokens = pos_tokens.permute(0, 2, 3, 1).flatten(1, 2).squeeze(0)
    new_pos_embed = pos_tokens #torch.cat((extra_tokens, pos_tokens), dim=0)
    return new_pos_embed


class SiglipVisionTower(VisionTower):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig, state_dict=None):
        super().__init__(model_name_or_path, config)
        # config.image_size = 490
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path, size={'height': config.image_size, 'width': config.image_size})
        self.vision_tower = SiglipVisionModel.from_pretrained(
            # TODO(ligeng): why pass config here leading to errors?
            model_name_or_path, torch_dtype=eval(config.model_dtype), state_dict=state_dict
        )
        
        if self.vision_tower.vision_model.embeddings.image_size != config.image_size:
            patch_size = self.vision_tower.vision_model.embeddings.patch_size
            num_patches = (config.image_size // patch_size)**2 
            new_size = config.image_size // patch_size

            self.vision_tower.vision_model.embeddings.image_size = config.image_size
            self.vision_tower.vision_model.embeddings.num_patches = num_patches
            self.vision_tower.vision_model.embeddings.num_positions = num_patches
            self.vision_tower.vision_model.embeddings.position_ids = torch.arange(num_patches).expand((1, -1))
            new_pos = interpolate_pos_embed_siglip(self.vision_tower, new_size)
            self.vision_tower.vision_model.embeddings.position_embedding = torch.nn.Embedding.from_pretrained(new_pos)
        
        self.is_loaded = True


class SiglipVisionTowerS2(VisionTowerS2):
    def __init__(self, model_name_or_path: str, config: PretrainedConfig):
        super().__init__(model_name_or_path, config)
        self.image_processor = SiglipImageProcessor.from_pretrained(model_name_or_path)
        self.vision_tower = SiglipVisionModel.from_pretrained(
            model_name_or_path, torch_dtype=eval(config.model_dtype)
        )

        # Make sure it crops/resizes the image to the largest scale in self.scales to maintain high-res information
        self.image_processor.size['height'] = self.image_processor.size['width'] = self.scales[-1]

        self.is_loaded = True


# AutoConfig.register("siglip_vision_model", SiglipVisionConfig)
# AutoModel.register(SiglipVisionConfig, SiglipVisionModel)