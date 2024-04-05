from calendar import c
from functools import reduce
from operator import mul
import string as str
import torch
import math
import torch.nn as nn
from torch.nn import Dropout

import timm

from typing import Callable, Dict, Optional, Sequence, Set, Tuple, Type, Union, List

from timm.models.vision_transformer import VisionTransformer, PatchEmbed, LayerType

from timm.models._registry import generate_default_cfgs, register_model, register_model_deprecations

from timm.models._builder import build_model_with_cfg

from copy import copy

class VisionTransformerVPT(VisionTransformer):
    """ Vision Transformer with Visual Prompt Tuning

    Args:
        VisionTransformer (_type_): _description_
    """
    def __init__(self,
                 base_model_name = None,
                 pretrained: bool = False,
                 VPT_Project_dim : int = -1, # understand what it does
                 VPT_Prompt_Token_Num : int = 1,
                 VPT_type = "Shallow",
                 VPT_Deep_Shared = "False",
                 VPT_Dropout: float = 0.1,
                 VPT_Initiation = 'random',
                 VPT_Location = 'prepend',
                 VPT_Num_Deep_Layers = None, 
                 VPT_Reverse_Deep: bool = False,
                 VPT_VIT_Pool_Type = 'original',
                 VPT_Forward_Deep_Noexpand = 'False',
                 **kwargs):# ,
                #  basic_state_dict=None):
        
        # init the VisionTransformer class with pretrained weights
        super(VisionTransformerVPT, self).__init__(**kwargs)

        # # Save the pretrained flag and base model name
        # self.pretrained = kwargs.get('pretrained', False)
        # self.base_model_name = kwargs.get('base_model_name', 'vit_base_patch16_224')

        # pretrained args
        self.base_model_name = base_model_name
        self.pretrained = pretrained
        self.num_classes = kwargs['num_classes']

        # Initialize VPT attributes
        self.VPT_Prompt_Token_Num = VPT_Prompt_Token_Num
        self.Prompt_Dropout = Dropout(VPT_Dropout)


        # Conditional projection for VPT embeddings
        if VPT_Project_dim > -1:
            prompt_dim = VPT_Project_dim
            self.VPT_proj = nn.Linear(prompt_dim, self.embed_dim)
            nn.init.kaiming_normal_(self.VPT_proj.weight, a = 0, mode='fan_out')
            self.VPT_proj.bias.data.fill_(0.01) # TODO do we really need this?
        else:
            prompt_dim = self.embed_dim
            self.VPT_proj = nn.Identity()
        
        # initiate prompt:
        if VPT_Initiation == "random": # what is this random? 
            val = math.sqrt(6. / float(3 * reduce(mul, (kwargs['patch_size'],kwargs['patch_size']), 
                                                  1) + prompt_dim)) 

            self.prompt_embeddings = nn.Parameter(torch.zeros(1, self.VPT_Prompt_Token_Num, self.embed_dim))

            # xavier_uniform_initialization
            nn.init.uniform_(self.prompt_embeddings.data, -val, val)

            if VPT_type == 'DEEP': 
                total_d_layer = self.depth-1 # since first layer is shallow
                self.deep_prompt_embeddings = nn.Parameter(torch.zeros(
                    total_d_layer, self.VPT_Prompt_Token_Num, prompt_dim))
                # xavier_uniform initialization
                nn.init.uniform_(self.deep_prompt_embeddings.data, -val, val)
        else:
            raise ValueError("No other init scheme supported by VPT authors")
        
        if pretrained:
            self._load_pretrained()
        

        
    def _incorporate_prompt(self, x: torch.Tensor) -> torch.Tensor:
        """ Combine prompts with image-patch embeddings

        Args:
            x (Input): Input img patches

        Returns:
            _type_: embeddings with VPT prompts
        """
        batch_size = x.shape[0] # check the shape here once, but it should be the batch of embeddings
        # x = self.embeddings(x) # this is handled by the functions patch embedding process 
        
        # get the VPT embeddings
        VPT_Embeddings = self.VPT_proj(self.prompt_embeddings.expand(batch_size, -1, -1)) # self.Prompt_Dropout
        
        # Prepend the embeddings
        x = torch.cat((x, VPT_Embeddings), dim=1)
        # (batch_size, cls_token + n_prompt + n_patches, hidden_dim)
        # x = torch.cat((x[:, :1, :],VPT_embeddings, x[;, 1:, :],), dim=1)
        
        # x = torch.cat((
        #     x[;, :1, :],
        #     self.Prompt_Dropout(self.VPT_proj(self.VPT_Embeddings)).expand(batch_size, -1, -1),
        #     x[:, 1:, :]
        # ), dim =1)

        return x

    def _load_pretrained(self):
        """Loads the pretrained ViT model 

        Returns:
            _type_: None
        """
        # Load the pre-trained ViT model
        pretrained_model = timm.create_model(model_name=self.base_model_name, 
                                             pretrained=self.pretrained, 
                                             num_classes=self.num_classes)

        # Get the state dict of the pre-trained model
        pretrained_state_dict = pretrained_model.state_dict()

        # Get the state dict of the current model
        current_state_dict = self.state_dict()

        # Filter out the keys that don't match between the two state dicts
        pretrained_state_dict = {k: v for k, v in pretrained_state_dict.items() if k in current_state_dict}

        # Update the current model's state dict with the pre-trained weights
        current_state_dict.update(pretrained_state_dict)

        # Load the updated state dict into the current model
        self.load_state_dict(current_state_dict, strict=False)









        # # Implement or invoke pre-trained weight loading logic if not handled by super().__init__
        # # Example: self.load_state_dict(torch.load(PATH_TO_PRETRAINED_WEIGHTS), strict=False)
        # self.base_vit = timm.create_model(model_name=self.base_model_name, 
        #                                   pretrained=self.pretrained, 
        #                                   num_classes=self.num_classes)

    def _freeze_all_parameters(self):
        for param in self.parameters():
            param.requires_grad = False

    def _unfreeze_vpt_parameters(self):
        self.prompt_embeddings.requires_grad = True
        if hasattr(self, 'deep_prompt_embeddings'):
            self.deep_prompt_embeddings.requires_grad = True


    def _train(self, mode=True):
        """ set train status for this class: disable all but the prompt-related modules
        
        Args:
            mode (bool, optional): Train mode or eval mode. Defaults to True.

        Returns:
            _type_: _description_
        """
        if mode: # check with the named blocks
            # training
            self.base_model.encoder.eval()
            self.embeddings.eval()
            self.prompt_proj.train() # understand why?
            self.prompt_dropout.train() # understand why?
        else:
            # eval:
            for module in self.children():
                module.train(mode)

    def forward_features(self, x):
        # Apply parent class's patch embedding process
        embeddings = super().forward_features(x)

        # Process and prepend VPT embeddings
        x = self._incorporate_prompt(embeddings)

        return x

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
    def get_classifier(self) -> nn.Module:
        return self.head

# def _create_vision_transformer_vpt(pretrained: bool = False, **kwargs) -> VisionTransformerVPT:
#     return build_model_with_cfg(VisionTransformerVPT, pretrained, **kwargs) # noqa

# def _create_vision_transformer_vpt(model_cls, variant, pretrained: bool = False, **kwargs):
#     return build_model_with_cfg(
#         model_cls=model_cls,
#         variant=variant,
#         pretrained=pretrained,
#         **kwargs
#     )

def _vpt_cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000,  # Default number of output classes
        'input_size': (3, 224, 224),  # Input image size (C, H, W)
        'pool_size': None,  # Global pooling size, if applicable
        'crop_pct': 0.875,  # Crop percentage for validation
        'interpolation': 'bicubic',  # Interpolation method
        'mean': (0.485, 0.456, 0.406),  # Normalization mean
        'std': (0.229, 0.224, 0.225),  # Normalization std
        'classifier': 'head',  # Name of the classifier layer
        **kwargs
    }

default_cfgs = generate_default_cfgs({
    'vit_base_patch16_224.augreg2_in21k_ft_in1k': _vpt_cfg(hf_hub_id='timm/')
})


# def _create_vision_transformer_vpt(model_cls, variant, pretrained=False, **kwargs):
#     return build_model_with_cfg(model_cls=model_cls,
#                                 variant=variant,
#                                 pretrained=pretrained, 
#                                 **kwargs)

def _create_vision_transformer_vpt(variant, pretrained=False, **kwargs):
    if kwargs.get('features_only', None):
        raise RuntimeError('features_only not implemented for Vision Transformer VPT models.')

    model_cls = VisionTransformerVPT  # Use the VPT variant of VisionTransformer

    # Get the default configuration for the specified variant
    default_cfg = default_cfgs['vit_base_patch16_224.augreg2_in21k_ft_in1k']

    # Update the default configuration with the provided keyword arguments
    cfg = default_cfg

    cfg.update(kwargs)

    # Create the model using the build_model_with_cfg function from TIMM
    model = build_model_with_cfg(
        model_cls,
        variant,
        pretrained,
        default_cfg=default_cfg,
        **cfg
    )

    return model

# @register_model
# def vision_transformer_vpt_small_patch16_224(pretrained: bool = False, **kwargs):
#     cfg = _vpt_cfg(url='https://example.com/path/to/your/pretrained/weights.pth')
#     # Assume this function returns an instance of your model for a specific configuration
#     return _create_vision_transformer_vpt(pretrained, **kwargs)

# @register_model
# def vision_transformer_vpt_small_patch16_224(pretrained: bool = False, **kwargs):
#     model_variant = 'VisionTransformer'  # You need to specify the variant name based on your model naming conventions
#     cfg = _vpt_cfg(url='https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k')
#     return _create_vision_transformer_vpt(VisionTransformerVPT, model_variant, pretrained=pretrained, **kwargs)

@register_model
def vision_transformer_vpt_base_patch16_224(pretrained=False, **kwargs):
    cfg = _vpt_cfg(url='https://huggingface.co/timm/vit_base_patch16_224.augreg2_in21k_ft_in1k')
    return _create_vision_transformer_vpt(VisionTransformer, pretrained=pretrained, cfg=cfg)