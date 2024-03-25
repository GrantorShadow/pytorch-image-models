import timm
import torch
import torch.nn as nn

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model

import argparse
import csv
import glob
import json
import logging
import os
import time
from collections import OrderedDict
from contextlib import suppress
from functools import partial

import torch
import torch.nn as nn
import torch.nn.parallel

from timm.data import create_dataset, create_loader, resolve_data_config, RealLabelsImagenet
from timm.layers import apply_test_time_pool, set_fast_norm, PatchEmbed, resample_abs_pos_embed
from timm.models import create_model, load_checkpoint, is_model, list_models
from timm.utils import accuracy, AverageMeter, natural_key, setup_default_logging, set_jit_fuser, \
    decay_batch_step, check_batch_size_retry, ParseKwargs, reparameterize_model
from timm.models._manipulate import checkpoint_seq


try:
    from apex import amp
    has_apex = True
except ImportError:
    has_apex = False

has_native_amp = False
try:
    if getattr(torch.cuda.amp, 'autocast') is not None:
        has_native_amp = True
except AttributeError:
    pass

try:
    from functorch.compile import memory_efficient_fusion
    has_functorch = True
except ImportError as e:
    has_functorch = False

has_compile = hasattr(torch, 'compile')

_logger = logging.getLogger('validate')

parser = argparse.ArgumentParser(description='PyTorch ImageNet Validation')
parser.add_argument('--model_eval', metavar = "torch_tensor",
                    help='model used for eval')
parser.add_argument('data', nargs='?', metavar='DIR', const=None,
                    help='path to dataset (*deprecated*, use --data-dir)')
parser.add_argument('--data-dir', metavar='DIR',
                    help='path to dataset (root dir)')
parser.add_argument('--dataset', metavar='NAME', default='',
                    help='dataset type + name ("<type>/<name>") (default: ImageFolder or ImageTar if empty)')
parser.add_argument('--split', metavar='NAME', default='validation',
                    help='dataset split (default: validation)')
parser.add_argument('--num-samples', default=None, type=int,
                    metavar='N', help='Manually specify num samples in dataset split, for IterableDatasets.')
parser.add_argument('--dataset-download', action='store_true', default=False,
                    help='Allow download of dataset for torch/ and tfds/ datasets that support it.')
parser.add_argument('--class-map', default='', type=str, metavar='FILENAME',
                    help='path to class to idx mapping file (default: "")')
parser.add_argument('--input-key', default=None, type=str,
                   help='Dataset key for input images.')
parser.add_argument('--input-img-mode', default=None, type=str,
                   help='Dataset image conversion mode for input images.')
parser.add_argument('--target-key', default=None, type=str,
                   help='Dataset key for target labels.')

parser.add_argument('--model', '-m', metavar='NAME', default='dpn92',
                    help='model architecture (default: dpn92)')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('-b', '--batch-size', default=256, type=int,
                    metavar='N', help='mini-batch size (default: 256)')
parser.add_argument('--img-size', default=None, type=int,
                    metavar='N', help='Input image dimension, uses model default if empty')
parser.add_argument('--in-chans', type=int, default=None, metavar='N',
                    help='Image input channels (default: None => 3)')
parser.add_argument('--input-size', default=None, nargs=3, type=int,
                    metavar='N N N', help='Input all image dimensions (d h w, e.g. --input-size 3 224 224), uses model default if empty')
parser.add_argument('--use-train-size', action='store_true', default=False,
                    help='force use of train input size, even when test size is specified in pretrained cfg')
parser.add_argument('--crop-pct', default=None, type=float,
                    metavar='N', help='Input image center crop pct')
parser.add_argument('--crop-mode', default=None, type=str,
                    metavar='N', help='Input image crop mode (squash, border, center). Model default if None.')
parser.add_argument('--crop-border-pixels', type=int, default=None,
                    help='Crop pixels from image border.')
parser.add_argument('--mean', type=float, nargs='+', default=None, metavar='MEAN',
                    help='Override mean pixel value of dataset')
parser.add_argument('--std', type=float,  nargs='+', default=None, metavar='STD',
                    help='Override std deviation of of dataset')
parser.add_argument('--interpolation', default='', type=str, metavar='NAME',
                    help='Image resize interpolation type (overrides model)')
parser.add_argument('--num-classes', type=int, default=None,
                    help='Number classes in dataset')
parser.add_argument('--gp', default=None, type=str, metavar='POOL',
                    help='Global pool type, one of (fast, avg, max, avgmax, avgmaxc). Model default if None.')
parser.add_argument('--log-freq', default=10, type=int,
                    metavar='N', help='batch logging frequency (default: 10)')
parser.add_argument('--checkpoint', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--num-gpu', type=int, default=1,
                    help='Number of GPUS to use')
parser.add_argument('--test-pool', dest='test_pool', action='store_true',
                    help='enable test time pool')
parser.add_argument('--no-prefetcher', action='store_true', default=False,
                    help='disable fast prefetcher')
parser.add_argument('--pin-mem', action='store_true', default=False,
                    help='Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.')
parser.add_argument('--channels-last', action='store_true', default=False,
                    help='Use channels_last memory layout')
parser.add_argument('--device', default='cuda', type=str,
                    help="Device (accelerator) to use.")
parser.add_argument('--amp', action='store_true', default=False,
                    help='use NVIDIA Apex AMP or Native AMP for mixed precision training')
parser.add_argument('--amp-dtype', default='float16', type=str,
                    help='lower precision AMP dtype (default: float16)')
parser.add_argument('--amp-impl', default='native', type=str,
                    help='AMP impl to use, "native" or "apex" (default: native)')
parser.add_argument('--tf-preprocessing', action='store_true', default=False,
                    help='Use Tensorflow preprocessing pipeline (require CPU TF installed')
parser.add_argument('--use-ema', dest='use_ema', action='store_true',
                    help='use ema version of weights if present')
parser.add_argument('--fuser', default='', type=str,
                    help="Select jit fuser. One of ('', 'te', 'old', 'nvfuser')")
parser.add_argument('--fast-norm', default=False, action='store_true',
                    help='enable experimental fast-norm')
parser.add_argument('--reparam', default=False, action='store_true',
                    help='Reparameterize model')
parser.add_argument('--model-kwargs', nargs='*', default={}, action=ParseKwargs)

scripting_group = parser.add_mutually_exclusive_group()
scripting_group.add_argument('--torchscript', default=False, action='store_true',
                             help='torch.jit.script the full model')
scripting_group.add_argument('--torchcompile', nargs='?', type=str, default=None, const='inductor',
                             help="Enable compilation w/ specified backend (default: inductor).")
scripting_group.add_argument('--aot-autograd', default=False, action='store_true',
                             help="Enable AOT Autograd support.")

parser.add_argument('--results-file', default='', type=str, metavar='FILENAME',
                    help='Output csv file for validation results (summary)')
parser.add_argument('--results-format', default='csv', type=str,
                    help='Format for results file one of (csv, json) (default: csv).')
parser.add_argument('--real-labels', default='', type=str, metavar='FILENAME',
                    help='Real labels JSON file for imagenet evaluation')
parser.add_argument('--valid-labels', default='', type=str, metavar='FILENAME',
                    help='Valid label indices txt file for validation of partial label space')
parser.add_argument('--retry', default=False, action='store_true',
                    help='Enable batch size decay & retry for single model validation')




class ViTB16_32(nn.Module):
    def __init__(self, base_model_name: str, img_size: int, patch_size: int, num_classes: int = 1000, pretrained: bool = False):
        super(ViTB16_32, self).__init__()
        # load the base model without any weights, just a place holder
        self.base_model = timm.create_model(model_name=base_model_name, pretrained=False, num_classes=num_classes)

        # get the total patches and then init the new patch embeds
        self.num_patches = (img_size// patch_size) ** 2 # create a tuple (x,x)
        self.patch_size = patch_size

        embed_args = {} 
        # init the new patch embedding wieght
        self.base_model.patch_embed = PatchEmbed( img_size=224,
                                                 patch_size=32,
                                                 in_chans=3,
                                                 embed_dim=768,
                                                 bias=not False,  # disable bias if pre-norm is used (e.g. CLIP)
                                                 dynamic_img_pad=False,
                                                 **embed_args)

        # now get the new pos embed
        # pos_embed_shape = (1, self.num_patches + 1, self.base_model.embed_dim)
        #nn.Parameter(torch.randn(pos_embed_shape))

        embed_len = self.num_patches if self.base_model.no_embed_class else self.num_patches + self.base_model.num_prefix_tokens
        self.base_model.pos_embed = nn.Parameter(torch.randn(1, embed_len, self.base_model.embed_dim) * .02)

        # init the pos embed and patch embed in he init
        nn.init.kaiming_uniform_(self.base_model.pos_embed)

        if pretrained:
            self._init_pretrained_weights(base_model_name, num_classes)

    def _init_pretrained_weights(self, base_model_name: str, num_classes: int):
        pretrained_weights = timm.create_model(model_name=base_model_name, pretrained=True, num_classes=num_classes).state_dict()
        current_state_dict = self.base_model.state_dict()

        # Filter out the weights that we don't want to copy
        pretrained_weights = {k: v for k, v in pretrained_weights.items() if k not in ['pos_embed', 'patch_embed.proj.weight', 'patch_embed.proj.bias']}

        # Update current state dict with filtered pretrained weights
        current_state_dict.update(pretrained_weights)

        # Load the updated state dict
        self.base_model.load_state_dict(current_state_dict)

    def _pos_embed(self, x: torch.Tensor) -> torch.Tensor:
        if self.base_model.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.no_embed_class else self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.base_model.pos_embed

        to_cat = []
        if self.base_model.cls_token is not None:
            to_cat.append(self.base_model.cls_token.expand(x.shape[0], -1, -1))
        if self.base_model.reg_token is not None:
            to_cat.append(self.base_model.reg_token.expand(x.shape[0], -1, -1))

        if self.base_model.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.base_model.pos_drop(x)
    # def forward(self, x):
    #     # Forward pass through the modified base model
    #     # This assumes the base model's forward method is equipped to handle the input correctly
    #     # after the architectural adjustments made during initialization.
    #     return self.base_model(x)
    
    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.base_model.patch_embed(x)
        x = self._pos_embed(x)
        x = self.base_model.patch_drop(x)
        x = self.base_model.norm_pre(x)
        if self.base_model.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.base_model.blocks, x)
        else:
            x = self.base_model.blocks(x)
        x = self.base_model.norm(x)
        return x

    def forward_head(self, x: torch.Tensor, pre_logits: bool = False) -> torch.Tensor:
        if self.base_model.attn_pool is not None:
            x = self.base_model.attn_pool(x)
        elif self.base_model.global_pool == 'avg':
            x = x[:, self.base_model.num_prefix_tokens:].mean(dim=1)
        elif self.base_model.global_pool:
            x = x[:, 0]  # class token
        x = self.base_model.fc_norm(x)
        x = self.base_model.head_drop(x)
        return x if pre_logits else self.base_model.head(x)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x


    
def validate(args):
    # might as well try to validate something
    args.pretrained = args.pretrained or not args.checkpoint
    args.prefetcher = not args.no_prefetcher
    
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.benchmark = True

    device = torch.device(args.device)

    amp_autocast = suppress

    _logger.info('Validating in float32. AMP not enabled.')
    
    # if args.num_classes is None:
    #     assert hasattr(model, 'num_classes'), 'Model must have `num_classes` attr if not set on cmd line/config.'
    #     args.num_classes = model.num_classes

    model = args.model_eval

    param_count = sum([m.numel() for m in model.parameters()])
    _logger.info('Model %s created, param count: %d' % (args.model, param_count))

    
    model = model.to(device)
    if args.channels_last:
        model = model.to(memory_format=torch.channels_last)

    criterion = nn.CrossEntropyLoss().to(device)

    root_dir = args.data or args.data_dir

    data_config = resolve_data_config(
        vars(args),
        model=model,
        use_test_size=not args.use_train_size,
        verbose=True,
    )

    if args.input_img_mode is None:
        input_img_mode = 'RGB' if data_config['input_size'][0] == 3 else 'L'
    else:
        input_img_mode = args.input_img_mode

    dataset = create_dataset(
        root=root_dir,
        name=args.dataset,
        split=args.split,
        download=args.dataset_download,
        load_bytes=args.tf_preprocessing,
        class_map=args.class_map,
        num_samples=args.num_samples,
        input_key=args.input_key,
        input_img_mode=input_img_mode,
        target_key=args.target_key,
    )

    # if args.valid_labels:
    #     with open(args.valid_labels, 'r') as f:
    #         valid_labels = [int(line.rstrip()) for line in f]
    # else:
    valid_labels = None

    # if args.real_labels:
    #     real_labels = RealLabelsImagenet(dataset.filenames(basename=True), real_json=args.real_labels)
    # else:
    real_labels = None

    crop_pct = 1.0 if False else data_config['crop_pct']
    loader = create_loader(
        dataset,
        input_size=data_config['input_size'],
        batch_size=args.batch_size,
        use_prefetcher=args.prefetcher,
        interpolation=data_config['interpolation'],
        mean=data_config['mean'],
        std=data_config['std'],
        num_workers=args.workers,
        crop_pct=crop_pct,
        crop_mode=data_config['crop_mode'],
        crop_border_pixels=args.crop_border_pixels,
        pin_memory=args.pin_mem,
        device=device,
        tf_preprocessing=args.tf_preprocessing,
    )

    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    model.eval()

    with torch.no_grad():
        # warmup, reduce variability of first batch time, especially for comparing torchscript vs non
        input = torch.randn((args.batch_size,) + tuple(data_config['input_size'])).to(device)
        if args.channels_last:
            input = input.contiguous(memory_format=torch.channels_last)
        with amp_autocast():
            model(input)

        end = time.time()
        for batch_idx, (input, target) in enumerate(loader):
            if args.no_prefetcher:
                target = target.to(device)
                input = input.to(device)
            if args.channels_last:
                input = input.contiguous(memory_format=torch.channels_last)

            # compute output
            with amp_autocast():
                output = model(input)

                if valid_labels is not None:
                    output = output[:, valid_labels]
                loss = criterion(output, target)

            if real_labels is not None:
                real_labels.add_result(output)

            # measure accuracy and record loss
            acc1, acc5 = accuracy(output.detach(), target, topk=(1, 5))
            # print(loss.item(), input.size(0))
            losses.update(loss.item(), input.size(0))
            top1.update(acc1.item(), input.size(0))
            top5.update(acc5.item(), input.size(0))

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if batch_idx % args.log_freq == 0:
                _logger.info(
                    'Test: [{0:>4d}/{1}]  '
                    'Time: {batch_time.val:.3f}s ({batch_time.avg:.3f}s, {rate_avg:>7.2f}/s)  '
                    'Loss: {loss.val:>7.4f} ({loss.avg:>6.4f})  '
                    'Acc@1: {top1.val:>7.3f} ({top1.avg:>7.3f})  '
                    'Acc@5: {top5.val:>7.3f} ({top5.avg:>7.3f})'.format(
                        batch_idx,
                        len(loader),
                        batch_time=batch_time,
                        rate_avg=input.size(0) / batch_time.avg,
                        loss=losses,
                        top1=top1,
                        top5=top5
                    )
                )

    if real_labels is not None:
        # real labels mode replaces topk values at the end
        top1a, top5a = real_labels.get_accuracy(k=1), real_labels.get_accuracy(k=5)
    else:
        top1a, top5a = top1.avg, top5.avg
    results = OrderedDict(
        model=args.model,
        top1=round(top1a, 4), top1_err=round(100 - top1a, 4),
        top5=round(top5a, 4), top5_err=round(100 - top5a, 4),
        param_count=round(param_count / 1e6, 2),
        img_size=data_config['input_size'][-1],
        crop_pct=crop_pct,
        interpolation=data_config['interpolation'],
    )

    _logger.info(' * Acc@1 {:.3f} ({:.3f}) Acc@5 {:.3f} ({:.3f})'.format(
       results['top1'], results['top1_err'], results['top5'], results['top5_err']))

    return results 


def main():
    # Initialize the model
    patch_32_model = ViTB16_32(base_model_name='vit_base_patch16_224', img_size=224, patch_size=32, num_classes=1000, pretrained=True)

    setup_default_logging()
    args = parser.parse_args()
    
    args.model_eval = patch_32_model

    results = validate(args)

    if args.results_file:
        write_results(args.results_file, results, format=args.results_format)

    # output results in JSON to stdout w/ delimiter for runner script
    print(f'--result\n{json.dumps(results, indent=4)}')


def write_results(results_file, results, format='csv'):
    with open(results_file, mode='w') as cf:
        if format == 'json':
            json.dump(results, cf, indent=4)
        else:
            if not isinstance(results, (list, tuple)):
                results = [results]
            if not results:
                return
            dw = csv.DictWriter(cf, fieldnames=results[0].keys())
            dw.writeheader()
            for r in results:
                dw.writerow(r)
            cf.flush()



if __name__ == '__main__':
    main()






# class CustomViT(nn.Module):
#     def __init__(self, base_model_name: str, img_size: int, new_patch_size: int, num_classes: int = 1000, pretrained: bool = False):
#         super(CustomViT, self).__init__()
#         # Load the base model configuration from timm but without the pretrained weights for now
#         self.base_model = timm.create_model(base_model_name, img_size=img_size, num_classes=num_classes, pretrained=False)
        
#         # Calculate the new number of patches
#         self.num_patches = (img_size // new_patch_size) ** 2
#         self.new_patch_size = new_patch_size
        
#         # Replace the patch embedding convolution to the new patch size
#         self.base_model.patch_embed = nn.Conv2d(3, self.base_model.embed_dim, kernel_size=new_patch_size, stride=new_patch_size)

#         # Adjust positional embeddings size
#         new_pos_embed_shape = (1, self.num_patches + 1, self.base_model.embed_dim)  # +1 for the CLS token
#         self.base_model.pos_embed = nn.Parameter(torch.zeros(new_pos_embed_shape))

#         # Reinitialize positional embeddings and any other necessary parameters
#         self.reset_parameters()

#         if pretrained:
#             self.load_pretrained_weights(base_model_name)

#     def reset_parameters(self):
#         nn.init.normal_(self.base_model.pos_embed, std=0.02)
#         # TODO add patch embedding here.
#         # Reinitialize other parameters if necessary

#     def forward(self, x):
#         return self.base_model(x)

#     def load_pretrained_weights(self, base_model_name):
#         # Load the original pre-trained weights
#         pretrained_weights = timm.create_model(base_model_name, pretrained=True).state_dict()

#         # Prepare the custom model's state dict for selective weight loading
#         custom_state_dict = self.state_dict()

#         # Iterate through the pre-trained weights and transfer them if compatible
#         for name, param in pretrained_weights.items():
#             if name not in ["pos_embed", "patch_embed.proj.weight", "patch_embed.proj.bias"]: # CHECK bias term
#                 custom_state_dict[name] = param
#             else:
#                 print(f"Skipping {name} due to being reinit " ) #to size mismatch or it does not exist in the custom model.")

#         # Load the updated state dict into the custom model
#         self.load_state_dict(custom_state_dict)

# Initialize your custom model
# custom_model = (base_model_name='vit_base_patch16_224', img_size=224, new_patch_size=32, num_classes=1000, pretrained=True)

# Now custom_model can be used for further training or inference as required
