import torch.backends.cudnn as cudnn
import torch.multiprocessing as mp
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import os
import collections
from pathlib import Path
from packaging import version

import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
import logging
import shutil
from pprint import pprint

from utils import load_state_dict, LossMeter, set_global_logging_level
import wandb
from pprint import pformat

proj_dir = Path(__file__).resolve().parent.parent

_use_native_amp = False
_use_apex = False

# Check if Pytorch version >= 1.6 to switch between Native AMP and Apex
if version.parse(torch.__version__) < version.parse("1.6"):
    from transormers.file_utils import is_apex_available
    if is_apex_available():
        from apex import amp
    _use_apex = True
else:
    _use_native_amp = True
    from torch.cuda.amp import autocast


class TrainerBase(object):
    def __init__(self, args, train_loader=None, val_loader=None, test_loader=None, train=True):
        self.args = args

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader

        self.best_epoch = 0.0
        self.best_valid = 0.0
        self.verbose = True
        if self.args.distributed:
            if self.args.gpu != 0:
                self.verbose = False

        if self.args.tokenizer is None:
            self.args.tokenizer = self.args.backbone

        if not self.verbose:
            set_global_logging_level(logging.ERROR, ["transformers"])

    def create_config(self):
        from transformers import T5Config, BartConfig, BertConfig

        if 't5' in self.args.backbone:
            config_class = T5Config
        elif 'bart' in self.args.backbone:
            config_class = BartConfig
        else:
            return None

        config = config_class.from_pretrained(self.args.backbone)
        bert_config = BertConfig.from_pretrained('bert-base-uncased')

        args = self.args

        config.feat_dim = args.feat_dim
        config.pos_dim = args.pos_dim
        config.n_images = 2

        config.use_vis_order_embedding = args.use_vis_order_embedding

        config.dropout_rate = args.dropout
        config.dropout = args.dropout
        config.attention_dropout = args.dropout
        config.activation_dropout = args.dropout

        config.use_vis_layer_norm = args.use_vis_layer_norm
        config.individual_vis_layer_norm = args.individual_vis_layer_norm
        config.losses = args.losses
        config.losses_weight = args.losses_weight

        config.share_vis_lang_layer_norm = args.share_vis_lang_layer_norm
        config.classifier = args.classifier

        #####adapter
        config.bert_type = args.bert_type
        config.ft_large = args.ft_large
        config.l_layers = args.num_l_layers
        config.r_layers = args.num_r_layers
        config.x_layers = args.num_x_layers
        config.hidden_dropout_prob = args.dropout
        config.MV_size = args.MV_size
        config.ML_size = args.ML_size
        config.cross_type = args.cross_type

        config.intermediate_size = bert_config.intermediate_size
        config.hidden_act = bert_config.hidden_act
        config.attention_probs_dropout_prob = bert_config.attention_probs_dropout_prob
        config.initializer_range = bert_config.initializer_range

        return config


    def create_model(self, model_class, config=None, **kwargs):
        print(f'Building Model at GPU {self.args.gpu}')

        model_name = self.args.backbone

        model = model_class.from_pretrained(
            model_name,
            config=config,
            **kwargs
        )
        return model

    def create_tokenizer(self, **kwargs):
        from transformers import T5Tokenizer, BartTokenizer, T5TokenizerFast, BartTokenizerFast
        from tokenization import VLT5Tokenizer, VLT5TokenizerFast

        if 't5' in self.args.tokenizer:
            if self.args.use_vision:
                # tokenizer_class = VLT5Tokenizer
                tokenizer_class = VLT5TokenizerFast
            else:
                # tokenizer_class = T5Tokenizer
                tokenizer_class = T5TokenizerFast
        elif 'bart' in self.args.tokenizer:
            tokenizer_class = BartTokenizer
            # tokenizer_class = BartTokenizerFast

        tokenizer_name = self.args.backbone

        tokenizer = tokenizer_class.from_pretrained(
            tokenizer_name,
            max_length=self.args.max_text_length,
            do_lower_case=self.args.do_lower_case,
            **kwargs
            )

        return tokenizer

    def create_optimizer_and_scheduler(self):
        if self.verbose:
            print('Building Optimizer')

        lr_scheduler = None

        if 'adamw' in self.args.optim:
            from transformers.optimization import AdamW, get_linear_schedule_with_warmup
            batch_per_epoch = len(self.train_loader)
            t_total = batch_per_epoch // self.args.gradient_accumulation_steps * self.args.epochs
            warmup_ratio = self.args.warmup_ratio
            warmup_iters = int(t_total * warmup_ratio)
            if self.verbose:
                print("Batch per epoch: %d" % batch_per_epoch)
                print("Total Iters: %d" % t_total)
                print('Warmup ratio:', warmup_ratio)
                print("Warm up Iters: %d" % warmup_iters)

            no_decay = ["bias", "LayerNorm.weight"]
            optimizer_grouped_parameters = [
                {
                    "params": [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
                    "weight_decay": self.args.weight_decay,
                },
                {
                    "params": [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)],
                    "weight_decay": 0.0,
                },
            ]

            optim = AdamW(optimizer_grouped_parameters,
                          lr=self.args.lr, eps=self.args.adam_eps)
            lr_scheduler = get_linear_schedule_with_warmup(
                optim, warmup_iters, t_total)

        else:
            optim = self.args.optimizer(
                list(self.model.parameters()), self.args.lr)

        return optim, lr_scheduler

    def load_checkpoint(self, ckpt_path):
        state_dict = load_state_dict(ckpt_path, 'cpu')

        original_keys = list(state_dict.keys())
        if 'model' in original_keys:
            state_dict = state_dict['model']
        else:
            for key in original_keys:
                if key.startswith("vis_encoder."):
                    new_key = 'encoder.' + key[len("vis_encoder."):]
                    state_dict[new_key] = state_dict.pop(key)

                if key.startswith("model.vis_encoder."):
                    new_key = 'model.encoder.' + key[len("model.vis_encoder."):]
                    state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', ckpt_path)
            pprint(results)

    def init_weights(self):

        def init_bert_weights(module):
            """ Initialize the weights."""
            if isinstance(module, (nn.Linear, nn.Embedding)):
                # Slightly different from the TF version which uses truncated_normal for initialization
                module.weight.data.normal_(mean=0.0, std=1)
            elif isinstance(module, nn.LayerNorm):
                module.bias.data.zero_()
                module.weight.data.fill_(1.0)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        self.model.apply(init_bert_weights)
        self.model.init_weights()

    def predict(self):
        pass

    def evaluate(self):
        pass

    def save(self, name=None, epoch=-1):
        if not os.path.isdir(self.args.output):
            os.makedirs(self.args.output, exist_ok=True)
        if epoch != -1:
            checkpoint_paths = [self.args.output + '/checkpoint.pth', self.args.output + f'/checkpoint{epoch:03}.pth']
        elif name is not None:
            checkpoint_paths = [self.args.output + f'/checkpoint_{name}.pth']
        for checkpoint_path in checkpoint_paths:
            weights = {
                'model': self.model_without_ddp.state_dict(),
                'optimizer': self.optim.state_dict(),
                'lr_scheduler': self.lr_scheduler.state_dict(),
                'epoch': epoch,
                'args': self.args,
            }
            torch.save(weights, checkpoint_path)

    def load(self, path, loc=None):
        if loc is None and hasattr(self.args, 'gpu'):
            loc = f'cuda:{self.args.gpu}'
        state_dict = torch.load("%s.pth" % path, map_location=loc)

        original_keys = list(state_dict.keys())
        for key in original_keys:
            if key.startswith("module.vis_encoder."):
                new_key = 'module.encoder.' + key[len("module.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

            if key.startswith("module.model.vis_encoder."):
                new_key = 'module.model.encoder.' + key[len("module.model.vis_encoder."):]
                state_dict[new_key] = state_dict.pop(key)

        results = self.model.load_state_dict(state_dict, strict=False)
        if self.verbose:
            print('Model loaded from ', path)
            pprint(results)

    def masking(self, img, txt, vq):
        (_B, _T, _, _H, _W), (_, _X) = img.shape, txt.shape
        _h, _w = _H // 32, _W // 32
        spc_txt = torch.logical_or(torch.logical_or(txt == 101, txt == 102), txt == 0)

        ans_mtm, ans_mvm = torch.ones(txt.shape).long() * -1, torch.ones(vq.shape).long() * -1
        for i in range(_B):
            mask_mtm = torch.where(torch.logical_and(torch.logical_not(spc_txt[i]), torch.rand(_X) < 0.15))[0]
            while len(mask_mtm) == 0:
                mask_mtm = torch.where(torch.logical_and(torch.logical_not(spc_txt[i]), torch.rand(_X) < 0.15))[0]

            mask_mvm = set()
            for _ in range(_T):
                t, h, w = [np.random.randint(1, _T) if _T > 1 else 1,
                           np.random.randint(1, _h * 2 // 3), np.random.randint(1, _w * 2 // 3)]
                t1, h1, w1 = [np.random.randint(0, _T - t + 1),
                              np.random.randint(0, _h - h + 1), np.random.randint(0, _w - w + 1)]
                for i_t in range(t1, t1 + t):
                    for i_h in range(h1, h1 + h):
                        for i_w in range(w1, w1 + w):
                            mask_mvm.add((i_t, i_h, i_w))
            mask_mvm = list(mask_mvm)

            for p in mask_mtm:
                ans_mtm[i][p], txt[i][p] = txt[i][p], 103

            cov = torch.zeros(_T, _h, _w)
            for i_t, i_h, i_w in mask_mvm:
                cov[i_t][i_h][i_w] = 1.0
                p = (1 + _h * _w) * i_t + 1 + i_h * _w + i_w
                ans_mvm[i][p] = vq[i][p]
            cov = cov.unsqueeze(1).unsqueeze(3).unsqueeze(5).expand([-1, 3, -1, 32, -1, 32])
            cov = cov.flatten(2, 3).flatten(3, 4)
            img[i] *= (1.0 - cov)
        return img, txt, ans_mtm, ans_mvm
