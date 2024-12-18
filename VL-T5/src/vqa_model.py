from pathlib import Path
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from modeling_t5 import VLT5
class VLT5VQA(VLT5):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        lm_input_ids = batch['lm_input_ids'].to(device)
        itm_input_ids = batch['itm_input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )
            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            assert 'loss' in output
            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()
            loss = output['loss']
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
            loss = loss * batch['scores'].to(device=device)
            loss = loss.mean()

            #####lm_loss
            lm_lm_labels = batch["lm_target_ids"].to(device)
            lm_output = self(
                input_ids=lm_input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_lm_labels,
                return_dict=True
            )
            assert 'loss' in lm_output
            lm_lm_mask = (lm_lm_labels != -100).float()
            B, L = lm_lm_labels.size()
            aux_loss_0 = lm_output['loss']
            aux_loss_0 = aux_loss_0.view(B, L) * lm_lm_mask
            aux_loss_0 = aux_loss_0.sum(dim=1) / lm_lm_mask.sum(dim=1).clamp(min=1)  # B
            aux_loss_0 = aux_loss_0.to(device=device)
            aux_loss_0 = aux_loss_0.mean()

            #####itm_loss
            itm_lm_labels = batch["itm_target_ids"].to(device)
            itm_output = self(
                input_ids=itm_input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=itm_lm_labels,
                return_dict=True
            )
            assert 'loss' in itm_output
            itm_lm_mask = (itm_lm_labels != -100).float()
            B, L = itm_lm_labels.size()
            aux_loss_1 = itm_output['loss']
            aux_loss_1 = aux_loss_1.view(B, L) * itm_lm_mask
            aux_loss_1 = aux_loss_1.sum(dim=1) / itm_lm_mask.sum(dim=1).clamp(min=1)  # B
            aux_loss_1 = aux_loss_1.to(device=device)
            aux_loss_1 = aux_loss_1.mean()

        result = {
            'loss': loss * self.config.losses_weight['vqa'],
            'lm_loss': aux_loss_0 * self.config.losses_weight['lm'],
            'itm_loss': aux_loss_1 * self.config.losses_weight['itm']
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.ones(
                B, 1, dtype=torch.long, device=device) * self.config.decoder_start_token_id

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:
            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result

from modeling_bart import VLBart
class VLBartVQA(VLBart):
    def __init__(self, config, num_answers=None, label2ans=None):
        super().__init__(config)

        if config.classifier:
            self.answer_head = nn.Sequential(
                nn.Linear(config.d_model, config.d_model * 2),
                nn.GELU(),
                nn.LayerNorm(config.d_model * 2),
                nn.Linear(config.d_model * 2, num_answers)
            )

        self.num_answers = num_answers
        self.label2ans = label2ans
        self.bce_loss = nn.BCEWithLogitsLoss()

    def train_step(self, batch):

        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        lm_input_ids = batch['lm_input_ids'].to(device)
        itm_input_ids = batch['itm_input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            target = batch['targets'].to(device)

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            loss = self.bce_loss(logit, target)

        else:
            lm_labels = batch["target_ids"].to(device)
            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_labels,
                return_dict=True
            )
            assert 'loss' in output

            lm_mask = (lm_labels != -100).float()
            B, L = lm_labels.size()
            loss = output['loss']
            loss = loss.view(B, L) * lm_mask
            loss = loss.sum(dim=1) / lm_mask.sum(dim=1).clamp(min=1)  # B
            loss = loss * batch['scores'].to(device=device)
            loss = loss.mean()

            #####lm_loss
            lm_lm_labels = batch["lm_target_ids"].to(device)
            lm_output = self(
                input_ids=lm_input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=lm_lm_labels,
                return_dict=True
            )
            assert 'loss' in lm_output
            lm_lm_mask = (lm_lm_labels != -100).float()
            B, L = lm_lm_labels.size()
            aux_loss_0 = lm_output['loss']
            aux_loss_0 = aux_loss_0.view(B, L) * lm_lm_mask
            aux_loss_0 = aux_loss_0.sum(dim=1) / lm_lm_mask.sum(dim=1).clamp(min=1)  # B
            aux_loss_0 = aux_loss_0.to(device=device)
            aux_loss_0 = aux_loss_0.mean()

            #####itm_loss
            itm_lm_labels = batch["itm_target_ids"].to(device)
            itm_output = self(
                input_ids=itm_input_ids,
                vis_inputs=(vis_feats, vis_pos),
                labels=itm_lm_labels,
                return_dict=True
            )
            assert 'loss' in itm_output
            itm_lm_mask = (itm_lm_labels != -100).float()
            B, L = itm_lm_labels.size()
            aux_loss_1 = itm_output['loss']
            aux_loss_1 = aux_loss_1.view(B, L) * itm_lm_mask
            aux_loss_1 = aux_loss_1.sum(dim=1) / itm_lm_mask.sum(dim=1).clamp(min=1)  # B
            aux_loss_1 = aux_loss_1.to(device=device)
            aux_loss_1 = aux_loss_1.mean()

        result = {
            'loss': loss * self.config.losses_weight['vqa'],
            'lm_loss': aux_loss_0 * self.config.losses_weight['lm'],
            'itm_loss': aux_loss_1 * self.config.losses_weight['itm']
        }

        return result

    @torch.no_grad()
    def test_step(self, batch, **kwargs):
        self.eval()
        device = next(self.parameters()).device
        vis_feats = batch['vis_feats'].to(device)
        input_ids = batch['input_ids'].to(device)
        vis_pos = batch['boxes'].to(device)

        result = {}
        if self.config.classifier:
            B = len(input_ids)

            decoder_input_ids = torch.tensor(
                [self.config.decoder_start_token_id, self.config.bos_token_id],
                dtype=torch.long, device=device).unsqueeze(0).expand(B, 2)

            output = self(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                decoder_input_ids=decoder_input_ids,
                output_hidden_states=True,
                return_dict=True
            )

            last_layer_hidden_state = output.decoder_hidden_states[-1]
            last_hidden_state = last_layer_hidden_state.view(B, -1, self.config.d_model)[:, -1]

            # [B, num_answers]
            logit = self.answer_head(last_hidden_state)

            score, pred_ans_id = logit.max(1)
            pred_ans_id = pred_ans_id.cpu().numpy()
            pred_ans = [self.label2ans[ans_id] for ans_id in pred_ans_id]

            result['pred_ans'] = pred_ans

        else:

            output = self.generate(
                input_ids=input_ids,
                vis_inputs=(vis_feats, vis_pos),
                **kwargs
            )
            generated_sents = self.tokenizer.batch_decode(output, skip_special_tokens=True)
            result['token_ids'] = output
            result['pred_ans'] = generated_sents

        return result