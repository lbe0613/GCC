import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from utils.Bert import BertForMaskedLM

class Net(BertForMaskedLM):
    def __init__(self, config):
        super().__init__(config)
        self.stru_embeddings = nn.Embedding(15, config.hidden_size)
        self.stru_lm_head = StruLMHead(config, 15)

        self.char_embedding = nn.Linear(config.hidden_size, 20902)
        self.apply(self._init_weights)
        self.tie_rel_weights()

    def tie_rel_weights(self):
        self.stru_lm_head.decoder.weight = self.stru_embeddings.weight
        if getattr(self.stru_lm_head.decoder, "bias", None) is not None:
            self.stru_lm_head.decoder.bias.data = torch.nn.functional.pad(
                self.stru_lm_head.decoder.bias.data,
                (0, self.stru_lm_head.decoder.weight.shape[0] - self.stru_lm_head.decoder.bias.shape[0],),
                "constant",
                0,
            )

    def forward(self, batch):
        input_ids_origin = batch['input_ids']
        stru_mask = batch['stru_mask']
        char_label = batch['char_label']
        comp_label = batch['comp_label']
        stru_label = batch['stru_label']

        strc_ids = input_ids_origin * stru_mask - 1176
        y = torch.full_like(strc_ids, 0)
        strc_ids = torch.where(strc_ids >= 0, strc_ids, y)
        input_ids = input_ids_origin * (1 - stru_mask)

        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        stru_embeddings = self.stru_embeddings(strc_ids)
        inputs_embeds = stru_embeddings * stru_mask.unsqueeze(-1) + word_embeddings * (1 - stru_mask).unsqueeze(-1)


        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        type_ids = batch['type_ids']
        char_sep_ids = batch['char_sep_ids']

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            type_ids=type_ids,
            char_sep_ids=char_sep_ids,
            inputs_embeds=inputs_embeds,
            batch=batch
        )
        sequence_output = outputs[0]  # batch x seq_len x hidden_size

        loss_fct = CrossEntropyLoss(ignore_index=-100, reduction='mean')

        char_logits = self.char_embedding(sequence_output)
        char_predict = torch.argmax(char_logits, dim=-1)
        char_masked_lm_loss = loss_fct(char_logits.view(-1, char_logits.size(-1)), char_label.view(-1))
        char_mat = char_label != -100
        char_correct = (char_predict == char_label) & char_mat

        comp_logits = self.cls(sequence_output)
        comp_predict = torch.argmax(comp_logits, dim=-1)
        masked_lm_loss = loss_fct(comp_logits.view(-1, self.config.vocab_size), comp_label.view(-1))
        comp_mat = comp_label != -100
        comp_correct = (comp_predict == comp_label) & comp_mat

        stru_logits = self.stru_lm_head(sequence_output)
        stru_predict = torch.argmax(stru_logits, dim=-1)
        stru_masked_lm_loss = loss_fct(stru_logits.view(-1, stru_logits.size(-1)), stru_label.view(-1))
        stru_mat = stru_label != -100
        stru_correct = (stru_predict == stru_label) & stru_mat

        loss = masked_lm_loss + char_masked_lm_loss + stru_masked_lm_loss
        return (loss, char_predict, comp_predict, stru_predict, char_correct.sum(), comp_correct.sum(), stru_correct.sum(),
                char_mat.sum(), comp_mat.sum(), stru_mat.sum())




class StruLMHead(nn.Module):
    def __init__(self, config, num_stru):
        super().__init__()
        self.dense = nn.Linear(config.hidden_size, config.hidden_size)
        self.layer_norm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)

        self.decoder = nn.Linear(config.hidden_size, num_stru, bias=False)
        self.bias = nn.Parameter(torch.zeros(num_stru), requires_grad=True)

        self.decoder.bias = self.bias

    def forward(self, features, **kwargs):
        x = self.dense(features)
        x = F.gelu(x)
        x = self.layer_norm(x)
        x = self.decoder(x)
        return x
