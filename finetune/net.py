import torch
from torch import nn
from torch.nn import MultiLabelSoftMarginLoss
from utils.Bert import BertForMaskedLM

class Sememe_Net(BertForMaskedLM):
    def __init__(self, config, args, embedding_matrix):
        super().__init__(config)
        self.stru_embeddings = nn.Embedding(15, config.hidden_size)
        self.prj_pre_train = nn.Linear(config.hidden_size, args.hidden_size, bias=False)
        self.prj_word = nn.Linear(args.embed_dim, args.hidden_size, bias=False)

        self.encoder = nn.Embedding(args.word_size, args.embed_dim, padding_idx=0, max_norm=5, sparse=True)
        self.encoder.weight.data.copy_(torch.from_numpy(embedding_matrix))
        self.encoder.weight.requires_grad = False

        self.lstm = nn.LSTM(args.hidden_size,
                            args.hidden_size,
                            batch_first=True,
                            bidirectional=True)

        self.linear = nn.Linear(args.hidden_size * 2, args.num_classes)


    def forward(self, batch):
        src = batch["src"]
        mask = torch.gt(src, 0).to(torch.int64)
        x_len = torch.sum(mask, dim=1)
        x_word_embedding = self.encoder(src)
        x = self.prj_word(x_word_embedding)

        # load pretrain GCC
        input_ids_origin = batch['input_ids']
        stru_mask = batch['stru_mask']
        attention_mask = batch['attention_mask']
        position_ids = batch['position_ids']
        type_ids = batch['type_ids']
        char_sep_ids = batch['char_sep_ids']

        strc_ids = input_ids_origin * stru_mask - 1176
        y = torch.full_like(strc_ids, 0)
        strc_ids = torch.where(strc_ids >= 0, strc_ids, y)
        input_ids = input_ids_origin * (1 - stru_mask)

        word_embeddings = self.bert.embeddings.word_embeddings(input_ids)
        stru_embeddings = self.stru_embeddings(strc_ids)
        inputs_embeds = stru_embeddings * stru_mask.unsqueeze(-1) + word_embeddings * (1 - stru_mask).unsqueeze(-1)

        outputs = self.bert(
            input_ids=None,
            attention_mask=attention_mask,
            position_ids=position_ids,
            type_ids=type_ids,
            char_sep_ids=char_sep_ids,
            inputs_embeds=inputs_embeds,
            batch=batch
        )
        sequence_output = outputs[0]

        char_label = batch['char_label']
        char_label_mask = ~char_label.eq(-100)
        # use the representation of [CHAR] only
        sequence_output = sequence_output * char_label_mask.unsqueeze(-1)

        char_pos = batch['char_position']
        char_pos = char_pos.unsqueeze(-1).repeat(1, 1, sequence_output.shape[-1])
        pretrain_char = torch.gather(sequence_output, 1, char_pos.to(torch.long))
        pretrain_char = self.prj_pre_train(pretrain_char)

        x = x + pretrain_char

        packed_embedd = nn.utils.rnn.pack_padded_sequence(x,
                                                          x_len,
                                                          batch_first=True,
                                                          enforce_sorted=False)
        output, hidden = self.lstm(packed_embedd)
        output, output_lengths = nn.utils.rnn.pad_packed_sequence(output, batch_first=True)

        cls = output[:, 0, :]

        output = self.linear(cls)

        loss_fct = MultiLabelSoftMarginLoss()
        loss = loss_fct(output, batch["label"])

        return loss, output
