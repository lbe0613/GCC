import os
import json
import argparse
import time
import sys
sys.path.append('../')

from torch.utils.data import Dataset, DataLoader
from torch.nn.utils.rnn import pad_sequence
from utils.progressbar import ProgressBar
from utils.common import init_logger, logger, seed_everything
from utils.data_parallel import DataParallel
from model import *

from transformers import BertConfig


class InputExample(object):
    def __init__(self, input_id, stru_mask, pos_id, seg_id, type_id, char_label, char_pos):
        self.input_id = input_id
        self.input_len = len(input_id)
        self.stru_mask = stru_mask
        self.pos_id = pos_id
        self.seg_id = seg_id
        self.type_id = type_id
        self.char_label = char_label
        self.char_pos = char_pos

class SememeDataset(Dataset):
    def __init__(self, features):
        self.data = features

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


def get_dataset(data_type='train'):
    if data_type == 'train':
        path = args.train_dir
    elif data_type == 'dev':
        path = args.dev_dir

    with open(path , 'r')as f:
        data = f.readlines()

    with open(args.char_dict,'r')as f:
        char_dict = json.load(f)

    examples = []
    for line in data:
        input_id = []
        stru_mask = []
        pos_id = []
        seg_id = []
        type_id = []
        line = line.strip()
        char_label = []
        char_pos = [0] # The first token must be [CHAR]

        for i, char in enumerate(line):
            sample = char_dict[char]
            input_id.extend(sample['src_idx'])
            stru_mask.extend(sample['type_mask'])
            pos = list(range(1, len(sample['src_idx'])+1))
            pos_id.extend(pos)
            seg_id.extend([i+1] * len(sample['src_idx']))

            char_label.append(sample['word_id'])
            char_label.extend([-100] * (len(sample['src_idx'])-1))
            char_pos.append(char_pos[-1]+len(sample['src_idx']))

            # 1 for [CHAR], 2 for CPN, 3 for STC
            type_x = [x + 2 for x in sample['type_mask']]
            type_x[0] = 1
            type_id.extend(type_x)

        examples.append(InputExample(input_id, stru_mask, pos_id, seg_id, type_id, char_label, char_pos))

    dataset = SememeDataset(examples)
    return dataset

def get_special_tokens_mask(token_ids_0):
    all_special_ids = [0, 1, 2, 3, 4]

    special_tokens_mask = [1 if token in all_special_ids else 0 for token in token_ids_0]
    return special_tokens_mask

def mask_tokens(inputs, mask):
    """
    Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original.
    """
    mlm_probability = 0.15
    pad_token_id = 0
    mask_token_id = 3
    labels = inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability)
    special_tokens_mask = [
        get_special_tokens_mask(val) for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool), value=0.0)

    padding_mask = labels.eq(pad_token_id)
    probability_matrix.masked_fill_(padding_mask, value=0.0)
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens
    y = torch.full_like(labels, -100)

    # 0-1175 is for component tokens, 1176-1190 is for structure tokens, but in our model these two types will be embeded and predicted separately.
    # That's why we map stru_label from 1176-1190 to 0-14.
    # For convenience, the range of inputs here is still 0-1190, and we map strc_ids when forwarding the model.
    stru_label = torch.where(labels > 1175, labels - 1176, y)
    comp_label = torch.where((labels > 4) & (labels < 1176), labels, y)

    # 80% of the time, we replace masked input tokens with mask_token
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8)).bool() & masked_indices
    inputs[(indices_replaced * mask).to(bool)] = 1190 # [S_M]

    mask_comp = (indices_replaced.to(int) - mask) == 1
    inputs[mask_comp] = mask_token_id # [C_M]

    # 10% of the time, we replace masked input tokens with random token
    indices_random = torch.bernoulli(torch.full(labels.shape, 0.5)).bool() & masked_indices & ~indices_replaced
    random_comp = torch.randint(5, 1176, labels.shape, dtype=torch.long).to(inputs.dtype)
    mask_r = (indices_random.to(int) - mask) == 1
    inputs[mask_r] = random_comp[mask_r]
    random_stru = torch.randint(1176, 1190, labels.shape, dtype=torch.long).to(inputs.dtype)
    inputs[(indices_random * mask).to(bool)] = random_stru[(indices_random * mask).to(bool)]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, stru_label, comp_label



def collate(batch):
    batch_id = [torch.Tensor(e.input_id) for e in batch]
    stru_mask = [torch.Tensor(e.stru_mask) for e in batch]
    batch_id = pad_sequence(batch_id, batch_first=True, padding_value=0)
    stru_mask = pad_sequence(stru_mask, batch_first=True, padding_value=0)

    inputs, stru_label, comp_label = mask_tokens(batch_id, stru_mask)

    char_sep_ids = [torch.Tensor(e.seg_id) for e in batch]
    position_ids = [torch.Tensor(e.pos_id) for e in batch]
    type_ids = [torch.Tensor(e.type_id) for e in batch]
    char_label = [torch.Tensor(e.char_label) for e in batch]
    char_sep_ids = pad_sequence(char_sep_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
    char_label = pad_sequence(char_label, batch_first=True, padding_value=-100)

    attention_mask = []
    for e in batch:
        mat = torch.zeros((inputs.shape[1],inputs.shape[1]))
        char_pos = e.char_pos
        for i in range(len(char_pos)-1):
            b = char_pos[i]
            end = char_pos[i+1]
            mat[b:end, b:end] = 1
            mat[b, char_pos[:-1]] = 1

        attention_mask.append(mat)

    attention_mask = torch.stack(attention_mask)


    return {"input_ids": inputs.to(torch.long).to(device), "stru_mask": stru_mask.to(torch.long).to(device), "attention_mask": attention_mask.to(device),
             "position_ids": position_ids.to(torch.long).to(device), "type_ids": type_ids.to(torch.long).to(device),
            "char_label": char_label.to(torch.long).to(device), "stru_label": stru_label.to(torch.long).to(device), "comp_label": comp_label.to(torch.long).to(device),
            "char_sep_ids": char_sep_ids.to(torch.long).to(device)}


def evaluate(args, eval_dataset, model):
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    total_correct_char, total_denominator_char = 0., 0.
    total_correct_comp, total_denominator_comp = 0., 0.
    total_correct_stru, total_denominator_stru = 0., 0.
    loss_eval = 0.

    pbar = ProgressBar(n_total=len(eval_dataloader), desc='Evaluating')
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        with torch.no_grad():
            loss, char_pred, comp_pred, stru_pred, char_cor_num, comp_cor_num, stru_cor_num, char_num, comp_num, stru_num = model(batch)

        total_correct_char += char_cor_num.sum().item()
        total_correct_comp += comp_cor_num.sum().item()
        total_correct_stru += stru_cor_num.sum().item()
        total_denominator_char += char_num.sum().item()
        total_denominator_comp += comp_num.sum().item()
        total_denominator_stru += stru_num.sum().item()
        loss_eval += loss.mean().item()
        pbar(step)

    print(' ')

    return total_correct_char/total_denominator_char, total_correct_comp/total_denominator_comp, total_correct_stru/total_denominator_stru, loss_eval/len(eval_dataloader)


def main():
    model = Net(config)
    model.to(device)

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
        {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
    ]
    optimizer = torch.optim.AdamW(optimizer_grouped_parameters, lr=args.learning_rate)

    if args.n_gpu > 1:
        model = DataParallel(model)

    total_correct_char, total_denominator_char = 0., 0.
    total_correct_comp, total_denominator_comp = 0., 0.
    total_correct_stru, total_denominator_stru = 0., 0.

    max_dev = [0]
    step_cnt = 0
    while True:
        eval_dataset = get_dataset(data_type='dev')
        train_dataset = get_dataset(data_type='train')
        logger.info(f"NUM train:{len(train_dataset)}\tNUM dev:{len(eval_dataset)}")
        train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

        train_loss = 0.
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for batch in train_dataloader:
            model.train()
            loss, char_pred, comp_pred, stru_pred, char_cor_num, comp_cor_num, stru_cor_num, char_num, comp_num, stru_num = model(batch)

            if args.n_gpu > 1:
                loss = loss.mean()
            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()
            step_cnt += 1

            if step_cnt % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()


            train_loss += loss.item()
            total_correct_char += char_cor_num.sum().item()
            total_correct_comp += comp_cor_num.sum().item()
            total_correct_stru += stru_cor_num.sum().item()
            total_denominator_char += char_num.sum().item()
            total_denominator_comp += comp_num.sum().item()
            total_denominator_stru += stru_num.sum().item()


            if step_cnt % args.logging_steps == 0:
                char_acc, comp_acc, stru_acc, loss_eval = evaluate(args, eval_dataset, model)

                logger.info(f"STEP:{step_cnt}\tLOSS:{train_loss / args.logging_steps:.8f}\tDEV LOSS:{loss_eval:.8f}")
                logger.info(f"TRAIN:\tchar_acc:{total_correct_char / total_denominator_char:.6f}\tcomp_acc:{total_correct_comp / total_denominator_comp:.6f}\tstru_acc:{total_correct_stru / total_denominator_stru:.6f}")
                logger.info(f"DEV:\tchar_acc:{char_acc:.6f}\tcomp_acc:{comp_acc:.6f}\tstru_acc:{stru_acc:.6f}")
                train_loss = 0.
                total_correct_char, total_denominator_char = 0., 0.
                total_correct_comp, total_denominator_comp = 0., 0.
                total_correct_stru, total_denominator_stru = 0., 0.

                if char_acc > max_dev[-1]:
                    if not os.path.exists(args.output_dir + '/model'):
                        os.makedirs(args.output_dir + '/model')
                    state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step_cnt}
                    torch.save(state,
                               os.path.join(args.output_dir + '/model', localtime + '_' + args.name + '_' + str(args.seed) + '.ckp'))
                    logger.info(f"save model")
                    max_dev.append(char_acc)

                if 'cuda' in str(device):
                    torch.cuda.empty_cache()

            if step_cnt % args.saving_steps == 0:
                if not os.path.exists(args.output_dir + '/model'):
                    os.makedirs(args.output_dir + '/model')
                state = {'model': model.state_dict(), 'optimizer': optimizer.state_dict(), 'step': step_cnt}
                torch.save(state,
                           os.path.join(args.output_dir + '/model',
                                        localtime + '_' + args.name + '_' + str(step_cnt) +'_' + str(args.seed) + '.ckp'))
                logger.info(f"save model")

            pbar(step_cnt, {'loss': loss.item()})


        print(" ")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

        if step_cnt == args.total_train_steps * args.gradient_accumulation_steps:
            break


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--train_dir", default="corpus/train.txt", type=str,
                        help="train data")
    parser.add_argument("--dev_dir", default="corpus/dev.txt", type=str,
                        help="dev data")
    parser.add_argument("--char_dict", default="../data/han_seq.json", type=str,
                        help="convert Chinese character into sequence")
    parser.add_argument("--name", default="pretrain", type=str,
                        help="name")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument("--total_train_steps", default=100000, type=int,
                        help="Total number of training steps to perform.")
    parser.add_argument("--num_train_epochs", default=2, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=8, help="gradient_accumulation_steps")
    parser.add_argument('--logging_steps', type=int, default=1000,
                        help="Log every X updates steps.")
    parser.add_argument('--saving_steps', type=int, default=10000,
                        help="Save every X updates steps.")

    args = parser.parse_args()

    seed_everything(args.seed)

    localtime = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.n_gpu = torch.cuda.device_count()
    output_dir = args.output_dir
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    init_logger(log_file=args.output_dir + '/' + localtime + '_' + args.name + '_' + str(args.seed) + '.log')
    logger.info("Training/evaluation parameters %s", args)
    # The number of component tokens is 1176, ../data/GCC_vocab.json is the dictionary
    config = BertConfig.from_pretrained('../data/config.json', vocab_size=1176)

    main()
