import argparse
import json
import numpy as np
import sys
import os
import time

sys.path.append('../')

from utils.progressbar import ProgressBar
from utils.common import init_logger, logger, seed_everything
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import Dataset, DataLoader
from transformers import BertConfig
from net import *

class InputExample(object):
    def __init__(self, word_idx, src, label, input_id=None, stru_mask=None, pos_id=None, seg_id=None, type_id=None, char_label=None, char_pos=None):
        self.src = src
        self.label = label
        self.word = word_idx
        self.input_id = input_id
        self.input_len = len(input_id)
        self.stru_mask = stru_mask
        self.pos_id = pos_id
        self.seg_id = seg_id
        self.type_id = type_id
        self.char_pos = char_pos
        self.char_label = char_label

class SememeDataset(Dataset):
    def __init__(self, features):
        self.data = features

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)

def load_examples(char_dict, data_type='train'):
    with open(args.word2index_path, 'r') as f:
        word2index_data = json.load(f)

    if data_type == 'train':
        path = args.train_dir
    elif data_type == 'dev':
        path=args.dev_dir
    else:
        path=args.test_dir

    with open(path, 'r')as f:
        data = json.load(f)

    examples = []
    for values in data:
        word = values['word']
        word_idx = word2index_data[word]
        src = [word2index_data[word]]

        input_id = [1]
        stru_mask = [0]
        pos_id = [0]
        seg_id = [0]
        type_id = [0]
        char_label = [-100]
        char_pos = [0, 1]

        for i, char in enumerate(word):
            if char in word2index_data:
                src.append(word2index_data[char])
            else:
                src.append(1) # OOV

            sample = char_dict[char]
            input_id.extend(sample['src_idx'])
            stru_mask.extend(sample['type_mask'])
            pos = list(range(1, len(sample['src_idx']) + 1))
            pos_id.extend(pos)
            seg_id.extend([i + 1] * len(sample['src_idx']))
            char_label.append(sample['word_id'])
            char_label.extend([-100] * (len(sample['src_idx']) - 1))
            char_pos.append(char_pos[-1] + len(sample['src_idx']))

            type_x = [x + 2 for x in sample['type_mask']]
            type_x[0] = 1 # the first token is for tag [CHAR]
            type_id.extend(type_x)
        assert len(src) == len(char_pos) - 1
        examples.append(InputExample(word_idx, src, values["sememe_idx"], input_id=input_id, stru_mask=stru_mask, pos_id=pos_id,
                                     seg_id=seg_id, type_id=type_id, char_label=char_label, char_pos=char_pos))

    dataset = SememeDataset(examples)


    return dataset


def collate(batch):
    src = [torch.Tensor(e.src) for e in batch]
    src = pad_sequence(src, batch_first=True, padding_value=0)
    label = []
    ori_label = []
    word = [e.word for e in batch]

    batch_id = [torch.Tensor(e.input_id) for e in batch]
    stru_mask = [torch.Tensor(e.stru_mask) for e in batch]
    inputs = pad_sequence(batch_id, batch_first=True, padding_value=0)
    stru_mask = pad_sequence(stru_mask, batch_first=True, padding_value=0)

    char_sep_ids = [torch.Tensor(e.seg_id) for e in batch]
    position_ids = [torch.Tensor(e.pos_id) for e in batch]
    type_ids = [torch.Tensor(e.type_id) for e in batch]
    char_label = [torch.Tensor(e.char_label) for e in batch]
    char_sep_ids = pad_sequence(char_sep_ids, batch_first=True, padding_value=0)
    position_ids = pad_sequence(position_ids, batch_first=True, padding_value=0)
    type_ids = pad_sequence(type_ids, batch_first=True, padding_value=0)
    char_label = pad_sequence(char_label, batch_first=True, padding_value=-100)

    attention_mask = []
    char_position = []

    for e in batch:
        label.append(torch.zeros(1, 1400).scatter_(1, torch.tensor([e.label]), 1))
        ori_label.append(e.label)
        mat = torch.zeros((inputs.shape[1], inputs.shape[1]))
        char_pos = e.char_pos
        for i in range(len(char_pos) - 1):
            b = char_pos[i]
            end = char_pos[i + 1]
            mat[b:end, b:end] = 1
            mat[b, char_pos[:-1]] = 1

        char_pos = char_pos[:-1]
        char_position.append(torch.Tensor(char_pos))
        attention_mask.append(mat)

    char_position = pad_sequence(char_position, batch_first=True, padding_value=0)
    attention_mask = torch.stack(attention_mask)

    return {"input_ids": inputs.to(torch.long).to(device), "stru_mask": stru_mask.to(torch.long).to(device), "attention_mask": attention_mask.to(device),
        "position_ids": position_ids.to(torch.long).to(device), "type_ids": type_ids.to(torch.long).to(device), "char_sep_ids": char_sep_ids.to(torch.long).to(device),
        "label": torch.cat(label).to(device), "ori_label": ori_label, "src": torch.tensor(src).to(torch.long).to(device), "char_label": char_label.to(torch.long).to(device),
        'char_position': char_position.to(device), "word": word}


def load_sense_embedding(sense_embedding_file):
    word_mat = np.load(sense_embedding_file)
    return word_mat

def evaluate(args, eval_dataset, model):
    eval_dataloader = DataLoader(eval_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=collate)
    preds = None
    labels = []
    for step, batch in enumerate(eval_dataloader):
        model.eval()
        with torch.no_grad():
            loss, output = model(batch)
        labels.extend(batch["ori_label"])
        if preds is None:
            preds = output.detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)

    scores = []
    for i in range(preds.shape[0]):
        score = []
        answer_sememes = labels[i]
        for id, s in enumerate(preds[i]):
            score.append((id, s))
        score.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in score]

        index = 1
        correct = 0
        point = 0
        for item in result:
            if item in answer_sememes:
                correct += 1
                point += float(correct) / (index)
            index += 1
        point /= len(answer_sememes)
        scores.append(point)

    return sum(scores)/len(scores)

class MyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        else:
            return super(MyEncoder, self).default(obj)


def test(args):
    with open(args.char_dict, 'r') as f:
        char_dict = json.load(f)

    args.num_classes = 1400
    pred_dataset = load_examples(char_dict, data_type='test')
    embedding_matrix = load_sense_embedding(args.embedding_matrix)
    args.word_size = embedding_matrix.shape[0]

    pred_dataloader = DataLoader(pred_dataset, batch_size=32, shuffle=False, collate_fn=collate)

    config = BertConfig.from_pretrained('../data/config.json', vocab_size=1176)
    model = Sememe_Net(config, args, embedding_matrix)
    model.load_state_dict(torch.load(args.SP_model_path))

    model.to(device)

    preds = None
    labels = []
    word_id = []
    predictions = dict()
    for step, batch in enumerate(pred_dataloader):
        model.eval()
        with torch.no_grad():
            loss, output = model(batch)
        labels.extend(batch["ori_label"])
        word_id.extend(batch["word"])
        if preds is None:
            preds = output.detach().cpu().numpy()
        else:
            preds = np.append(preds, output.detach().cpu().numpy(), axis=0)

    scores = []
    for i in range(preds.shape[0]):
        score = []
        answer_sememes = labels[i]
        word = word_id[i]
        for id, s in enumerate(preds[i]):
            score.append((id, s))
        predictions[word] = score
        score.sort(key=lambda x: x[1], reverse=True)
        result = [x[0] for x in score]

        index = 1
        correct = 0
        point = 0
        for item in result:
            if (item in answer_sememes):
                correct += 1
                point += float(correct) / (index)
            index += 1

        point /= len(answer_sememes)
        scores.append(point)

    print(sum(scores) / len(scores))

def main(localtime):
    with open(args.char_dict, 'r') as f:
        char_dict = json.load(f)

    train_dataset = load_examples(char_dict, data_type='train')
    eval_dataset = load_examples(char_dict, data_type='dev')
    pred_dataset = load_examples(char_dict, data_type='test')

    args.num_classes = 1400

    embedding_matrix = load_sense_embedding(args.embedding_matrix)
    args.word_size = embedding_matrix.shape[0]


    train_dataloader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=collate)

    config = BertConfig.from_pretrained('../data/config.json', vocab_size=1176)

    model = Sememe_Net(config, args, embedding_matrix)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(args.pretrain_model_path)
    pretrained_dict = {k.replace('module.', ''): v for k, v in pretrained_dict.items() if k.replace('module.', '') in model_dict}
    if len(pretrained_dict) == 0:
        print('pretrain is empty!')
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    model.to(device)

    optimizer_grouped_parameters = []
    for n, p in model.named_parameters():
        params_group = {}
        # params_group['weight_decay'] = 0.0 if any(nd in n for nd in no_decay) else args.weight_decay
        if 'bert.' in n or 'cls.' in n or 'stru_embeddings' in n:
            p.requires_grad = False

        params_group['params'] = p
        params_group['lr'] = args.learning_rate
        optimizer_grouped_parameters.append(params_group)

    optimizer = torch.optim.Adam(optimizer_grouped_parameters, lr=args.learning_rate)

    max_dev =[0]
    for epoch in range(args.num_train_epochs):

        train_loss = 0
        pbar = ProgressBar(n_total=len(train_dataloader), desc='Training')
        for step, batch in enumerate(train_dataloader):
            model.train()
            loss, output = model(batch)

            if args.gradient_accumulation_steps > 1:
                loss = loss / args.gradient_accumulation_steps

            loss.backward()

            if (step + 1) % args.gradient_accumulation_steps == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                optimizer.step()
                optimizer.zero_grad()

            train_loss += loss.item()

            pbar(step, {'loss': loss.item()})
        print(" ")
        if 'cuda' in str(device):
            torch.cuda.empty_cache()

        results_dev = evaluate(args, eval_dataset, model)
        results_test = evaluate(args, pred_dataset, model)
        logger.info(
            f"STEP:{epoch}\tLOSS:{train_loss/len(train_dataset.data)}, \tDEV ACC:{results_dev:.6f}, \tTEST ACC:{results_test:.6f}")
        if results_dev > max_dev[-1]:
            if not os.path.exists(args.output_dir + '/model'):
                os.makedirs(args.output_dir + '/model')
            torch.save(model.state_dict(), os.path.join(args.output_dir + '/model', localtime + '_' + args.name + '_' +str(args.seed)+'.bin'))
            logger.info(f"save model")
            max_dev.append(results_dev)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--output_dir", default='output', type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")
    parser.add_argument("--char_dict", default="../data/han_seq.json", type=str,
                        help="convert Chinese character into sequence")
    parser.add_argument("--pretrain_model_path", default="../pretrain/pytorch_model.bin", type=str,
                        help="Path of the pretrained GCC model")

    # DATA file given by SCorP
    parser.add_argument("--train_dir", default="data/train_data.json", type=str,
                        help="train data")
    parser.add_argument("--dev_dir", default="data/valid_data.json", type=str,
                        help="dev data")
    parser.add_argument("--test_dir", default="data/test_data.json", type=str,
                        help="test data")
    parser.add_argument("--word2index_path", default="data/word2index.json", type=str,
                        help="word2index.json given by SCorP")
    parser.add_argument("--embedding_matrix", default="data/word_vector.npy", type=str,
                        help="word_vector.npy given by SCorP")

    parser.add_argument("--name", default="train", type=str,
                        help="Where do you want to store the pre-trained models downloaded from s3")
    parser.add_argument("--max_grad_norm", default=1.0, type=float,
                        help="Max gradient norm.")
    parser.add_argument('--seed', type=int, default=4,
                        help="random seed for initialization")
    parser.add_argument("--num_train_epochs", default=200, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument("--learning_rate", default=0.0001, type=float,
                        help="The initial learning rate for Adam.")
    parser.add_argument('--batch_size', type=int, default=128, help="batch_size")
    parser.add_argument('--gradient_accumulation_steps', type=int, default=1, help="gradient_accumulation_steps")
    parser.add_argument('--hidden_size', type=int, default=512, help="hidden_size")
    parser.add_argument('--embed_dim', type=int, default=200, help="embed_dim")
    parser.add_argument('--do_train', type=int, default=1, help="do_train")
    parser.add_argument('--do_test', type=int, default=0, help="do_test")
    parser.add_argument("--SP_model_path", default="SP_model.bin", type=str,
                        help="Path of the Sememe Prediction model")

    args = parser.parse_args()

    seed_everything(args.seed)
    localtime = time.strftime("%Y%m%d-%H%M%S", time.localtime())

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    if args.do_train:
        output_dir = args.output_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        init_logger(log_file=output_dir + '/' + localtime + '_' + args.name + '_' + str(args.seed) + '.log')
        logger.info("Training/evaluation parameters %s", args)

        main(localtime)

    if args.do_test:
        test(args)