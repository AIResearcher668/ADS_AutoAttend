'''
train the supernet by Jimmy
'''

import numpy as np
from model.ops import PRIMITIVES
import os
import yaml
import argparse
from tqdm import tqdm

import torch
import torch.optim
import torch.nn.functional as F
from torch.utils.data import DataLoader

from model.utils import get_edge_node_op, sample_valid_archs
from model.supernet import TextClassifier
from task.utils import logger, set_seed

### Jimmy: for reading datasets
import logging
import pickle
from collections import Counter
from torch.utils import data
import sys

LOGGER = logger.get_logger('sst5-search')

### Jimmy: start1
### logger = logging.getLogger()


class PTBTree:
    WORD_TO_WORD_MAPPING = {
        "{": "-LCB-",
        "}": "-RCB-"
    }

    def __init__(self):
        self.subtrees = []
        self.word = None
        self.label = ""
        self.parent = None
        self.span = (-1, -1)
        self.word_vector = None  # HOS, store dx1 RNN word vector
        self.prediction = None  # HOS, store Kx1 prediction vector

    def is_leaf(self):
        return len(self.subtrees) == 0

    def set_by_text(self, text, pos=0, left=0):
        depth = 0
        right = left
        for i in range(pos + 1, len(text)):
            char = text[i]
            # update the depth
            if char == "(":
                depth += 1
                if depth == 1:
                    subtree = PTBTree()
                    subtree.parent = self
                    subtree.set_by_text(text, i, right)
                    right = subtree.span[1]
                    self.span = (left, right)
                    self.subtrees.append(subtree)
            elif char == ")":
                depth -= 1
                if len(self.subtrees) == 0:
                    pos = i
                    for j in range(i, 0, -1):
                        if text[j] == " ":
                            pos = j
                            break
                    self.word = text[pos + 1:i]
                    self.span = (left, left + 1)

            # we've reached the end of the category that is the root of this subtree
            if depth == 0 and char == " " and self.label == "":
                self.label = text[pos + 1:i]
            # we've reached the end of the scope for this bracket
            if depth < 0:
                break

        # Fix some issues with variation in output, and one error in the treebank
        # for a word with a punctuation POS
        self.standardise_node()

    def standardise_node(self):
        if self.word in self.WORD_TO_WORD_MAPPING:
            self.word = self.WORD_TO_WORD_MAPPING[self.word]

    def __repr__(self, single_line=True, depth=0):
        ans = ""
        if not single_line and depth > 0:
            ans = "\n" + depth * "\t"
        ans += "(" + self.label
        if self.word is not None:
            ans += " " + self.word
        for subtree in self.subtrees:
            if single_line:
                ans += " "
            ans += subtree.__repr__(single_line, depth + 1)
        ans += ")"
        return ans


def read_tree(source):
    cur_text = []
    depth = 0
    while True:
        line = source.readline()
        # Check if we are out of input
        if line == "":
            return None
        # strip whitespace and only use if this contains something
        line = line.strip()
        if line == "":
            continue
        cur_text.append(line)
        # Update depth
        for char in line:
            if char == "(":
                depth += 1
            elif char == ")":
                depth -= 1
        # At depth 0 we have a complete tree
        if depth == 0:
            tree = PTBTree()
            tree.set_by_text(" ".join(cur_text))
            return tree
    return None


def read_trees(source, max_sents=-1):
    with open(source, encoding='utf8') as fp:
        trees = []
        while True:
            tree = read_tree(fp)
            if tree is None:
                break
            trees.append(tree)
            if len(trees) >= max_sents > 0:
                break
        return trees


class SSTDataset(data.Dataset):
    def __init__(self, sents, mask, labels):
        self.sents = sents
        self.labels = labels
        self.mask = mask

    def __getitem__(self, index):
        return self.sents[index], self.labels[index], self.mask[index]

    def __len__(self):
        return len(self.sents)


def sst_get_id_input(content, word_id_dict, max_input_length):
    words = content.split(" ")
    sentence = [word_id_dict["<pad>"]] * max_input_length
    mask = [0] * max_input_length
    unknown = word_id_dict["<unknown>"]
    for i, word in enumerate(words[:max_input_length]):
        sentence[i] = word_id_dict.get(word, unknown)
        mask[i] = 1
    return sentence, mask


def sst_get_phrases(trees, sample_ratio=1.0, is_binary=False, only_sentence=False):
    all_phrases = []
    for tree in trees:
        if only_sentence:
            sentence = get_sentence_by_tree(tree)
            label = int(tree.label)
            pair = (sentence, label)
            all_phrases.append(pair)
        else:
            phrases = get_phrases_by_tree(tree)
            sentence = get_sentence_by_tree(tree)
            pair = (sentence, int(tree.label))
            all_phrases.append(pair)
            all_phrases += phrases
    if sample_ratio < 1.:
        np.random.shuffle(all_phrases)
    result_phrases = []
    for pair in all_phrases:
        if is_binary:
            phrase, label = pair
            if label <= 1:
                pair = (phrase, 0)
            elif label >= 3:
                pair = (phrase, 1)
            else:
                continue
        if sample_ratio == 1.:
            result_phrases.append(pair)
        else:
            rand_portion = np.random.random()
            if rand_portion < sample_ratio:
                result_phrases.append(pair)
    return result_phrases


def get_phrases_by_tree(tree):
    phrases = []
    if tree is None:
        return phrases
    if tree.is_leaf():
        pair = (tree.word, int(tree.label))
        phrases.append(pair)
        return phrases
    left_child_phrases = get_phrases_by_tree(tree.subtrees[0])
    right_child_phrases = get_phrases_by_tree(tree.subtrees[1])
    phrases.extend(left_child_phrases)
    phrases.extend(right_child_phrases)
    sentence = get_sentence_by_tree(tree)
    pair = (sentence, int(tree.label))
    phrases.append(pair)
    return phrases


def get_sentence_by_tree(tree):
    if tree is None:
        return ""
    if tree.is_leaf():
        return tree.word
    left_sentence = get_sentence_by_tree(tree.subtrees[0])
    right_sentence = get_sentence_by_tree(tree.subtrees[1])
    sentence = left_sentence + " " + right_sentence
    return sentence.strip()


def get_word_id_dict(word_num_dict, word_id_dict, min_count):
    z = [k for k in sorted(word_num_dict.keys())]
    for word in z:
        count = word_num_dict[word]
        if count >= min_count:
            index = len(word_id_dict)
            if word not in word_id_dict:
                word_id_dict[word] = index
    return word_id_dict


def load_word_num_dict(phrases, word_num_dict):
    for sentence, _ in phrases:
        words = sentence.split(" ")
        for cur_word in words:
            word = cur_word.strip()
            word_num_dict[word] += 1
    return word_num_dict


def init_trainable_embedding(embedding_path, word_id_dict, embed_dim=300):
    word_embed_model = load_glove_model(embedding_path, embed_dim)
    assert word_embed_model["pool"].shape[1] == embed_dim
    embedding = np.random.random([len(word_id_dict), embed_dim]).astype(np.float32) / 2.0 - 0.25
    embedding[0] = np.zeros(embed_dim)  # PAD
    embedding[1] = (np.random.rand(embed_dim) - 0.5) / 2  # UNK
    for word in sorted(word_id_dict.keys()):
        idx = word_id_dict[word]
        if idx == 0 or idx == 1:
            continue
        if word in word_embed_model["mapping"]:
            embedding[idx] = word_embed_model["pool"][word_embed_model["mapping"][word]]
        else:
            embedding[idx] = np.random.rand(embed_dim) / 2.0 - 0.25
    return embedding


def sst_get_trainable_data(phrases, word_id_dict, max_input_length):
    texts, labels, mask = [], [], []

    for phrase, label in phrases:
        if not phrase.split():
            continue
        phrase_split, mask_split = sst_get_id_input(phrase, word_id_dict, max_input_length)
        texts.append(phrase_split)
        labels.append(int(label))
        mask.append(mask_split)  # field_input is mask
    labels = np.array(labels, dtype=np.long)
    texts = np.reshape(texts, [-1, max_input_length]).astype(np.long)
    mask = np.reshape(mask, [-1, max_input_length]).astype(bool)

    return SSTDataset(texts, mask, labels)


def load_glove_model(filename, embed_dim):
    if os.path.exists(filename + ".cache"):
        ### logger.info("Found cache. Loading...")
        with open(filename + ".cache", "rb") as fp:
            return pickle.load(fp)
    embedding = {"mapping": dict(), "pool": []}
    with open(filename) as f:
        for i, line in enumerate(f):
            line = line.rstrip("\n")
            vocab_word, *vec = line.rsplit(" ", maxsplit=embed_dim)
            assert len(vec) == 300, "Unexpected line: '%s'" % line
            embedding["pool"].append(np.array(list(map(float, vec)), dtype=np.float32))
            embedding["mapping"][vocab_word] = i
    embedding["pool"] = np.stack(embedding["pool"])
    with open(filename + ".cache", "wb") as fp:
        pickle.dump(embedding, fp)
    return embedding


def read_data_sst(data_path, max_input_length=64, min_count=1, train_with_valid=False,
                  train_ratio=1., valid_ratio=1., is_binary=False, only_sentence=False):
    word_id_dict = dict()
    word_num_dict = Counter()

    sst_path = os.path.join(data_path, "sst")
    ### logger.info("Reading SST data...")
    train_file_name = os.path.join(sst_path, "trees", "train.txt")
    valid_file_name = os.path.join(sst_path, "trees", "dev.txt")
    test_file_name = os.path.join(sst_path, "trees", "test.txt")
    train_trees = read_trees(train_file_name)
    train_phrases = sst_get_phrases(train_trees, train_ratio, is_binary, only_sentence)
    ### logger.info("Finish load train phrases.")
    valid_trees = read_trees(valid_file_name)
    valid_phrases = sst_get_phrases(valid_trees, valid_ratio, is_binary, only_sentence)
    ### logger.info("Finish load valid phrases.")
    if train_with_valid:
        train_phrases += valid_phrases
    test_trees = read_trees(test_file_name)
    test_phrases = sst_get_phrases(test_trees, valid_ratio, is_binary, only_sentence=True)
    ### logger.info("Finish load test phrases.")

    # get word_id_dict
    word_id_dict["<pad>"] = 0
    word_id_dict["<unknown>"] = 1
    load_word_num_dict(train_phrases, word_num_dict)
    ### logger.info("Finish load train words: %d.", len(word_num_dict))
    load_word_num_dict(valid_phrases, word_num_dict)
    load_word_num_dict(test_phrases, word_num_dict)
    ### logger.info("Finish load valid+test words: %d.", len(word_num_dict))
    word_id_dict = get_word_id_dict(word_num_dict, word_id_dict, min_count)
    ### logger.info("After trim vocab length: %d.", len(word_id_dict))

    ### logger.info("Loading embedding...")
    embedding = init_trainable_embedding(os.path.join(data_path, "glove.840B.300d.txt"), word_id_dict)
    ### logger.info("Finish initialize word embedding.")

    dataset_train = sst_get_trainable_data(train_phrases, word_id_dict, max_input_length)
    ### logger.info("Loaded %d training samples.", len(dataset_train))
    dataset_valid = sst_get_trainable_data(valid_phrases, word_id_dict, max_input_length)
    ### logger.info("Loaded %d validation samples.", len(dataset_valid))
    dataset_test = sst_get_trainable_data(test_phrases, word_id_dict, max_input_length)
    ### logger.info("Loaded %d test samples.", len(dataset_test))

    return dataset_train, dataset_valid, dataset_test, torch.from_numpy(embedding)

def convert_to_file(data_path, train_with_valid=False, train_ratio=1., valid_ratio=1., is_binary=False, only_sentence=False, folder=None):
    if folder:
        os.makedirs(folder, exist_ok=True)

    sst_path = os.path.join(data_path, "sst")
    ### logger.info("Reading SST data...")
    train_file_name = os.path.join(sst_path, "trees", "train.txt")
    valid_file_name = os.path.join(sst_path, "trees", "dev.txt")
    test_file_name = os.path.join(sst_path, "trees", "test.txt")
    train_trees = read_trees(train_file_name)
    train_phrases = sst_get_phrases(train_trees, train_ratio, is_binary, only_sentence)
    ### logger.info("Finish load train phrases.")
    valid_trees = read_trees(valid_file_name)
    valid_phrases = sst_get_phrases(valid_trees, valid_ratio, is_binary, only_sentence)
    ### logger.info("Finish load valid phrases.")
    if train_with_valid:
        train_phrases += valid_phrases
    test_trees = read_trees(test_file_name)
    test_phrases = sst_get_phrases(test_trees, valid_ratio, is_binary, only_sentence=True)
    ### logger.info("Finish load test phrases.")

    def write_to_file(phrases, path):
        with open(path, 'w') as f:
            for sent, label in phrases:
                if not sent.split(): continue
                f.write('__label__{} {}\n'.format(label, sent))

    write_to_file(train_phrases, os.path.join(folder or sst_path, 'fast-train.txt'))
    if not train_with_valid:
        write_to_file(valid_phrases, os.path.join(folder or sst_path, 'fast-val.txt'))
    write_to_file(test_phrases, os.path.join(folder or sst_path, 'fast-test.txt'))
### Jimmy: end 1




if __name__ == '__main__':

    logger.LEVEL = logger.INFO
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--dim', type=int, help='dimension', default=64)
    parser.add_argument('--head', type=int, help='default attn head number', default=8)
    parser.add_argument('--layer', type=int, help='total layer number', default=24)
    parser.add_argument('--space', type=int, help='search space type', default=1)
    parser.add_argument('--context', choices=['sc', 'fc', 'tc', 'nc'], default='fc')
    parser.add_argument('--dataset', type=str, help='path to dataset', default='/media/timeclr/JLearning/1_PhD_research/Projects/Recommendation_System/rs_code_2/AutoAttend/data')

    parser.add_argument('--arch_batch_size', type=int, help='batch size to train the base', default=16)
    parser.add_argument('--epoch', type=int, help='total num of epoch to train the supernet', default=10)
    parser.add_argument('--lr', type=float, help='learning rate', default=5e-4)
    parser.add_argument('--weight_decay', type=float, help='weight decay', default=0)
    parser.add_argument('--dropout', type=float, help='dropout', default=0.1)
    parser.add_argument('--gradient_clip', type=float, help='gradient clip', default=5.0)
    
    parser.add_argument('--path', type=str, help='path to save logs & models', default='./searched')

    parser.add_argument('--seed', type=int, help='random seed', default=2021)
    parser.add_argument('--device', type=int, default=0, help='the main progress')

    args = parser.parse_args()

    set_seed(args.seed)

    if args.path is not None:
        os.makedirs(args.path, exist_ok=True)
        LOGGER.set_path(os.path.join(args.path, 'log.txt'))

    LOGGER.info('hyper parameters')
    for k,v in args.__dict__.items():
        LOGGER.info('{} - {}'.format(k, v))
    LOGGER.info('end of hyper parameters')

    input('hyperparameter confirmed')

    LOGGER.info('load dataset...')
    edgeop, nodeop = get_edge_node_op(PRIMITIVES, args.space)
    print(args.dataset)

    ### Jimmy2
    train, val, test, embedding = read_data_sst('data')
    print("train_dataset路经", os.path.join(args.dataset, 'sst/train.pt'))
    train_dataset = train
    valid_dataset = val
    test_dataset  = test
    ### embedding = embedding
    ### Jimmy2 end

    print(type(train))
    ### sys.exit(0)

    train_dataloader = DataLoader(train_dataset, batch_size=128, shuffle=True, drop_last=True)
    device = torch.device('cuda:{}'.format(args.device))

    LOGGER.info('build model...')
    model = TextClassifier(
        embeddings=embedding,
        dim=args.dim,
        head=args.head,
        nclass=5,
        layer=args.layer,
        edgeops=edgeop,
        nodeops=nodeop,
        dropout=args.dropout,
        context=args.context, aug_dropouts=[0.1, 0.1]).to(device)

    LOGGER.info('train supernet...')
    model.train()
    eval_id = 0
    time_list = []
    loss_item = []

    opt = torch.optim.Adam(model.parameters(), lr=args.lr / args.arch_batch_size, weight_decay=args.weight_decay)

    for idx in range(args.epoch):
        t = tqdm(train_dataloader)
        for batch in t:
            opt.zero_grad()
            
            for arch in sample_valid_archs(args.layer, edgeop, nodeop, args.arch_batch_size, PRIMITIVES):                
                logit = model(batch[0].to(device), batch[2].to(device), arch)
                loss = F.cross_entropy(logit, batch[1].long().to(device))
                loss.backward()
                loss_item.append(loss.item())
                loss_item = loss_item[-100:]

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.gradient_clip)
            
            opt.step()
            t.set_postfix(loss=round(np.mean(loss_item), 4))
        if args.path:
            torch.save(model, os.path.join(args.path, 'model_epoch_{}.full'.format(idx)))
