#!/usr/bin/env python
# encoding: utf-8
import re
from argparse import ArgumentParser

import torch

from metrics import evaluate_result
from model_joint_cut import JointCut
from preprocess import label_best_str
from utils import *

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


def label_best_str(best_str):
    # remove entity labels
    best_str = re.sub(r"<\s*/?\s*[A-Za-z]+\s*>", "", best_str)
    tokens = [w for w in best_str.split("|") if len(w) > 0]

    char_with_label = [
        (c, 'I' if i > 0 else 'B')
        for w in tokens for i, c in enumerate(w)
    ]

    return char_with_label


def make_samples(sentences, n_gram):
    pad_len = n_gram // 2
    pad_char = [' '] * pad_len
    pad_type = ['p'] * pad_len

    other_idx = CHARS_MAP.get('other')

    samples = []
    for sent in sentences:
        pad_sent_char = pad_char + [c for c, ll in sent] + pad_char
        pad_sent_type = pad_type + [CHAR_TYPE_FLATTEN.get(c, 'p') for c, ll in sent] + pad_type

        pad_sent_char = [CHARS_MAP.get(c, other_idx) for c in pad_sent_char]
        pad_sent_type = [CHAR_TYPES_MAP.get(t) for t in pad_sent_type]

        sent_samples = [
            (
                pad_sent_char[i: i + n_gram],
                pad_sent_type[i: i + n_gram],
                LABELS_MAP[ll]
            )
            for i, (_, ll) in enumerate(sent)
        ]

        samples.extend(sent_samples)

    return samples


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='models/base.pt',
                        help='path to checkpoint file')
    parser.add_argument('--test', type=str, default='data/BEST_2010/TEST.txt',
                        help='path to test file')

    parser.add_argument('--target', type=str, default='word',
                        help='word or syllable')

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def clean_str(best_str):
    best_str = re.sub(r"<\s*/?\s*[A-Za-z]+\s*>", "", best_str)
    best_str = re.sub(r"^\|", "", best_str)
    return best_str


def test(cfg):
    ckpt = torch.load(cfg.model, map_location=torch.device(cfg.device))

    model_cfg = ckpt['config']

    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(model_cfg).items()])

    model = JointCut(model_cfg).to(cfg.device)
    model.load_state_dict(ckpt['state'])

    print(parameters_string(model))

    if os.path.isfile(cfg.test):
        logger.info("reading %s" % cfg.test)
        with open(cfg.test, 'r', encoding='utf8') as best_file:
            test_sents = [sent.strip() for sent in best_file]
    else:
        text_files = glob(os.path.join(cfg.test, '*.txt'))
        test_sents = []
        for txt_file in text_files:
            logger.info("reading %s" % txt_file)
            with open(txt_file, 'r', encoding='utf8') as best_file:
                test_sents.extend([sent.strip() for sent in best_file])

    # test_sents = test_sents[:10]
    logger.info("preprocessing ...")
    test_sents = [clean_str(sent) for sent in test_sents if len(sent) > 0]

    test_sents_with_label = [label_best_str(sent) for sent in test_sents]
    test_sent_samples = [make_samples([sent], model_cfg.n_gram) for sent in test_sents_with_label]

    y_target, y_word_pred, y_syllable_pred = [], [], []
    logger.info("predict sentences ...")
    for sent_samples, sent in zip(test_sent_samples, test_sents):
        if len(sent_samples) == 0:
            continue

        field_num = len(sent_samples[0])
        batch = tuple([row[f] for row in sent_samples] for f in range(field_num))
        model.eval()
        with torch.no_grad():
            x_char, x_type, y = batch
            try:
                x_char = torch.tensor(x_char, dtype=torch.long).to(cfg.device)
                x_type = torch.tensor(x_type, dtype=torch.long).to(cfg.device)
            except Exception:
                print(x_char)

            logits_word, logits_syllable = model(x_char, x_type)

            y_target.append(y)
            # if not isinstance(word_pred, list) or len(word_pred) > 500:
            #     print(word_pred)
            #     continue

            word_pred = (logits_word > 0.5).long().tolist()
            y_word_pred.append(word_pred)
            assert len(word_pred) == len(y)

            syllable_pred = (logits_syllable > 0.5).long().tolist()
            y_syllable_pred.append(syllable_pred)
            assert len(syllable_pred) == len(y)

    if cfg.target == 'word':
        logger.info("benchmark for word segmentation ...")
        evaluate_result(y_target, y_word_pred)
    else:
        logger.info("benchmark for syllable segmentation ...")
        evaluate_result(y_target, y_syllable_pred)


if __name__ == "__main__":
    logger = logging.getLogger('main')

    config = get_args()
    test(config)
    exit(0)
