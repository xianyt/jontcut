#!/usr/bin/env python
# encoding: utf-8
import time
from argparse import ArgumentParser
from datetime import timedelta

import torch

from model_joint_cut import JointCut
from utils import *

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--model', type=str, default='models/base.pt',
                        help='path to model file')
    parser.add_argument('--in_file', type=str, default='', help='input file')
    parser.add_argument('--out_file', type=str, default='std', help='output file')
    parser.add_argument('--separator', type=str, default=' ', help='output file')
    parser.add_argument('--sep_replace', type=str, default='_', help='output file')

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def clean_str(best_str):
    # best_str = re.sub(r"<\s*/?\s*[A-Za-z]+\s*>", "", best_str)
    # best_str = re.sub(r"^\|", "", best_str)
    return best_str


def make_samples(sent, n_gram):
    pad_len = n_gram // 2
    pad_char = [' '] * pad_len
    pad_type = ['p'] * pad_len

    other_idx = CHARS_MAP.get('other')

    pad_sent_char = pad_char + [c for c in sent] + pad_char
    pad_sent_type = pad_type + [CHAR_TYPE_FLATTEN.get(c, 'p') for c in sent] + pad_type

    pad_sent_char = [CHARS_MAP.get(c, other_idx) for c in pad_sent_char]
    pad_sent_type = [CHAR_TYPES_MAP.get(t) for t in pad_sent_type]

    sent_samples = [(pad_sent_char[i: i + n_gram], pad_sent_type[i: i + n_gram]) for i, c in enumerate(sent)]

    return sent_samples


def segment(sent, cfg):
    sent = sent.strip()
    if len(sent) == 0:
        return ""
    sent = clean_str(sent)
    sent_samples = make_samples(sent, cfg.n_gram)

    x_char, x_type = tuple([row[f] for row in sent_samples] for f in range(2))

    model.eval()
    with torch.no_grad():
        try:
            x_char = torch.tensor(x_char, dtype=torch.long).to(cfg.device)
            x_type = torch.tensor(x_type, dtype=torch.long).to(cfg.device)
        except Exception:
            print(x_char)

        logits_word, logits_syllable = model(x_char, x_type)

        logits_word = (logits_word > 0.5).cpu().detach().tolist()
        result = [
            cfg.separator + char.replace(cfg.separator, cfg.sep_replace) if label and i > 0 else char
            for label, char, i
            in zip(logits_word, sent, range(len(x_char)))
        ]

        return "".join(result)


def segment_file(cfg):
    logger.info("segment %s ..." % cfg.in_file)
    with open(cfg.in_file, 'r', encoding='utf8') as in_file:
        with open(cfg.out_file, 'w', encoding='utf8') as out_file:
            start_time = time.time()
            char_count = 0
            for sent in in_file:
                result = segment(sent, cfg)
                out_file.write(result + "\n")
                char_count += len(sent)

    end_time = time.time()
    delta_time = timedelta(seconds=int(round(end_time - start_time)))
    return delta_time.seconds, char_count


if __name__ == "__main__":
    logger = logging.getLogger('main')

    config = get_args()

    ckpt = torch.load(config.model, map_location=torch.device(config.device))

    model_cfg = ckpt['config']

    model_cfg.device = config.device
    config.n_gram = model_cfg.n_gram

    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(model_cfg).items()])

    model = JointCut(model_cfg).to(config.device)
    model.load_state_dict(ckpt['state'])

    if os.path.isfile(config.in_file):
        tm_delta, ch_count = segment_file(config)
        logger.info("segment finished, %d sec, %d characters, %d char/sec."
                    % (tm_delta, ch_count, round(ch_count / tm_delta)))

    exit(0)
