#!/usr/bin/env python
# encoding: utf-8
import csv
import logging
import math
import os
import random
from glob import glob

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')

logger = logging.getLogger('utils')

CHAR_TYPE = {
    u'กขฃคฆงจชซญฎฏฐฑฒณดตถทธนบปพฟภมยรลวศษสฬอ': 'c',
    u'ฅฉผฟฌหฮ': 'n',
    u'ะาำิีืึุู': 'v',  # า ะ ำ ิ ี ึ ื ั ู ุ
    u'เแโใไ': 'w',
    u'่้๊๋': 't',  # วรรณยุกต์ ่ ้ ๊ ๋
    u'์ๆฯ.': 's',  # ์  ๆ ฯ .
    u'0123456789๑๒๓๔๕๖๗๘๙': 'd',
    u'"': 'q',
    u"‘": 'q',
    u"’": 'q',
    u"'": 'q',
    u' ': 'p',
    u'abcdefghijklmnopqrstuvwxyz': 's_e',
    u'ABCDEFGHIJKLMNOPQRSTUVWXYZ': 'b_e'
}

CHAR_TYPE_FLATTEN = {}
for ks, v in CHAR_TYPE.items():
    for k in ks:
        CHAR_TYPE_FLATTEN[k] = v

# create map of dictionary to character
CHARS = [
    u'\n', u' ', u'!', u'"', u'#', u'$', u'%', u'&', "'", u'(', u')', u'*', u'+',
    u',', u'-', u'.', u'/', u'0', u'1', u'2', u'3', u'4', u'5', u'6', u'7', u'8',
    u'9', u':', u';', u'<', u'=', u'>', u'?', u'@', u'A', u'B', u'C', u'D', u'E',
    u'F', u'G', u'H', u'I', u'J', u'K', u'L', u'M', u'N', u'O', u'P', u'Q', u'R',
    u'S', u'T', u'U', u'V', u'W', u'X', u'Y', u'Z', u'[', u'\\', u']', u'^', u'_',
    u'a', u'b', u'c', u'd', u'e', u'f', u'g', u'h', u'i', u'j', u'k', u'l', u'm',
    u'n', u'o', u'other', u'p', u'q', u'r', u's', u't', u'u', u'v', u'w', u'x', u'y',
    u'z', u'}', u'~', u'ก', u'ข', u'ฃ', u'ค', u'ฅ', u'ฆ', u'ง', u'จ', u'ฉ', u'ช',
    u'ซ', u'ฌ', u'ญ', u'ฎ', u'ฏ', u'ฐ', u'ฑ', u'ฒ', u'ณ', u'ด', u'ต', u'ถ', u'ท',
    u'ธ', u'น', u'บ', u'ป', u'ผ', u'ฝ', u'พ', u'ฟ', u'ภ', u'ม', u'ย', u'ร', u'ฤ',
    u'ล', u'ว', u'ศ', u'ษ', u'ส', u'ห', u'ฬ', u'อ', u'ฮ', u'ฯ', u'ะ', u'ั', u'า',
    u'ำ', u'ิ', u'ี', u'ึ', u'ื', u'ุ', u'ู', u'ฺ', u'เ', u'แ', u'โ', u'ใ', u'ไ',
    u'ๅ', u'ๆ', u'็', u'่', u'้', u'๊', u'๋', u'์', u'ํ', u'๐', u'๑', u'๒', u'๓',
    u'๔', u'๕', u'๖', u'๗', u'๘', u'๙', u'‘', u'’', u'\ufeff'
]
CHARS_MAP = {v: k for k, v in enumerate(CHARS)}

CHAR_TYPES = [
    'b_e', 'c', 'd', 'n', 'o',
    'p', 'q', 's', 's_e', 't',
    'v', 'w'
]
CHAR_TYPES_MAP = {v: k for k, v in enumerate(CHAR_TYPES)}

LABELS_MAP = {'B': 1, 'I': 0}


def load_csv_data(data_path):
    train_files = [data_path] if os.path.isfile(data_path) else glob(os.path.join(data_path, '*.csv'))
    logger.info("loading data from %s" % data_path)
    sentences = []
    for f_test in train_files:
        with open(f_test, 'r', encoding='utf8') as csv_file:
            reader = csv.reader(csv_file)
            sent = []
            for row in reader:
                _, wl, sl = row
                if wl == '-' or sl == '-':
                    if len(sent) > 0:
                        sentences.append(sent)
                    sent = []
                else:
                    sent.append(row)

    return sentences


def make_samples(sentences, n_gram):
    pad_len = n_gram // 2
    pad_char = [' '] * pad_len
    pad_type = ['p'] * pad_len

    other_idx = CHARS_MAP.get('other')

    samples = []
    for sent in sentences:
        pad_sent_char = pad_char + [c for c, wl, sl in sent] + pad_char
        pad_sent_type = pad_type + [CHAR_TYPE_FLATTEN.get(c, 'p') for c, wl, sl in sent] + pad_type

        pad_sent_char = [CHARS_MAP.get(c, other_idx) for c in pad_sent_char]
        pad_sent_type = [CHAR_TYPES_MAP.get(t) for t in pad_sent_type]

        sent_samples = [
            (
                pad_sent_char[i: i + n_gram],
                pad_sent_type[i: i + n_gram],
                LABELS_MAP[wl],
                LABELS_MAP[sl]
            )
            for i, (_, wl, sl) in enumerate(sent)
        ]

        samples.extend(sent_samples)

    return samples


def data_generator(data, batch_size, shuffle=False, repeat=False):
    batch_num = math.ceil(len(data) / batch_size)
    field_num = len(data[0])
    while True:
        if shuffle:
            shuffled_idx = [i for i in range(len(data))]
            random.shuffle(shuffled_idx)
            data = [data[i] for i in shuffled_idx]

        batch_idx = 0
        while batch_idx < batch_num:
            offset = batch_idx * batch_size
            batch_data = data[offset: offset + batch_size]

            batch = tuple([row[f] for row in batch_data] for f in range(field_num))
            yield batch

            batch_idx += 1
        if repeat:
            continue
        else:
            break


def width(text):
    return sum([2 if '\u4E00' <= c <= '\u9FA5' else 1 for c in text])


def print_table(tab):
    col_width = [max(width(x) for x in col) for col in zip(*tab)]
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")
    for line in tab:
        print("| " + " | ".join("{:{}}".format(x, col_width[i]) for i, x in enumerate(line)) + " |")
    print("+-" + "-+-".join("{:-^{}}".format('-', col_width[i]) for i, x in enumerate(tab[0])) + "-+")


def parameters_string(module):
    lines = [
        "",
        "List of model parameters:",
        "=========================",
    ]

    row_format = "{name:<55} {shape:>20} ={total_size:>12,d}"
    params = list(module.named_parameters())
    for name, param in params:
        lines.append(row_format.format(
            name=name,
            shape=" * ".join(str(p) for p in param.size()),
            total_size=param.numel()
        ))
    lines.append("=" * 90)
    lines.append(row_format.format(
        name="all parameters",
        shape="sum of above",
        total_size=sum(int(param.numel()) for name, param in params)
    ))
    lines.append("")
    return "\n".join(lines)
