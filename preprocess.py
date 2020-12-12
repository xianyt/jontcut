#!/usr/bin/env python
# encoding: utf-8
import argparse
import csv
import math
import os
import random
import re
from glob import glob

from ssg import syllable_tokenize

BLANK_ROW = ('-', '-', '-')


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", type=str, default="data/BEST_2010")
    parser.add_argument("--sub_datasets", type=str, nargs='+', default=['article', 'encyclopedia', 'news', 'novel'])
    parser.add_argument("--output_path", type=str, default="data/best_2010_csv")
    parser.add_argument("--split_ratio", type=float, default=0.1)

    cfg = parser.parse_args()

    return cfg


def label_best_str(best_str):
    best_str = re.sub(r"<\s*/?\s*[A-Za-z]+\s*>", "", best_str)
    words = [w for w in best_str.split("|") if len(w) > 0]

    word_labels = [
        (c, 'I' if i > 0 else 'B')
        for w in words for i, c in enumerate(w)
    ]

    syllables = [syllable_tokenize(w) for w in words]
    syllable_labels = [(c, 'I' if i > 0 else 'B') for ws in syllables for s in ws for i, c in enumerate(s)]

    assert len(word_labels) == len(syllable_labels)
    char_with_label = [(c, wl, sl) for (c, wl), (_, sl) in zip(word_labels, syllable_labels)]

    return char_with_label


def best_to_csv(doc_sents):
    doc_rows = []
    for sent in doc_sents:
        sent = sent.strip()
        if len(sent) == 0:
            continue
        rows = label_best_str(sent)
        doc_rows.extend(rows)
        doc_rows.append(BLANK_ROW)

    return doc_rows


def process_dataset(doc_sents, csv_file_name, out_path, split_ratio):
    if split_ratio > 0.0:
        sent_num = len(doc_sents)
        indices = [i for i in range(sent_num)]
        random.shuffle(indices)

        split_num = math.floor(sent_num * split_ratio)
        train_indices = sorted(indices[:-split_num])
        valid_indices = sorted(indices[-split_num:])

        for name, sent_indices in [('train', train_indices), ('valid', valid_indices)]:
            os.makedirs(os.path.join(out_path, name), exist_ok=True)

            sents = [doc_sents[i] for i in sent_indices]
            rows = best_to_csv(sents)

            with open(os.path.join(out_path, name, csv_file_name + '.csv'), 'w', encoding='utf8') as csv_file:
                csv_writer = csv.writer(csv_file)
                csv_writer.writerows(rows)
    else:
        rows = best_to_csv(doc_sents)
        with open(os.path.join(out_path, csv_file_name + '.csv'), 'w', encoding='utf8') as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerows(rows)


def preprocess(in_dir, out_dir, sub_datasets=None, split_ratio=0.0):
    if not os.path.isdir(out_dir):
        os.mkdir(out_dir)

    no_sub_datasets = not sub_datasets or len(sub_datasets) == 0
    sub_dataset_path = [in_dir] if no_sub_datasets else [os.path.join(in_dir, sd) for sd in sub_datasets]

    for ds_path, ds in zip(sub_dataset_path, sub_datasets):
        print("processing %s ..." % ds)
        csv_file_name = re.sub(r"\.[^.]+$", '', ds).lower()

        if os.path.isfile(ds_path):
            with open(ds_path, 'r', encoding='utf8') as best_file:
                sub_dataset_sents = [sent for sent in best_file]
        else:
            text_files = glob(os.path.join(ds_path, '*.txt'))
            sub_dataset_sents = []
            for txt_file in text_files:
                with open(txt_file, 'r', encoding='utf8') as best_file:
                    sub_dataset_sents.extend([sent for sent in best_file])

        process_dataset(sub_dataset_sents, csv_file_name, out_dir, split_ratio)


if __name__ == '__main__':
    cfg = get_args()
    preprocess(cfg.data_path, cfg.output_path, cfg.sub_datasets, cfg.split_ratio)
    exit(0)
