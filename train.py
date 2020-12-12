#!/usr/bin/env python
# encoding: utf-8
import re
import shutil
from argparse import ArgumentParser
from os import path

import torch
import torch.optim as optim
from ignite.contrib.handlers import ProgressBar
from ignite.engine import Engine, Events
from ignite.handlers import EarlyStopping
from ignite.metrics import RunningAverage, Precision, Recall, Average, Accuracy, MetricsLambda
from torch.optim.lr_scheduler import StepLR

from model_joint_cut import JointCut
from utils import *

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(name)s %(levelname)s %(message)s",
                    datefmt='%Y-%m-%d  %H:%M:%S %a')


def get_args():
    parser = ArgumentParser()

    parser.add_argument('--train', type=str, default='data/best_2010_csv/train/',
                        help='path of training files')
    parser.add_argument('--valid', type=str, default='data/best_2010_csv/valid/',
                        help='path of validation files')
    parser.add_argument('--test', type=str, default='data/best_2010_csv/test.csv', help='test file '),

    parser.add_argument('--checkpoint', type=str, default='',
                        help='path to checkpoint file')

    parser.add_argument("--cache", type=str, default='cache/best_2010')

    parser.add_argument("--n_gram", type=int, default=21)
    parser.add_argument("--char_embed_dim", type=int, default=64)
    parser.add_argument("--type_embed_dim", type=int, default=16)

    parser.add_argument("--disable_type_embed", default=False, action="store_true")
    parser.add_argument("--disable_type_embed_re", default=False, action="store_true")

    parser.add_argument("--char_embed_dropout", type=float, default=0.15)

    parser.add_argument("--transformer_d_model", type=int, default=80)
    parser.add_argument("--transformer_n_head", type=int, default=8)
    parser.add_argument("--transformer_dim_feedforward", type=int, default=128)
    parser.add_argument("--transformer_num_layers", type=int, default=1)
    parser.add_argument("--transformer_activation", type=str, default="relu", help="relu or gelu")

    parser.add_argument("--syllable_dense_dim", type=int, default=50)
    parser.add_argument("--word_dense_dim", type=int, default=100)

    parser.add_argument("--hidden_activation", type=str, default="relu", help="relu or gelu")
    parser.add_argument("--hidden_dropout", type=float, default=0.1, help="dropout rate for features")

    parser.add_argument("--syllable_loss_lambda", type=float, default=0.5)
    parser.add_argument("--align_loss_lambda", type=float, default=0.5)

    parser.add_argument("--train_batch_size", type=int, default=32768, help="Batch size for training")
    parser.add_argument("--valid_batch_size", type=int, default=32768, help="Batch size for validation")
    parser.add_argument("--test_batch_size", type=int, default=32768, help="Batch size for test")

    parser.add_argument("--lr", type=float, default=1e-3, help="Learning rate")
    parser.add_argument("--weight_decay", type=float, default=1e-5, help="weight decay")

    parser.add_argument("--max_epochs", type=int, default=100, help="Number of training epochs")

    parser.add_argument("--output_path", type=str, default='./output', help="checkpoint directory")

    parser.add_argument("--early_stop_patience", type=int, default=5)
    parser.add_argument("--early_stop_metric", type=str, default='loss', help='loss, acc, MP, MR, F1')

    parser.add_argument("--device", type=str, default="cuda:0" if torch.cuda.is_available() else "cpu",
                        help="Device (cuda or cpu)")

    cfg = parser.parse_args()

    return cfg


def get_model(cfg):
    return JointCut(cfg)
    # return JointCutAtt(cfg)


def initialize(cfg, num_train_batch):
    model = get_model(cfg).to(cfg.device)
    optimizer = optim.Adam(model.parameters(), lr=cfg.lr, weight_decay=cfg.weight_decay)

    lr_scheduler = StepLR(optimizer, step_size=num_train_batch, gamma=0.9)

    return model, optimizer, lr_scheduler


def batch_to_tensor(batch, cfg):
    return (torch.tensor(t, dtype=torch.long) for t in batch)


def create_trainer(model, optimizer, lr_scheduler, cfg):
    # Define any training logic for iteration update
    def _train_step(engine, batch):
        x_char, x_type, y_word, y_syllable = batch_to_tensor(batch, cfg)
        x_char, x_type, y_word, y_syllable = (t.to(cfg.device) for t in [x_char, x_type, y_word, y_syllable])

        model.train()

        logits_word, logits_syllable = model(x_char, x_type)
        loss, word_loss, syllable_loss, align_loss = model.joint_loss(logits_word, y_word, logits_syllable, y_syllable)

        optimizer.zero_grad()
        loss.backward()
        # torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.max_norm)
        optimizer.step()
        lr_scheduler.step()

        return ((logits_word > 0.5).long(), y_word, (logits_syllable > 0.5).long(), y_syllable,
                loss, word_loss, syllable_loss, align_loss)

    # Define trainer engine
    trainer = Engine(_train_step)

    # Define metrics for trainer
    w_loss = Accuracy(lambda x: x[0:2])
    RunningAverage(w_loss).attach(trainer, 'w_acc')

    s_acc = Accuracy(lambda x: x[2:4])
    RunningAverage(s_acc).attach(trainer, 's_acc')

    RunningAverage(output_transform=lambda x: x[4]).attach(trainer, 'loss')
    RunningAverage(output_transform=lambda x: x[5]).attach(trainer, 'w_loss')
    RunningAverage(output_transform=lambda x: x[6]).attach(trainer, 's_loss')
    RunningAverage(output_transform=lambda x: x[7]).attach(trainer, 'a_loss')

    # Add progress bar showing trainer metrics
    metrics = ['w_acc', 's_acc', 'loss', 'w_loss', 's_loss', 'a_loss']
    ProgressBar(persist=True).attach(trainer, metrics)

    return trainer


def create_evaluator(model, cfg):
    def _validation_step(_, batch):
        model.eval()
        with torch.no_grad():
            x_char, x_type, y_word, y_syllable = batch_to_tensor(batch, cfg)
            x_char, x_type, y_word, y_syllable = (t.to(cfg.device) for t in [x_char, x_type, y_word, y_syllable])

            logits_word, logits_syllable = model(x_char, x_type)
            loss, word_loss, syllable_loss, align_loss = model.joint_loss(logits_word, y_word, logits_syllable,
                                                                          y_syllable)

            return ((logits_word > 0.5).long(), y_word, (logits_syllable > 0.5).long(), y_syllable,
                    loss, word_loss, syllable_loss, align_loss)

    evaluator = Engine(_validation_step)

    w_loss = Accuracy(lambda x: x[0:2])
    w_loss.attach(evaluator, 'w_acc')

    s_acc = Accuracy(lambda x: x[2:4])
    s_acc.attach(evaluator, 's_acc')

    Average(lambda x: x[4]).attach(evaluator, 'loss')
    Average(lambda x: x[5]).attach(evaluator, 'w_loss')
    Average(lambda x: x[6]).attach(evaluator, 's_loss')
    Average(lambda x: x[7]).attach(evaluator, 'a_loss')

    accuracy = Accuracy(lambda x: x[0:2])
    accuracy.attach(evaluator, "acc")

    w_precision = Precision(lambda x: x[0:2])
    w_precision.attach(evaluator, 'WP')
    MetricsLambda(lambda t: torch.mean(t).item(), w_precision).attach(evaluator, "WMP")

    s_precision = Precision(lambda x: x[2:4])
    s_precision.attach(evaluator, 'SP')
    MetricsLambda(lambda t: torch.mean(t).item(), s_precision).attach(evaluator, "SMP")

    w_recall = Recall(lambda x: x[0:2])
    w_recall.attach(evaluator, 'WR')
    MetricsLambda(lambda t: torch.mean(t).item(), w_recall).attach(evaluator, "WMR")

    s_recall = Recall(lambda x: x[2:4])
    s_recall.attach(evaluator, 'SR')
    MetricsLambda(lambda t: torch.mean(t).item(), s_recall).attach(evaluator, "SMR")

    w_f1 = 2. * w_precision * w_recall / (w_precision + w_recall + 1e-20)
    w_f1 = MetricsLambda(lambda t: torch.mean(t).item(), w_f1)
    w_f1.attach(evaluator, "WF1")

    s_f1 = 2. * s_precision * s_recall / (s_precision + s_recall + 1e-20)
    s_f1 = MetricsLambda(lambda t: torch.mean(t).item(), s_f1)
    s_f1.attach(evaluator, "SF1")

    return evaluator


def training(cfg, ds):
    # create data generators
    train, valid, test = ds

    # train = train[:100]
    # valid = valid[:100]
    # test = test[:100]

    train_batch_num = math.ceil(len(train) / cfg.train_batch_size)
    train_gen = data_generator(train, cfg.train_batch_size, True, True)

    valid_batch_num = math.ceil(len(valid) / cfg.valid_batch_size)
    valid_gen = data_generator(valid, cfg.valid_batch_size, False, True)

    test_batch_num = math.ceil(len(test) / cfg.test_batch_size)
    test_gen = data_generator(test, cfg.test_batch_size)

    # create model, optimizer and learning rate scheduler
    model, optimizer, lr_scheduler = initialize(cfg, train_batch_num)

    if len(cfg.checkpoint) > 0:
        logger.info("loading checkpoint from %s" % cfg.checkpoint)
        ckpt = torch.load(cfg.checkpoint, map_location=torch.device(cfg.device))

        model_cfg = ckpt['config']
        model = JointCut(model_cfg).to(cfg.device)
        model.load_state_dict(ckpt['state'])

    # print settings
    print_table([(k, str(v)[0:60]) for k, v in vars(cfg).items()])

    # print model parameters
    print(parameters_string(model))

    # Setup model trainer and evaluator
    trainer = create_trainer(model, optimizer, lr_scheduler, cfg)
    evaluator = create_evaluator(model, cfg)
    tester = create_evaluator(model, cfg)

    # Run model evaluation every epochs and show results
    @trainer.on(Events.EPOCH_COMPLETED(every=1))
    def evaluate_model():
        val_state = evaluator.run(valid_gen, epoch_length=valid_batch_num)
        eval_metrics = [(m, val_state.metrics[m]) for m in ['WP', 'WR', 'WF1', 'SP', 'SR', 'SF1', 'loss']]
        eval_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in eval_metrics])
        logger.info(eval_metrics)

    def score_function(_):
        if cfg.early_stop_metric == 'loss':
            return - evaluator.state.metrics['loss']
        elif cfg.early_stop_metric in evaluator.state.metrics:
            return evaluator.state.metrics[cfg.early_stop_metric]
        else:
            raise Exception('unsupported metric %s' % cfg.early_stop_metric)

    @evaluator.on(Events.EPOCH_COMPLETED)
    def save_checkpoint():
        best_score = getattr(evaluator.state, 'best_score', None)
        epoch_score = score_function(evaluator)
        epoch = trainer.state.epoch

        os.makedirs(cfg.output_path, exist_ok=True)
        if best_score is None or epoch_score > best_score:
            checkpoint = "checkpoint_%03d_%.4f.pt" % (epoch, epoch_score)
            checkpoint = path.join(cfg.output_path, checkpoint)
            torch.save({'state': model.state_dict(), 'config': cfg}, checkpoint)

            evaluator.state.best_score = epoch_score
            evaluator.state.best_epoch = epoch

            best_checkpoint = path.join(cfg.output_path, "best.pt")
            shutil.copy(checkpoint, best_checkpoint)

    hdl_early_stop = EarlyStopping(patience=cfg.early_stop_patience, score_function=score_function, trainer=trainer)
    # Note: the handler is attached to an *Evaluator* (runs one epoch on validation dataset).
    evaluator.add_event_handler(Events.COMPLETED, hdl_early_stop)

    # hdl_pred_saver = PredSaver(lambda x: (x[1], x[0] if cfg.multi_label else torch.max(x[0], dim=-1)[1]))
    # tester.add_event_handler(Events.ITERATION_COMPLETED, hdl_pred_saver)

    trainer.run(train_gen, max_epochs=cfg.max_epochs, epoch_length=train_batch_num)

    best_ckpt = path.join(cfg.output_path, "best.pt")
    best_epoch = evaluator.state.best_epoch
    logger.info("Load best model checkpoint from Epoch[%d], '%s'" % (best_epoch, best_ckpt))

    ckpt = torch.load(best_ckpt)
    model.load_state_dict(ckpt['state'])

    logger.info("Evaluate model on test data, %s" % cfg.test)
    test_state = tester.run(test_gen, epoch_length=test_batch_num)
    test_metrics = [(m, test_state.metrics[m]) for m in ['WP', 'WR', 'WF1', 'SP', 'SR', 'SF1', 'loss']]
    test_metrics = ", ".join([("%s: %.4f" % (m, v)) for m, v in test_metrics])
    logger.info(test_metrics)


def preprocess(cfg):
    train = load_csv_data(cfg.train)
    valid = load_csv_data(cfg.valid)
    test = load_csv_data(cfg.test)

    train = make_samples(train, cfg.n_gram)
    valid = make_samples(valid, cfg.n_gram)
    test = make_samples(test, cfg.n_gram)

    logger.info("saving data to cache %s" % cfg.cache)
    if not os.path.exists(cfg.cache):
        os.makedirs(re.sub("/[^/]+$", "", cfg.cache), exist_ok=True)

    torch.save([train, valid, test], cfg.cache)

    return train, valid, test


# --- Single computation device ---
if __name__ == "__main__":
    logger = logging.getLogger('main')

    config = get_args()

    # add vocab args
    config.char_vocab_size = len(CHARS)
    config.type_vocab_size = len(CHAR_TYPES)

    if not path.isfile(config.cache):
        dataset = preprocess(config)
    else:
        logger.info("Load data from cache: '%s'" % config.cache)
        dataset = torch.load(config.cache)

    torch.manual_seed(2357)
    # np.random.seed(2357)

    if 'cuda' in config.device:
        torch.cuda.manual_seed(2357)

    training(config, dataset)
    exit(0)
