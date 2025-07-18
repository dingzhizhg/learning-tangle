"""Script to run the baselines."""
import argparse
import importlib
from itertools import cycle, islice

import numpy as np
import os
import sys
import random
import math
import tensorflow as tf
import multiprocessing as mp
import timeit

sys.path.insert(1, './leaf/models')

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel

from tanglecta import Tangle, Transaction, PoisonType, train_single, test_single

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

FLIP_FROM_CLASS = 3
FLIP_TO_CLASS = 8

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def main():
    mp.set_start_method('spawn')

    args = parse_args()

    start_from_round = 0

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = 'leaf/models/%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)

    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = (args.num_rounds if args.num_rounds != -1 else tup[0]) + start_from_round
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    # Suppress tf warnings
    tf.logging.set_verbosity(tf.logging.WARN)

    # Create 2 models
    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)

    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(args.seed, *model_params)

    # Create tangle and tangle metrics
    os.makedirs('tangle_data/transactions', exist_ok=True)

    # Legacy metrics variables (not used)
    global_loss = [],
    global_accuracy = [],
    norm = []

    tangle_tag = f'{clients_per_round}_clients_'

    if start_from_round == 0:
        # genesis = Transaction(client_model.get_params(), [], tag=0)
        genesis = Transaction(client_model.get_params(), None, [], tag=0)
        # genesis = Transaction(client_model.get_params(), None, None, [], tag=0)
        tangle = Tangle({genesis.name(): genesis}, genesis.name())
        tangle.save(tangle_tag + str(0), global_loss, global_accuracy, norm)
    else:
        tangle = Tangle.fromfile(tangle_tag + str(start_from_round))

    # Create server
    server = Server(client_model)

    # Create clients
    poison_type = PoisonType[args.poison_type]
    poison_from = args.poison_from - 1 # We define the first round as number 1, but the loop counter starts with 0
    clients, malicious_clients = setup_clients(args.dataset, client_model, args.use_val_set, args.poison_fraction, poison_type)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    start_time = timeit.default_timer()
    print_stats(0, tangle, random_sample(clients, int(len(clients) * 0.1)), client_num_samples, args, stat_writer_fn, args.use_val_set, (poison_type != PoisonType.NONE))

    # Set up execution timing
    avg_eval_duration = timeit.default_timer() - start_time
    eval_count = 1
    avg_round_duration = 1000

    # Simulate training
    for i in range(start_from_round, num_rounds):
        rounds_remaining = num_rounds - i
        time_remaining = avg_eval_duration * (rounds_remaining // eval_every) + avg_round_duration * rounds_remaining
        print('--- Round %d of %d: Training %d Clients --- Time remaining: ~ %d min' % (i + 1, num_rounds, clients_per_round, time_remaining // 60))
        start_time = timeit.default_timer()

        # Select clients to train this round
        if i >= poison_from:
            if i == poison_from and poison_type != PoisonType.NONE:
                print('Started poisoning in round %d' % (i + 1))
            server.select_clients(i, online(clients), num_clients=clients_per_round)
        else:
            server.select_clients(i, online(clients, exclude_clients=malicious_clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = tangle.run_nodes(train_single, server.selected_clients, i + 1,
                                       num_epochs=args.num_epochs, batch_size=args.batch_size,
                                       malicious_clients=malicious_clients, poison_type=poison_type)
        # norm.append(np.array(norm_this_round).mean(axis=0).tolist() if len(norm_this_round) else [])
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)

        # Update global metrics
        # Todo: Add global accuracy & loss to json files
        #global_accuracy.append(sys_metrics)

        # Update tangle on disk
        tangle.save(tangle_tag + str(i+1), global_loss, global_accuracy, norm)

        avg_round_duration = (avg_round_duration * i / (i+1)) + ((timeit.default_timer() - start_time) / (i+1))

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            start_time = timeit.default_timer()
            average_test_metrics = print_stats(i + 1, tangle, random_sample(clients, int(len(clients) * 0.1)), client_num_samples, args, stat_writer_fn, args.use_val_set, (poison_type != PoisonType.NONE))
            eval_count = eval_count + 1
            avg_eval_duration = (avg_eval_duration * (eval_count-1) / eval_count) + ((timeit.default_timer() - start_time) / eval_count)
            if average_test_metrics['accuracy'] >= args.target_accuracy:
                print("Reached test_accuracy: %g after %d rounds" % (average_test_metrics['accuracy'], i + 1))
                break

    # Close models
    # server.close_model()

def online(clients, exclude_clients=None):
    """We assume all users are always online. However we abuse this method to avoid selecting poisoning clients """
    if exclude_clients is not None:
        return [client for client in clients if client.id not in exclude_clients]
    else:
        return clients

def random_sample(clients, sample_size):
    """Choose a subset of clients to perform the model validation. Only to be used during development to speed up experiment run times"""
    return np.random.choice(clients, min(sample_size, len(clients)), replace=False)

def create_clients(users, groups, train_data, test_data, model):
    if len(groups) == 0:
        groups = [[] for _ in users]
    clients = [Client(u, g, train_data[u], test_data[u], model) for u, g in zip(users, groups)]
    return clients

def setup_clients(dataset, model=None, use_val_set=False, poison_fraction=0, poison_type=PoisonType.NONE):
    """Instantiates clients based on given train and test data directories.

    Return:
        all_clients: list of Client objects.
    """
    eval_set = 'test' if not use_val_set else 'val'
    train_data_dir = os.path.join('leaf', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('leaf', 'data', dataset, 'data', eval_set)

    users, groups, train_data, test_data = read_data(train_data_dir, test_data_dir)

    clients = create_clients(users, groups, train_data, test_data, model)

    num_malicious_clients = math.floor(len(clients) * poison_fraction)
    malicious_clients = [client.id for client in clients[:num_malicious_clients]]
    if poison_type == PoisonType.LABELFLIP:
        # get reference fipping data in case a client possesses no data of class FLIP_FROM_CLASS
        reference_data = []
        for j in range(len(clients)):
            if len([clients[j].train_data['x'][i] for i in range(len(clients[j].train_data['y'])) if clients[j].train_data['y'][i] == FLIP_FROM_CLASS]) > 10:
                reference_data = [clients[j].train_data['x'][i] for i in range(len(clients[j].train_data['y'])) if clients[j].train_data['y'][i] == FLIP_FROM_CLASS]
                break
        for client in clients[:num_malicious_clients]:
            # flip labels
            client_label_counter = len(client.train_data['y'])
            flip_data = [client.train_data['x'][i] for i in range(client_label_counter) if client.train_data['y'][i] == FLIP_FROM_CLASS]
            if len(flip_data) == 0:
                flip_data = reference_data
            client.train_data['x'] = (flip_data * math.ceil(client_label_counter / len(flip_data)))[:client_label_counter]
            client.train_data['y'] = [FLIP_TO_CLASS] * client_label_counter

    return clients, malicious_clients


def get_stat_writer_function(ids, groups, num_samples, args):

    def writer_fn(num_round, metrics, partition):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, partition, 'leaf/models/metrics', '{}_{}'.format('stat', args.metrics_name))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', 'leaf/models/metrics', '{}_{}'.format('sys', args.metrics_name))

    return writer_fn


def print_stats(
    num_round, tangle, clients, num_samples, args, writer, use_val_set, print_conf_matrix):

    train_stat_metrics = tangle.test_model(test_single, clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, num_round, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = tangle.test_model(test_single, clients, set_to_use=eval_set)
    average_test_metrics = print_metrics(test_stat_metrics, num_samples, num_round, prefix='{}_'.format(eval_set), print_conf_matrix=print_conf_matrix)
    writer(num_round, test_stat_metrics, eval_set)

    with open('results.txt', 'a+') as file:
        # file.write('Round %d \n' % num_round)
        file.write(str(average_test_metrics) + ',\n')
    return average_test_metrics



def print_metrics(metrics, weights, num_round, prefix='', print_conf_matrix=False):
    """Prints weighted averages of the given metrics.

    Args:
        metrics: dict with client ids as keys. Each entry is a dict
            with the metrics of that client.
        weights: dict with client ids as keys. Each entry is the weight
            for that client.
    """
    ordered_weights = [weights[c] for c in sorted(weights) if c in metrics]
    # metric_names = metrics_writer.get_metrics_names(metrics)
    # to_ret = None
    metric_names = ['accuracy', 'loss']  # 只保留accuracy和loss
    average_metrics = {}
    for metric in metric_names:
        if metric == 'conf_matrix':
            continue
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]

        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        average_metrics[metric] = np.average(ordered_metric, weights=ordered_weights)

    # print confusion matrix
    if print_conf_matrix:
        if 'conf_matrix' in metric_names:
            full_conf_matrix = sum([metrics[c]['conf_matrix'] for c in sorted(metrics)])
            misclassification = full_conf_matrix[FLIP_FROM_CLASS, FLIP_TO_CLASS] / np.sum(full_conf_matrix[FLIP_FROM_CLASS]) * 100
            average_metrics['misclassification'] = misclassification
            print('Misclassification percentage: %.2f%%' % misclassification)
            #print(full_conf_matrix)
            np.savetxt('conf_matrix.txt', full_conf_matrix, fmt='%4u')

    return average_metrics


if __name__ == '__main__':
    main()
