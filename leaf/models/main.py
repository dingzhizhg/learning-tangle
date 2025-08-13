"""Script to run the baselines."""
import argparse
import importlib
import numpy as np
import os
import sys
import random
import tensorflow as tf
import math
import timeit

import metrics.writer as metrics_writer

from baseline_constants import MAIN_PARAMS, MODEL_PARAMS
from client import Client
from server import Server
from model import ServerModel
from poison_type import PoisonType

from utils.args import parse_args
from utils.model_utils import read_data

STAT_METRICS_PATH = 'metrics/stat_metrics.csv'
SYS_METRICS_PATH = 'metrics/sys_metrics.csv'

FLIP_FROM_CLASS = 3
FLIP_TO_CLASS = 8
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'

def random_sample(clients, sample_size):
    """Choose a subset of clients to perform the model validation. Only to be used during development to speed up experiment run times"""
    return np.random.choice(clients, min(sample_size, len(clients)), replace=False)

def main():

    args = parse_args()

    # Set the random seed if provided (affects client sampling, and batching)
    random.seed(1 + args.seed)
    np.random.seed(12 + args.seed)
    tf.set_random_seed(123 + args.seed)

    model_path = '%s/%s.py' % (args.dataset, args.model)
    if not os.path.exists(model_path):
        print('Please specify a valid dataset and a valid model.')
    model_path = '%s.%s' % (args.dataset, args.model)
    
    print('############################## %s ##############################' % model_path)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = 100

    eval_every = 1
    clients_per_round = 10

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

    # 添加全局步数管理
    global_step = tf.Variable(0, trainable=False, name='global_step')
    client_model.global_step = global_step

    # Clear delay.txt file for new run
    with open('delay.txt', 'w') as delay_file:
        delay_file.write('Training Round Delays:\n')
        delay_file.write('=' * 30 + '\n')

    # Clear train.txt file for new run
    with open('train.txt', 'w') as train_file:
        train_file.write('Training Duration (All Selected Nodes per Round):\n')
        train_file.write('=' * 30 + '\n')

    # Create server
    server = Server(client_model)

    # Create clients
    poison_type = PoisonType[args.poison_type]
    poison_from = args.poison_from
    clients, malicious_clients = setup_clients(args.dataset, client_model, args.use_val_set, args.poison_fraction, poison_type)
    client_ids, client_groups, client_num_samples = server.get_clients_info(clients)
    print('Clients in Total: %d' % len(clients))

    # Initial status
    print('--- Random Initialization ---')
    stat_writer_fn = get_stat_writer_function(client_ids, client_groups, client_num_samples, args)
    sys_writer_fn = get_sys_writer_function(args)
    start_time = timeit.default_timer()
    print_stats(0, server, random_sample(clients, int(len(clients) * 0.1)), client_num_samples, args, stat_writer_fn, args.use_val_set)

    # Simulate training
    # 记录训练起始（不含初始化与第0轮评估）
    training_start_time = timeit.default_timer()
    total_training_duration = 0
    avg_training_duration = 0
    avg_round_duration = 1000

    for i in range(num_rounds):
        print('--- Round %d of %d: Training %d Clients ---' % (i + 1, num_rounds, clients_per_round))
        round_start_time = timeit.default_timer()

        # Select clients to train this round
        if i >= poison_from:
            if i == poison_from and poison_type != PoisonType.NONE:
                print('Started poisoning in round %d' % (i + 1))
            server.select_clients(i, online(clients), num_clients=clients_per_round)
        else:
            server.select_clients(i, online(clients, exclude_clients=malicious_clients), num_clients=clients_per_round)
        c_ids, c_groups, c_num_samples = server.get_clients_info(server.selected_clients)

        # Simulate server model training on selected clients' data
        sys_metrics = measure_and_log_train_duration(
            i + 1,
            server.train_model,
            num_epochs=args.num_epochs,
            batch_size=args.batch_size,
            minibatch=args.minibatch,
            malicious_clients=malicious_clients,
            poison_type=poison_type
        )
        sys_writer_fn(i + 1, c_ids, sys_metrics, c_groups, c_num_samples)
        
        # 更新训练时长跟踪：从 train.txt 中读取本轮训练耗时并累计
        try:
            with open('train.txt', 'r') as train_file:
                lines = train_file.readlines()
                if lines:
                    last_line = lines[-1].strip()
                    if last_line.startswith(f'Round {i + 1}:'):
                        round_training_duration = float(last_line.split(':')[1].split()[0])
                        total_training_duration += round_training_duration
                        avg_training_duration = total_training_duration / (i + 1)
        except Exception:
            pass

        # Update server model
        server.update_model()
        
        # 更新全局步数
        with tf.Session() as sess:
            sess.run(global_step.assign(i + 1))

        # Calculate round duration and write to delay.txt
        round_duration = timeit.default_timer() - round_start_time
        with open('delay.txt', 'a+') as delay_file:
            delay_file.write(f'Round {i + 1}: {round_duration:.4f} seconds\n')

        # 更新平均轮耗时
        avg_round_duration = (avg_round_duration * i / (i + 1)) + (round_duration / (i + 1))

        # Test model
        if (i + 1) % eval_every == 0 or (i + 1) == num_rounds:
            print_stats(i + 1, server, random_sample(clients, int(len(clients) * 0.1)), client_num_samples, args, stat_writer_fn, args.use_val_set)
    
    # Write total training time to delay.txt（仅训练阶段，不含初始化与评估）
    total_training_time = timeit.default_timer() - training_start_time
    with open('delay.txt', 'a+') as delay_file:
        delay_file.write('=' * 30 + '\n')
        delay_file.write(f'Total Training Time: {total_training_time:.4f} seconds\n')
        delay_file.write(f'Average Round Time: {avg_round_duration:.4f} seconds\n')
    
    # 写入 train.txt 汇总
    with open('train.txt', 'a+') as train_file:
        train_file.write('=' * 30 + '\n')
        train_file.write(f'Total Training Duration: {total_training_duration:.4f} seconds\n')
        train_file.write(f'Average Training Duration per Round: {avg_training_duration:.4f} seconds\n')
    
    print(f'Training completed. Total time: {total_training_time:.4f} seconds')
    print(f'Delay information saved to delay.txt')

    # Save server model
    ckpt_path = os.path.join('checkpoints', args.dataset)
    if not os.path.exists(ckpt_path):
        os.makedirs(ckpt_path)
    save_path = server.save_model(os.path.join(ckpt_path, '{}.ckpt'.format(args.model)))
    print('Model saved in path: %s' % save_path)

    # Close models
    server.close_model()

def online(clients, exclude_clients=None):
    """We assume all users are always online. However we abuse this method to avoid selecting poisoning clients """
    if exclude_clients is not None:
        return [client for client in clients if client.id not in exclude_clients]
    else:
        return clients


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
    train_data_dir = os.path.join('..', 'data', dataset, 'data', 'train')
    test_data_dir = os.path.join('..', 'data', dataset, 'data', eval_set)

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
            num_round, ids, metrics, groups, num_samples, partition, args.metrics_dir, '{}_{}'.format(args.metrics_name, 'stat'))

    return writer_fn


def get_sys_writer_function(args):

    def writer_fn(num_round, ids, metrics, groups, num_samples):
        metrics_writer.print_metrics(
            num_round, ids, metrics, groups, num_samples, 'train', args.metrics_dir, '{}_{}'.format(args.metrics_name, 'sys'))

    return writer_fn


def measure_and_log_train_duration(round_number, training_callable, *args, **kwargs):
    start_time = timeit.default_timer()
    result = training_callable(*args, **kwargs)
    training_duration = timeit.default_timer() - start_time
    with open('train.txt', 'a+') as train_file:
        train_file.write(f'Round {round_number}: {training_duration:.4f} seconds\n')
    return result


def print_stats(num_round, server, clients, num_samples, args, writer, use_val_set):

    train_stat_metrics = server.test_model(clients, set_to_use='train')
    print_metrics(train_stat_metrics, num_samples, num_round, prefix='train_')
    writer(num_round, train_stat_metrics, 'train')

    eval_set = 'test' if not use_val_set else 'val'
    test_stat_metrics = server.test_model(clients, set_to_use=eval_set)
    average_test_metrics = print_metrics(test_stat_metrics, num_samples, num_round, prefix='{}_'.format(eval_set), print_conf_matrix=False)
    writer(num_round, test_stat_metrics, eval_set)

    with open('results.txt', 'a+') as file:
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
    metric_names = ['accuracy', 'loss']  # 只保留accuracy和loss
    average_metrics = {}
    
    for metric in metric_names:
        if metric not in metrics[next(iter(metrics))]:  # 检查指标是否存在
            continue
        ordered_metric = [metrics[c][metric] for c in sorted(metrics)]
        print('%s: %g, 10th percentile: %g, 50th percentile: %g, 90th percentile %g' \
              % (prefix + metric,
                 np.average(ordered_metric, weights=ordered_weights),
                 np.percentile(ordered_metric, 10),
                 np.percentile(ordered_metric, 50),
                 np.percentile(ordered_metric, 90)))
        average_metrics[metric] = np.average(ordered_metric, weights=ordered_weights)

    return average_metrics


if __name__ == '__main__':
    main()
