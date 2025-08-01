import argparse

DATASETS = ['mnist', 'fmnist', 'cifar10', 'cifar100']
SIM_TIMES = ['small', 'medium', 'large']
POISON_TYPES = ['NONE', 'RANDOM', 'LABELFLIP', 'LAZY']


def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument('-dataset',
                    help='name of dataset;',
                    type=str,
                    choices=DATASETS,
                    required=True)
    parser.add_argument('-model',
                    help='name of model;',
                    type=str,
                    required=True)
    parser.add_argument('--num-rounds',
                    help='number of rounds to simulate;',
                    type=int,
                    default=-1)
    parser.add_argument('--eval-every',
                    help='evaluate every ____ rounds;',
                    type=int,
                    default=-1)
    parser.add_argument('--clients-per-round',
                    help='number of clients trained per round;',
                    type=int,
                    default=-1)
    parser.add_argument('--batch-size',
                    help='batch size when clients train on data;',
                    type=int,
                    default=10)
    parser.add_argument('--seed',
                    help='seed for random client sampling and batch splitting',
                    type=int,
                    default=0)
    parser.add_argument('--metrics-name', 
                    help='name for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--metrics-dir', 
                    help='dir for metrics file;',
                    type=str,
                    default='metrics',
                    required=False)
    parser.add_argument('--use-val-set', 
                    help='use validation set;', 
                    action='store_true')

    # Minibatch doesn't support num_epochs, so make them mutually exclusive
    epoch_capability_group = parser.add_mutually_exclusive_group()
    epoch_capability_group.add_argument('--minibatch',
                    help='None for FedAvg, else fraction;',
                    type=float,
                    default=None)
    epoch_capability_group.add_argument('--num-epochs',
                    help='number of epochs when clients train on data;',
                    type=int,
                    default=1)

    parser.add_argument('-t',
                    help='simulation time: small, medium, or large;',
                    type=str,
                    choices=SIM_TIMES,
                    default='large')
    parser.add_argument('-lr',
                    help='learning rate for local optimizers;',
                    type=float,
                    default=-1,
                    required=False)

    parser.add_argument('--poison-type',
                    help='type of malicious clients considered',
                    type=str,
                    choices=POISON_TYPES,
                    default='NONE',
                    required=False)
    parser.add_argument('--poison-fraction',
                    help='fraction of clients being malicious',
                    type=float,
                    default=0,
                    required=False)
    parser.add_argument('--poison-from',
                    help='epoch to start poisoning from',
                    type=float,
                    default=1,
                    required=False)

    parser.add_argument('--num-tips',
                    help='number of tips being selected per round',
                    type=int,
                    default=2,
                    required=False)
    parser.add_argument('--sample-size',
                    help='number of possible tips being sampled per round',
                    type=int,
                    default=2,
                    required=False)

    parser.add_argument('--target-accuracy',
                    help='stop training after reaching this test accuracy',
                    type=float,
                    default=1,
                    required=False)
    parser.add_argument('--reference-avg-top',
                    help='number models to average when picking reference model',
                    type=int,
                    default=1,
                    required=False)

    return parser.parse_args()
