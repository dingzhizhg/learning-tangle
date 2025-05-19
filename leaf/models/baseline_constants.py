"""Configuration file for common models/experiments"""

MAIN_PARAMS = { 
    'femnist': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
    'cifar10': {
        'small': (6, 2, 2),
        'medium': (8, 2, 2),
        'large': (20, 1, 2)
        },
    'cifar100': {
        'small': (30, 10, 2),
        'medium': (100, 10, 2),
        'large': (400, 20, 2)
        },
}
"""dict: Specifies execution parameters (tot_num_rounds, eval_every_num_rounds, clients_per_round)"""

MODEL_PARAMS = {
    'femnist.cnn': (0.0003, 62), # lr, num_classes
    'cifar10.cnn': (0.0001, 10), # lr, num_classes
    'cifar100.cnn': (0.0001, 10), # lr, num_classes
}
"""dict: Model specific parameter specification"""

ACCURACY_KEY = 'accuracy'
BYTES_WRITTEN_KEY = 'bytes_written'
BYTES_READ_KEY = 'bytes_read'
LOCAL_COMPUTATIONS_KEY = 'local_computations'
NUM_ROUND_KEY = 'round_number'
NUM_SAMPLES_KEY = 'num_samples'
CLIENT_ID_KEY = 'client_id'
