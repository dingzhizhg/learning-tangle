import os
import shutil
from sklearn.model_selection import ParameterGrid

params = {
    'dataset': ['femnist'],
    'model': ['cnn'],
    'num_rounds': [100],
    'eval_every': [10],

    'num_epochs': [1],
    'batch_size': [16],
    # 'num_epochs': [5],
    # 'batch_size': [64],

    'clients_per_round':  [5],
    'num_tips':  [3],
    'sample_size':  [3],
    'reference_avg_top':  [3],
    # tanglect, 这个参数和num_tips保持一致
    'target_accuracy':  [1.0],
    
    'learning_rate':  [0.0003],
    # 'learning_rate':  [0.0001],
    'poison_type':  ['NONE'],
    'poison_fraction':  [0],
    'poison_from':  [0],
    # 'poison_type':  ['LABELFLIP'],
    # 'poison_fraction':  [0.1],
    # 'poison_from':  [100],
}

file_name = 'results.txt'
    
for p in ParameterGrid(params):
    os.system('rm -rf tangle_data')
    command = "python3 main.py -dataset %s -model %s --num-rounds %s --eval-every %s --num-epochs %s --batch-size %s --clients-per-round %s --num-tips %s --sample-size %s --reference-avg-top %s --target-accuracy %s -lr %s --poison-type %s --poison-fraction %s --poison-from %s"
    parameters = (p['dataset'], p['model'], p['num_rounds'], p['eval_every'], p['num_epochs'], p['batch_size'], p['clients_per_round'], p['num_tips'], p['sample_size'], p['reference_avg_top'], p['target_accuracy'], p['learning_rate'], p['poison_type'], p['poison_fraction'], p['poison_from'])
    command = command % parameters
    # with open(file_name, 'a+') as file:
    #     file.write('\n\n' + command + '\n')
    os.system(command)
