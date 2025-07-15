import importlib
import random
import tensorflow as tf
import numpy as np
import time

from utils.args import parse_args
from baseline_constants import MAIN_PARAMS, MODEL_PARAMS, BYTES_WRITTEN_KEY, BYTES_READ_KEY, LOCAL_COMPUTATIONS_KEY
from client import Client

from .node import Node
from .blockchain import Blockchain

def build_client(u, g, flops, train_data, eval_data):
    args = parse_args()

    model_path = '%s.%s' % (args.dataset, args.model)
    mod = importlib.import_module(model_path)
    ClientModel = getattr(mod, 'ClientModel')

    tup = MAIN_PARAMS[args.dataset][args.t]
    num_rounds = args.num_rounds if args.num_rounds != -1 else tup[0]
    eval_every = args.eval_every if args.eval_every != -1 else tup[1]
    clients_per_round = args.clients_per_round if args.clients_per_round != -1 else tup[2]

    model_params = MODEL_PARAMS[model_path]
    if args.lr != -1:
        model_params_list = list(model_params)
        model_params_list[0] = args.lr
        model_params = tuple(model_params_list)


    # Create client model, and share params with server model
    tf.reset_default_graph()
    client_model = ClientModel(1234, *model_params)
    client_model.flops = flops
    return Client(u, g, train_data, eval_data, client_model)

def train_single(u, g, flops, seed, train_data, eval_data, tangle_name, malicious_node, poison_type, blockchain=None):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    random.seed(1 + seed)
    np.random.seed(12 + seed)
    tf.compat.v1.set_random_seed(123 + seed)

    client = build_client(u, g, flops, train_data, eval_data)

    tangle = Tangle.fromfile(tangle_name)
    node_id = f"device_{u}"
    
    if malicious_node:
        node = Node(client, tangle, poison_type, blockchain, node_id)
    else:
        node = Node(client, tangle, blockchain=blockchain, node_id=node_id)

    args = parse_args()
    
    # BAFL 10步流程
    # 步骤1：本地模型更新
    local_model_data = node.update_local_model(args.num_epochs, args.batch_size)
    
    # 步骤2：本地模型上传
    if node.upload_local_model(local_model_data):
        # 步骤3：本地模型交叉验证（在upload_local_model中完成）
        pass
    
    # 步骤4：模型和分数下载
    global_info = node.download_global_model_and_score()
    
    # 步骤5：全局模型和分数上传（如果需要）
    if global_info and global_info['global_model']:
        # 计算新的全局模型
        global_model = global_info['global_model']
        new_score = global_info['current_score'] + 0.1  # 简化的分数更新
        
        node.upload_global_model_and_score(global_model, new_score)
        
        # 步骤7：全局模型交叉验证（在upload_global_model_and_score中完成）
        pass
    
    # 步骤8：区块生成
    new_block = node.mine_block()
    
    # 步骤9：区块传播
    if new_block:
        node.propagate_block(new_block)
    
    # 步骤10：全局模型下载
    latest_global_model = node.download_latest_global_model()
    
    # 兼容原有的tangle处理
    tx, metrics, comp = node.process_next_batch(args.num_epochs, args.batch_size, args.num_tips, args.sample_size, args.reference_avg_top)

    sys_metrics = {BYTES_WRITTEN_KEY: 0,
                   BYTES_READ_KEY: 0,
                   LOCAL_COMPUTATIONS_KEY: 0}
    sys_metrics[BYTES_READ_KEY] += node.client.model.size
    sys_metrics[BYTES_WRITTEN_KEY] += node.client.model.size
    sys_metrics[LOCAL_COMPUTATIONS_KEY] = comp

    return tx, metrics, u, sys_metrics

def test_single(u, g, flops, seed, train_data, eval_data, tangle_name, set_to_use):
    # Suppress tf warnings
    tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)

    random.seed(1 + seed)
    np.random.seed(12 + seed)
    tf.compat.v1.set_random_seed(123 + seed)

    client = build_client(u, g, flops, train_data, eval_data)

    tangle = Tangle.fromfile(tangle_name)
    node = Node(client, tangle)
    args = parse_args()
    reference_txs, reference, reference_poison_score = node.obtain_reference_params(avg_top=args.reference_avg_top)
    node.client.model.set_params(reference)
    metrics = node.client.test(set_to_use)

    metrics['consensus_round'] = np.average([tangle.transactions[tx].tag for tx in reference_txs])
    metrics['consensus_poisoning'] = reference_poison_score

    metrics['norm'] = 0
    parents = [tangle.transactions[tx].parents for tx in reference_txs]
    parents = set.union(*parents)
    if len(parents) == 2:
        p1, p2 = parents
        pw1 = tangle.transactions[p1].load_weights()
        pw2 = tangle.transactions[p2].load_weights()
        partial_norms = [np.linalg.norm(np.array(weights)[0] - np.array(weights)[1]) for weights in zip(pw1, pw2)]
        metrics['norm'] = np.linalg.norm(partial_norms)

    return u, metrics

def create_blockchain_manager():
    """创建区块链管理器"""
    return Blockchain(difficulty=2)  # 降低难度以简化挖矿

def run_bafl_epoch(clients, blockchain, rnd, num_epochs=1, batch_size=10, malicious_clients=None, poison_type=None):
    """运行一个BAFL epoch"""
    results = []
    
    for client in clients:
        malicious_node = client.id in malicious_clients if malicious_clients else False
        poison_type = poison_type if malicious_node else None
        
        result = train_single(
            client.id, client.group, client.model.flops, 
            random.randint(0, 4294967295), client.train_data, client.eval_data, 
            "bafl_tangle", malicious_node, poison_type, blockchain
        )
        results.append(result)
    
    return results
