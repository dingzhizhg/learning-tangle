#!/usr/bin/env python3
"""
BAFL (Blockchain-based Asynchronous Federated Learning) 使用示例
演示如何使用区块链作为账本的联邦学习框架
"""

import numpy as np
import tensorflow as tf
from .blockchain import Blockchain
from .node import Node
from .node_process import create_blockchain_manager, run_bafl_epoch

def example_bafl_workflow():
    """BAFL工作流程示例"""
    
    print("=== BAFL 区块链联邦学习示例 ===")
    
    # 1. 创建区块链网络
    print("1. 创建区块链网络...")
    blockchain = create_blockchain_manager()
    print(f"区块链创建成功，难度: {blockchain.difficulty}")
    
    # 2. 模拟设备节点
    print("\n2. 创建设备节点...")
    # 这里需要实际的client对象，简化示例
    # 在实际使用中，这些client应该包含真实的模型和数据
    
    # 3. 运行BAFL epoch
    print("\n3. 运行BAFL epoch...")
    print("执行10步BAFL流程:")
    print("  步骤1: 本地模型更新")
    print("  步骤2: 本地模型上传")
    print("  步骤3: 本地模型交叉验证")
    print("  步骤4: 模型和分数下载")
    print("  步骤5: 全局模型和分数上传")
    print("  步骤6: 全局模型和分数上传（到全局模型）")
    print("  步骤7: 全局模型交叉验证")
    print("  步骤8: 区块生成")
    print("  步骤9: 区块传播")
    print("  步骤10: 全局模型下载")
    
    # 4. 查看区块链状态
    print("\n4. 区块链状态:")
    print(f"  区块数量: {len(blockchain.chain)}")
    print(f"  待处理模型: {len(blockchain.pending_models)}")
    print(f"  区块链有效: {blockchain.is_chain_valid()}")
    
    # 5. 获取全局模型
    print("\n5. 获取全局模型:")
    global_model = blockchain.get_global_model()
    if global_model:
        print(f"  全局模型类型: {global_model['type']}")
        print(f"  设备ID: {global_model['device_id']}")
        print(f"  分数: {global_model['score']}")
    else:
        print("  暂无全局模型")
    
    print("\n=== BAFL示例完成 ===")

def example_blockchain_operations():
    """区块链操作示例"""
    
    print("\n=== 区块链操作示例 ===")
    
    # 创建区块链
    blockchain = Blockchain(difficulty=2)
    
    # 添加一些测试模型数据
    test_model_data = {
        'type': 'local_model',
        'weights': [np.random.randn(10, 10) for _ in range(3)],
        'score': 85.5,
        'device_id': 'test_device_1',
        'timestamp': 1234567890
    }
    
    blockchain.add_pending_model(test_model_data)
    
    # 挖矿
    new_block = blockchain.mine_pending_models("miner_1")
    if new_block:
        print(f"新区块生成成功: {new_block.hash[:10]}...")
        print(f"区块索引: {new_block.index}")
        print(f"矿工ID: {new_block.miner_id}")
        
        # 添加到区块链
        if blockchain.add_block(new_block):
            print("区块已添加到区块链")
        else:
            print("区块添加失败")
    
    print(f"区块链长度: {len(blockchain.chain)}")
    print("=== 区块链操作示例完成 ===")

if __name__ == "__main__":
    # 运行示例
    example_bafl_workflow()
    example_blockchain_operations() 