import json
import time
import hashlib
from typing import List, Dict, Any, Optional, Tuple
from .block import Block

class Blockchain:
    """区块链类，管理整个区块链网络"""
    
    def __init__(self, difficulty: int = 4):
        self.chain: List[Block] = []
        self.pending_models: List[Dict[str, Any]] = []
        self.difficulty = difficulty
        self.miners: List[Miner] = []
        self.validator = ModelValidator()
        
        # 创建创世区块
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """创建创世区块"""
        genesis_block = Block(
            index=0,
            previous_hash="0",
            timestamp=time.time(),
            model_data={
                'type': 'genesis',
                'weights': None,
                'score': 0.0,
                'device_id': 'genesis'
            }
        )
        self.chain.append(genesis_block)
    
    def get_latest_block(self) -> Block:
        """获取最新的区块"""
        return self.chain[-1]
    
    def add_miner(self, miner: Miner):
        """添加矿工到网络"""
        self.miners.append(miner)
    
    def add_pending_model(self, model_data: Dict[str, Any]):
        """添加待处理的模型到待处理队列"""
        self.pending_models.append(model_data)
    
    def mine_pending_models(self, miner_id: str) -> Optional[Block]:
        """挖矿：处理待处理的模型并创建新区块"""
        if not self.pending_models:
            return None
        
        # 获取待处理的模型
        model_data = self.pending_models.pop(0)
        
        # 创建新区块
        new_block = Block(
            index=len(self.chain),
            previous_hash=self.get_latest_block().hash,
            timestamp=time.time(),
            model_data=model_data,
            miner_id=miner_id
        )
        
        # 挖矿
        if new_block.mine_block(self.difficulty):
            return new_block
        
        return None
    
    def add_block(self, block: Block) -> bool:
        """添加区块到区块链"""
        # 验证区块
        if not self.is_valid_block(block):
            return False
        
        # 验证模型数据
        if not self.validator.validate_model(block.model_data):
            return False
        
        self.chain.append(block)
        return True
    
    def is_valid_block(self, block: Block) -> bool:
        """验证区块的有效性"""
        # 检查区块索引
        if block.index != len(self.chain):
            return False
        
        # 检查前一个区块的哈希
        if block.previous_hash != self.get_latest_block().hash:
            return False
        
        # 检查区块哈希
        if not block.is_valid():
            return False
        
        return True
    
    def is_chain_valid(self) -> bool:
        """验证整个区块链的有效性"""
        for i in range(1, len(self.chain)):
            current_block = self.chain[i]
            previous_block = self.chain[i-1]
            
            # 检查区块哈希
            if not current_block.is_valid():
                return False
            
            # 检查区块链接
            if current_block.previous_hash != previous_block.hash:
                return False
        
        return True
    
    def get_global_model(self) -> Optional[Dict[str, Any]]:
        """获取最新的全局模型"""
        if len(self.chain) <= 1:
            return None
        
        # 获取最新的模型数据
        latest_block = self.get_latest_block()
        if latest_block.model_data['type'] == 'global_model':
            return latest_block.model_data
        
        return None
    
    def get_model_history(self, device_id: str) -> List[Dict[str, Any]]:
        """获取特定设备的模型历史"""
        history = []
        for block in self.chain:
            if (block.model_data.get('device_id') == device_id and 
                block.model_data['type'] != 'genesis'):
                history.append(block.model_data)
        
        return history
    
    def broadcast_block(self, block: Block):
        """广播区块到所有矿工"""
        for miner in self.miners:
            miner.receive_block(block)
    
    def handle_fork(self, competing_block: Block) -> bool:
        """处理分叉情况"""
        # 简单的分叉处理：选择最长的链
        # 在实际实现中，这里会有更复杂的共识机制
        
        # 如果竞争区块的索引更高，接受它
        if competing_block.index > len(self.chain) - 1:
            # 验证竞争区块
            if self.is_valid_block(competing_block):
                self.chain.append(competing_block)
                return True
        
        return False
    
    def save_to_file(self, filename: str):
        """保存区块链到文件"""
        data = {
            'chain': [block.to_dict() for block in self.chain],
            'difficulty': self.difficulty,
            'pending_models': self.pending_models
        }
        
        with open(filename, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    @classmethod
    def load_from_file(cls, filename: str) -> 'Blockchain':
        """从文件加载区块链"""
        with open(filename, 'r') as f:
            data = json.load(f)
        
        blockchain = cls(difficulty=data['difficulty'])
        blockchain.chain = [Block.from_dict(block_data) for block_data in data['chain']]
        blockchain.pending_models = data['pending_models']
        
        return blockchain 