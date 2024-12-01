from typing import Tuple, List, Dict, Optional, Any
from tqdm import tqdm
from ..data_structures.merkle_tree import MerkleTree
from ..optimizers.neural_cache import NeuralCache
from ..optimizers.attention_mechanism import MultiHeadAttention 
import torch