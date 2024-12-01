from dataclasses import dataclass
from typing import Optional

@dataclass
class EditorConfig:
    # Model settings
    model_name: str = "gpt-2"
    max_tokens: int = 1000
    temperature: float = 0.0
    
    # Cache settings
    cache_size: int = 1000
    neural_cache_dim: int = 768
    
    # Optimization settings
    num_quantum_iterations: int = 3
    genetic_population_size: int = 100
    rl_learning_rate: float = 0.001
    
    # Performance settings
    batch_size: Optional[int] = 16
    use_gpu: bool = True
    num_threads: int = 4

    # Merkle tree settings
    merkle_depth: int = 8
    hash_algorithm: str = "sha256"

    @classmethod
    def from_dict(cls, config_dict: dict) -> "EditorConfig":
        return cls(**{
            k: v for k, v in config_dict.items()
            if k in cls.__dataclass_fields__
        }) 