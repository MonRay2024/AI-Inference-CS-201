from dataclasses import dataclass
from typing import Dict, List, Optional
import time
import torch
import psutil

@dataclass
class PerformanceMetrics:
    generation_time: float
    cache_hits: int
    cache_misses: int
    memory_usage: float
    gpu_usage: Optional[float]
    successful_speculations: int
    failed_speculations: int

class MetricsCollector:
    def __init__(self):
        self.metrics_history: List[PerformanceMetrics] = []
        
    def collect_metrics(self) -> PerformanceMetrics:
        gpu_usage = None
        if torch.cuda.is_available():
            gpu_usage = torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated()
            
        return PerformanceMetrics(
            generation_time=time.time(),
            cache_hits=self.cache_hits,
            cache_misses=self.cache_misses,
            memory_usage=psutil.Process().memory_info().rss / (1024 * 1024),
            gpu_usage=gpu_usage,
            successful_speculations=self.successful_speculations,
            failed_speculations=self.failed_speculations
        ) 