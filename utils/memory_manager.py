'''
# author: Yuchen Hui
# email: huiyc@stu.xidian.edu.cn
# date: 2025-10-11
# description: 内存管理
'''

import torch
import gc
import psutil
import os

class MemoryManager:
    def __init__(self, logger):
        self.logger = logger
        self.initial_gpu_memory = torch.cuda.memory_allocated()
        self.initial_ram = psutil.Process(os.getpid()).memory_info().rss
        
    def check_memory(self, step):
        current_gpu = torch.cuda.memory_allocated()
        peak_gpu = torch.cuda.max_memory_allocated()
        gpu_growth = current_gpu - self.initial_gpu_memory
        
        current_ram = psutil.Process(os.getpid()).memory_info().rss
        ram_growth = current_ram - self.initial_ram
        
        self.logger.info(
            f"\nMemory Status at {step}:\n"
            f"GPU Memory: Current={current_gpu/1024**3:.2f}GB, "
            f"Peak={peak_gpu/1024**3:.2f}GB, "
            f"Growth={gpu_growth/1024**3:.2f}GB\n"
            f"RAM: Current={current_ram/1024**3:.2f}GB, "
            f"Growth={ram_growth/1024**3:.2f}GB"
        )
        
        if gpu_growth > 1024**3 * 5:  # 5GB
            self.logger.warning(f"The GPU memory has grown too much: {gpu_growth/1024**3:.2f}GB")
        if ram_growth > 1024**3 * 10:  # 10GB
            self.logger.warning(f"RAM growth too large: {ram_growth/1024**3:.2f}GB")
    
    def clean(self):
        torch.cuda.empty_cache()
        gc.collect()
        self.logger.info("Memory cleaned")