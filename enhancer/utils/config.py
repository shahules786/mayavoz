from dataclasses import dataclass

@dataclass
class Paths:
    log : str
    data : str

@dataclass
class Files:
    train_clean : str
    train_noisy : str
    test_clean : str
    test_noisy : str

@dataclass
class EnhancerConfig:
    path : Paths
    files: Files