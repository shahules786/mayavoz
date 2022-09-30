from dataclasses import dataclass

@dataclass
class Files:
    root_dir : str
    train_clean : str
    train_noisy : str
    test_clean : str
    test_noisy : str


