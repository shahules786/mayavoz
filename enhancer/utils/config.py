from dataclasses import dataclass


@dataclass
class Files:
    train_clean: str
    train_noisy: str
    test_clean: str
    test_noisy: str
