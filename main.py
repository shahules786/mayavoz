import hydra
import torch
from hydra.core.config_store import ConfigStore

from enhancer.utils.config import EnhancerConfig

cs = ConfigStore.instance()
cs.store(name="enhancer_config", node=EnhancerConfig)


@hydra.main(config_path=".",config_name="conf")
def main(cfg: EnhancerConfig):

    print(cfg.paths.data)



if __name__=="__main__":
    main()