import os
import random
import torch



def create_unique_rng(epoch:int):
    """create unique random number generator for each (worker_id,epoch) combination"""

    rng = random.Random()

    global_seed = int(os.environ.get("PL_GLOBAL_SEED","0"))
    global_rank = int(os.environ.get('GLOBAL_RANK',"0"))
    local_rank = int(os.environ.get('LOCAL_RANK',"0"))
    node_rank = int(os.environ.get('NODE_RANK',"0"))
    world_size = int(os.environ.get('WORLD_SIZE',"0"))

    worker_info = torch.utils.data.get_worker_info()
    if worker_info is not None:
        num_workers = worker_info.num_workers
        worker_id = worker_info.worker_id
    else:
        num_workers = 1
        worker_id = 0

    seed = (
            global_seed
            + worker_id
            + local_rank * num_workers
            + node_rank * num_workers * global_rank
            + epoch * num_workers * world_size
        )

    rng.seed(seed)

    return rng




