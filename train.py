import os
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
import torch
import numpy as np
import random
from model_noz0 import AdaptModel
import torch.multiprocessing as mp
from torch.nn.parallel import DistributedDataParallel as DDP
import dataset
import setproctitle
setproctitle.setproctitle("Fusion")

data_path = 'Fusion/train'

epochs = 5000

def main(rank, world_size):
    setup(rank, world_size)

    seed = 61 + rank
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)

    DATASET = dataset.DATASET(data_path, batchsize=1)
    print("=> dataset length:{}".format(len(DATASET.train_dataset)))
    train_sampler, train_loader = DATASET.get_loaders()

    print("=> creating fusion model...")
    model = AdaptModel(rank, num_class=9)
    model.ae.encoder = model.ae.encoder.to(rank)
    model.ae.decoder = model.ae.decoder.to(rank)
    model.IKRModule = model.IKRModule.to(rank)
    model.CKGModule = model.CKGModule.to(rank)
    model.FusionModule = DDP(model.FusionModule, device_ids=[rank])
    model.train(epochs, train_loader, train_sampler) # load_pretrained={'path':'checkpoints', 'epoch': 3}

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = '12353' 
    torch.distributed.init_process_group(
        backend="nccl",
        rank=rank,
        world_size=world_size
    )
    torch.cuda.set_device(rank)


if __name__ == "__main__":
    world_size = torch.cuda.device_count()
    for i in range(world_size):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
    mp.spawn(main, args=(world_size, ), nprocs=world_size, join=True)
   