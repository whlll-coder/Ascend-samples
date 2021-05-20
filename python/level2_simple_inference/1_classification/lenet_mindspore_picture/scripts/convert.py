from mindspore.train.serialization import load_checkpoint, save_checkpoint, export
from src.lenet import LeNet5
import numpy as np
from mindspore import Tensor
network = LeNet5()
load_checkpoint("./checkpoint_lenet-1_1875.ckpt", network)
input = np.random.uniform(0.0, 1.0, size=[1, 1, 32, 32]).astype(np.float32)
export(network,Tensor(input),  file_name='./mnist', file_format='AIR') 