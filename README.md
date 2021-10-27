### RPC-based Parameter Server Framework
An implementation of [parameter server](https://www.usenix.org/system/files/conference/osdi14/osdi14-paper-li_mu.pdf) (PS) framework in [PyTorch](https://pytorch.org/). The implementation is based on [torch.distributed.rpc](https://pytorch.org/docs/1.9.0/rpc.html).
***
### PS-based Architecture
The figure below shows the PS-based architecture. The consists of two logical entities: one (or multiple) PS(s) and multiple workers. The whole dataset is partitioned among workers and the PS maintains model parameters. During training, each worker pulls model parameters from the PS, computes gradients on a mini-batch from its data partition, and pushes the gradients to the PS. The PS updates model parameters with gradients from the workers according to a synchronization strategy and sends the updated parameters back to the workers.

<div align=center><img width="80%" src="./architecture.jpg"/></div>
