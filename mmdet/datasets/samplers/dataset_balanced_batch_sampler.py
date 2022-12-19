import math
import torch
from torch.utils.data import Sampler, ConcatDataset
from typing import Sequence as Seq

from mmcv.runner import get_dist_info
from mmdet.core.utils import sync_random_seed


class DatasetBalancedBatchSampler(Sampler):
    """
    DatasetBalancedBatchSampler is designed to get sample indices from multiple datasets for each batch.

    For multi-datasets joint training, balancing ratio of samples from different datasets in equivalent to 
    oversampling small datasets. 

    Only support DistributedDataParallel.

    Args:
        dataset: ConcatDataset used for sampling.
        samples_per_dataset (Iterable[int]): The number of samples of different datasets used to form the batch.
            The order is consistent with `dataset.datasets`, so must satisfy len(samples_per_dataset) == len(dataset.datasets).
            Thus, `samples_per_gpu = sum(samples_per_dataset)`.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self, 
                 dataset: ConcatDataset, 
                 samples_per_dataset: Seq[int], 
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True
                 ):
        assert isinstance(dataset, ConcatDataset), "Only support ConcatDataset"
        assert len(dataset.datasets) == len(samples_per_dataset)
        for i, num in enumerate(samples_per_dataset):
            assert num > 0, f"num of samples must be > 0 for dataset{i}, but got {num}"
        
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank

        self.dataset = dataset
        self.datasets = dataset.datasets
        self.samples_per_dataset = samples_per_dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        # Must be the same across all workers. If None, will use a
        # random seed shared among workers
        # (require synchronization among all workers)
        self.seed = sync_random_seed(seed)
        # Each dataset has its own seed
        self._set_random_seed_per_dataset()


        # calculate the # of samples of each dataset that should be loaded on this rank
        # map from dataset idx to # of samples per gpu / total samples / num batches
        self.num_samples_dict = dict()
        self.total_size_dict = dict()
        self.num_batches_dict = dict()
        for i, (ds, sp_per_gpu) in enumerate(zip(self.datasets, self.samples_per_dataset)):
            num_samples = int(math.ceil(
                    len(ds) * 1.0 / self.num_replicas /
                    sp_per_gpu)) * sp_per_gpu
            self.num_samples_dict[i] = num_samples
            self.total_size_dict[i] = int(num_samples * self.num_replicas)
            self.num_batches_dict[i] = int(num_samples / sp_per_gpu)
        
        self.sp_idx_queue = {i : [] for i in range(len(self.datasets))}


    def _set_random_seed_per_dataset(self):
        self.seed_per_ds = {i : self.seed for i in range(len(self.datasets))}

    def _fill_idx_queue(self):
        for ds_idx in self.sp_idx_queue:
            if len(self.sp_idx_queue[ds_idx]) < self.samples_per_dataset[ds_idx]:
                if self.shuffle:
                    g = torch.Generator()
                    g.manual_seed(self.seed_per_ds[ds_idx])
                    indices = torch.randperm(len(self.datasets[ds_idx]), generator=g)
                    
                    # increment random seed for next shuffle if hitting the end of a dataset
                    self.seed_per_ds[ds_idx] += 1
                
                else:
                    indices = torch.arange(len(self.datasets[ds_idx]))
                
                if ds_idx != 0:
                    indices += self.dataset.cummulative_sizes[ds_idx - 1]
                indices = indices.tolist()

                extra = self.total_size_dict[ds_idx] - len(indices)
                tmp = indices.copy()
                for _ in range(extra // len(indices)):
                    indices.extend(tmp)
                indices.extend(tmp[:extra % len(indices)])

                start = self.rank * self.num_samples_dict[ds_idx]
                end = (self.rank + 1) * self.num_samples_dict[ds_idx]
                self.sp_idx_queue[ds_idx].extend(indices[start : end])
                assert len(self.sp_idx_queue[ds_idx]) == self.num_samples_dict[ds_idx]
    
    def __iter__(self):
        self.num_batch_returned = 0
        while self.num_batch_returned < len(self):
            # make sure the # of available idx in idx queue is at least self.samples_per_dataset[ds_idx] for 
            # dataset ds_idx
            self._fill_idx_queue()

            batch_buffer = []
            for ds_idx, sp_per_gpu in enumerate(self.samples_per_dataset):
                for _ in range(sp_per_gpu):
                    batch_buffer.append(self.sp_idx_queue[ds_idx].pop(0))
            yield batch_buffer
            self.num_batch_returned += 1

    def __len__(self):
        return max(list(self.num_batches_dict.values()))
    
    @property
    def sampler(self):
        return self
    


        




