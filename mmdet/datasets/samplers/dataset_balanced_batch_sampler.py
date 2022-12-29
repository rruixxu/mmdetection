import enum
import math
import torch
from torch.utils.data import Sampler, ConcatDataset
from typing import Sequence as Seq
from typing import Mapping
from collections import defaultdict

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
    


        
class DatasetRankBalancedBatchSampler(Sampler):
    """
    DatasetRankBalancedBatchSampler is designed to load data from distinct datasets for different rank.
    For example, rank0 may load data from dataset0, while rank1 can load from dataset1 and dataset2. If
    mutliple ranks load from the same datasets and same sample ratio, they form a loading group that should 
    partition datasets evenly.

    Same as DatasetBatchBalancedBatchSampler, the class is to balance data ratio for joint training. However, 
    synchronized BN should be used to avoid apparent shifts of BN statistics among ranks.

    Only support DistributedDataParallel.

    Args:
        dataset: ConcatDataset used for sampling.
        load_group_info (Mapping[int, Mapping[str, Seq[int]]]): Specify infomation of loading group. Keys are
            loading group ids starting from 0. Each group infomation contains `ranks`, `ds_ids` and `sp_per_ds`.
            For example, {0: {'ranks': [0,1], 'ds_ids': [0, 2], 'sp_per_ds': [1, 3]}} means loading_group0 is
            working on rank0 and rank1 and loads dataset0 with 1 sample and dataset2 with 3 smples.
        num_replicas (optional): Number of processes participating in
            distributed training.
        rank (optional): Rank of the current process within num_replicas.
        seed (int, optional): random seed used to shuffle the sampler if
            ``shuffle=True``. This number should be identical across all
            processes in the distributed group. Default: 0.
    """

    def __init__(self, 
                 dataset: ConcatDataset, 
                 load_group_info: Mapping[int, Mapping[str, Seq[int]]], # ranks, ds_ids, sp_per_ds
                 num_replicas=None,
                 rank=None,
                 seed=0,
                 shuffle=True
                 ):
        assert isinstance(dataset, ConcatDataset), "Only support ConcatDataset"
        
        # some checks
        all_ds_id = set()
        all_ranks = set()
        self.ds_id_to_num_sp = dict()
        self.rank_to_load_group = dict()
        for g_id, g_info in load_group_info.items():
            assert len(g_info["ds_ids"]) == len(g_info["sp_per_ds"])
            # make sure do not specify repeated datasets in one group
            assert len(set(g_info["ds_ids"])) == len(g_info["ds_ids"]), "duplicated datasets in one loading group"
            for _id, num_sp in zip(g_info["ds_ids"], g_info['sp_per_ds']):
                # make sure one dataset only loaded in one loading group
                assert _id not in all_ds_id, f"dataset{_id} will be loaded in multiple loading groups"
                all_ds_id.add(_id)
                assert num_sp > 0, f"dataset{_id} will be loaded {num_sp} samples, which is less or equal to 0"
                self.ds_id_to_num_sp[_id] = num_sp
            
            assert len(set(g_info["ranks"])) == len(g_info["ranks"]), "duplicated ranks in one loading group"
            for r in g_info['ranks']:
                assert r not in all_ranks, f"rank{r} is specified in multiple loading groups"
                all_ranks.add(r)
                self.rank_to_load_group[r] = g_id

        not_load_ds = []
        for i in range(len(dataset.datasets)):
            if i not in all_ds_id:
                not_load_ds.append(i)
        if len(not_load_ds) > 0:
            raise ValueError("{} datasets will not be loaded".format(" ".join(not_load_ds)))
        
        
        _rank, _num_replicas = get_dist_info()
        if num_replicas is None:
            num_replicas = _num_replicas
        if rank is None:
            rank = _rank
        
        not_work_rank = []
        for i in range(len(num_replicas)):
            if i not in all_ranks:
                not_work_rank.append(i)
        if len(not_work_rank) > 0:
            raise ValueError("{} ranks are not specified in load_group_info".format(" ".join(not_work_rank)))

        self.dataset = dataset
        self.datasets = dataset.datasets
        self.load_group_info = load_group_info
        self.num_replicas = num_replicas
        self.rank = rank
        self.shuffle = shuffle
        # Must be the same across all workers. If None, will use a
        # random seed shared among workers
        # (require synchronization among all workers)
        self.seed = sync_random_seed(seed)

        self.load_g_id_of_the_rank = self.rank_to_load_group[self.rank]
        self.load_g_info_of_the_rank = self.load_group_info[self.load_g_id_of_the_rank]
        for local_id, r in enumerate(self.load_g_info_of_the_rank['ranks']):
            if r == self.rank:
                self.local_rank_id = local_id
                break

        # calculate the # of samples of each dataset that should be loaded on this rank
        # map from dataset idx to # of samples per gpu / total samples / num batches
        self.num_samples_dict = dict()
        self.total_size_dict = dict()
        self.num_batches_dict = dict()
        for g_id, g_info in self.load_group_info.items():
            if g_id != self.load_g_id_of_the_rank:
                continue
            for i, _id in enumerate(g_info["ds_ids"]):
                ds = self.datasets[_id]
                sp_per_gpu = g_info["sp_per_ds"][i]
                num_rep = len(g_info['ranks'])
                num_samples = int(math.ceil(
                        len(ds) * 1.0 / num_rep /
                        sp_per_gpu)) * sp_per_gpu
                self.num_samples_dict[_id] = num_samples
                self.total_size_dict[_id] = int(num_samples * num_rep)
                self.num_batches_dict[_id] = int(num_samples / sp_per_gpu)
        
        self.sp_idx_queue = {i : [] for i in self.load_g_info_of_the_rank['ds_ids']}
        # Each dataset has its own seed
        self._set_random_seed_per_dataset()


    def _set_random_seed_per_dataset(self):
        self.seed_per_ds = {i : self.seed for i in self.load_g_info_of_the_rank['ds_ids']}

    def _fill_idx_queue(self):
        for ds_idx in self.sp_idx_queue:
            if len(self.sp_idx_queue[ds_idx]) < self.ds_id_to_num_sp[ds_idx]:
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

                start = self.local_rank_id * self.num_samples_dict[ds_idx]
                end = (self.local_rank_id + 1) * self.num_samples_dict[ds_idx]
                self.sp_idx_queue[ds_idx].extend(indices[start : end])
                assert len(self.sp_idx_queue[ds_idx]) == self.num_samples_dict[ds_idx]
    
    def __iter__(self):
        self.num_batch_returned = 0
        while self.num_batch_returned < len(self):
            # make sure the # of available idx in idx queue is at least self.samples_per_dataset[ds_idx] for 
            # dataset ds_idx
            self._fill_idx_queue()

            batch_buffer = []
            for ds_idx, sp_per_gpu in zip(self.load_g_info_of_the_rank['ds_ids'], self.load_g_id_of_the_rank['sp_per_ds']):
                for _ in range(sp_per_gpu):
                    batch_buffer.append(self.sp_idx_queue[ds_idx].pop(0))
            yield batch_buffer
            self.num_batch_returned += 1

    def __len__(self):
        return max(list(self.num_batches_dict.values()))
    
    @property
    def sampler(self):
        return self



