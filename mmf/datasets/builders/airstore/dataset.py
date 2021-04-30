# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.

import itertools
from typing import Any, Callable, Iterator, Optional
import importlib.resources as pkg_resources

import torch
from airstore.client.airstore_tabular import AIRStorePathHandler
from iopath.common.file_io import PathManager
# from iopath.fb.manifold import ManifoldPathHandler

# from torch.utils.data import DataLoader
from torch.utils.data.dataset import IterableDataset


def _get_batch_iterator(
    row_iterator: Iterator,
    batch_size: int,
    transform: Optional[Callable[[dict], Any]] = None,
    drop_last_batch: bool = True,
) -> Iterator:
    """
    This is a wrapper that converts a row iterator to a batch iterator.

    Args:
        row_iterator (Iterator): The row iterator, which returns one instance at a time.
        batch_size (str): Expected batch size.
        transform (Callable): Transform that is applied to the entire batch.
        drop_last_batch (bool): Whether to drop the last batch.
    """
    while True:
        batch = list(itertools.islice(row_iterator, batch_size))
        if len(batch) == 0 or (drop_last_batch and len(batch) < batch_size):
            break
        if transform is None:
            yield batch
        else:
            yield transform(batch)


class AirStoreDataset(IterableDataset):
    """
    Video dataset for AirStore. Call the iterator() to get the data loader of the
    dataset. Airstore dataset reads all the columns so make sure to add all the
    required columns in the airstore ingestion process. The data loader assumes
    distributed training is available.
    """
    TEST_DS_CATALOG_PATH = "airstore/data/airstore_test_dataset_catalog.csv"

    def __init__(
        self,
        airstore_uri: str,
        batchsize_per_replica: int = 4,
        shuffle: bool = False,
        transform: Callable[[dict], Any] = None,
        keep_last_batch: bool = False,
        *,
        dataloader_num_workers: int = 4,
        airstore_num_threads: int = 10,
        airstore_prefetch_factor: int = 1,
        airstore_max_holding_bundles: int = 4,
        airstore_bundle_download_parallel: int = 10,
    ):
        """
        Args:
            airstore_uri (str): The url in for airstore. For example, FCC training data
                is "airstore://aml_fcc_v3p1_video_train".
            batchsize_per_replica (int): Batch size per replica.
            shuffle (bool): Whether to shuffle dataset.
            transform (Callable): The transform that is applied to the video.
            keep_last_batch (bool): Whether to keep the last batch.
            dataloader_num_workers (int): Number of workers in the dataloader.
            airstore_num_threads (int): IO threads for  airstore c++ client.
            airstore_prefetch_factor (int): How many bundles to prefetch.
            airstore_max_holding_bundles (int): How many bundles to keep in the memory,
                bigger value helps random access performance but will increase memory.
            airstore_bundle_download_parallel (int): Parallel bundle download size.
        """
        self.pathmgr = PathManager()
        self.pathmgr.register_handler(AIRStorePathHandler())
        # self.pathmgr.register_handler(ManifoldPathHandler())

        self.airstore_uri = airstore_uri
        self.batchsize_per_replica = batchsize_per_replica
        self.shuffle = shuffle
        self.transform = transform
        self.num_workers = dataloader_num_workers
        self.num_threads = airstore_num_threads
        self.prefetch_factor = airstore_prefetch_factor
        self.max_holding_bundles = airstore_max_holding_bundles
        self.bundle_download_parallel = airstore_bundle_download_parallel
        self.keep_last_batch = keep_last_batch
        self.worker_info = None
        self.shuffle_seed = 123456
        self.epoch = 1
        # Set in iterator method.
        # self.rank = torch.distributed.get_rank()
        # self.world_size = torch.distributed.get_world_size()

    def get_default_dataset_catalog_path(self):
        dspath = self.TEST_DS_CATALOG_PATH
        try:
            with pkg_resources.path(
                "airstore.data", "airstore_test_dataset_catalog.csv"
            ) as dsc_path:
                dspath = str(dsc_path)
        except Exception:
            dspath = self.TEST_DS_CATALOG_PATH

        return dspath

    def __iter__(self) -> Iterator:
        """
        Getting the batch iterator for the dataset.
        """
        assert self.shuffle_seed is not None
        assert self.epoch is not None
        # assert self.worker_info is not None
        # assert self.rank is not None
        # assert self.world_size is not None

        # num_workers = self.worker_info.num_workers
        # worker_id = self.worker_info.id
        num_workers = 0
        worker_id = 0

        # Data iterator from airstore for current data split.
        # Data are sharded by global total number of workers after shuffling.
        iterator = self.pathmgr.opent(
            self.airstore_uri,
            offset=0,
            limit=128,
            binary_to_tensor=True,
            env="OSS",
            dataset_catalog_path=self.get_default_dataset_catalog_path(),
        )

        # Get batch iterator.
        batch_iterator = _get_batch_iterator(
            iterator,
            self.batchsize_per_replica,
            transform=self.transform,
            drop_last_batch=not self.keep_last_batch,
        )

        return batch_iterator

    # def _worker_init_fn(self, worker_id) -> None:
    #     """
    #     Args:
    #         worker_id (int): Current worker_id. This is used to partition data.
    #     """
    #     worker_info = torch.utils.data.get_worker_info()
    #     dataset = worker_info.dataset
    #     dataset.worker_info = worker_info

    # def iterator(self, *args, **kwargs) -> DataLoader:
    #     """
    #     Get iterator for data reading. Return DataLoader.
    #     """

    #     # We need to set rank and world_size here instead of __init__.
    #     # Dataset creation is before distributed context is initialized.

    #     self.rank = torch.distributed.get_rank()
    #     self.world_size = torch.distributed.get_world_size()
    #     return self._get_multiprocess_loader(*args, **kwargs)

    # def _get_multiprocess_loader(self, *args, **kwargs) -> DataLoader:
    #     """
    #     Args:
    #         kwargs: kwargs can optionally take "pin_memory", "multiprocessing_context",
    #             "num_workers" for dataloader.
    #     """
    #     self.shuffle_seed = kwargs.get("shuffle_seed", 0)
    #     assert isinstance(self.shuffle_seed, int), "Shuffle seed must be an int"
    #     self.epoch = kwargs.get("current_phase_id", 0)
    #     assert isinstance(self.epoch, int), "Epoch must be an int"
    #     num_workers_override = kwargs.get("num_workers", self.num_workers)
    #     if num_workers_override == 0:
    #         kwargs["multiprocessing_context"] = None

    #     pin_memory = kwargs.get("pin_memory", False)
    #     multiprocessing_context = kwargs.get("multiprocessing_context", None)

    #     dl = torch.utils.data.DataLoader(
    #         self,
    #         num_workers=kwargs.get("num_workers", self.num_workers),
    #         batch_size=None,
    #         worker_init_fn=self._worker_init_fn,
    #         pin_memory=pin_memory,
    #         multiprocessing_context=multiprocessing_context,
    #     )
    #     return dl
