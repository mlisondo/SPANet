from glob import glob
from typing import Optional, Union, Tuple

import numpy as np
import torch
from torch.utils._pytree import tree_map

from rich import progress

import time

from spanet.options import Options
from spanet.dataset.types import Source, AssignmentTargets
from spanet.network.jet_reconstruction.jet_scm_eval_test import SCM_Eval_Test

from collections import defaultdict

class Timer:
    def __init__(self):
        self.start_time = 0
        self.total_time = 0
        self.running = False

    def start(self):
        if not self.running:
            self.start_time = time.time()
            self.running = True

    def stop(self):
        if self.running:
            self.total_time += time.time() - self.start_time
            self.running = False

    def reset(self):
        self.start_time = 0
        self.total_time = 0
        self.running = False

    def get_time(self):
        if self.running:
            return self.total_time + time.time() - self.start_time
        else:
            return self.total_time
        



def load_model(
    log_directory: str,
    testing_file: Optional[str] = None,
    event_info_file: Optional[str] = None,
    batch_size: Optional[int] = None,
    cuda: bool = False,
    checkpoint: Optional[str] = None
) -> SCM_Eval_Test:                                                                       # CHANGED
    # Load the best-performing checkpoint on validation data
    if checkpoint is None:
        checkpoint = sorted(glob(f"{log_directory}/checkpoints/epoch*"))[-1]
        print(f"Loading: {checkpoint}")

    checkpoint = torch.load(checkpoint, map_location='cpu')
    checkpoint = checkpoint["state_dict"]

    # Load the options that were used for this run and set the testing-dataset value
    options = Options.load(f"{log_directory}/options.json")

    # Override options from command line arguments
    if testing_file is not None:
        options.testing_file = testing_file

    if event_info_file is not None:
        options.event_info_file = event_info_file

    if batch_size is not None:
        options.batch_size = batch_size

    # Create model and disable all training operations for speed
    model = SCM_Eval_Test(options)
    model.load_state_dict(checkpoint)
    model = model.eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)

    if cuda:
        model = model.cuda()

    return model


def evaluate_on_test_dataset(
        model: SCM_Eval_Test,
        progress=progress,
        return_full_output: bool = False
) -> Union[dict, Tuple[dict, Tuple[dict, ...]]]:
    # Loops over model.test_dataloader(), calls model.evaluate_scm_batch(batch)     (which runs extract_predictions internally)

    bank = defaultdict(list)
    full_outputs = []

    dataloader = model.test_dataloader()
    if progress:
        dataloader = progress.track(dataloader, description="Evaluating SCM model")

    timer = Timer()
    for batch in dataloader:
        # sources = tuple(Source(x[0].to(model.device), x[1].to(model.device)) for x in batch.sources)
        probe(batch, "batch")
        batch = move_batch_to_device(batch, model.device)
        probe(batch, "batch")
        timer.start()
        outputs = model.evaluate_scm_batch(batch)
        timer.stop()

        for k, v in outputs.items():
            bank[k].append(v.cpu().numpy())

        if return_full_output:
            full_outputs.append(tree_map(lambda x: x.cpu().numpy(), outputs))

    print(f"Total combinatorics time: {timer.get_time():.2f} s")

    arrays = {k: np.concatenate(v, axis=0) for k, v in bank.items()}

    if return_full_output:
        return arrays, tuple(full_outputs)
    return arrays


def move_batch_to_device(batch, device):
    sources = [Source(x[0].to(device), x[1].to(device)) for x in batch[0]]
    integers = batch[1].to(device)
    assignments = []
    for item in batch[2]:
        idx = item[0].to(device)
        mask = item[1].to(device)
        assignments.append((idx, mask))
    return (sources, integers, [AssignmentTargets(*a) for a in assignments], batch[3], batch[4])




def probe(o, name=None):
    obj = type(o)
    header = f"Object '{name}'"
    print(f"\n{header}: {obj.__module__}.{obj.__name__}")

    # NumPy-style introspection
    if hasattr(o, 'shape'):
        print(f"shape: {o.shape}")
    if hasattr(o, 'ndim'):
        print(f"ndim: {o.ndim}")
    if hasattr(o, 'dtype'):
        print(f"dtype: {o.dtype}")

    # size attribute
    if hasattr(o, 'size') and not callable(o.size):
        print(f"size: {o.size}")

    # Pythonic length
    try:
        print(f"len: {len(o)}")
    except Exception:
        pass

    # Recursive descent into lists
    try:
        if isinstance(o, (list, tuple)):
            for idx, item in enumerate(o):
                probe(item, f"{name}[{idx}]")
    except Exception:
        pass

    # PyTorch tensors
    if isinstance(o, torch.Tensor):
        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")

        print(f"shape: {tuple(o.size())}")
        print(f"dtype: {o.dtype}")
        print(f"numel: {o.numel()}")
        print(f"device: {o.device}")