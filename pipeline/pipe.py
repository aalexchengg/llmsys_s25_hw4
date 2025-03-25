from typing import Any, Iterable, Iterator, List, Optional, Union, Sequence, Tuple, cast

import torch
from torch import Tensor, nn
import torch.autograd
import torch.cuda
from .worker import Task, create_workers
from .partition import _split_module

# ASSIGNMENT 4.2
def _clock_cycles(num_batches: int, num_partitions: int) -> Iterable[List[Tuple[int, int]]]:
    '''Generate schedules for each clock cycle.

    An example of the generated schedule for m=3 and n=3 is as follows:
    
    k (i,j) (i,j) (i,j)
    - ----- ----- -----
    0 (0,0)
    1 (1,0) (0,1)
    2 (2,0) (1,1) (0,2)
    3       (2,1) (1,2)
    4             (2,2)

    where k is the clock number, i is the index of micro-batch, and j is the index of partition.

    Each schedule is a list of tuples. Each tuple contains the index of micro-batch and the index of partition.
    This function should yield schedules for each clock cycle.
    '''
    # BEGIN SOLUTION
    k = 0
    result = []
    state = dict()
    # initialize empty state dict
    for i in range(num_partitions):
        state[i] = 0
    while(state[num_partitions - 1] < num_batches):
        # generate empty step
        curr = []
        # for each partition
        for partition, idx in state.items():
            # if we can increment
            if partition <= k and idx < num_batches:
                curr.append((idx, partition))
                state[partition] += 1
        # update
        result.append(curr)
        k += 1
    return result
    # END SOLUTION

class Pipe(nn.Module):
    def __init__(
        self,
        module: nn.ModuleList,
        split_size: int = 1,
    ) -> None:
        super().__init__()

        self.split_size = int(split_size)
        self.partitions, self.devices = _split_module(module)
        self.verbose = False
        if self.verbose:
            print("in init")
            print(self.partitions)
            print(self.devices)
        (self.in_queues, self.out_queues) = create_workers(self.devices)

    # ASSIGNMENT 4.2
    def forward(self, x):
        ''' Forward the input x through the pipeline. The return value should be put in the last device.

        Hint:
        1. Divide the input mini-batch into micro-batches.
        2. Generate the clock schedule.
        3. Call self.compute to compute the micro-batches in parallel.
        4. Concatenate the micro-batches to form the mini-batch and return it.
        
        Please note that you should put the result on the last device. Putting the result on the same device as input x will lead to pipeline parallel training failing.
        '''
        # BEGIN SOLUTION
        batches = list(torch.split(tensor = x, split_size_or_sections = self.split_size))
        cycle = _clock_cycles(len(batches), len(self.partitions))
        if self.verbose:
            print("before running the cycle")
            print(batches)
        for schedule in cycle:
            self.compute(batches, schedule)
            if self.verbose:
                print("what batches looks like now")
                print(batches)
        if self.verbose:
            print("finished running cycle")
        result = [batch.to(self.devices[-1]) for batch in batches]
        return torch.cat(result)
        # END SOLUTION

    # ASSIGNMENT 4.2
    def compute(self, batches, schedule: List[Tuple[int, int]]) -> None:
        if self.verbose:
            print("="*20)
            print("schedule for this step is")
            print(schedule)
        '''Compute the micro-batches in parallel.

        Hint:
        1. Retrieve the partition and microbatch from the schedule.
        2. Use Task to send the computation to a worker. 
        3. Use the in_queues and out_queues to send and receive tasks.
        4. Store the result back to the batches.
        '''
        partitions = self.partitions
        devices = self.devices

        # BEGIN SOLUTION
        trace_idx = 0
        tasks = dict()
        for microbatch_idx, partition_idx in schedule:
            if self.verbose:
                print(f"inputting microbatch {microbatch_idx} into partition {partition_idx}")
                print(f"before moving to {devices[partition_idx]}: {batches[microbatch_idx]}")
            batches[microbatch_idx] = batches[microbatch_idx].to(devices[partition_idx])
            if self.verbose:
                print(f"after: {batches[microbatch_idx]}")
            microbatch = batches[microbatch_idx]
            partition = partitions[partition_idx]
            task = Task(lambda: partition(microbatch), partition_idx = partition_idx, microbatch_idx = microbatch_idx)
            if self.verbose:
                print("*"*20)
                print(f"observing microbatch {microbatch_idx}")
                print(f"current batch shape is {microbatch.shape}")
                print(f"batch is {microbatch}")
                print(f"about to be put into {partition} on device {devices[partition_idx]}")
                print(f"task is {task}")
                print("*"*20)
            self.in_queues[partition_idx].put(task)
            tasks[partition] = task
        for microbatch_idx, partition_idx in schedule:
            if self.verbose:
                print(f"receiving microbatch {microbatch_idx} from partition {partition_idx}")
                print("getting result from out queue")
            status, result = self.out_queues[partition_idx].get()
            if self.verbose:
                print(f"status {status} result {result}")
            if status:
                temp = result[1].to(batches[microbatch_idx].device)
                batches[microbatch_idx] = temp
                if self.verbose:
                    print("*"*20)
                    print(f"observing microbatch {microbatch_idx}")
                    print(f"result is {result[1]}")
                    print(f"result shape is {result[1]}.shape")
                    print("*"*20)
            elif result[0] != tasks[partition]:
                raise AssertionError("Task is wrong ...")
            else:
                raise AssertionError(f"Something went wrong... status: {status} result: {result} idx: {microbatch_idx} partition: {partition_idx}")
        if self.verbose:
            print("step complete")
            print("="*20)
        # END SOLUTION
