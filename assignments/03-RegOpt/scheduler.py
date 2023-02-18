from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Decay LR over steps, resetting when it hits some floor"""

    def __init__(
        self,
        optimizer: Optimizer,
        gamma_batch: float = 0.8,
        gamma_max: float = 0.8,
        reset_floor: float = 1e-6,
        last_epoch: int = -1,
    ):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.gamma_batch = gamma_batch
        self.gamma_max = gamma_max
        self.reset_floor = reset_floor
        self.max_lrs = [group["lr"] for group in self.optimizer.param_groups]

    def get_lr(self) -> List[float]:
        """Multiply LR by gamma each step, resetting upon hitting some floor"""
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        outs = []
        new_max = []
        for group, cycle_max in zip(self.optimizer.param_groups, self.max_lrs):
            if group["lr"] <= self.reset_floor:
                outs.append(cycle_max)
                new_max.append(cycle_max * self.gamma_max)
            else:
                outs.append(group["lr"] * self.gamma_batch)
                new_max.append(cycle_max)

        self.max_lrs = new_max
        return outs
