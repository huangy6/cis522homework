from typing import List

from torch.optim import Optimizer
from torch.optim.lr_scheduler import _LRScheduler


class CustomLRScheduler(_LRScheduler):
    """Gamma exponential LR scheduler"""

    def __init__(self, optimizer: Optimizer, gamma: float = 0.8, last_epoch: int = -1):
        """
        Create a new scheduler.

        Note to students: You can change the arguments to this constructor,
        if you need to add new parameters.

        """
        super(CustomLRScheduler, self).__init__(optimizer, last_epoch)
        self.gamma = gamma

    def get_lr(self) -> List[float]:
        """Multiply LR by gamma each epoch"""
        # Note to students: You CANNOT change the arguments or return type of
        # this function (because it is called internally by Torch)

        # ... Your Code Here ...
        if self.last_epoch == 0:
            return [group["lr"] for group in self.optimizer.param_groups]
        return [group["lr"] * self.gamma for group in self.optimizer.param_groups]
