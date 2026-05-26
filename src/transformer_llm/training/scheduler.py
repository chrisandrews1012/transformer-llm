"""
Learning rate schedulers: warmup-cosine, linear, constant, and cosine annealing.
"""

import math
from typing import Optional

import torch.optim as optim


class _SchedulerMixin:
    """
    Small helper so trainer checkpoints can save and restore scheduler state.
    """

    def state_dict(self) -> dict:
        """
        Return scheduler state as a dictionary.

        :returns: Dictionary containing current_step.
        :rtype: dict
        """
        return {"current_step": self.current_step}

    def load_state_dict(self, state_dict: dict) -> None:
        """
        Restore scheduler state from a dictionary.

        :param state_dict: Dictionary produced by state_dict().
        :type state_dict: dict
        """
        self.current_step = state_dict.get("current_step", 0)


class WarmupScheduler(_SchedulerMixin):
    """
    Linear warmup learning rate scheduler.

    Linearly increases the learning rate from 0 to base_lr over warmup_steps,
    then holds it constant.
    Formula: lr = base_lr * (step / warmup_steps) for step < warmup_steps,
    lr = base_lr for step >= warmup_steps.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        base_lr: Optional[float] = None,
    ):
        """
        Initialize warmup scheduler.

        :param optimizer: PyTorch optimizer to update.
        :type optimizer: optim.Optimizer
        :param warmup_steps: Number of warmup steps.
        :type warmup_steps: int
        :param base_lr: Target learning rate after warmup (defaults to the
            optimizer's current lr).
        :type base_lr: Optional[float]
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> None:
        """
        Advance one step and update the optimizer's learning rate.
        """
        self.current_step += 1

        if self.warmup_steps <= 0:
            lr = self.base_lr
        elif self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            lr = self.base_lr

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """
        Return the current learning rate.

        :returns: Current learning rate.
        :rtype: float
        """
        return self.optimizer.param_groups[0]["lr"]


class CosineAnnealingScheduler(_SchedulerMixin):
    """
    Cosine annealing learning rate scheduler.

    Decreases the learning rate following a cosine curve from base_lr to min_lr.
    Formula: lr = min_lr + 0.5 * (base_lr - min_lr) * (1 + cos(pi * step / total_steps)).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        """
        Initialize cosine annealing scheduler.

        :param optimizer: PyTorch optimizer to update.
        :type optimizer: optim.Optimizer
        :param total_steps: Total number of training steps.
        :type total_steps: int
        :param min_lr: Minimum learning rate at the end of annealing.
        :type min_lr: float
        :param base_lr: Starting learning rate (defaults to optimizer's current lr).
        :type base_lr: Optional[float]
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> None:
        """
        Advance one step and update the optimizer's learning rate.
        """
        self.current_step += 1

        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
            1 + math.cos(math.pi * progress)
        )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """
        Return the current learning rate.

        :returns: Current learning rate.
        :rtype: float
        """
        return self.optimizer.param_groups[0]["lr"]


class WarmupCosineScheduler(_SchedulerMixin):
    """
    Combined linear warmup followed by cosine annealing scheduler.

    The most commonly used scheduler in modern transformer training.
    Phase 1: linear warmup for the first warmup_steps.
    Phase 2: cosine annealing for the remaining steps.
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        warmup_steps: int,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        """
        Initialize warmup-cosine scheduler.

        :param optimizer: PyTorch optimizer to update.
        :type optimizer: optim.Optimizer
        :param warmup_steps: Number of linear warmup steps.
        :type warmup_steps: int
        :param total_steps: Total number of training steps.
        :type total_steps: int
        :param min_lr: Minimum learning rate at end of cosine decay.
        :type min_lr: float
        :param base_lr: Peak learning rate (defaults to optimizer's current lr).
        :type base_lr: Optional[float]
        """
        self.optimizer = optimizer
        self.warmup_steps = warmup_steps
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> None:
        """
        Advance one step and update the optimizer's learning rate.
        """
        self.current_step += 1

        if self.warmup_steps > 0 and self.current_step < self.warmup_steps:
            lr = self.base_lr * (self.current_step / self.warmup_steps)
        else:
            decay_steps = max(self.total_steps - self.warmup_steps, 1)
            progress = min(
                max(self.current_step - self.warmup_steps, 0) / decay_steps,
                1.0,
            )
            lr = self.min_lr + 0.5 * (self.base_lr - self.min_lr) * (
                1 + math.cos(math.pi * progress)
            )

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """
        Return the current learning rate.

        :returns: Current learning rate.
        :rtype: float
        """
        return self.optimizer.param_groups[0]["lr"]


class LinearScheduler(_SchedulerMixin):
    """
    Linear learning rate decay scheduler.

    Linearly decreases the learning rate from base_lr to min_lr over total_steps.
    Formula: lr = base_lr - (base_lr - min_lr) * (step / total_steps).
    """

    def __init__(
        self,
        optimizer: optim.Optimizer,
        total_steps: int,
        min_lr: float = 0.0,
        base_lr: Optional[float] = None,
    ):
        """
        Initialize linear scheduler.

        :param optimizer: PyTorch optimizer to update.
        :type optimizer: optim.Optimizer
        :param total_steps: Total number of training steps.
        :type total_steps: int
        :param min_lr: Minimum learning rate at the end of decay.
        :type min_lr: float
        :param base_lr: Starting learning rate (defaults to optimizer's current lr).
        :type base_lr: Optional[float]
        """
        self.optimizer = optimizer
        self.total_steps = total_steps
        self.min_lr = min_lr
        self.base_lr = base_lr or optimizer.param_groups[0]["lr"]
        self.current_step = 0

    def step(self) -> None:
        """
        Advance one step and update the optimizer's learning rate.
        """
        self.current_step += 1

        progress = min(self.current_step / max(self.total_steps, 1), 1.0)
        lr = self.base_lr - (self.base_lr - self.min_lr) * progress

        for param_group in self.optimizer.param_groups:
            param_group["lr"] = lr

    def get_lr(self) -> float:
        """
        Return the current learning rate.

        :returns: Current learning rate.
        :rtype: float
        """
        return self.optimizer.param_groups[0]["lr"]


def create_scheduler(
    optimizer: optim.Optimizer,
    scheduler_type: str,
    **kwargs,
) -> _SchedulerMixin:
    """
    Create a learning rate scheduler by name.

    :param optimizer: PyTorch optimizer to schedule.
    :type optimizer: optim.Optimizer
    :param scheduler_type: Scheduler variant ("warmup", "cosine", "warmup_cosine",
        or "linear").
    :type scheduler_type: str
    :returns: Learning rate scheduler instance.
    :rtype: _SchedulerMixin
    :raises ValueError: If scheduler_type is not recognized.
    """
    if scheduler_type == "warmup":
        return WarmupScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "cosine":
        return CosineAnnealingScheduler(
            optimizer,
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "warmup_cosine":
        return WarmupCosineScheduler(
            optimizer,
            warmup_steps=kwargs.get("warmup_steps", 0),
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    elif scheduler_type == "linear":
        return LinearScheduler(
            optimizer,
            total_steps=kwargs.get("total_steps", 1),
            min_lr=kwargs.get("min_lr", 0.0),
            base_lr=kwargs.get("base_lr"),
        )
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def visualize_scheduler(
    scheduler_type: str,
    total_steps: int = 1000,
    warmup_steps: int = 100,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    save_path: str = "lr_schedule.png",
) -> None:
    """
    Plot a learning rate schedule and save the figure.

    :param scheduler_type: Type of scheduler to visualize.
    :type scheduler_type: str
    :param total_steps: Total training steps to simulate.
    :type total_steps: int
    :param warmup_steps: Number of warmup steps.
    :type warmup_steps: int
    :param base_lr: Peak learning rate.
    :type base_lr: float
    :param min_lr: Minimum learning rate.
    :type min_lr: float
    :param save_path: File path for the saved figure.
    :type save_path: str
    """
    import matplotlib.pyplot as plt
    import torch.nn as nn

    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=base_lr)

    if scheduler_type == "warmup":
        scheduler = WarmupScheduler(optimizer, warmup_steps, base_lr)
    elif scheduler_type == "cosine":
        scheduler = CosineAnnealingScheduler(optimizer, total_steps, min_lr, base_lr)
    elif scheduler_type == "warmup_cosine":
        scheduler = WarmupCosineScheduler(
            optimizer, warmup_steps, total_steps, min_lr, base_lr
        )
    elif scheduler_type == "linear":
        scheduler = LinearScheduler(optimizer, total_steps, min_lr, base_lr)
    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")

    lrs = []
    for _ in range(total_steps):
        lrs.append(scheduler.get_lr())
        scheduler.step()

    plt.figure(figsize=(10, 6))
    plt.plot(lrs, linewidth=2)
    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title(f"{scheduler_type.replace('_', ' ').title()} Learning Rate Schedule")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Learning rate schedule visualization saved to {save_path}")


def compare_schedulers(
    total_steps: int = 1000,
    warmup_steps: int = 100,
    base_lr: float = 1e-3,
    min_lr: float = 1e-5,
    save_path: str = "lr_schedules_comparison.png",
) -> None:
    """
    Plot all schedulers on the same axes and save the comparison figure.

    :param total_steps: Total training steps to simulate.
    :type total_steps: int
    :param warmup_steps: Number of warmup steps.
    :type warmup_steps: int
    :param base_lr: Peak learning rate.
    :type base_lr: float
    :param min_lr: Minimum learning rate.
    :type min_lr: float
    :param save_path: File path for the saved figure.
    :type save_path: str
    """
    import matplotlib.pyplot as plt
    import torch.nn as nn

    schedulers = {
        "Warmup": WarmupScheduler,
        "Cosine": CosineAnnealingScheduler,
        "Warmup + Cosine": WarmupCosineScheduler,
        "Linear": LinearScheduler,
    }

    plt.figure(figsize=(12, 6))

    for name, scheduler_class in schedulers.items():
        model = nn.Linear(10, 10)
        optimizer = optim.Adam(model.parameters(), lr=base_lr)

        if name == "Warmup":
            scheduler = scheduler_class(optimizer, warmup_steps, base_lr)
        elif name == "Cosine":
            scheduler = scheduler_class(optimizer, total_steps, min_lr, base_lr)
        elif name == "Warmup + Cosine":
            scheduler = scheduler_class(
                optimizer, warmup_steps, total_steps, min_lr, base_lr
            )
        elif name == "Linear":
            scheduler = scheduler_class(optimizer, total_steps, min_lr, base_lr)

        lrs = []
        for _ in range(total_steps):
            lrs.append(scheduler.get_lr())
            scheduler.step()

        plt.plot(lrs, label=name, linewidth=2)

    plt.xlabel("Training Step")
    plt.ylabel("Learning Rate")
    plt.title("Learning Rate Schedulers Comparison")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"Scheduler comparison saved to {save_path}")


def test_schedulers() -> None:
    """
    Test learning rate schedulers by printing step-by-step learning rates.
    """
    import torch.nn as nn

    model = nn.Linear(10, 10)
    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("Testing Learning Rate Schedulers:\n")

    print("1. Warmup Scheduler:")
    scheduler = WarmupScheduler(optimizer, warmup_steps=10)
    for step in range(15):
        print(f"   Step {step}: lr = {scheduler.get_lr():.6f}")
        scheduler.step()

    optimizer = optim.Adam(model.parameters(), lr=1e-3)

    print("\n2. Cosine Annealing Scheduler:")
    scheduler = CosineAnnealingScheduler(optimizer, total_steps=20, min_lr=1e-5)
    for step in range(0, 20, 5):
        print(f"   Step {step}: lr = {scheduler.get_lr():.6f}")
        scheduler.step()

    print("\nVisualize schedulers with visualize_scheduler() and compare_schedulers()")
