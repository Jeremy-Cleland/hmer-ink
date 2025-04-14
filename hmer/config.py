"""
Configuration handling for HMER-Ink.
"""

import os
from typing import Any, Dict

import yaml


def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load configuration from a YAML file.

    Args:
        config_path: Path to the configuration file

    Returns:
        Configuration dictionary
    """
    if not os.path.exists(config_path):
        raise ValueError(f"Config file not found: {config_path}")

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return config


def save_config(config: Dict[str, Any], config_path: str) -> None:
    """
    Save configuration to a YAML file.

    Args:
        config: Configuration dictionary
        config_path: Path to save the configuration file
    """
    # Ensure directory exists
    os.makedirs(os.path.dirname(config_path), exist_ok=True)

    with open(config_path, "w") as f:
        yaml.dump(config, f, default_flow_style=False)


def update_config(config: Dict[str, Any], updates: Dict[str, Any]) -> Dict[str, Any]:
    """
    Update configuration with new values.

    Args:
        config: Original configuration dictionary
        updates: Dictionary with updates

    Returns:
        Updated configuration dictionary
    """

    # Recursive update
    def _update(d, u):
        for k, v in u.items():
            if isinstance(v, dict) and k in d and isinstance(d[k], dict):
                d[k] = _update(d[k], v)
            else:
                d[k] = v
        return d

    # Create a copy of the original config
    updated_config = config.copy()

    # Apply updates
    updated_config = _update(updated_config, updates)

    return updated_config


def get_optimizer(config: Dict[str, Any], model_parameters):
    """
    Create optimizer based on configuration.

    Args:
        config: Configuration dictionary
        model_parameters: Model parameters to optimize

    Returns:
        Optimizer instance
    """
    import torch.optim as optim

    optimizer_name = config.get("optimizer", "adam").lower()
    lr = config.get("learning_rate", 0.001)
    weight_decay = config.get("weight_decay", 0.0)

    if optimizer_name == "adam":
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "adamw":
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == "sgd":
        momentum = config.get("momentum", 0.9)
        return optim.SGD(
            model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
    elif optimizer_name == "rmsprop":
        return optim.RMSprop(model_parameters, lr=lr, weight_decay=weight_decay)
    else:
        raise ValueError(f"Unknown optimizer: {optimizer_name}")


def get_scheduler(config: Dict[str, Any], optimizer):
    """
    Create learning rate scheduler based on configuration.

    Args:
        config: Configuration dictionary
        optimizer: Optimizer instance

    Returns:
        Scheduler instance or None
    """
    import torch.optim.lr_scheduler as lr_scheduler

    scheduler_config = config.get("lr_scheduler", {})
    if not scheduler_config or not scheduler_config.get("type"):
        return None

    scheduler_type = scheduler_config.get("type", "").lower()

    if scheduler_type == "step":
        step_size = scheduler_config.get("step_size", 10)
        gamma = scheduler_config.get("gamma", 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)

    elif scheduler_type == "multistep":
        milestones = scheduler_config.get("milestones", [30, 60, 90])
        gamma = scheduler_config.get("gamma", 0.1)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    elif scheduler_type == "exponential":
        gamma = scheduler_config.get("gamma", 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)

    elif scheduler_type == "cosine":
        T_max = scheduler_config.get("T_max", 100)
        eta_min = scheduler_config.get("eta_min", 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)

    elif scheduler_type == "reduce_on_plateau":
        mode = scheduler_config.get("mode", "min")
        factor = scheduler_config.get("factor", 0.1)
        patience = scheduler_config.get("patience", 10)
        threshold = scheduler_config.get("threshold", 1e-4)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold
        )

    elif scheduler_type == "linear_warmup":
        from torch.optim.lr_scheduler import LambdaLR

        warmup_steps = scheduler_config.get("warmup_steps", 100)

        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0

        return LambdaLR(optimizer, lr_lambda)
        
    elif scheduler_type == "one_cycle":
        from torch.optim.lr_scheduler import OneCycleLR
        
        max_lr = scheduler_config.get("max_lr", 0.01)
        total_steps = scheduler_config.get("total_steps", None)
        epochs = scheduler_config.get("epochs", 30)
        
        # Get steps_per_epoch from config if available
        if hasattr(optimizer, 'param_groups') and len(optimizer.param_groups) > 0:
            lr = optimizer.param_groups[0]['lr']
            print(f"Initial learning rate: {lr}, Max learning rate for OneCycleLR: {max_lr}")
        
        # Try to estimate steps per epoch
        steps_per_epoch = scheduler_config.get("steps_per_epoch", 100)
        
        pct_start = scheduler_config.get("pct_start", 0.3)
        div_factor = scheduler_config.get("div_factor", 25.0)
        final_div_factor = scheduler_config.get("final_div_factor", 10000.0)
        
        if total_steps is None:
            total_steps = epochs * steps_per_epoch
            print(f"OneCycleLR configured with total_steps={total_steps} (epochs={epochs} × steps_per_epoch={steps_per_epoch})")
            
        return OneCycleLR(
            optimizer, 
            max_lr=max_lr,
            total_steps=total_steps,
            pct_start=pct_start,
            div_factor=div_factor,
            final_div_factor=final_div_factor
        )

    else:
        raise ValueError(f"Unknown scheduler type: {scheduler_type}")


def get_device(config: Dict[str, Any]):
    """
    Get torch device based on configuration.

    Args:
        config: Configuration dictionary

    Returns:
        torch.device
    """
    import torch

    # 1. Check top-level 'device' key
    device_name = config.get("device")

    # 2. If not found, check 'training' section's 'device' key
    if device_name is None:
        device_name = config.get("training", {}).get("device")

    # 3. If still not found, attempt auto-detection (MPS > CUDA > CPU)
    if device_name is None:
        print("Device not specified in config, attempting auto-detection...")
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS available, using MPS.")
            return torch.device("mps")
        elif torch.cuda.is_available():
            print("CUDA available, using CUDA.")
            return torch.device("cuda")
        else:
            print("No GPU available, using CPU.")
            return torch.device("cpu")

    # 4. If device_name was specified, use it (checking availability)
    device_name = str(device_name).lower()  # Ensure lowercase string
    print(f"Device specified in config: '{device_name}'")
    if device_name == "mps":
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            print("MPS available, using MPS.")
            return torch.device("mps")
        else:
            print("Warning: MPS requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_name == "cuda":
        if torch.cuda.is_available():
            print("CUDA available, using CUDA.")
            return torch.device("cuda")
        else:
            print("Warning: CUDA requested but not available. Falling back to CPU.")
            return torch.device("cpu")
    elif device_name == "cpu":
        print("Using CPU as specified.")
        return torch.device("cpu")
    else:
        print(
            f"Warning: Unknown device '{device_name}' requested. Falling back to CPU."
        )
        return torch.device("cpu")
