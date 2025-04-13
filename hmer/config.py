"""
Configuration handling for HMER-Ink.
"""

import os
import yaml
from typing import Dict, Any, Optional


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
    
    with open(config_path, 'r') as f:
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
    
    with open(config_path, 'w') as f:
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
    
    optimizer_name = config.get('optimizer', 'adam').lower()
    lr = config.get('learning_rate', 0.001)
    weight_decay = config.get('weight_decay', 0.0)
    
    if optimizer_name == 'adam':
        return optim.Adam(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'adamw':
        return optim.AdamW(model_parameters, lr=lr, weight_decay=weight_decay)
    elif optimizer_name == 'sgd':
        momentum = config.get('momentum', 0.9)
        return optim.SGD(model_parameters, lr=lr, momentum=momentum, weight_decay=weight_decay)
    elif optimizer_name == 'rmsprop':
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
    
    scheduler_config = config.get('lr_scheduler', {})
    if not scheduler_config or not scheduler_config.get('type'):
        return None
    
    scheduler_type = scheduler_config.get('type', '').lower()
    
    if scheduler_type == 'step':
        step_size = scheduler_config.get('step_size', 10)
        gamma = scheduler_config.get('gamma', 0.1)
        return lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    
    elif scheduler_type == 'multistep':
        milestones = scheduler_config.get('milestones', [30, 60, 90])
        gamma = scheduler_config.get('gamma', 0.1)
        return lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)
    
    elif scheduler_type == 'exponential':
        gamma = scheduler_config.get('gamma', 0.95)
        return lr_scheduler.ExponentialLR(optimizer, gamma=gamma)
    
    elif scheduler_type == 'cosine':
        T_max = scheduler_config.get('T_max', 100)
        eta_min = scheduler_config.get('eta_min', 0)
        return lr_scheduler.CosineAnnealingLR(optimizer, T_max=T_max, eta_min=eta_min)
    
    elif scheduler_type == 'reduce_on_plateau':
        mode = scheduler_config.get('mode', 'min')
        factor = scheduler_config.get('factor', 0.1)
        patience = scheduler_config.get('patience', 10)
        threshold = scheduler_config.get('threshold', 1e-4)
        return lr_scheduler.ReduceLROnPlateau(
            optimizer, mode=mode, factor=factor, patience=patience, threshold=threshold)
    
    elif scheduler_type == 'linear_warmup':
        from torch.optim.lr_scheduler import LambdaLR
        
        warmup_steps = scheduler_config.get('warmup_steps', 100)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            return 1.0
        
        return LambdaLR(optimizer, lr_lambda)
    
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
    
    device_name = config.get('device', 'cuda').lower()
    
    if device_name == 'cuda' and torch.cuda.is_available():
        return torch.device('cuda')
    elif device_name == 'mps' and hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device('mps')
    else:
        return torch.device('cpu')