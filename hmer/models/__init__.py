"""
Model initialization and factory functions.
"""

from .encoder import get_encoder
from .decoder import get_decoder
from .model import HMERModel

__all__ = ['get_encoder', 'get_decoder', 'HMERModel', 'create_model']


def create_model(config, vocab_size):
    """
    Create a HMER model based on the configuration.
    
    Args:
        config: Model configuration
        vocab_size: Size of the target vocabulary
        
    Returns:
        Configured HMERModel instance
    """
    model_config = config.get('model', {})
    
    # Create encoder
    encoder = get_encoder(model_config)
    
    # Create decoder
    decoder = get_decoder(model_config, vocab_size)
    
    # Create full model
    model = HMERModel(
        encoder=encoder,
        decoder=decoder,
        config=model_config
    )
    
    return model