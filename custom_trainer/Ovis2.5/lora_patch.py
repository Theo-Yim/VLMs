"""
LoRA compatibility fix for Ovis2.5-9B
Patches the model to support PEFT LoRA training
"""

import logging
from typing import Optional

import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


def patch_ovis_for_lora(model) -> None:
    """
    Patch Ovis2.5 model to be compatible with PEFT LoRA
    Adds missing get_input_embeddings and set_input_embeddings methods
    """
    logger.info("Patching Ovis2.5 model for LoRA compatibility...")

    def find_embedding_layer(model):
        """Find the embedding layer in the Ovis2.5 model"""
        # Try common patterns for finding embeddings
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model.embed_tokens
        elif hasattr(model, "embed_tokens"):
            return model.embed_tokens
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer.wte
        elif hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            return model.transformer.word_embeddings

        # Search through the model hierarchy
        for name, module in model.named_modules():
            if isinstance(module, nn.Embedding) and "embed" in name.lower():
                logger.info(f"Found embedding layer: {name}")
                return module

        # If we can't find it automatically, try to get it from the language model part
        if hasattr(model, "llm"):
            if hasattr(model.llm, "model") and hasattr(model.llm.model, "embed_tokens"):
                return model.llm.model.embed_tokens
            elif hasattr(model.llm, "embed_tokens"):
                return model.llm.embed_tokens

        return None

    def find_embedding_parent_and_attr(model):
        """Find the parent module and attribute name for the embedding layer"""
        if hasattr(model, "model") and hasattr(model.model, "embed_tokens"):
            return model.model, "embed_tokens"
        elif hasattr(model, "embed_tokens"):
            return model, "embed_tokens"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "wte"):
            return model.transformer, "wte"
        elif hasattr(model, "transformer") and hasattr(model.transformer, "word_embeddings"):
            return model.transformer, "word_embeddings"
        elif hasattr(model, "llm"):
            if hasattr(model.llm, "model") and hasattr(model.llm.model, "embed_tokens"):
                return model.llm.model, "embed_tokens"
            elif hasattr(model.llm, "embed_tokens"):
                return model.llm, "embed_tokens"

        return None, None

    # Find the embedding layer
    embedding_layer = find_embedding_layer(model)
    parent, attr_name = find_embedding_parent_and_attr(model)

    if embedding_layer is None:
        logger.warning("Could not find embedding layer - LoRA may not work properly")
        return

    logger.info(
        f"Found embedding layer: {type(embedding_layer).__name__} with vocab_size={embedding_layer.num_embeddings}"
    )

    def get_input_embeddings(self) -> Optional[nn.Module]:
        """Get input embeddings for PEFT compatibility"""
        return find_embedding_layer(self)

    def set_input_embeddings(self, value: nn.Module) -> None:
        """Set input embeddings for PEFT compatibility"""
        parent_module, attr_name = find_embedding_parent_and_attr(self)
        if parent_module is not None and attr_name is not None:
            setattr(parent_module, attr_name, value)
            logger.info(f"Set input embeddings: {attr_name}")
        else:
            logger.warning("Could not set input embeddings - parent not found")

    def get_output_embeddings(self) -> Optional[nn.Module]:
        """Get output embeddings for PEFT compatibility"""
        # For decoder-only models, output embeddings are often the same as input embeddings
        # or located in the language model head
        if hasattr(model, "lm_head"):
            return model.lm_head
        elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
            return model.model.lm_head
        elif hasattr(model, "llm") and hasattr(model.llm, "lm_head"):
            return model.llm.lm_head
        return None

    def set_output_embeddings(self, value: nn.Module) -> None:
        """Set output embeddings for PEFT compatibility"""
        if hasattr(model, "lm_head"):
            model.lm_head = value
        elif hasattr(model, "model") and hasattr(model.model, "lm_head"):
            model.model.lm_head = value
        elif hasattr(model, "llm") and hasattr(model.llm, "lm_head"):
            model.llm.lm_head = value

    def resize_token_embeddings(self, new_num_tokens: int) -> nn.Embedding:
        """Resize token embeddings for PEFT compatibility"""
        embedding_layer = find_embedding_layer(self)
        parent_module, attr_name = find_embedding_parent_and_attr(self)

        if embedding_layer is None or parent_module is None:
            raise RuntimeError("Could not find embedding layer to resize")

        old_num_tokens, embedding_dim = embedding_layer.weight.shape

        if new_num_tokens == old_num_tokens:
            return embedding_layer

        # Create new embedding layer
        new_embedding = nn.Embedding(new_num_tokens, embedding_dim)

        # Copy existing weights
        with torch.no_grad():
            new_embedding.weight[:old_num_tokens] = embedding_layer.weight
            # Initialize new tokens with small random values
            if new_num_tokens > old_num_tokens:
                new_embedding.weight[old_num_tokens:] = (
                    torch.randn(new_num_tokens - old_num_tokens, embedding_dim) * 0.02
                )

        # Replace the embedding layer
        setattr(parent_module, attr_name, new_embedding)

        # Also update output embeddings if they exist and are tied
        output_embeddings = get_output_embeddings(self)
        if output_embeddings is not None and output_embeddings.weight.shape[0] == old_num_tokens:
            new_output = nn.Linear(embedding_dim, new_num_tokens, bias=False)
            with torch.no_grad():
                new_output.weight[:old_num_tokens] = output_embeddings.weight
                if new_num_tokens > old_num_tokens:
                    new_output.weight[old_num_tokens:] = (
                        torch.randn(new_num_tokens - old_num_tokens, embedding_dim) * 0.02
                    )
            set_output_embeddings(self, new_output)

        logger.info(f"Resized token embeddings from {old_num_tokens} to {new_num_tokens}")
        return new_embedding

    # Monkey patch the methods onto the model
    import types

    model.get_input_embeddings = types.MethodType(get_input_embeddings, model)
    model.set_input_embeddings = types.MethodType(set_input_embeddings, model)
    model.get_output_embeddings = types.MethodType(get_output_embeddings, model)
    model.set_output_embeddings = types.MethodType(set_output_embeddings, model)
    model.resize_token_embeddings = types.MethodType(resize_token_embeddings, model)

    logger.info("✅ Successfully patched Ovis2.5 model for LoRA compatibility")


def validate_lora_compatibility(model) -> bool:
    """
    Validate that the model is compatible with LoRA
    """
    required_methods = [
        "get_input_embeddings",
        "set_input_embeddings",
        "get_output_embeddings",
        "set_output_embeddings",
        "resize_token_embeddings",
    ]

    for method in required_methods:
        if not hasattr(model, method):
            logger.error(f"Missing required method: {method}")
            return False

    # Test that get_input_embeddings actually works
    try:
        embeddings = model.get_input_embeddings()
        if embeddings is None:
            logger.error("get_input_embeddings returned None")
            return False
        logger.info(f"✅ Input embeddings validation passed: {type(embeddings).__name__}")
    except Exception as e:
        logger.error(f"get_input_embeddings failed: {e}")
        return False

    return True
