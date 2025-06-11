from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfg import config_logger, logger, settings
from src.utils import setup_device

config_logger(log_level=settings.log_level)
device = setup_device()


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    use_peft: bool = False,
    peft_config: dict = None,
    evaluation_mode: bool = False,
    dtype=torch.bfloat16,
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """
    Load and configure the model and tokenizer for evaluation.

    Args:
        model_id (str): The HuggingFace model identifier
        device (torch.device): The device to load the model on

    Returns:
        Tuple[AutoModelForCausalLM, AutoTokenizer]: The loaded model and tokenizer
    """
    logger.info(f"Loading tokenizer and model: {model_id}...")

    tokenizer = AutoTokenizer.from_pretrained(model_id)

    # Gemma models are often in bfloat16, which MPS supports.
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,  # Use bfloat16 for Gemma and MPS
        device_map="auto",  # This might automatically use MPS, but we'll explicitly move it too
    )

    if use_peft:
        if peft_config is None:
            raise ValueError("PEFT is enabled but no `peft_config` was provided.")

        # Optionally prepare for k-bit (QLoRA etc.)
        model = prepare_model_for_kbit_training(model)
        # TODO: export defaults to settings
        lora_config = LoraConfig(
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            bias=peft_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()

    model.to(device)  # Explicitly move model to MPS device
    if evaluation_mode:
        model.eval()

    return model, tokenizer
