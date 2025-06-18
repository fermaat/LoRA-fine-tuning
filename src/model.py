import os
import re
from typing import Tuple

import torch
from peft import LoraConfig, get_peft_model
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfg import config_logger, logger, settings
from src.utils import setup_device

config_logger(log_level=settings.log_level)
device = setup_device()


def get_latest_checkpoint(checkpoint_dir: str) -> str | None:
    if not os.path.exists(checkpoint_dir):
        return None

    checkpoints = []
    for entry in os.listdir(checkpoint_dir):
        full_path = os.path.join(checkpoint_dir, entry)
        if os.path.isdir(full_path) and re.match(r"checkpoint-\d+", entry):
            checkpoints.append(full_path)

    if not checkpoints:
        return None

    # Ordenar por número (paso) descendente y coger el mayor
    checkpoints = sorted(
        checkpoints,
        key=lambda x: int(re.findall(r"\d+", os.path.basename(x))[0]),
        reverse=True,
    )
    return checkpoints[0]


def load_model_and_tokenizer(
    model_id: str,
    device: torch.device,
    use_peft: bool = False,
    peft_config: dict = None,
    evaluation_mode: bool = False,
    dtype=torch.float16,  # Use float16 for better MPS support
    model_checkpoint_enable: bool = False,
    compile_model: bool = False,
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        logger.info("Tokenizer `pad_token` was None. Set to `eos_token`.")

    # Use float16 as it's generally more stable on MPS than bfloat16
    logger.info(f"Loading model with torch_dtype: {dtype} and device_map: 'auto'")
    model = AutoModelForCausalLM.from_pretrained(
        model_id,
        torch_dtype=dtype,
        device_map="auto",
    )
    if model_checkpoint_enable and not evaluation_mode:
        logger.info("Enabling model gradient checkpointing.")
        model.gradient_checkpointing_enable()

    if use_peft:
        logger.info("Configuring model for PEFT (LoRA)...")
        if peft_config is None:
            raise ValueError("PEFT is enabled but no `peft_config` was provided.")

        lora_config = LoraConfig(
            r=peft_config.get("r", 8),
            lora_alpha=peft_config.get("lora_alpha", 16),
            target_modules=peft_config.get("target_modules", ["q_proj", "v_proj"]),
            lora_dropout=peft_config.get("lora_dropout", 0.05),
            bias=peft_config.get("bias", "none"),
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        logger.info("PEFT model created.")
        model.print_trainable_parameters()

    model.to(device)
    logger.info(f"Model moved to device: {model.device}")

    if evaluation_mode:
        logger.info("Setting model to evaluation mode.")
        model.eval()
    elif compile_model:
        logger.info("Attempting to compile the model with torch.compile()...")
        try:
            model = torch.compile(model)
            logger.info("Model compiled successfully.")
        except Exception as e:
            logger.warning(f"torch.compile() is not supported or failed: {e}")

    return model, tokenizer
