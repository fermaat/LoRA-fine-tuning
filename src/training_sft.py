import torch
from trl import SFTConfig, SFTTrainer

from cfg import (
    config_logger,
    logger,
    settings,
)
from src.data_loading import (
    load_and_prepare_sft_dataset,
)
from src.model import load_model_and_tokenizer
from src.utils import setup_device

config_logger(log_level=settings.log_level)


def sft_training():
    model_id = settings.default_model_id

    peft_config = {
        "r": settings.peft_r,
        "lora_alpha": settings.peft_lora_alpha,
        "target_modules": settings.peft_target_modules,
        "lora_dropout": settings.peft_lora_dropout,
        "bias": settings.peft_bias,
    }
    use_peft = True

    device = setup_device()

    # Check if we should use fp16
    use_fp16 = settings.sft_fp16_training and device.type == "mps"

    logger.info(f"Starting SFT training for model: {model_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Using PEFT (LoRA): {use_peft}")
    logger.info(f"Using FP16 Mixed-Precision: {use_fp16}")

    logger.info("Loading model and tokenizer...")
    model, tokenizer = load_model_and_tokenizer(
        model_id,
        device,
        use_peft=use_peft,
        peft_config=peft_config,
        dtype=torch.bfloat16,
    )

    logger.info("Loading and preparing datasets...")
    train_dataset = load_and_prepare_sft_dataset(
        settings.default_dataset_name, "train_sft", tokenizer=tokenizer
    )
    eval_dataset = load_and_prepare_sft_dataset(
        settings.default_dataset_name,
        settings.default_evaluation_dataset_split,
        tokenizer=tokenizer,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")

    training_args = SFTConfig(
        output_dir=f"{settings.model_output_dir}/lora-sft-gemma",
        max_seq_length=settings.max_tokens_length_sft,
        learning_rate=settings.sft_learning_rate,
        num_train_epochs=settings.sft_num_train_epochs,
        packing=settings.sft_packing,
        per_device_train_batch_size=settings.sft_per_device_train_batch_size,
        gradient_accumulation_steps=settings.sft_gradient_accumulation_steps,
        logging_steps=settings.sft_logging_steps,
        save_steps=settings.sft_save_steps,
        save_total_limit=settings.sft_save_total_limit,
        report_to="none",
        # Set fp16 flag directly in the trainer. This is the correct way.
        fp16=use_fp16,
        # Ensure bf16 is false
        bf16=False,
        run_name=f"lora-sft-gemma-lr{settings.sft_learning_rate}-r{settings.peft_r}",
        push_to_hub=False,
    )

    logger.info("Initializing SFTTrainer...")
    effective_batch_size = (
        settings.sft_per_device_train_batch_size
        * settings.sft_gradient_accumulation_steps
    )
    logger.info(f"Effective Batch Size: {effective_batch_size}")

    trainer = SFTTrainer(
        model=model,
        processing_class=tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
    )

    logger.info("Starting the SFT training with LoRA and FP16...")
    trainer.train()

    logger.info("Training complete. Saving final adapter model...")
    final_model_path = f"{settings.model_output_dir}/lora-sft-gemma-final"
    trainer.save_model(final_model_path)
    logger.info(f"Adapter model saved to {final_model_path}")


if __name__ == "__main__":
    sft_training()
