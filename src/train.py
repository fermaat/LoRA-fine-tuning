import argparse
import json
import os

import torch
from transformers import TrainerCallback, TrainingArguments
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from cfg import (
    config_logger,
    logger,
    settings,
)
from src.data_loading import (
    load_and_prepare_alignment_dataset,
    load_and_prepare_sft_dataset,
)
from src.model import get_latest_checkpoint, load_model_and_tokenizer
from src.plotting_utils import plot_training_logs
from src.utils import get_dtype, setup_device


class JSONLoggerCallback(TrainerCallback):
    def __init__(self, log_path: str):
        self.log_path = log_path

    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None:
            with open(self.log_path, "a") as f:
                json.dump(logs, f)
                f.write("\n")


def create_training_args(training_type: str, output_dir: str) -> TrainingArguments:
    base_args = dict(
        output_dir=output_dir,
        per_device_train_batch_size=settings.per_device_train_batch_size,
        gradient_accumulation_steps=settings.gradient_accumulation_steps,
        logging_steps=settings.logging_steps,
        save_steps=settings.save_steps,
        save_total_limit=settings.save_total_limit,
        # logging_dir=settings.results_path,
        report_to="none",
        fp16=get_dtype() == torch.float16,
        bf16=False,
        remove_unused_columns=False,
        push_to_hub=False,
    )

    if training_type == "dpo":
        return DPOConfig(
            beta=settings.dpo_beta,
            max_length=settings.dpo_max_length,
            learning_rate=settings.dpo_learning_rate,
            num_train_epochs=settings.dpo_num_train_epochs,
            run_name=f"{training_type}-gemma-lr{settings.dpo_learning_rate}-r{settings.peft_r}",
            **base_args,
        )
    elif training_type == "sft":
        return SFTConfig(
            learning_rate=settings.sft_learning_rate,
            max_seq_length=settings.sft_max_tokens_length,
            num_train_epochs=settings.sft_num_train_epochs,
            packing=settings.sft_packing,
            run_name=f"{training_type}-gemma-lr{settings.sft_learning_rate}-r{settings.peft_r}",
            **base_args,
        )
    else:
        raise ValueError(f"Unsupported training type: {training_type}")


def model_training(training_type: str, max_samples=None, resume_from_checkpoint=False):
    assert training_type in ["sft", "dpo"], (
        f"Unsupported training_type: {training_type}"
    )

    dataset_load_function = (
        load_and_prepare_sft_dataset
        if training_type == "sft"
        else load_and_prepare_alignment_dataset
    )

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

    logger.info(f"Starting {training_type} training for model: {model_id}")
    logger.info(f"Device: {device}")
    logger.info(f"Using PEFT (LoRA): {use_peft}")
    dtype = get_dtype()
    logger.info(f"Using {str(dtype)} Precision")

    logger.info("Loading model and tokenizer...")

    model, tokenizer = load_model_and_tokenizer(
        model_id, device, use_peft=use_peft, peft_config=peft_config, dtype=dtype
    )

    logger.info("Loading and preparing datasets...")
    train_dataset = dataset_load_function(
        settings.default_dataset_name,
        f"train_{training_type}" if training_type == "sft" else "train_prefs",
        tokenizer=tokenizer,
        max_length=settings.sft_max_tokens_length
        if training_type == "sft"
        else settings.dpo_max_length,
        max_samples=max_samples,
    )
    eval_dataset = dataset_load_function(
        settings.default_dataset_name,
        f"test_{training_type}" if training_type == "sft" else "test_prefs",
        tokenizer=tokenizer,
        max_length=settings.sft_max_tokens_length
        if training_type == "sft"
        else settings.dpo_max_length,
        max_samples=max_samples,
    )
    logger.info(f"Training dataset size: {len(train_dataset)}")
    logger.info(f"Evaluation dataset size: {len(eval_dataset)}")
    model_sub_folder = (
        f"lora-{training_type}-{settings.default_model_id.split('/')[-1]}"
    )
    output_dir = f"{settings.model_output_dir}/{model_sub_folder}/"
    args = create_training_args(training_type, output_dir)
    log_folder = f"{settings.results_path}/{model_sub_folder}/"
    os.makedirs(os.path.dirname(log_folder), exist_ok=True)
    log_file_path = f"{log_folder}{training_type}_logs.json"
    callbacks = [JSONLoggerCallback(log_file_path)]

    if training_type == "dpo":
        trainer = DPOTrainer(
            model=model,
            ref_model=None,  # use another reference model just in case
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )
    else:  # Assume SFT
        trainer = SFTTrainer(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=tokenizer,
            callbacks=callbacks,
        )

    logger.info(f"Starting the {training_type} training with LoRA and FP16...")
    if resume_from_checkpoint:
        last_ckpt = get_latest_checkpoint(output_dir)

        if last_ckpt:
            logger.info(f"Resuming training from checkpoint: {last_ckpt}")
            trainer.train(resume_from_checkpoint=last_ckpt)
        else:
            logger.info("No checkpoint found, starting training from scratch")
            trainer.train()
    else:
        trainer.train()

    logger.info("Training complete. Saving final adapter model...")

    trainer.save_model(f"{output_dir}/final")
    logger.info(f"Adapter model saved to {output_dir}/final")

    # # plot
    plot_training_logs(log_file_path=log_file_path, output_dir=output_dir)


if __name__ == "__main__":
    config_logger(log_level=settings.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_type", type=str, choices=["sft", "dpo"], required=True
    )
    args = parser.parse_args()

    model_training(args.training_type, max_samples=50, resume_from_checkpoint=True)
