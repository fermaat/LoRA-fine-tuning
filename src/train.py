import argparse
import json
import os
from typing import Any, Dict

import torch
from transformers import TrainerCallback, TrainingArguments
from trl import DPOConfig, DPOTrainer, ORPOConfig, ORPOTrainer, SFTConfig, SFTTrainer

from cfg import (
    config_logger,
    logger,
    settings,
)
from src.data_loading import (
    load_and_prepare_alignment_dataset,
    load_and_prepare_orpo_alignment_dataset,
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


class CleanJSONLoggerCallback(TrainerCallback):
    def __init__(self, log_path: str, log_steps: bool = True, log_epochs: bool = True):
        """
        Improved JSON logger for HuggingFace Trainer

        Args:
            log_path: Path to save the JSON log file
            log_steps: Whether to log individual training steps
            log_epochs: Whether to log epoch summaries
        """
        self.log_path = log_path
        self.log_steps = log_steps
        self.log_epochs = log_epochs

        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(log_path), exist_ok=True)

        # Initialize log file with proper structure
        with open(self.log_path, "w") as f:
            f.write("")  # Clear file

    def on_log(self, args, state, control, logs=None, **kwargs):
        """Log training metrics during training steps"""
        if logs is None:
            return

        # Filter out training summary (which contains train_runtime)
        if "train_runtime" in logs:
            return

        # Determine if this is an evaluation log or training log
        is_eval_log = "eval_loss" in logs

        if is_eval_log:
            # Handle evaluation logs
            log_entry = {
                "type": "eval",
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 4),
                "eval_loss": round(logs.get("eval_loss", 0), 4),
            }

            # Add standard eval metrics
            eval_metrics = [
                "eval_accuracy",
                "eval_f1",
                "eval_precision",
                "eval_recall",
                "eval_runtime",
                "eval_samples_per_second",
                "eval_steps_per_second",
            ]

            for metric in eval_metrics:
                if metric in logs:
                    value = logs[metric]
                    log_entry[metric] = (
                        round(value, 4) if isinstance(value, float) else value
                    )

            # Add DPO-specific reward metrics during evaluation if they exist
            dpo_eval_metrics = [
                "eval_rewards/chosen",
                "eval_rewards/rejected",
                "eval_rewards/accuracies",
                "eval_rewards/margins",
            ]

            for metric in dpo_eval_metrics:
                if metric in logs:
                    value = logs[metric]
                    log_entry[metric.replace("eval_", "")] = (
                        round(value, 4) if isinstance(value, float) else value
                    )

        elif self.log_steps:
            # Handle training step logs
            log_entry = {
                "type": "step",
                "step": state.global_step,
                "epoch": round(logs.get("epoch", 0), 4),
                "loss": round(logs.get("loss", 0), 4),
                "learning_rate": logs.get("learning_rate", 0),
                "grad_norm": round(logs.get("grad_norm", 0), 4)
                if logs.get("grad_norm")
                else None,
            }

            # Add optional training metrics if they exist
            optional_metrics = [
                "num_tokens",
                "mean_token_accuracy",
                "rewards/chosen",
                "rewards/rejected",
                "rewards/accuracies",
                "rewards/margins",
                "logps/chosen",
                "logps/rejected",
                "logits/chosen",
                "logits/rejected",
                "sft_loss",  # For ORPO
                "odds_ratio_loss",  # For ORPO
                "nll_loss",  # For ORPO - negative log likelihood
            ]

            for metric in optional_metrics:
                if metric in logs:
                    value = logs[metric]
                    log_entry[metric] = (
                        round(value, 4) if isinstance(value, float) else value
                    )
        else:
            return  # Skip if we're not logging steps and it's not an eval log

        self._write_log_entry(log_entry)

    def on_train_end(self, args, state, control, **kwargs):
        """Log final training summary"""
        if not self.log_epochs:
            return

        summary_entry = {
            "type": "summary",
            "total_steps": state.global_step,
            "total_epochs": state.epoch,
            "best_metric": getattr(state, "best_metric", None),
            "training_completed": True,
        }

        self._write_log_entry(summary_entry)

    def _write_log_entry(self, entry: Dict[str, Any]):
        """Write a single log entry to file"""
        with open(self.log_path, "a") as f:
            json.dump(entry, f, separators=(",", ":"))
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
        # evaluation
        eval_strategy="steps",  # Run evaluation every eval_steps
        eval_steps=settings.save_steps,  # Evaluate at same frequency as saving
        per_device_eval_batch_size=settings.per_device_train_batch_size,
        # Best checkpoint saving
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        save_strategy="steps",
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
    elif training_type == "orpo":
        return ORPOConfig(
            beta=settings.orpo_beta,
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


def model_training(
    training_type: str,
    max_samples=None,
    resume_from_checkpoint=False,
    max_samples_eval=None,
):
    assert training_type in ["sft", "dpo", "orpo"], (
        f"Unsupported training_type: {training_type}"
    )

    dataset_load_function = (
        load_and_prepare_sft_dataset
        if training_type == "sft"
        else load_and_prepare_orpo_alignment_dataset
        if training_type == "orpo"
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
    # Determine dataset split names and max_length based on training type
    if training_type == "sft":
        train_split = "train_sft"
        test_split = "test_sft"
        max_length = settings.sft_max_tokens_length
    else:  # dpo or orpo
        train_split = "train_prefs"
        test_split = "test_prefs"
        max_length = getattr(
            settings, f"{training_type}_max_length", settings.dpo_max_length
        )

    train_dataset = dataset_load_function(
        settings.default_dataset_name,
        train_split,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples,
    )
    eval_dataset = dataset_load_function(
        settings.default_dataset_name,
        test_split,
        tokenizer=tokenizer,
        max_length=max_length,
        max_samples=max_samples_eval,
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
    callbacks = [CleanJSONLoggerCallback(log_file_path)]

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
    elif training_type == "orpo":
        trainer = ORPOTrainer(
            model=model,
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

    metrics_to_plot = getattr(settings, f"{training_type}_metrics_to_plot")

    # # plot
    plot_training_logs(
        jsonl_filepath=log_file_path,
        output_dir=f"{settings.results_path}/{model_sub_folder}",
        metrics_to_plot=metrics_to_plot,
        smoothing_window=10,
        downsample_factor=5,
    )


if __name__ == "__main__":
    config_logger(log_level=settings.log_level)

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--training_type", type=str, choices=["sft", "dpo", "orpo"], required=True
    )
    args = parser.parse_args()

    model_training(
        args.training_type,
        max_samples=2000,
        max_samples_eval=50,
        resume_from_checkpoint=False,
    )
