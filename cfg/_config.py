import datetime
import os
import sys
from pathlib import Path
from typing import List

from dotenv import load_dotenv

# Load environment variables from .env file
from loguru import logger
from pydantic_settings import BaseSettings

# Get the project root directory
# This is the directory where the main.py file is located
PROJECT_ROOT = Path().resolve()
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Get the library root directory
# This is the directory where the library files are located
# TODO: check if this is the correct way to get the library root
LIBRARY_ROOT = PROJECT_ROOT.joinpath("src", "LoRA-fine-tuning")


class Settings(BaseSettings, case_sensitive=False, extra="ignore"):
    """
    Project settings.

    Uses pydantic BaseSettings for loading configuration from environment variables.
    """

    # local IP address of the server
    local_ip: str = os.environ.get("LOCAL_IP", "")
    # Library name (used for logging and storage)
    # TODO: fix this setting to be inmutable
    library_name: str = "lora_fine_tuning"

    # Library name (used for logging and storage)
    # TODO: fix this setting to be inmutable
    library_prefix: str = "lora"

    # # Library version
    # version: str = metadata.version(library_name)

    # Logging configuration
    log_level: str = "INFO"

    # Base folder for storing data
    storage_folder: str = str(PROJECT_ROOT.joinpath(library_prefix, "storage"))

    # Base folder for storing logs (default: lora/logs)
    logs_storage_folder: str = str(PROJECT_ROOT.joinpath(library_prefix, "logs"))

    # model id
    default_model_id: str = "google/gemma-2b"  # "mistralai/Mistral-7B-v0.1"

    # default dataset name
    default_dataset_name: str = "HuggingFaceH4/ultrafeedback_binarized"

    # Default split for evaluation
    default_evaluation_dataset_split: str = "test_sft"

    # Maximum number of new tokens to generate during evaluation
    max_new_tokens_evaluation: int = 128

    evaluation_batch_size: int = 4

    version: str = "0.1.0"  # Default version, can be overridden by environment variable

    results_path: str = str(PROJECT_ROOT.joinpath("results"))

    model_output_dir: str = str(PROJECT_ROOT.joinpath("models"))

    # PEFT / LoRA Configuration
    peft_r: int = 8
    # Common rule of thumb is lora_alpha = 2 * r.
    # In underfitting case, increase LoRA rank to 16 or 32 and adjust lora_alpha accordingly.
    peft_lora_alpha: int = 16
    # will just PEFT on those two layers for now. Just for testing
    peft_target_modules: list[str] = ["q_proj", "v_proj"]
    peft_lora_dropout: float = 0.05
    peft_bias: str = "none"
    # Use fp16 for MPS mixed-precision training
    fp16_training: bool = True
    bf16_training: bool = True

    per_device_train_batch_size: int = 2
    gradient_accumulation_steps: int = 4
    logging_steps: int = 1
    save_steps: int = 10
    save_total_limit: int = 3

    # SFT specific Training Configuration
    sft_num_train_epochs: int = 1
    # 5e-5 is also a safe choice if the higher rate proves unstable.
    sft_learning_rate: float = 2e-4
    sft_max_tokens_length: int = 512
    # pack similar examples together for avoiding max_length padding inefficiences.
    # please note examples will be concatenated so that
    # max_length is filled with multilpe examples. This is interesting
    # for domain adaptation but not on task resolution
    sft_packing: bool = False
    sft_metrics_to_plot: List[str] = [
        "loss",
        "eval_loss",
        "grad_norm",
        "learning_rate",
        "num_tokens",
        "mean_token_accuracy",
    ]

    # # quantization
    # use_quantization: bool = True
    # quant_load_in_4bit: bool = True
    # quant_bnb_4bit_quant_type: str = "nf4"
    # quant_bnb_4bit_compute_dtype: str = "bfloat16"
    # quant_bnb_4bit_use_double_quant: bool = True

    dpo_beta: float = 0.1
    dpo_max_length: int = 512
    dpo_learning_rate: float = 5e-5
    dpo_num_train_epochs: int = 1
    dpo_metrics_to_plot: List[str] = [
        "loss",
        "eval_loss",
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
        "logits/chosen",
        "logits/rejected",
    ]

    orpo_beta: float = 0.1  # Default beta for ORPO
    orpo_max_length: int = 512
    orpo_learning_rate: float = 5e-6
    orpo_num_train_epochs: int = 1
    orpo_metrics_to_plot: List[str] = [
        "loss",
        "eval_loss",
        "sft_loss",  # SFT (supervised fine-tuning loss)
        "eval_sft_loss",  # Eval SFT
        "odds_ratio_loss",  # odds ratio (ORPO)
        "eval_odds_ratio_loss",  # Eval odds ratio
        "nll_loss",  # Negative log likelihood loss
        "eval_nll_loss",  # Eval NLL loss
        "rewards/chosen",
        "rewards/rejected",
        "rewards/accuracies",
        "rewards/margins",
        "logps/chosen",
        "logps/rejected",
        "logits/chosen",
        "logits/rejected",
    ]


def config_logger(
    log_level="DEBUG",
    log_retention="7 days",
    log_rotation="00:00",
    stderr_log_level="INFO",
    activate_global_exception_handler=True,
):
    """
    Configures the logger with the library name, log level, retention, and rotation.

    :param log_level: Log level for file output (DEBUG, INFO, WARNING, ERROR, CRITICAL).
    :param log_retention: Log file retention (e.g. '7 days', '30 days').
    :param log_rotation: Log file rotation (e.g. '00:00' for midnight).
    :param stderr_log_level: Log level for the console (DEBUG, INFO, WARNING, ERROR, CRITICAL). If None, disables console output.
    :param activate_global_exception_handler: If True, sets up a global exception handler to log uncaught exceptions.
    :return: None

    Example 1: With console output enabled (default 'INFO' level for console):
    ```python
    config_logger(log_level="DEBUG", log_retention="7 days", log_rotation="00:00", stderr_log_level="INFO")
    ```

    Example 2: Without console output (setting `stderr_log_level=None`):
    ```python
    config_logger(log_level="DEBUG", log_retention="7 days", log_rotation="00:00", stderr_log_level=None)
    ```
    """

    # Logs directory (next to main.py)

    logs_dir = Path(settings.logs_storage_folder)
    logs_dir.mkdir(parents=True, exist_ok=True)

    # Log file path (one log per day)
    date_suffix = datetime.datetime.now().strftime("%Y-%m-%d")
    log_file_path = os.path.join(logs_dir, f"{settings.library_name}_{date_suffix}.log")

    # Remove previous handlers (by default Loguru writes to console)
    logger.remove()

    # Log to file (daily, rotation at midnight)
    logger.add(
        str(log_file_path),
        rotation=log_rotation,  # Rotation according to the parameter
        retention=log_retention,  # Retention according to the parameter
        compression="zip",  # Compress old logs
        level=log_level,  # Log level for the file according to the parameter
        format="{time:YYYY-MM-DD HH:mm:ss} | {level: <8} | {message} | Module: {module} | Function: {function}",
    )

    # If stderr_log_level is not None, add console output
    if stderr_log_level is not None:
        logger.add(
            sys.stderr,
            level=stderr_log_level,  # Log level for console according to the parameter
            format="<green>{time:HH:mm:ss}</green> | <level>{level: <8}</level> | {message}",
        )

    if activate_global_exception_handler:
        register_global_exception_handler()

    logger.info(
        f"Log configured: {log_file_path} | Log level: {log_level} | Retention: {log_retention} | Rotation: {log_rotation}"
    )
    logger.info(f" version: {settings.version}")


def register_global_exception_handler():
    """
    Set up a global exception handler to log uncaught exceptions.
    This handler will log the exception type, value, and traceback.
    It will also allow the program to exit gracefully on KeyboardInterrupt (Ctrl+C).
    """

    def exception_handler(exc_type, exc_value, exc_traceback):
        if issubclass(exc_type, KeyboardInterrupt):
            # Allow program to exit on Ctrl+C without stacktrace
            sys.__excepthook__(exc_type, exc_value, exc_traceback)
            return
        logger.opt(exception=(exc_type, exc_value, exc_traceback)).error(
            "Uncaught exception"
        )

    sys.excepthook = exception_handler


# Load settings
settings = Settings()

# Remove default logger
logger.remove()
