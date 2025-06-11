import torch

from cfg import config_logger, logger, settings

config_logger(log_level=settings.log_level)


def setup_device() -> torch.device:
    """
    Set up and return the appropriate device for model inference.

    Returns:
        torch.device: The device to use (MPS if available, otherwise CPU)
    """
    if torch.backends.mps.is_available():
        device = torch.device("mps")
        logger.info("MPS (Metal Performance Shaders) is available and will be used.")
    else:
        device = torch.device("cpu")
        logger.info("MPS is not available, falling back to CPU.")

    return device


def generate_response(prompts, tokenizer, model, device, max_new_tokens):
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            num_beams=1,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded = [
        tokenizer.decode(output[inputs.input_ids.shape[1] :], skip_special_tokens=True)
        for output in outputs
    ]
    return decoded


def save_predictions(preds, refs, filename="predictions.txt"):
    with open(filename, "w") as f:
        for i, (p, r) in enumerate(zip(preds, refs)):
            f.write(f"Example {i + 1}\n")
            f.write(f"Prediction: {p}\n")
            f.write(f"Reference: {r}\n")
            f.write("-" * 40 + "\n")
