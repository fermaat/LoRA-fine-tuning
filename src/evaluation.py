import evaluate
import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from cfg import config_logger, logger, settings

config_logger(log_level=settings.log_level)

if torch.backends.mps.is_available():
    device = torch.device("mps")
    logger.info("MPS (Metal Performance Shaders) is available and will be used.")
else:
    device = torch.device("cpu")
    logger.info("MPS is not available, falling back to CPU.")

logger.info(f"Loading tokenizer and model: {settings.default_model_id}...")
tokenizer = AutoTokenizer.from_pretrained(settings.default_model_id)

# Gemma models are often in bfloat16, which MPS supports.
model = AutoModelForCausalLM.from_pretrained(
    settings.default_model_id,
    torch_dtype=torch.bfloat16,  # Use bfloat16 for Gemma and MPS
    device_map="auto",  # This might automatically use MPS, but we'll explicitly move it too
)
model.to(device)  # Explicitly move model to MPS device
model.eval()

logger.info(
    f"Loading dataset: {settings.default_dataset_name}, split: {settings.default_evaluation_dataset_split}..."
)

dataset = load_dataset(
    settings.default_dataset_name, split=settings.default_evaluation_dataset_split
)
logger.info(f"Dataset loaded. Number of examples: {len(dataset)}")

logger.info("Loading evaluation metrics (BLEU, ROUGE, BERTScore)...")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
try:
    bertscore_metric = evaluate.load("bertscore")
except Exception as e:
    logger.warning(
        f"Warning: Could not load BERTScore. Ensure 'sentence-transformers' is installed. Error: {e}"
    )
    bertscore_metric = None


def generate_response(prompts):
    # Ensure pad_token is set for batch inference
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token  # A common fallback for Gemma

    inputs = tokenizer(prompts, return_tensors="pt", padding=True, truncation=True).to(
        device
    )

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=settings.max_new_tokens_evaluation,
            num_beams=1,  # For faster, deterministic generation
            do_sample=False,  # Disable sampling
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
        )

    decoded_responses = []
    for i, output in enumerate(outputs):
        # Decode only the newly generated tokens
        generated_text = tokenizer.decode(
            output[inputs.input_ids.shape[1] :], skip_special_tokens=True
        )
        decoded_responses.append(generated_text)
    return decoded_responses


logger.info("Performing inference and collecting responses...")
predictions = []
references = []

# Iterate over the dataset in batches
for i in tqdm(
    range(0, len(dataset), settings.evaluation_batch_size), desc="Generating responses"
):
    batch_indices = list(
        range(i, min(i + settings.evaluation_batch_size, len(dataset)))
    )

    batch_data = dataset.select(batch_indices)

    batch_prompts = batch_data["prompt"]
    batch_references = [item[0]["content"] for item in batch_data["chosen"]]

    generated_texts = generate_response(batch_prompts)

    predictions.extend(generated_texts)
    references.extend(batch_references)

logger.info("Inference complete.")
logger.info(f"Total generated responses: {len(predictions)}")
logger.info(f"Total references collected: {len(references)}")


logger.info("\nCalculating evaluation metrics...")

# BLEU
bleu_results = bleu_metric.compute(
    predictions=predictions, references=[[ref] for ref in references]
)
logger.info(f"BLEU: {bleu_results}")

# ROUGE
rouge_results = rouge_metric.compute(predictions=predictions, references=references)
logger.info(f"ROUGE: {rouge_results}")

# BERTScore
if bertscore_metric:
    logger.info("Calculating BERTScore (this might take some time)...")
    bertscore_results = bertscore_metric.compute(
        predictions=predictions, references=references, lang="en"
    )
    avg_bertscore_f1 = sum(bertscore_results["f1"]) / len(bertscore_results["f1"])
    logger.info(f"BERTScore (Average F1): {avg_bertscore_f1}")

logger.info("Evaluation complete!")

logger.info("\n--- Examples (Prediction vs. Reference) ---")
for i in range(min(5, len(predictions))):
    logger.info(f"\n--- Example {i + 1} ---")
    logger.info(f"Prompt: {dataset[i]['prompt']}")
    logger.info(f"Reference (Chosen): {references[i]}")
    logger.info(f"Model Prediction: {predictions[i]}")
    logger.info("-" * 30)
