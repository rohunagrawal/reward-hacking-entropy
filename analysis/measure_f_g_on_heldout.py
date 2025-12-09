import logging
import json
import os
from concurrent.futures import Future

import chz
import datasets
import tinker
from tinker import types
from transformers import AutoTokenizer
import numpy as np

from reward_function.leetcode import LeetCode
from train import SimpleRenderer, TrainingConfig

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)
logging.getLogger("httpx").setLevel(logging.WARN)


@chz.chz
class Config:
    dataset_path: str = "data/leetcode"  # Contains held-out dev set; should be prepared with prep_dataset.py
    training_output_dir: str = "outputs/rl-leetcode/llama-3.2-1b/Easy"  # where config, checkpoint_names.txt and latest_checkpoint.txt are stored
    sandbox_url: str = "http://localhost:8000/run_code"
    eval_batch_size: int = 16
    eval_ckpt_num: int = 10


def plot_scores_across_checkpoints(scores, ckpt_names, output_fn: str, y_label: str, title: str):
    import matplotlib.pyplot as plt
    step_names = [name.split("_checkpoint_")[-1] if name != "baseline" else "baseline" for name in ckpt_names]

    plt.figure(figsize=(10, 6))
    # plt.plot(step_names, scores["avg_f"], marker='o', label='Avg F Score')
    # plt.plot(step_names, scores["avg_g"], marker='o', label='Avg G Score')
    plt.plot(step_names, scores["avg_g_minus_f"], marker='o', label='Avg g - f Score')
    plt.xlabel('Step')
    plt.ylabel(y_label)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(output_fn)
    plt.close()

def main(config: Config):
    with open(os.path.join(config.training_output_dir, "config.json"), "r") as f:
        # Remove keys that start with "X_" and "wandb_run_name" (init_property)
        config_dict = json.load(f)
        config_dict = {k: v for k, v in config_dict.items() if not k.startswith("X_") and k != "wandb_run_name"}
        training_config = TrainingConfig(**config_dict)

    # Get tokenizer and renderer
    logger.info(f"Loading tokenizer for {training_config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(training_config.model_name)
    renderer = SimpleRenderer(tokenizer)

    # Load held-out dataset
    logger.info(f"Loading held-out dataset from {config.dataset_path}...")
    try:
        dataset = datasets.load_from_disk(config.dataset_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {config.dataset_path}. Please run prep_dataset.py first.")
        return

    if isinstance(dataset, datasets.DatasetDict):
        dataset = dataset["dev"]
    else:
        dataset = dataset

    # Filter by difficulty if specified
    if training_config.filter_by == "difficulty":
        logger.info(f"Filtering dataset for difficulty: {training_config.filter_difficulty}")
        dataset = dataset.filter(lambda x: x["difficulty"] == training_config.filter_difficulty)

    n_train_batches = len(dataset) // config.eval_batch_size

    # Initialize Reward Function
    leetcode_eval = LeetCode(sandbox_fusion_url=config.sandbox_url)

    # Setup training client
    service_client = tinker.ServiceClient(base_url=training_config.base_url)

    # Read in and iterate through ckpts
    ckpt_paths = ["baseline"]
    with open(os.path.join(config.training_output_dir, "checkpoint_names.txt"), "r") as f:
        ckpt_paths += [line.strip() for line in f.readlines() if line.strip()]
    # Given eval_ckpt_num, sample evenly from ckpt_paths
    if config.eval_ckpt_num < len(ckpt_paths):
        step_size = int(len(ckpt_paths) / config.eval_ckpt_num)
        ckpt_paths = [ckpt_paths[int(i * step_size)] for i in range(config.eval_ckpt_num)]

    sampling_params = tinker.types.SamplingParams(
        max_tokens=training_config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=0.0,  # Use greedy for evaluation
    )
    ckpt_to_avg_f_g_vals = {}
    for ckpt_path in ckpt_paths:
        if ckpt_path == "baseline":
            logger.info(f"Evaluating baseline model...")
            sampling_client = service_client.create_sampling_client(model_name=training_config.model_name)
        else:
            logger.info(f"Evaluating model from checkpoint: {ckpt_path}...")
            sampling_client = service_client.create_sampling_client(model_name=ckpt_path)

        ckpt_f_vals = []
        ckpt_g_vals = []
        ckpt_g_minus_f_vals = []

        for batch_idx in range(0, n_train_batches):
            # Get training batch
            batch_start = batch_idx * config.eval_batch_size
            batch_end = min((batch_idx + 1) * config.eval_batch_size, len(dataset))
            batch_rows = dataset.select(range(batch_start, batch_end))

            batch_futures: list[list[Future[types.SampleResponse]]] = []
            batch_prompts: list[list[int]] = []

            # Prepare prompts
            for i in range(len(batch_rows)):
                row = batch_rows[i]
                query = row["query"]

                # Construct conversation
                convo = [
                    {"role": "user", "content": query},
                ]
                model_input = renderer.build_generation_prompt(convo)
                prompt_tokens = model_input.to_ints()

                # Generate response
                batch_futures.append(sampling_client.sample(
                    prompt=model_input,
                    num_samples=1,
                    sampling_params=sampling_params,
                ))
                batch_prompts.append(prompt_tokens)

            # Process Responses
            for i, (future, prompt_tokens) in enumerate(zip(batch_futures, batch_prompts)):
                row = batch_rows[i]
                sample_result = future.result()
                sampled_tokens = sample_result.sequences[0].tokens
                sampled_logprobs = sample_result.sequences[0].logprobs
                assert sampled_logprobs is not None

                # Calculate entropy
                # avg_nll = -sum(sampled_logprobs) / len(sampled_logprobs) if sampled_logprobs else 0.0

                parsed_message, _ = renderer.parse_response(sampled_tokens)
                completion = parsed_message["content"]

                # Calculate Reward
                res = {
                    "query": row["query"],
                    "difficulty": row["difficulty"],
                    "import_prefix": row["import_prefix"],
                    "test_inputs_outputs": row["test_inputs_outputs"],
                    "completion": completion
                }

                # Call LeetCode reward function
                try:
                    res = leetcode_eval.process_code_result(res, g_type=training_config.g_type)
                    f_score = res.get("f_score", 0.0)
                    g_score = res.get("g_score", 0.0)
                except Exception as e:
                    logger.error(f"Error processing code result: {e}")
                    f_score = 0.0
                    g_score = 0.0

                ckpt_f_vals.append(f_score)
                ckpt_g_vals.append(g_score)
                ckpt_g_minus_f_vals.append(g_score - f_score)

        ckpt_to_avg_f_g_vals[ckpt_path] = {
            "avg_f": float(np.mean(ckpt_f_vals)) if ckpt_f_vals else 0.0,
            "avg_g": float(np.mean(ckpt_g_vals)) if ckpt_g_vals else 0.0,
            "avg_g_minus_f": float(np.mean(ckpt_g_minus_f_vals)) if ckpt_g_minus_f_vals else 0.0,
        }

    # Log final results
    # Save to file
    output_metrics_path = os.path.join(config.training_output_dir, "avg_heldout_f_g_scores.json")
    with open(output_metrics_path, "w") as f:
        json.dump(ckpt_to_avg_f_g_vals, f, indent=4)

    # Plot scores across checkpoints
    plot_scores_across_checkpoints(
        scores=ckpt_to_avg_f_g_vals,
        ckpt_names=list(ckpt_to_avg_f_g_vals.keys()),
        output_fn=os.path.join(config.training_output_dir, "heldout_g_minus_f_scores.png"),
        y_label="Avg g - f Score",
        title="Held-out Avg g - f Scores Across Checkpoints"
    )



if __name__ == "__main__":
    chz.nested_entrypoint(main)
