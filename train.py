import logging
import time
import random
import json
import os
from concurrent.futures import Future
from typing import List, Dict, Any

import chz
import datasets
import tinker
import torch
from tinker import types
from tinker.types.tensor_data import TensorData
from transformers import AutoTokenizer
import wandb
import numpy as np

from reward_function.leetcode import LeetCode

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
    base_url: str | None = None
    log_path: str = "outputs/rl-leetcode/llama-3.2-1b"
    model_name: str = "meta-llama/Llama-3.2-1B"
    epochs: int = 5
    batch_size: int = 64
    group_size: int = 16
    learning_rate: float = 1e-5
    max_length: int = 4096
    lora_rank: int = 16
    save_every: int = 1
    max_tokens: int = 1024
    sandbox_url: str = "http://localhost:8000/run_code"
    dataset_path: str = "data/leetcode"
    filter_difficulty: str | None = None # e.g., "Easy", "Medium", "Hard"
    max_train_samples: int | None = 600     # Based on split size of leetcode: Easy: 638 | Medium: 1397 | Hard: 606. Keep each bin to have the same sample size
    seed: int = 42
    g_type: str = "is_compilable"  # TODO type of g reward to use, may change to llm as a judge

    use_wandb: bool = True
    wandb_entity: str | None = "lorena-yantianyi1020"
    wandb_project: str | None = "COMS4705-reward-hacking-entropy"

    @chz.init_property
    def wandb_run_name(self) -> str:
        return self.log_path.split("outputs/")[-1]

def set_seed(seed: int):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

class SimpleRenderer:
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer
        self.stop_sequences = ["<end_of_turn>", "<eos>", "<|eot_id|>"] # Default for Gemma-2 and Llama-3
        
        if not self.tokenizer.chat_template:
            # Default Llama-3 style template
            self.tokenizer.chat_template = (
                "{% for message in messages %}"
                "<|start_header_id|>{{ message['role'] }}<|end_header_id|>\n\n"
                "{{ message['content'] }}<|eot_id|>"
                "{% endfor %}"
                "{% if add_generation_prompt %}"
                "<|start_header_id|>assistant<|end_header_id|>\n\n"
                "{% endif %}"
            )

    def build_generation_prompt(self, messages: List[Dict[str, str]]) -> types.ModelInput:
        prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        tokens = self.tokenizer.encode(prompt, add_special_tokens=False)
        return types.ModelInput.from_ints(tokens=tokens)

    def parse_response(self, tokens: List[int]) -> tuple[Dict[str, str], Any]:
        text = self.tokenizer.decode(tokens, skip_special_tokens=True)
        return {"role": "assistant", "content": text}, None

    def get_stop_sequences(self) -> List[str]:
        return self.stop_sequences


def log_metrics(metrics: Dict[str, float], step: int, log_dir: str):
    os.makedirs(log_dir, exist_ok=True)
    metrics["step"] = step
    metrics["timestamp"] = time.time()
    
    # Print to console
    logger.info(f"Step {step}: {json.dumps(metrics)}")
    
    # Append to jsonl
    with open(os.path.join(log_dir, "metrics.jsonl"), "a") as f:
        f.write(json.dumps(metrics) + "\n")
    # Also log to wandb if initialized
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.log(metrics, step=step)
    except Exception as e:
        logger.warning(f"wandb logging failed: {e}")


def main(config: Config):
    logger.info(f"Starting training with config: {config}")
    set_seed(config.seed)

    # Get tokenizer and renderer
    logger.info(f"Loading tokenizer for {config.model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(config.model_name)
    renderer = SimpleRenderer(tokenizer)

    # Load LeetCode dataset
    logger.info(f"Loading dataset from {config.dataset_path}...")
    try:
        dataset = datasets.load_from_disk(config.dataset_path)
    except FileNotFoundError:
        logger.error(f"Dataset not found at {config.dataset_path}. Please run prep_dataset.py first.")
        return
    
    if isinstance(dataset, datasets.DatasetDict):
        train_dataset = dataset["train"]
    else:
        train_dataset = dataset
    
    # Filter by difficulty if specified
    if config.filter_difficulty:
        logger.info(f"Filtering dataset for difficulty: {config.filter_difficulty}")
        train_dataset = train_dataset.filter(lambda x: x["difficulty"] == config.filter_difficulty)
    
    # Limit training samples if specified
    if config.max_train_samples is not None and len(train_dataset) > config.max_train_samples:
        logger.info(f"Limiting training dataset from {len(train_dataset)} to {config.max_train_samples} samples")
        train_dataset = train_dataset.select(range(config.max_train_samples))
    
    # Init wandb
    if config.use_wandb:
        try:
            wandb.init(
                entity=config.wandb_entity,
                project=config.wandb_project,
                name=config.wandb_run_name,
            config=vars(config),
            )
            logger.info(f"Initialized wandb run: {wandb.run}")
        except Exception as e:
            logger.warning(f"Failed to initialize wandb: {e}")
    
    logger.info(f"Dataset loaded: {len(train_dataset)} examples")

    # Initialize Reward Function
    leetcode_eval = LeetCode(sandbox_fusion_url=config.sandbox_url)

    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Training for {config.epochs} epochs, {n_train_batches} batches per epoch")

    # Setup training client
    service_client = tinker.ServiceClient(base_url=config.base_url)
    
    training_client = service_client.create_lora_training_client(
        base_model=config.model_name, rank=config.lora_rank
    )

    # Check for existing checkpoint (simple check)
    if os.path.exists(os.path.join(config.log_path, "latest_checkpoint.txt")):
        with open(os.path.join(config.log_path, "latest_checkpoint.txt"), "r") as f:
            latest_ckpt_path = f.read().strip()
        training_client.load_state(latest_ckpt_path)
        logger.info(f"Resuming from checkpoint: {latest_ckpt_path}")
    else:
        logger.info("No checkpoint found, starting fresh training")
    # Note: Resume logic simplified as we removed checkpoint_utils. 
    # If needed, one would load state from a file and pass to create_training_client_from_state.

    sampling_params = tinker.types.SamplingParams(
        max_tokens=config.max_tokens,
        stop=renderer.get_stop_sequences(),
        temperature=1.0, # Ensure exploration for entropy
    )
    # Optimizer step
    adam_params = types.AdamParams(
        learning_rate=config.learning_rate, beta1=0.9, beta2=0.95, eps=1e-8
    )

    #  Main training loop
    step = 0
    for epoch in range(0, config.epochs):
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        for batch_idx in range(0, n_train_batches):
            t_start = time.time()
            metrics: dict[str, float] = {
                "progress/epoch": epoch,
                "progress/step": step,
                "optim/lr": config.learning_rate,
                "progress/done_frac": (epoch * n_train_batches + batch_idx + 1) / (config.epochs * n_train_batches),
            }

            # Save checkpoint
            if step % config.save_every == 0 and step > 0:
                saved_path = training_client.save_state(f"{config.log_path.replace('/', '_')}_checkpoint_{step:06d}").result().path
                logger.info(f"Saved checkpoint at step {step} to {saved_path}")
                # Output to txt file
                with open(os.path.join(config.log_path, f"checkpoint_names.txt"), "a") as f:
                    f.write(saved_path + "\n")
                with open(os.path.join(config.log_path, "latest_checkpoint.txt"), "w") as f:
                    f.write(saved_path + "\n")
            # Get training batch
            batch_start = batch_idx * config.batch_size
            batch_end = min((batch_idx + 1) * config.batch_size, len(train_dataset))
            batch_rows = train_dataset.select(range(batch_start, batch_end))

            # Get sampling client with current weights
            sampling_path = training_client.save_weights_for_sampler(name=f"{step:06d}").result().path
            sampling_client = service_client.create_sampling_client(model_path=sampling_path)

            training_datums: list[types.Datum] = []
            batch_rewards: list[float] = []
            batch_correctness: list[float] = [] # f
            batch_compilable: list[float] = [] # g
            batch_entropies: list[float] = []
            
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
                sample_futures: list[Future[types.SampleResponse]] = []
                for _ in range(config.group_size):
                    sample_futures.append(
                        sampling_client.sample(
                            prompt=model_input,
                            num_samples=1,
                            sampling_params=sampling_params,
                        )
                    )

                batch_futures.append(sample_futures)
                batch_prompts.append(prompt_tokens)

            # Process results
            for i, (sample_futures, prompt_tokens) in enumerate(zip(batch_futures, batch_prompts)):
                row = batch_rows[i]
                
                group_rewards: list[float] = []
                group_tokens: list[list[int]] = []
                group_logprobs: list[list[float]] = []
                group_ob_lens: list[int] = []
                
                # Metrics for this group
                group_correctness = []
                group_compilable = []
                group_entropy = []

                for future in sample_futures:
                    sample_result = future.result()
                    sampled_tokens = sample_result.sequences[0].tokens
                    sampled_logprobs = sample_result.sequences[0].logprobs
                    assert sampled_logprobs is not None

                    all_tokens = prompt_tokens + sampled_tokens
                    group_tokens.append(all_tokens)
                    group_ob_lens.append(len(prompt_tokens) - 1)
                    group_logprobs.append(sampled_logprobs)
                    
                    # Calculate entropy (approximate from logprobs)
                    avg_nll = -sum(sampled_logprobs) / len(sampled_logprobs) if sampled_logprobs else 0.0
                    group_entropy.append(avg_nll)

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
                        res = leetcode_eval.process_code_result(res, g_type=config.g_type)
                        f_score = res.get("f_score", 0.0)
                        g_score = res.get("g_score", 0.0)
                    except Exception as e:
                        logger.error(f"Error processing code result: {e}")
                        f_score = 0.0
                        g_score = 0.0

                    # Total reward (can be weighted combination, for now just g)
                    # reward = f_score + 0.5 * g_score
                    reward = g_score
                    
                    group_rewards.append(reward)
                    group_correctness.append(f_score)
                    group_compilable.append(g_score)

                # Normalize rewards (advantage)
                mean_reward = sum(group_rewards) / len(group_rewards)
                advantages = [r - mean_reward for r in group_rewards]
                
                batch_rewards.append(mean_reward)
                batch_correctness.extend(group_correctness)
                batch_compilable.extend(group_compilable)
                batch_entropies.extend(group_entropy)

                # check if all advantages are zero
                if all(advantage == 0.0 for advantage in advantages):
                    continue

                for tokens, logprob, advantage, ob_len in zip(
                    group_tokens, group_logprobs, advantages, group_ob_lens
                ):
                    input_tokens = tokens[:-1]
                    input_tokens = [int(token) for token in input_tokens]
                    target_tokens = tokens[1:]
                    all_logprobs = [0.0] * ob_len + logprob
                    all_advantages = [0.0] * ob_len + [advantage] * (len(input_tokens) - ob_len)
                    
                    datum = types.Datum(
                        model_input=types.ModelInput.from_ints(tokens=input_tokens),
                        loss_fn_inputs={
                            "target_tokens": TensorData.from_torch(torch.tensor(target_tokens)),
                            "logprobs": TensorData.from_torch(torch.tensor(all_logprobs)),
                            "advantages": TensorData.from_torch(torch.tensor(all_advantages)),
                        },
                    )
                    training_datums.append(datum)

            # Training step (after processing all problems in batch)
            if training_datums:
                fwd_bwd_future = training_client.forward_backward(
                    training_datums, loss_fn="importance_sampling"
                )
                optim_step_future = training_client.optim_step(adam_params)
                _fwd_bwd_result = fwd_bwd_future.result()
                _optim_result = optim_step_future.result()
            else:
                logger.warning("No training datums generated for this batch (all zero advantage?)")

            # Log metrics (once per batch)
            metrics["time/total"] = time.time() - t_start
            metrics["reward/mean"] = sum(batch_rewards) / len(batch_rewards) if batch_rewards else 0.0
            metrics["reward/correctness_f"] = sum(batch_correctness) / len(batch_correctness) if batch_correctness else 0.0
            metrics["reward/compilable_g"] = sum(batch_compilable) / len(batch_compilable) if batch_compilable else 0.0
            metrics["reward/f_minus_g"] = np.mean(np.array(batch_correctness) - np.array(batch_compilable)).item() if batch_correctness and batch_compilable else 0.0
            metrics["policy/entropy"] = sum(batch_entropies) / len(batch_entropies) if batch_entropies else 0.0
            
            log_metrics(metrics, step=step, log_dir=config.log_path)
            
            step += 1

    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    final_ckpt_path = training_client.save_state(f"{config.log_path.replace('/', '_')}_final").result().path
    with open(os.path.join(config.log_path, f"checkpoint_names.txt"), "a") as f:
        f.write(final_ckpt_path + "\n")
    logger.info("Training completed")
    # Finish wandb run if it was started
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception as e:
        logger.warning(f"wandb finish failed: {e}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
