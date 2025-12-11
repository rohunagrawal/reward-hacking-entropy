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

TRAIN_STATE_FILENAME = "training_state.json"


@chz.chz
class TrainingConfig:
    base_url: str | None = None
    log_path: str = "outputs/rl-leetcode/llama-3.2-3b"
    model_name: str = "meta-llama/Llama-3.2-3B"
    reward_type: str | None = "g"     # options: g, f_g, 100f_g
    epochs: int = 5
    batch_size: int = 64
    group_size: int = 8
    learning_rate: float = 1e-5
    max_length: int = 4096
    lora_rank: int = 16
    save_every: int = 1
    max_tokens: int = 1024
    sandbox_url: str = "http://localhost:8000/run_code"
    dataset_path: str = "data/leetcode"
    filter_by: str = "difficulty" # difficulty, init_rollout_entropy
    filter_difficulty: str | None = None # e.g., "Easy", "Medium", "Hard"
    filter_entropy_bin: int | None = None  # e.g., 1, 2, 3 when using init_rollout_entropy
    max_train_samples: int | None = 600     # Based on split size of leetcode: Easy: 638 | Medium: 1397 | Hard: 606. Keep each bin to have the same sample size
    seed: int = 42
    g_type: str = "llm_judge"  # use LLM-as-judge by default; change to "is_compilable" for syntax-only
    g_judge_model_name: str | None = "meta-llama/Llama-3.2-3B"
    g_judge_model_path: str | None = None  # if using a tinker:// URI for judge
    g_judge_base_url: str | None = None
    g_judge_temperature: float = 0.0
    g_judge_max_tokens: int = 8

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


def load_training_state(log_path: str):
    state_path = os.path.join(log_path, TRAIN_STATE_FILENAME)
    if not os.path.exists(state_path):
        return None
    try:
        with open(state_path, "r") as f:
            return json.load(f)
    except Exception as e:
        logger.warning(f"Failed to load training state from {state_path}: {e}")
        return None


def save_training_state(log_path: str, epoch: int, batch_idx: int, step: int):
    os.makedirs(log_path, exist_ok=True)
    state_path = os.path.join(log_path, TRAIN_STATE_FILENAME)
    state = {
        "epoch": epoch,
        "batch_idx": batch_idx,
        "step": step,
    }
    try:
        with open(state_path, "w") as f:
            json.dump(state, f)
    except Exception as e:
        logger.warning(f"Failed to save training state to {state_path}: {e}")


def main(config: TrainingConfig):
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

    # Filter dataset
    if config.filter_by == "difficulty":
        if config.filter_difficulty is None:
            logger.info("filter_by=difficulty but no filter_difficulty provided; using full dataset")
        else:
            train_dataset = train_dataset.filter(lambda x: x["difficulty"] == config.filter_difficulty)
            logger.info(f"Filtering dataset for difficulty: {config.filter_difficulty} -> {len(train_dataset)} samples")
    elif config.filter_by == "init_rollout_entropy":
        if "entropy_bin" not in train_dataset.column_names:
            logger.error("Dataset missing 'entropy_bin'. Ensure you are using the entropy-prepared dataset.")
            return
        if config.filter_entropy_bin is None:
            logger.info("filter_by=init_rollout_entropy but no filter_entropy_bin provided; using full dataset")
        else:
            train_dataset = train_dataset.filter(lambda x: x["entropy_bin"] == config.filter_entropy_bin)
            logger.info(f"Filtering dataset for entropy_bin={config.filter_entropy_bin} -> {len(train_dataset)} samples")
    else:
        logger.warning(f"Unknown filter_by value '{config.filter_by}', proceeding without filtering.")

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
    leetcode_eval = LeetCode(
        sandbox_fusion_url=config.sandbox_url,
        g_judge_model_name=config.g_judge_model_name,
        g_judge_model_path=getattr(config, "g_judge_model_path", None),
        g_judge_base_url=config.g_judge_base_url,
        g_judge_temperature=config.g_judge_temperature,
        g_judge_max_tokens=config.g_judge_max_tokens,
    )

    n_train_batches = len(train_dataset) // config.batch_size
    logger.info(f"Training for {config.epochs} epochs, {n_train_batches} batches per epoch")

    # Load training state if it exists
    start_epoch = 0
    start_batch_idx = 0
    step = 0
    loaded_state = load_training_state(config.log_path)
    if loaded_state is not None:
        start_epoch = int(loaded_state.get("epoch", 0))
        start_batch_idx = int(loaded_state.get("batch_idx", 0))
        step = int(loaded_state.get("step", 0))
        logger.info(f"Resuming from training state: epoch={start_epoch}, batch_idx={start_batch_idx}, step={step}")
    if start_epoch >= config.epochs:
        logger.info("Training already completed according to saved state.")
        return

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

    # Save config
    os.makedirs(config.log_path, exist_ok=True)
    config_json_path = os.path.join(config.log_path, "config.json")
    try:
        with open(config_json_path, "w") as f:
            json.dump(vars(config), f, indent=2)
    except Exception as e:
        logger.warning(f"Failed to save config to {config_json_path}: {e}")

    #  Main training loop (supports resume)
    for epoch in range(start_epoch, config.epochs):
        if epoch > start_epoch:
            start_batch_idx = 0  # reset after first resumed epoch
        logger.info(f"Starting epoch {epoch + 1}/{config.epochs}")
        if start_batch_idx >= n_train_batches:
            continue
        for batch_idx in range(start_batch_idx, n_train_batches):
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

                    # Total reward (default g-only; supports f_g / 100f_g for compatibility)
                    if config.reward_type in (None, "g"):
                        reward = g_score
                    elif config.reward_type == "f_g":
                        reward = f_score + g_score
                    elif config.reward_type == "100f_g":
                        reward = 100 * f_score + g_score
                    else:
                        raise ValueError(f"Unknown reward type: {config.reward_type}")

                    if step == 0 and i < 2:
                        logger.info(
                            f"[debug reward] f_score={f_score}, g_score={g_score}, "
                            f"reward={reward}, g_type={config.g_type}, reward_type={config.reward_type}"
                        )
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
            metrics["reward/f"] = sum(batch_correctness) / len(batch_correctness) if batch_correctness else 0.0
            metrics["reward/g"] = sum(batch_compilable) / len(batch_compilable) if batch_compilable else 0.0
            # metrics["reward/f_minus_g"] = np.mean(np.array(batch_correctness) - np.array(batch_compilable)).item() if batch_correctness and batch_compilable else 0.0
            metrics["reward/g_minus_f"] = np.mean(np.array(batch_compilable) - np.array(batch_correctness)).item() if batch_correctness and batch_compilable else 0.0
            metrics["policy/entropy"] = sum(batch_entropies) / len(batch_entropies) if batch_entropies else 0.0
            
            log_metrics(metrics, step=step, log_dir=config.log_path)
            
            step += 1

            # Save training state for resume
            if batch_idx + 1 >= n_train_batches:
                next_epoch = epoch + 1
                next_batch_idx = 0
            else:
                next_epoch = epoch
                next_batch_idx = batch_idx + 1
            save_training_state(config.log_path, epoch=next_epoch, batch_idx=next_batch_idx, step=step)

    # Save final checkpoint
    logger.info("Saving final checkpoint...")
    final_ckpt_path = training_client.save_state(f"{config.log_path.replace('/', '_')}_final").result().path
    with open(os.path.join(config.log_path, f"checkpoint_names.txt"), "a") as f:
        f.write(final_ckpt_path + "\n")
    # Save final training state
    save_training_state(config.log_path, epoch=config.epochs, batch_idx=0, step=step)
    logger.info("Training completed")
    # Finish wandb run if it was started
    try:
        if getattr(wandb, "run", None) is not None:
            wandb.finish()
    except Exception as e:
        logger.warning(f"wandb finish failed: {e}")


if __name__ == "__main__":
    chz.nested_entrypoint(main)
