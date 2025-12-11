
import ast
import logging
from typing import Optional

import tinker
from tinker import types
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)

# Reference: https://github.com/allenai/olmes/blob/main/oe_eval/tasks/oe_eval_tasks/deepseek_leetcode.py
class LeetCode:
    def __init__(
        self,
        sandbox_fusion_url: str = None,
        g_judge_model_name: Optional[str] = None,
        g_judge_model_path: Optional[str] = None,
        g_judge_base_url: Optional[str] = None,
        g_judge_temperature: float = 0.0,
        g_judge_max_tokens: int = 16,
    ):
        self.sandbox_fusion_url = sandbox_fusion_url
        self.g_judge_model_name = g_judge_model_name
        self.g_judge_model_path = g_judge_model_path
        self.g_judge_base_url = g_judge_base_url
        self.g_judge_temperature = g_judge_temperature
        self.g_judge_max_tokens = g_judge_max_tokens

        self._judge_tokenizer = None
        self._judge_sampling_client = None
        self._judge_sampling_params = None

    def _markdown_code_extraction(self, completion: str) -> str:
        if "```python" in completion:
            solution = completion.split("```python")[-1].split("```")[0]
        elif "```" in completion:
            # Handle cases like ```\ncode\n```
            parts = completion.split("```")
            if len(parts) >= 2:
                solution = parts[1]
                # Remove potential language specifier like 'python\n'
                if "\n" in solution:
                    first_line, rest = solution.split("\n", 1)
                    if first_line.strip().isalpha():  # Simple check for language name
                        solution = rest
        else:
            solution = None
        return solution

    def is_compilable(self, completion: str) -> float:
        """
        Check if the code is compilable (syntax check only).
        """
        try:
            ast.parse(completion)
            return 1.0
        except SyntaxError:
            return 0.0
        except Exception:
            return 0.0

    def _init_llm_judge(self):
        """
        Lazily create the LLM judge client/tokenizer when g_type == "llm_judge".
        """
        if self.g_judge_model_path is None and self.g_judge_model_name is None:
            raise ValueError("g_type='llm_judge' requires g_judge_model_path or g_judge_model_name to be set.")

        if self._judge_sampling_client is not None:
            return

        service_client = tinker.ServiceClient(base_url=self.g_judge_base_url)
        # Prefer explicit model_path if provided; otherwise treat name as base_model
        if self.g_judge_model_path:
            self._judge_sampling_client = service_client.create_sampling_client(model_path=self.g_judge_model_path)
        elif self.g_judge_model_name:
            # Use base_model to allow public HF names
            self._judge_sampling_client = service_client.create_sampling_client(base_model=self.g_judge_model_name)
        else:
            raise ValueError(
                "LLM judge requires either g_judge_model_path or g_judge_model_name."
            )
        self._judge_sampling_params = types.SamplingParams(
            max_tokens=self.g_judge_max_tokens,
            temperature=self.g_judge_temperature,
            stop=["<eos>", "<|eot_id|>", "</s>", "###", "Answer:"],
        )
        self._judge_tokenizer = AutoTokenizer.from_pretrained(self.g_judge_model_name)

    def _llm_judge_score(self, code_solution: str, problem_description: str | None = None) -> float:
        """
        Ask an LLM to judge whether the code is likely to compile/run.
        Returns 1.0 if the model responds with a leading '1'/'yes', else 0.0.
        """
        try:
            self._init_llm_judge()
        except Exception as e:
            logger.error(f"Failed to initialize LLM judge: {e}")
            return 0.0

        prompt = (
            "You are a strict coding evaluator. "
            "Given the problem description and the Python code, reply with EXACTLY one character: "
            "'1' if the code correctly solves the problem for its described inputs/outputs; "
            "'0' if the code is likely incorrect or incomplete. "
            "Do not explain. Only output a single digit 1 or 0.\n\n"
        )
        if problem_description:
            prompt += f"Problem:\n{problem_description}\n\n"
        prompt += f"Code:\n{code_solution}\n\nAnswer:"

        input_ids = self._judge_tokenizer.encode(prompt, add_special_tokens=False)
        model_input = types.ModelInput.from_ints(tokens=input_ids)

        try:
            future = self._judge_sampling_client.sample(
                prompt=model_input,
                num_samples=1,
                sampling_params=self._judge_sampling_params,
            )
            sample_result = future.result()
            sampled_tokens = sample_result.sequences[0].tokens
            text = self._judge_tokenizer.decode(sampled_tokens, skip_special_tokens=True).strip()
            text_lower = text.lower()
            if text_lower.startswith("1") or text_lower.startswith("yes"):
                return 1.0
            if text_lower.startswith("0") or text_lower.startswith("no"):
                return 0.0
            # Fallback: check for digits anywhere
            return 1.0 if "1" in text_lower and "0" not in text_lower else 0.0
        except Exception as e:
            logger.error(f"LLM judge scoring failed: {e}")
            return 0.0

    # Reference: https://github.com/volcengine/verl/blob/main/verl/utils/reward_score/__init__.py#L74
    def check_correctness(self, completion, test_inputs_outputs, import_prefix, concurrent_semaphore=None, memory_limit_mb=None):
        if self.sandbox_fusion_url:
            from . import sandbox_fusion

            # score: pass rate (passed / total_test_case_num)
            # final_metadata: some string description for the error. e.g. [{"error": "Invalid test_cases JSON format"}]
            score, final_metadata = sandbox_fusion.compute_score(
                self.sandbox_fusion_url, concurrent_semaphore, memory_limit_mb, completion, test_inputs_outputs,
                import_prefix, continuous=True
            )
        else:
            # If no sandbox URL is provided, fall back to prime_code or raise error
            from . import prime_code

            # Assuming prime_code doesn't need the URL
            score, final_metadata = prime_code.compute_score(completion, test_inputs_outputs, import_prefix, continuous=True)
        return score, final_metadata

    def process_code_result(self, res: dict, g_type: str) -> dict:
        """
        res:
        {
            "query": entry["query"],
            "difficulty": "Medium",
            "import_prefix": "import...",
            "test_inputs_outputs": {        # reformatted for SandboxFusion
                "inputs": ["in1", "in2", ...],
                "outputs": ["out1", "out2", ...],
                "fn_name": "Solution().maxArea"
            },
            "completion": model_generated_code_string,
        }
        """
        continuation = res["completion"]

        # Extract code from the continuation
        code_solution = self._markdown_code_extraction(continuation)

        # g: whether compilable
        if code_solution is None:
            res["g_score"] = 0.0
        else:
            if g_type == "is_compilable":
                res["g_score"] = self.is_compilable(code_solution)
            elif g_type == "llm_judge":
                res["g_score"] = self._llm_judge_score(code_solution, res.get("query"))
            else:
                # TODO: may change to llm as a judge or other weak reward models
                raise ValueError(f"Unsupported g_type: {g_type}")

        # f: correctness
        score, final_metadata = self.check_correctness(
            continuation,
            res["test_inputs_outputs"],
            res["import_prefix"]
        )
        res["f_score"] = score      # correctness_reward
        res["correctness_metadata"] = final_metadata
        
        return res
