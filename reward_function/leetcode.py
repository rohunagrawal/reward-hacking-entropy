
import re
import ast

# Reference: https://github.com/allenai/olmes/blob/main/oe_eval/tasks/oe_eval_tasks/deepseek_leetcode.py
class LeetCode:
    def __init__(self,  sandbox_fusion_url: str = None):
        self.sandbox_fusion_url = sandbox_fusion_url

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
