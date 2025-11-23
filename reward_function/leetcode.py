
import re
import ast

# Reference: https://github.com/allenai/olmes/blob/main/oe_eval/tasks/oe_eval_tasks/deepseek_leetcode.py
class LeetCode:
    def __init__(self,  sandbox_fusion_url: str = None):
        self.sandbox_fusion_url = sandbox_fusion_url

    def _fallback_code_extraction(self, continuation: str) -> str:
        codelist = re.split("\ndef|\nclass|\nif|\n#|\nprint", continuation)
        if len(codelist) > 0:
            return codelist[0]
        else:
            return ""

    def _markdown_code_extraction(self, continuation: str) -> str:
        p_code = re.compile(r"```python\n?(.*?)\n?```", flags=re.DOTALL)
        code_blocks = p_code.findall(continuation)
        if len(code_blocks) > 0:
            return code_blocks[0]
        else:
            return self._fallback_code_extraction(continuation)

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

    def process_code_result(self, res: dict) -> dict:
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
        if "```python" in continuation:
            code_solution = self._markdown_code_extraction(continuation)
        else:
            code_solution = self._fallback_code_extraction(continuation)

        # g: whether compilable
        res["is_compilable_reward"] = self.is_compilable(code_solution)

        # f: correctness
        score, final_metadata = self.check_correctness(
            code_solution,
            res["test_inputs_outputs"],
            res["import_prefix"]
        )
        res["correctness_reward"] = score
        res["correctness_metadata"] = final_metadata

        return res
