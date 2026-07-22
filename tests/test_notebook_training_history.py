import ast
import json
import unittest
from pathlib import Path
from types import SimpleNamespace


NOTEBOOK = (
    Path(__file__).resolve().parents[1]
    / "notebooks"
    / "lora_sft_smollm_greentext.ipynb"
)


class NotebookTrainingHistoryTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        notebook = json.loads(NOTEBOOK.read_text())
        cls.code_cells = [
            "".join(cell["source"])
            for cell in notebook["cells"]
            if cell["cell_type"] == "code"
        ]

    def test_evaluation_does_not_overwrite_training_trainer(self):
        training_cell = next(
            source
            for source in self.code_cells
            if "# Initialize the SFTTrainer" in source
        )
        evaluation_cell = next(
            source for source in self.code_cells if "eval_results =" in source
        )

        self.assertIn("trainer = SFTTrainer(", training_cell)
        self.assertIn("eval_trainer = SFTTrainer(", evaluation_cell)
        self.assertIn("eval_results = eval_trainer.evaluate()", evaluation_cell)

    def test_perplexity_uses_current_training_history(self):
        plotting_cell = next(
            source
            for source in self.code_cells
            if "Training Perplexity — LoRA r=16" in source
        )

        self.assertNotIn("wandb.Api", plotting_cell)
        self.assertNotIn("cb5330-new-york-university", plotting_cell)
        self.assertIn("s16, l16 = loss_curve(trainer)", plotting_cell)

        tree = ast.parse(plotting_cell)
        loss_curve_node = next(
            node
            for node in tree.body
            if isinstance(node, ast.FunctionDef) and node.name == "loss_curve"
        )
        namespace = {}
        exec(
            compile(
                ast.Module(body=[loss_curve_node], type_ignores=[]),
                filename="<notebook-loss-curve>",
                mode="exec",
            ),
            namespace,
        )
        trainer = SimpleNamespace(
            state=SimpleNamespace(
                log_history=[
                    {"step": 1, "loss": 3.0},
                    {"step": 2, "eval_loss": 2.5},
                    {"step": 3, "loss": 2.0},
                ]
            )
        )

        self.assertEqual(namespace["loss_curve"](trainer), ([1, 3], [3.0, 2.0]))


if __name__ == "__main__":
    unittest.main()
