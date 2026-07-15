"""系统学习课程的离线验收测试。"""

from examples.learning import real_model_lab, run_lab
from scripts import check_learning_docs


def test_learning_docs_structure_and_links():
    assert check_learning_docs.main() == 0


def test_all_learning_labs_smoke():
    for lab in run_lab.LABS.values():
        lab()


def test_missing_real_model_is_explicitly_skipped(capsys):
    missing_model = check_learning_docs.PROJECT_ROOT / "models" / "__missing_learning.gguf"
    assert not missing_model.exists()
    assert real_model_lab.generation_lab(missing_model) is False
    assert "SKIPPED generation: GGUF not found" in capsys.readouterr().out
