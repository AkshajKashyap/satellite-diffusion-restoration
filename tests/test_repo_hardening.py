import json
from pathlib import Path


def test_docker_and_ci_files_exist():
    assert Path("Dockerfile").is_file()
    assert Path(".dockerignore").is_file()
    assert Path(".github/workflows/ci.yml").is_file()


def test_makefile_contains_expected_commands():
    makefile = Path("Makefile").read_text(encoding="utf-8")

    for command in [
        "install:",
        "test:",
        "lint:",
        "smoke-data:",
        "train-unet-synthetic:",
        "eval-unet-eurosat:",
        "api:",
        "streamlit:",
    ]:
        assert command in makefile


def test_final_results_json_is_valid():
    payload = json.loads(Path("reports/final_results.json").read_text(encoding="utf-8"))

    assert payload["project"] == "satellite-diffusion-restoration"
    assert "eurosat" in payload
    assert "residual_unet" in payload["eurosat"]
    assert payload["eurosat"]["residual_unet"]["psnr_db"] > payload["eurosat"]["corrupted_input"]["psnr_db"]


def test_final_reports_exist():
    for report_path in [
        "reports/model_card.md",
        "reports/project_summary.md",
        "reports/interview_notes.md",
        "reports/repo_health_check.md",
        "reports/final_results.json",
    ]:
        assert Path(report_path).is_file()
