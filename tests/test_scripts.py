import importlib.util
from pathlib import Path


def test_training_scripts_are_import_safe():
    for script_name in [
        "train_unet_baseline.py",
        "evaluate_unet_baseline.py",
        "train_unet_eurosat.py",
        "evaluate_unet_eurosat.py",
        "train_ddpm_synthetic.py",
        "train_ddpm_eurosat.py",
        "evaluate_ddpm_eurosat.py",
        "overfit_ddpm_tiny.py",
        "diagnose_ddpm_reconstruction.py",
        "overfit_unet_tiny.py",
    ]:
        script_path = Path("scripts") / script_name
        spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "main")


def test_tiny_overfit_function_reduces_loss():
    script_path = Path("scripts") / "overfit_unet_tiny.py"
    spec = importlib.util.spec_from_file_location("overfit_unet_tiny", script_path)
    assert spec is not None
    assert spec.loader is not None

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    result = module.run_tiny_overfit(
        num_samples=4,
        image_size=16,
        steps=20,
        learning_rate=3e-3,
        base_channels=4,
        save_outputs=False,
    )

    assert result["ending_loss"] < result["starting_loss"]
