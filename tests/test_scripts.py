import importlib.util
from pathlib import Path


def test_training_scripts_are_import_safe():
    for script_name in ["train_unet_baseline.py", "evaluate_unet_baseline.py"]:
        script_path = Path("scripts") / script_name
        spec = importlib.util.spec_from_file_location(script_name.removesuffix(".py"), script_path)
        assert spec is not None
        assert spec.loader is not None

        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        assert hasattr(module, "main")
