from __future__ import annotations

import importlib
import subprocess
import sys
from pathlib import Path

ARONNAX_REPO_URL = "https://github.com/edoddridge/aronnax.git"
ARONNAX_COMMIT = "08f5a8ad88972974cd4b9fd833965701f52bf5ce"
ARONNAX_CHECKOUT_DIR = Path("data/interim/aronnax-src")


def ensure_aronnax_checkout() -> Path:
    checkout_dir = ARONNAX_CHECKOUT_DIR
    if not (checkout_dir / ".git").exists():
        checkout_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.check_call(
            ["git", "clone", "--recursive", ARONNAX_REPO_URL, str(checkout_dir)]
        )

    subprocess.check_call(["git", "-C", str(checkout_dir), "fetch", "--all"])
    subprocess.check_call(["git", "-C", str(checkout_dir), "checkout", ARONNAX_COMMIT])
    subprocess.check_call(
        ["git", "-C", str(checkout_dir), "submodule", "update", "--init", "--recursive"]
    )
    return checkout_dir


def load_aronnax_modules():
    checkout_dir = ensure_aronnax_checkout()
    checkout_str = str(checkout_dir.resolve())

    if checkout_str not in sys.path:
        sys.path.insert(0, checkout_str)

    for module_name in list(sys.modules):
        if module_name == "aronnax" or module_name.startswith("aronnax."):
            del sys.modules[module_name]

    aro = importlib.import_module("aronnax")
    driver = importlib.import_module("aronnax.driver")
    return aro, driver
