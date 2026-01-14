import importlib.util
import inspect
import sys
from pathlib import Path

# Runtime patch for scipy.optimize._optimize using the modified file from
# Optimizing_the_Optimizer_PINNs.

def _supports_method_bfgs():
    try:
        import scipy.optimize._optimize as _optimize
    except Exception:
        return False
    return "method_bfgs" in inspect.signature(_optimize._minimize_bfgs).parameters


def _default_patch_path():
    return Path(__file__).resolve().parent / "third_party" / "scipy_optimize" / "_optimize.py"


def apply_scipy_optimize_patch(patch_path=None):
    """
    Replace scipy.optimize._optimize at runtime with the patched implementation.
    """
    import scipy.optimize
    import scipy.optimize._minimize as _minimize

    patch_path = Path(patch_path) if patch_path is not None else _default_patch_path()
    if not patch_path.exists():
        raise FileNotFoundError(f"Patched SciPy file not found: {patch_path}")

    spec = importlib.util.spec_from_file_location("scipy.optimize._optimize", patch_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)

    scipy.optimize._optimize = module
    scipy.optimize.fmin_bfgs = module.fmin_bfgs
    _minimize._minimize_bfgs = module._minimize_bfgs


def ensure_scipy_bfgs_patch(patch_path=None):
    """
    Ensure scipy.optimize._minimize_bfgs accepts method_bfgs. Returns True if a patch was applied.
    """
    if _supports_method_bfgs():
        return False

    apply_scipy_optimize_patch(patch_path=patch_path)

    if not _supports_method_bfgs():
        raise RuntimeError(
            "SciPy BFGS patch failed. Replace your SciPy "
            "site-packages scipy/optimize/_optimize.py with the patched copy."
        )
    return True
