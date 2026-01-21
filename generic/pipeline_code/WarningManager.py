# WarningManager

import warnings
from sklearn.exceptions import ConvergenceWarning
import contextlib

@contextlib.contextmanager
def suppress_convergence_warnings():
    """Context manager to suppress convergence warnings"""
    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=ConvergenceWarning)
        yield

def check_convergence(model):
    """Check if a model has converged properly"""
    if hasattr(model, 'n_iter_'):
        if hasattr(model, 'max_iter'):
            if model.n_iter_ >= model.max_iter:
                return False, f"Model reached max iterations ({model.max_iter})"
            else:
                return True, f"Model converged in {model.n_iter_} iterations"
    return True, "Convergence check not applicable"