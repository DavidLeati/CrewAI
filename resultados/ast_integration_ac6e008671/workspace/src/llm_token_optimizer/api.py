from .reducer import reduce_code as _reduce_code_impl
from .reconstructor import reconstruct_code as _reconstruct_code_impl

def reduce_code(code: str, **kwargs) -> str:
    """
    Reduces the given code to optimize token usage for LLMs.

    This function acts as the public entry point for the code reduction process.
    It delegates the actual reduction logic to the internal implementation
    defined in `reducer.py`.

    Args:
        code: The input code string to be reduced.
        **kwargs: Additional arguments that can be passed to the underlying
                  reduction algorithm (e.g., specific strategies,
                  configuration options).

    Returns:
        The reduced code string, optimized for LLM token usage.
    """
    return _reduce_code_impl(code, **kwargs)

def reconstruct_code(reduced_code: str, **kwargs) -> str:
    """
    Reconstructs the original code or a semantically equivalent version
    from its reduced form.

    This function serves as the public entry point for the code reconstruction
    process. It delegates the actual reconstruction logic to the internal
    implementation defined in `reconstructor.py`.

    Args:
        reduced_code: The reduced code string that needs to be reconstructed.
        **kwargs: Additional arguments that can be passed to the underlying
                  reconstruction algorithm (e.g., specific strategies,
                  metadata from reduction).

    Returns:
        The reconstructed code string, aiming to be semantically equivalent
        to the original or a readable version of it.
    """
    return _reconstruct_code_impl(reduced_code, **kwargs)