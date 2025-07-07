"""
Initializes the llm_token_optimizer package.

This __init__.py file makes the directory a Python package and exposes
the main public functions from the 'api' module, allowing users to
import them directly from the top-level package.
"""

from .api import reduce_code, reconstruct_code

# Define what is exposed when 'from llm_token_optimizer import *' is used.
# It's good practice to explicitly list public API elements.
__all__ = ["reduce_code", "reconstruct_code"]