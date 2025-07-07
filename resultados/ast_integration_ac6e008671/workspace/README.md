# LLM Token Optimizer

The `llm-token-optimizer` is a Python library designed to optimize token usage for Large Language Models (LLMs) by reducing the size of code inputs. It achieves this by intelligently removing non-essential elements like comments, excessive whitespace, and simplifying expressions, while preserving the semantic integrity of the code. This allows for more efficient processing by LLMs, potentially reducing costs and improving performance for tasks like code analysis, generation, and transformation.

## Features

-   **Code Reduction:** Efficiently removes comments, unnecessary whitespace, and applies minor simplifications to reduce code size.
-   **Code Reconstruction:** Ability to reconstruct a semantically equivalent version of the original code from its reduced form.
-   **API Integration:** Provides a straightforward API for easy integration into existing workflows.

## Installation

To install the `llm-token-optimizer` library, you can use pip:
pip install llm-token-optimizer
*(Note: This is a placeholder. Replace with actual package name if different, or instructions for local installation if not yet published.)*

## Usage

Here's a basic example of how to use the `llm-token-optimizer` to reduce and reconstruct code:
from llm_token_optimizer.api import reduce_code, reconstruct_code

# Example code
original_code = """
def factorial(n): # Calculates factorial
    '''
    This function computes the factorial of a non-negative integer.
    '''
    if n == 0:
        return 1
    else:
        return n * factorial(n-1)

# Call the function
result = factorial(5) # Should be 120
"""

# Reduce the code
reduced_code = reduce_code(original_code)
print("--- Reduced Code ---")
print(reduced_code)

# Reconstruct the code
reconstructed_code = reconstruct_code(reduced_code)
print("\n--- Reconstructed Code ---")
print(reconstructed_code)
## Project Structure

The project is organized into the following main directories:

-   `src/llm_token_optimizer/`: Contains the core logic, including `reducer.py`, `reconstructor.py`, `utils.py`, and `api.py`.
-   `tests/`: Houses all unit and end-to-end tests for the library.
-   `docs/`: Contains project documentation, such as `usage.md`.

For a detailed overview of the project's architecture, please refer to `ARCHITECTURE.md`.

## Quality and Success Criteria

As an Engineer of Quality and Documentation, our primary goal is to ensure the robustness, test coverage, and clarity of this library.

-   **Robustness:** The `reducer` and `reconstructor` modules must handle a wide variety of Python code structures without introducing syntax errors or altering semantic meaning. Edge cases, including malformed code snippets, should be gracefully managed.
-   **Test Coverage:** Comprehensive unit tests (`test_reducer.py`, `test_reconstructor.py`) and end-to-end tests (`test_e2e.py`) are critical to validate the functionality and ensure no regressions occur with new changes. High test coverage is a key metric.
-   **Documentation Clarity:** The `usage.md` and `ARCHITECTURE.md` files, along with inline code comments, must be clear, concise, and easy to understand for developers integrating or contributing to the library. The `README.md` itself serves as the primary entry point for new users.
-   **Performance:** The reduction and reconstruction processes should be efficient, especially for large codebases, to ensure practical applicability.
-   **Maintainability:** The codebase should be well-structured, modular, and adhere to Python best practices, making it easy to extend and maintain.

Success is measured by the library's ability to consistently and reliably optimize token usage for LLMs while maintaining code integrity, supported by a strong testing suite and clear, user-friendly documentation.