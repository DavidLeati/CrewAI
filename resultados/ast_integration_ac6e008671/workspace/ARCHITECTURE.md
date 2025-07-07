# Architecture Overview

This document outlines the directory and file structure of the `python-llm-token-optimizer` project.

## Directory Structure
python-llm-token-optimizer/
├── src/
│   └── llm_token_optimizer/
│       ├── __init__.py
│       ├── reducer.py
│       ├── reconstructor.py
│       ├── utils.py
│       └── api.py
├── tests/
│   ├── __init__.py
│   ├── test_reducer.py
│   ├── test_reconstructor.py
│   └── test_e2e.py
├── docs/
│   └── usage.md
└── ARCHITECTURE.md
## File Descriptions

### `src/llm_token_optimizer/`
This directory contains the core logic of the LLM token optimizer library.

-   `__init__.py`: Initializes the `llm_token_optimizer` package.
-   `reducer.py`: Contains the logic for reducing code (e.g., removing comments, whitespace, simplifying expressions) to optimize token usage for LLMs.
-   `reconstructor.py`: Contains the logic for reconstructing the original code or a semantically equivalent version from its reduced form.
-   `utils.py`: Provides utility functions and helper methods used across `reducer.py` and `reconstructor.py`.
-   `api.py`: Defines the public API for interacting with the token optimizer, providing functions for reduction and reconstruction.

### `tests/`
This directory contains all unit and integration tests for the project.

-   `__init__.py`: Initializes the `tests` package.
-   `test_reducer.py`: Contains unit tests for the `reducer.py` module.
-   `test_reconstructor.py`: Contains unit tests for the `reconstructor.py` module.
-   `test_e2e.py`: Contains end-to-end tests to verify the complete reduction and reconstruction pipeline.

### `docs/`
This directory contains project documentation.

-   `usage.md`: Provides instructions and examples on how to use the `llm-token-optimizer` library.

### `ARCHITECTURE.md`
-   This file, detailing the project's directory and file structure.