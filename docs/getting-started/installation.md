# Installation

Dendron has **zero external dependencies** and is built entirely on the Python standard library.

## Requirements
- Python 3.9 or higher

## Installation Options

### 1. From PyPI (Standard)
```bash
# Core library
pip install dendron-ai

# With optional LangChain support
pip install "dendron-ai[langchain]"
```

### 2. Direct from GitHub
```bash
pip install git+https://github.com/sumanth1989/Dendron.git

# With LangChain support
pip install "dendron-ai[langchain] @ git+https://github.com/sumanth1989/Dendron.git"
```

### 3. From Local Source
```bash
# Standard local install
pip install .

# Editable development install with dev & doc dependencies
git clone https://github.com/sumanth1989/Dendron.git
cd Dendron
pip install -e ".[dev,docs,langchain]"
```

### 4. From Built Wheel (`.whl`)
```bash
python -m build
pip install dist/dendron_ai-0.1.0-py3-none-any.whl
```

### 5. Using Poetry
```bash
# From PyPI
poetry add dendron-ai

# From GitHub
poetry add git+https://github.com/sumanth1989/Dendron.git
```

## Zero-Dependency Direct Copy (Vendoring)
Because Dendron relies solely on the Python standard library, you can also copy the `dendron/` directory directly into your project without running any package installers:

```text
my_agent_project/
├── dendron/
│   ├── __init__.py
│   ├── models.py
│   ├── node.py
│   ├── tree.py
│   ├── retriever.py
│   └── exceptions.py
└── main.py
```

## Verifying Installation
In your Python environment, run:

```python
import dendron
print(dendron.__version__)
```
Output should display `0.1.0`.
