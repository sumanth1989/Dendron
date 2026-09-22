# Installation

Dendron has **zero external dependencies** and is built entirely on the Python standard library.

## Requirements
- Python 3.9 or higher

## Installing with pip

```bash
# Standard installation
pip install dendron-ai

# Editable / development installation from source
git clone https://github.com/sumanth1989/Dendron.git
cd Dendron
pip install -e ".[dev,docs]"
```

## Zero-Dependency Direct Copy
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
