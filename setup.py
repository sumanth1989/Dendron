from setuptools import setup, find_packages

setup(
    name="dendron-ai",
    version="0.1.0",
    description="Dendron: An adaptive, tree-based tool execution and dynamic discovery library for AI agents",
    long_description=open("README.md").read() if open("README.md") else "",
    long_description_content_type="text/markdown",
    packages=find_packages(include=["dendron*"]),
    python_requires=">=3.9",
    install_requires=[],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
