from setuptools import setup, find_packages

setup(
    name="code_editor",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        'torch>=2.0.0',
        'transformers>=4.30.0',
        'tensorboard>=2.13.0',
        'numpy>=1.24.0',
        'tqdm>=4.65.0',
        'networkx>=3.0.0',
        'qiskit>=0.39.0',
        'scikit-learn>=1.0.0',
        'matplotlib>=3.5.0'
    ],
    author="Your Name",
    author_email="your.email@example.com",
    description="An AI-powered code editor with speculative execution",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    python_requires=">=3.8",
) 