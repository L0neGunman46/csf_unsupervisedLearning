from setuptools import setup, find_packages

setup(
    name="csf_halfcheetah",
    version="0.1.0",
    description="Contrastive Successor Features implementation for HalfCheetah",
    packages=find_packages(),
    install_requires=[
        "torch>=1.9.0",
        "gymnasium>=0.26.0",
        "numpy>=1.20.0",
        "matplotlib>=3.3.0",
        "seaborn>=0.11.0",
        "wandb>=0.12.0",
        "mujoco>=2.2.0",
        "pyyaml>=5.4.0",
        "tqdm>=4.62.0",
        "scipy>=1.7.0",
        "imageio>=2.9.0",
    ],
    python_requires=">=3.8",
)