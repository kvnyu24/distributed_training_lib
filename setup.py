from setuptools import setup, find_packages

setup(
    name="distributed_training_lib",
    version="0.1.0",
    description="A library for distributed training with model and data parallelism",
    author="Kevin Yu",
    packages=find_packages(),
    install_requires=[
        "numpy>=1.26.4",
        "mpi4py>=3.1.5",
        "h5py>=3.10.0",
    ],
    python_requires=">=3.8",
)
