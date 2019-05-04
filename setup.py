""" pypi setup for track-ml """
from setuptools import setup, find_packages

install_requires = [
    "tensorflow==1.13.1",
    "pandas>=0.20.1",
    "skeletor-ml",
    "cleverhans",
]

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name="Adversarial RL Experiments",
    author="Gokul Swamy, Noah Golmant",
    version="0.1",
    description="Reproducing 'Adversarial Attacks on Neural Network Policies'",
    install_requires=install_requires,
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gkswamy98/adversarial_rl",
    license='MIT License',
    packages=find_packages(),
)
