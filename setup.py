"""Install the research package in the official Diffusion Planner environment."""

from setuptools import find_packages, setup

setup(
    name="score-function",
    version="0.1.0",
    description="Time-independent ego-trajectory score refinement for Diffusion Planner",
    packages=find_packages(include=["score_function", "score_function.*"]),
    python_requires=">=3.9",
    install_requires=["torch>=2.0,<3", "numpy>=1.23,<2", "matplotlib>=3.5,<4"],
    entry_points={"console_scripts": ["score-function=score_function.cli:main"]},
)
