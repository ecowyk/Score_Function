"""Prepare official nuPlan features for score learning."""

import sys

from score_function.cli import main

if __name__ == "__main__":
    main(["prepare", *sys.argv[1:]])
