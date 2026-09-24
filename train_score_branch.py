"""Dedicated training entry; equivalent to python -m score_function train."""

import sys

from score_function.cli import main

if __name__ == "__main__":
    main(["train", *sys.argv[1:]])
