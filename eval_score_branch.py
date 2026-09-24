"""Dedicated corrupted-GT diagnostic entry; no nuPlan simulation is launched."""

import sys

from score_function.cli import main

if __name__ == "__main__":
    main(["evaluate", *sys.argv[1:]])
