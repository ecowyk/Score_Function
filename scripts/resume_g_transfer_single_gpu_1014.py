#!/usr/bin/env python3
"""Resume the existing transfer suite with one Ray scenario per physical GPU.

Preserves the prepared plan, completed results and failed-attempt receipts.
Only changes Ray's per-scenario GPU reservation from 0.15 to 1.0.
"""
from __future__ import annotations

import argparse
import signal
import time
from pathlib import Path

import evaluate_g_transfer_1014 as transfer


def retry_command(job, plan, out, label, original_command):
    # Called by the original runner while holding all evaluation/training locks.
    command = original_command(job, plan, out, label)
    index = command.index('--gpu-fraction') + 1
    if command[index] != '0.15':
        raise ValueError('Unexpected original GPU reservation; inspect the runner')
    command[index] = '1.0'
    profile = dict(
        name='one_scenario_per_gpu_v1',
        original_gpu_fraction=0.15,
        gpu_fraction=1.0,
        gpus=command[command.index('--gpus') + 1],
        threads=int(command[command.index('--threads') + 1]),
        plan_sha256=transfer.suite.digest(out / 'plan.json'),
        runner_sha256=transfer.suite.digest(Path(transfer.__file__).resolve()),
        adapter_sha256=transfer.suite.digest(Path(__file__).resolve()),
        change='Ray GPU admission control only; planner/model/scenario parameters retained',
    )
    profile_path = out / 'execution_profiles/one_scenario_per_gpu_v1.json'
    if profile_path.exists():
        if transfer.suite.read(profile_path) != profile:
            raise ValueError('Existing retry profile differs; refusing to replace it')
    else:
        transfer.suite.write(profile_path, profile)
    previous_path = out / 'jobs' / f"{job['id']}.json"
    if previous_path.exists():
        previous = transfer.suite.read(previous_path)
        if previous.get('state') in ('complete', 'reused'):
            raise ValueError('Completed results must be skipped, never relaunched')
        receipt = dict(previous_record=previous, next_command=command,
                       execution_profile=str(profile_path), archived_unix=time.time())
        transfer.suite.write(out / 'retry_history' / f'{label}.json', receipt)
    print(f"EXECUTION: {job['id']} uses 1.0 GPU per scenario, at most 8 concurrent GPU scenarios", flush=True)
    return command


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('mode', choices=('run', 'status'))
    parser.add_argument('--source', type=Path, default=transfer.SOURCE)
    parser.add_argument('--output', type=Path, default=transfer.OUTPUT)
    args = parser.parse_args()
    if args.mode == 'status':
        print(transfer.suite.read(args.output / 'status.json'), flush=True)
        transfer.summarize(args.output, transfer.suite.read(args.output / 'plan.json'))
        return

    def interrupt(signum, frame):
        raise KeyboardInterrupt(f'Received signal {signum}')

    signal.signal(signal.SIGTERM, interrupt)
    original_command = transfer.command_for
    transfer.command_for = lambda job, plan, out, label: retry_command(
        job, plan, out, label, original_command
    )
    try:
        transfer.run(args)
    finally:
        transfer.command_for = original_command


if __name__ == '__main__':
    main()
