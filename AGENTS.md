# Score Function agent onboarding

Before working on this research project, read [docs/handoff/START_HERE.md](docs/handoff/START_HERE.md), then the files it links for the assigned task. The handoff is a dated research snapshot, not a live server monitor.

- Preserve the confirmed research objective: local trajectory refinement using a learned conditional score; evaluate planning quality in closed-loop benchmarks. Do not reframe it as proven DP few-step collapse, global MAP, RL, or a world model.
- Resolve implementation questions against the exact code/config/checkpoint used by the experiment. `main` is a core integration entry point; experimental adapters and queues also live on pinned branches and server snapshots.
- Distinguish verified source behavior, user-reported results, historical status, and hypotheses. Never turn a script's existence into a successful experiment or missing scores into zeros.
- Preserve paired-evaluation identities: scenario keys, NR/R, baseline and score checkpoints, seed policy, normalizer, inference settings, and simulator config. K is score-refinement iterations, not diffusion/flow sampling steps.
- The current default follow-up protocol is S10-G, sigma=0.1, gamma=0.1, K=20, evaluation seed=20260922; Flow CFG=1.8. Training seed is inherited from the original training config. These settings do not retroactively relabel older runs.
- Respect existing complete results and frozen running experiments. Resume the same run when appropriate; do not regenerate caches, change a prepared plan, or stop unrelated processes as part of routine diagnosis.
- Use separate task branches/worktrees for concurrent code changes. Record task ownership and outputs in the handoff. Existing authorized work can proceed without an extra confirmation ritual.
- After a substantive change, update the relevant handoff status/decisions/results with source links and validation. Run checks appropriate to the change; a documentation-only update does not require GPU simulation.
- Keep communication in Chinese unless requested otherwise. Explain reasoning and experimental implications clearly. In Markdown use `$...$` / `$$...$$` for math.

Research documentation is under `docs/handoff/`; historical documents elsewhere retain their original scope and date. The archived Python files under `docs/handoff/snapshots/` are provenance, not a new self-contained deployment package.
