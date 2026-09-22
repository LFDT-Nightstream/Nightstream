# Manual golden conformance

The selected goal is CPU–Lean conformance for 1→2→3: the first fold and one
recursive fold, with the complete input connection and terminal checks at
state 3. CI enforcement, Metal validation and a third independent fold are
out of scope by owner instruction.

Use fresh prepared roots for a new run. Restore all five recorded archives
and check source and input identities. Missing inputs and failed checks stop
the run. The existing producer interface supports these selected phases:

```sh
python3 -B formal/nightstream-fprime/scripts/replay_recursive_loop.py /path/to/first-root 1 first-fold
for phase in build prepare native ccs reductions successor terminal; do
  python3 -B formal/nightstream-fprime/scripts/replay_recursive_loop.py /path/to/second-root 2 "$phase" || exit
done
```

The second root must use inputs equal to the first Lean successor. Reuse of
the completed retained second fold requires complete equality of all 17
source witnesses and public values, package, context, state, message request,
carried parent, 17 successor digest frames, caller C/R/D links, and retained C
public input. Digests identify files; they do not replace these comparisons.
Producer commands alone do not establish this connection or the required
current CPU proof, parent, caller, physical, fresh-witness and child equality.

The current retained run uses the prepared task driver `selected_123.py` and
`bridge_first_second.py` in its evidence directory. They preserve the audited
source transition and passed second-fold checkpoints. Paths and execution
status are in
[GOLDEN_CONFORMANCE_CHECKS.json](../docs/reviews/nightstream-fprime-requirements/GOLDEN_CONFORMANCE_CHECKS.json).
Completion requires `independent-1-2-3-result.json`, the complete input-bridge
result, final-state-3 acceptance, and `ce-evaluation`, `ce-matrix-evaluation`
and `fresh-private` rejections. This computation can take many hours.

The older `golden_conformance_ci.py independent` command and the replay
coordinator's `2 all` mode include 3→4. They implement the broader registered
`fresh-recursive-loop` obligation, which remains open and is separate from
this selected goal. Do not use those modes to schedule this goal.

Each native child keeps the 300-second cap. Current production phases also
use the documented 16 GiB RSS guard. Each Lean child keeps the 1,500-second
cap. Use one build/execution queue per worktree, with separate writable build
outputs between worktrees. The outer coordinator does not hold the child lock.

For an explicit check plan after source changes, run
`python3 -B scripts/golden_conformance_changes.py --base BASE --head HEAD`.
Its dependency rules select work; they neither run checks nor prove success.
Keep the command receipts and complete comparison inputs with each result.
