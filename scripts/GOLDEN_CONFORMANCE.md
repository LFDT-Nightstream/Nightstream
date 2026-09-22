# Manual golden conformance

Run these checks explicitly. CI enforcement is out of scope by owner
instruction. No workflow starts these long computations automatically.

Use a fresh output directory for each command. The archive directory must
contain all five recorded archives. Missing inputs and failed checks stop
the run.

```sh
python3 -B scripts/golden_conformance_ci.py cpu --archives /path/to/archives --directory /path/to/cpu-run
python3 -B scripts/golden_conformance_ci.py independent --archives /path/to/archives --directory /path/to/independent-run --cpu-reference /path/to/cpu-run
python3 -B scripts/golden_conformance_ci.py metal --archives /path/to/archives --directory /path/to/metal-run --cpu-reference /path/to/cpu-run
```

The CPU command runs current native folds 1–3 and fresh Lean verifier,
caller, physical-witness and mutation checks. The independent command
generates the exact first fold and the 2→3→4 replay. Each producer uses its
own successor for its next input. This computation can take many hours.

Run Metal on macOS with an available Metal device. Copy the complete CPU
run and its Lean input snapshots to that host. The Metal command compares
the transported CPU files before and after its engine comparison. It does
not repeat Lean generation.

Each native child keeps the 300-second cap. Current production phases also
use the documented 16 GiB RSS guard. Each Lean child keeps the 1,500-second
cap. Use one build/execution queue. The outer coordinator does not hold the
child lock.

For an explicit check plan after source changes, run
`python3 -B scripts/golden_conformance_changes.py --base BASE --head HEAD`.
Its dependency rules select work; they neither run checks nor prove success.
Keep the command receipts and complete comparison inputs with each result.
