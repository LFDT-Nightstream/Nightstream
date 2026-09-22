# Manual golden conformance

The selected goal is CPU–Lean conformance for **1→2→3**: the first fold and
one recursive fold, their complete input connection, and terminal checks at
state 3. CI enforcement, Metal validation and a third independent fold are
out of scope by owner instruction.

## Fresh CPU run and Lean verification

Supply the five archives named by `restore_golden_inputs.py` in one directory.
The command checks their recorded identities and restores them into a new
directory. Missing archives, outputs or successful phase records stop the run.

```sh
python3 -B scripts/golden_conformance_ci.py cpu \
  --archives /path/to/archives --directory /path/to/new-cpu-run
```

This builds the current CPU producer, runs folds 1 and 2, checks terminal
acceptance and rejection at state 3, and checks both fresh proofs with Lean.
It compares complete proof bytes, caller values and physical witnesses.
The Lean verifier receives CPU C proof messages; this is verifier conformance,
not independent proof generation.

## Independent Lean results

To generate both selected folds and run their complete comparisons:

```sh
python3 -B scripts/golden_conformance_ci.py independent \
  --archives /path/to/archives --directory /path/to/new-lean-run \
  --cpu-reference /path/to/new-cpu-run
```

Lean receives original witnesses and public inputs. Its generators derive
proof messages and challenges. The first root is `independent-first`; the
second is `independent-loop`. The second starts from restored original inputs;
acceptance requires their complete equality with the first Lean successor.
This command can take many hours. It does not generate 3→4.

To compare already generated roots with a fresh CPU run:

```sh
python3 -B scripts/check_selected_replay.py \
  --first-root /path/to/independent-first \
  --second-root /path/to/independent-loop \
  --cpu-reference /path/to/new-cpu-run \
  --directory /path/to/new-selected-comparison
```

Keep each root's `producer-sources.json`, `original-sources`, package and
selected `step-N-to-M` directory. No external Python driver is needed.
The source audit requires the same Lean definitions, producer scripts,
toolchain and package. It permits only the exact `neo-fold-clean` to
`neo-fold-legacy` comparison-crate rename. Changed Lean computation requires
new generation. Digests record file custody; comparisons use actual values.

The check compares both complete parents, all child witnesses, proof bytes,
caller values, physical witnesses and fresh outputs with the same CPU run.
It projects all 17 first-successor sources and compares their complete bytes
with the second fold's original inputs and consumed projection. It also
checks package, context, state, message, carried parent, all 17 digest frames,
C/R/D links and the public input consumed by C. It runs fresh terminal
acceptance and `ce-evaluation`, `ce-matrix-evaluation` and `fresh-private`
rejections at state 3. Success is recorded only after every check passes.

## Limits and records

Each native or Python child keeps the repository's 300-second cap. Each Lean
child uses `validate.sh` and keeps the 1,500-second cap. Current CPU phases use
the 16 GiB RSS guard from `NIGHTSTREAM_CRATE_GOAL.md`, section
“Owner-approved engine extension”. Builds share a lock within each worktree;
other worktrees use their own locks and writable build outputs.

The formal project's five-round stop rule applies within
`formal/nightstream-fprime`; the root's three-round rule applies elsewhere.
These are existing scoped policies, not new exceptions.

Keep complete inputs, output directories and command receipts with each run.
See [the status record](../docs/reviews/nightstream-fprime-requirements/GOLDEN_CONFORMANCE.md)
for executed checks. The older `replay_recursive_loop.py 2 all` command still
serves the separate, broader `fresh-recursive-loop` obligation. It is not the
selected command above.

For a manual change plan, run
`python3 -B scripts/golden_conformance_changes.py --base BASE --head HEAD`.
Its flags request checks or a source audit; they do not certify a result or
enforce CI. Metal flags remain advisory and are outside this selected run.
