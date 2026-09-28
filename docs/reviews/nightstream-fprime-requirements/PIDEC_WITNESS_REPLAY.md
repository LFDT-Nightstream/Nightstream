# PiDEC private witness replay

All 4,048,180,416 coefficients of the 16 Rust child witnesses match Lean,
including each complete carrier tail. A changed final coefficient in child
15 fails with its child, block, lane and range. The two signed bit masks are
exact coefficient encodings. They are compared as values.

Lean receives its own PiRLC parent ranges and computes the existing signed
binary decomposition with `b = 2`, `k_rho = 16`, and strict bound `B = 65536`.
`StoredSplit.kernel_eq_spec` identifies the complete optional result with
the existing scalar specification: every bounded input succeeds, and every
unbounded input rejects. `PiDECStoredSplitHonestWitness` connects the stored
digits to the selected honest witness and message definitions.

The parent inherits PiRLC's current boundary: Lean checked the Rust PiCCS
messages and derived the challenges. Independent PiCCS prover replay is
still required. Rust child witnesses are never inputs to Lean's digit
computation. The comparison uses freshly generated native children.

| Invocation | Time | Peak RSS |
| --- | ---: | ---: |
| Lean blocks 0–74,271 | 8.38 s | 1,502,416 KiB |
| Lean blocks 74,272–4,685,393 | 142.15 s | 1,498,056 KiB |
| Fresh Rust split and commitment check | 141.00 s | 4,995,680 KiB |
| Complete comparison and tail mutation | 2.77 s | 535,392 KiB |

`PIDEC_WITNESS_REPLAY.json` records the exact files, SHA-256 values,
commands, gate results and limits. Times include the guard wrapper; the
comparison itself took 2.56 seconds. The second range uses the existing
PiRLC boundary; the first range measurement established its feasibility.

The existing lean-graph record is `pidec-witness-replay`. Its value gate
first runs the existing PiRLC commands in the same gate workspace, then
computes PiDEC from those Lean outputs and compares against a separate Rust
split. The required input manifest is the same as for `pirlc-witness-values`.
See `PIRLC_WITNESS_REPLAY.md` for the original input retrieval and R recipe.
Release upload and fresh-download verification remain pending.

For a retained PiRLC run, the D commands are:

1. `validate.sh build replayPiDECWitness`.
2. `validate.sh pi-dec-witness-replay-boundaries <new-results-directory>`.
3. Native fixture command `check-owned-parent <package> <original-sources> <Rust-parent-directory> <new-child-directory>`.
4. `validate.sh pi-dec-witness-replay <Lean-R-prefix> <new-D-prefix> 0 74272`.
5. `validate.sh pi-dec-witness-replay <Lean-R-suffix> <new-D-suffix> 74272 4685394`.
6. Native fixture command `compare-pidec-replay <new-child-directory> <new-D-prefix> <new-D-suffix>`.

Use the single build queue, the 1,500-second Lean cap and the 300-second
native cap. The boundary suite also has a 300-second cap. The registered
gate contains the complete commands and required completion markers.

This result closes the private digit comparison. Independent computation
of every child commitment and evaluation from these same digits remains
required for full PiDEC replay. No constraint, transcript, package or
cryptographic assumption changes here. No Ironwood source was copied.
