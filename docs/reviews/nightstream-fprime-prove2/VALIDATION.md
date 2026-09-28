# Validation record

Reviewed commit: `4fc02857c3aa4207f3290739cc62d794ba86f5f9`.

The tracked working tree was clean before and after this review. New review files are under the ignored `docs/` directory. Lean: `leanprover/lean4:v4.30.0`. Mathlib: `c5ea00351c28e24afc9f0f84379aa41082b1188f`.

## Commands and results

Run from `/Users/nicarq/starstream/develop/nightstream-clean-up/formal/nightstream-fprime`:

```sh
PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 1500 bash scripts/validate.sh all
```

This passed earlier in the same review on the unchanged source. It used the existing build cache. The relevant lines from the full log are retained in [package-validation-excerpt.log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/package-validation-excerpt.log). The full log remains at `/tmp/nightstream-fprime-review-validation-bash5.log`. The boundary check passed; the library completed 3,642 jobs in 4 s; the test/axiom library completed 3,677 jobs in 1 s. Job counts include cached/replayed jobs. These times do not measure a cold build.

The review's additional files were checked separately, in sequence:

```sh
PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 1500 bash scripts/validate.sh file \
  /Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/tests/RootSketch.lean
```

Result: exit 0, 2 s. Both declarations passed `#audit_axioms`. See [root-sketch.log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/root-sketch.log).

```sh
PATH="/opt/homebrew/bin:$PATH" timeout -s KILL 1500 bash scripts/validate.sh file \
  /Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/tests/EncodingCheck.lean
```

Result: exit 0, 1 s. All four declarations passed `#audit_axioms`. See [encoding-check.log](/Users/nicarq/starstream/develop/nightstream-clean-up/docs/reviews/nightstream-fprime-prove2/encoding-check.log).

The 1,500 s cap comes from project policy. The outer timeout also enforces that cap over the wrapper. Bash 5 from Homebrew was used because the boundary script requires features absent in macOS Bash 3.2. No Lean recursion or heartbeat limit was increased.

## Interpretation

The root proof has explicit open hypotheses. A passing audit establishes that its conclusion follows from those hypotheses under the allowed Lean axioms. It does not establish the hypotheses for accepted circuit assignments.

The encoding proof establishes the length, norm, reconstruction, and noncanonical form of a fixed example. It does not evaluate the full circuit on that example.

No Rust test, artifact generation, hosted submission, or cryptographic proof backend was run. The current artifact JSON is a Git LFS pointer. The sibling conformance evidence archive is absent. Recorded diagnostics in `CONSTRAINT_TREE.md` were read as records, not reported as new executions.
