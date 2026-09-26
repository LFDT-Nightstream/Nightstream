# Current-package golden conformance

This record checks the selected wide package against the four parity
interfaces of `FPRIME_LEAN_ARCHITECTURE_SPEC.md` (section "Parity surface").
Prover internals are outside that contract.

Source: commit `48c8e0b99`, with the two tool changes committed with this
record (optional reference archives; no physical-witness step). Host: the
development Mac, 2026-09-26. Engine: optimized CPU.

## Results

| Interface | Check | Result |
|---|---|---|
| 1. Primitives | `scripts/check_fprime_foundation_parity.sh`: fresh Lean ring-transform and `split_b` vectors against Rust | pass |
| 1. Primitives | Lean regenerates the Ajtai setup and sparse-commitment parity files; both equal the committed files | pass |
| 1. Primitives | `neo-ajtai` setup tests on the fresh file; `check_package_conformance primitive` on the fresh file (3 support coordinates, 1,188 coefficients) | pass |
| 2. Challenges | `neo-reductions --test pi_rlc_wide_lean_parity` (decoder, transcript states, sampler) | pass |
| 3. Running instance | Fold 1 and fold 2: Lean C/R/D verifier accepts; native comparison of every field; Lean-encoded proof bytes equal the native proof | pass |
| 3. Running instance | Recursive-step inputs: 177,326 private and 278 public words and all seven result fields equal Lean | pass |
| 3. Running instance | Rejections per fold: 55 native PiDEC mutation classes; Lean 10 decoder, 34 public-check, 1 PiCCS and 2 internal cases | pass |
| Terminal | Final state 3 accepted; mutation, reject, `opening-k` and `opening-a` phases | pass |
| 4. Package | `nightstream-fprime` package-loader (8) and production-binding (3) tests | pass |

The fresh fold-1 proof (945,983 bytes) is byte-identical to the committed
`crates/nightstream/tests/fixtures/stage1_actual_nifs/proof.native`.

All 20 native phases ran within the 300-second cap; the longest took 190 s,
and peak RSS was 9.7 GiB. Each Lean command ran within the 1,500-second cap.
The phase receipts are in `phases/`, the run summary is
`native-conformance.json`, and the Lean fold results are `lean-fold-1.json`
and `lean-fold-2.json`.

## What Lean computes, and what Rust supplies

Lean computes the transcript states, all challenges including the wide
sampler, the running instance after PiCCS, PiRLC and PiDEC, the recursive-step
inputs, and the primitive vectors. Rust supplies the PiCCS messages
(sum-check rounds, Eval_K and Eval_A) and the PiDEC child commitments and
evaluations. Lean checks those prover outputs; it does not generate them.

## Scope

- The complete physical witness is a prover internal and is not compared.
  Its Lean replay (`replayPhysicalWitness`) targets a retired package.
- Poseidon2 matches through every transcript state that both sides compute.
  The `neo-ccs` round-constant test reads a file from the removed
  `formal/nightstream-lean` project and fails; it is not used here.
- Field and extension arithmetic have no separate Lean vector check in this
  run; every check above depends on them.
- Recipe rejection was not checked.
- These are executed examples, not universal proofs of Rust semantics.

## Reproduce

From the repository root, with Homebrew `bash` and `python3` first on `PATH`:

```sh
cargo test -p nightstream --release --lib --no-run
python3 -B crates/nightstream/tests/run_golden_conformance.py --binary target/release/deps/nightstream-<hash> --directory RUN
cargo build -p neo-fold-legacy --release --bin generate_pi_ccs_fixture
python3 -B crates/nightstream/tests/check_lean_fold.py --directory RUN --step 1 --output OUT1 --native-checker target/release/generate_pi_ccs_fixture
python3 -B crates/nightstream/tests/check_lean_fold.py --directory RUN --step 2 --output OUT2 --native-checker target/release/generate_pi_ccs_fixture
```
