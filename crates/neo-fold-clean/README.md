# neo-fold-clean

`neo-fold-clean` owns the public folding lifecycle, SuperNeo NIFS composition,
and the application frontends. Arithmetic, commitments, reduction engines,
and accelerator device code have separate crates.

## Entry points

| Task | API or module |
|---|---|
| Fold caller-supplied CCS instances | `preprocess`, `prove`, `extend`, `finish_uncompressed`, `verify_uncompressed` |
| Load the canonical Lean Stage 1 package | `stage1::Poseidon2HashChainV1Package::load` |
| Convert native PiCCS messages to package inputs | `stage1::PiCcsV1_1ProofInputs::from_proof` |
| Use the memory frontend | `frontends::nebula::{NebulaFPrimePreparedProfile, NebulaFPrimeChainBuilder}` |

The package API exposes Lean-authored matrix rows and the package witness
program. It does not provide a complete recursive proving lifecycle.
The memory frontend still uses the native circuit builders.

## Ownership

- `stage1/`: canonical package loading and input serialization
- `lifecycle/`: preprocess, prove, extend, finish, and verify
- `paper/nifs/`: PiCCS, PiRLC, and PiDEC composition
- `paper/construction2/`: recursive state and terminal fold
- `frontends/direct_ccs/`: caller-supplied CCS instances
- `frontends/nebula/`: memory relation and application state wiring
- `frontends/f_prime/`: shared low-norm image construction
- `frontends/r1cs_f_prime/`: native R1CS construction, lowering, and diagnostics

## Prover engines

- `OptimizedCpuNifsProver`: optimized CPU execution
- `PaperExactNifsProver`: reference execution for small correctness checks
- `MetalNifsProver`: device oracle and commitments on supported Apple builds
- `CudaNifsProver`: interface reserved for the CUDA implementation; construction
  returns `BackendUnavailable` until its canonical kernel exists

Protocol code owns transcript order and verification. An engine selection
must preserve the proof messages and complete phase outputs.

## Checks

```sh
RUSTC_WRAPPER="" timeout 300s cargo test -p neo-fold-clean --release --test nifs_round_trip
RUSTC_WRAPPER="" timeout 300s cargo test -p neo-fold-clean --release --test f_prime_package_production
RUSTC_WRAPPER="" timeout 300s cargo test -p neo-fold-clean --release --test system_preprocessing_cache_hygiene
```

Tests that compare frozen Lean artifacts are read-only. A mismatch fails;
the tests do not write replacement artifacts. Package conformance tests live
in `nightstream-fprime` and in the registered `nifs_*_lean_*` targets.
