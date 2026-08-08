# neo-fold-clean

`neo-fold-clean` is Nightstream's protocol integrator. It owns the public
lifecycle, SuperNeo NIFS composition, HyperNova F' relations, Nebula
integration, and terminal proof statement.

It does not own field arithmetic, Ajtai commitments, the reduction engines,
accelerator device code, or the WHIR PCS.

## Main ownership boundaries

- `lifecycle/`: preprocess, prove, extend, finish, and verify
- `paper/nifs/`: PiCCS, PiRLC, and PiDEC composition
- `paper/construction2/`: recursive state and terminal fold
- `paper/f_prime/`: native and constrained F' semantics
- `frontends/f_prime/`: shared low-norm F' image
- `frontends/r1cs_f_prime/`: authoritative fixed-shape R1CS F' relation
- `frontends/nebula/`: Nebula memory relation and F' lifecycle
- `frontends/direct_ccs/`: caller-supplied CCS instances
- `engine/decider.rs`: full-history audit R1CS
- `frontends/r1cs_f_prime/terminal_r1cs/`: terminal R1CS and WIP Spartan bridge

The recursive R1CS and Nebula frontends carry the prior NIFS verification in
their compiled relation. Direct CCS does not add this F' induction, so its
multi-chunk verifier uses the audit trail.

## NIFS prover implementations

- `OptimizedCpuNifsProver`: canonical optimized CPU path
- `PaperExactNifsProver`: direct reference path for small checks
- `MetalNifsProver`: Metal one-joint oracle and device commitments on
  supported Apple builds
- `CudaNifsProver`: required CUDA target; construction returns
  `BackendUnavailable` until the canonical kernel exists

## Checks

```sh
timeout 300s cargo test -p neo-fold-clean --release --test nifs_round_trip
timeout 300s cargo test -p neo-fold-clean --release --test f_prime_r1cs
timeout 300s cargo test -p neo-fold-clean --release --test nebula_f_prime
timeout 300s cargo test -p neo-fold-clean --release --test system_r1cs_ivc_terminal
```

The ignored performance tests and their exact commands are in the root
[AGENTS.md](../../AGENTS.md).
