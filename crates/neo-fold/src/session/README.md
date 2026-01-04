# `neo_fold::session`

`neo_fold::session` is intended to be Neo’s single front-facing proving API. It wraps the shard
protocol, hides commitment-mixer details, and provides ergonomic helpers for defining circuits and
running end-to-end prove/verify flows.

## Workflows

### 1) “Direct IO” steps (CCS-only)

- Implement `NeoStep` (or use `ProveInput`/`FoldingSession::add_step_from_io`).
- Use `FoldingSession` to run the Π-CCS folding loop and verification.

Ergonomic shortcuts:

- `FoldingSession::<AjtaiSModule>::new_ajtai(...)` auto-picks `NeoParams` and builds an Ajtai committer.
- `FoldingSession::add_step_io(...)` avoids constructing `ProveInput`.
- `FoldingSession::prove_and_verify_collected(...)` folds+proves+verifies in one call.

### 2) Shared CPU-bus R1CS circuits (Twist + Shout)

This is the ergonomic path for zkVM-like traces where Twist/Shout semantics are proven via the
shared CPU bus (Route-A sidecar), but the CPU witness is still a CCS/R1CS-style object.

- Define a typed witness layout with `witness_layout!` using `Lane<N>` and ports:
  - `TwistPort<N>` / `TwistPortWithInc<N>`
  - `ShoutPort<N>`
- Implement `NeoCircuit` to:
  - declare resources (`SharedBusResources`): Twist layouts/init + Shout tables/specs
  - define CPU constraints with `CcsBuilder`
  - build the CPU witness prefix per chunk, typically:
    - fill CPU-local lanes, then
    - call `TwistPort::fill_from_trace(...)` / `ShoutPort::fill_from_trace(...)` to auto-fill CPU binding columns
- Preprocess once with `preprocess_shared_bus_r1cs(Arc<C>)` to compute the base CCS and witness width.
- Build a prover artifact with `SharedBusR1csPreprocessing::into_prover(params, committer)`.
- Execute into a session with `SharedBusR1csProver::execute_into_session(...)`.
- Prove/verify using `FoldingSession` helpers (including output binding helpers in `neo_fold::output_binding`).

See `crates/neo-fold/tests/twist_shout_fibonacci_cycle_trace.rs` for a complete example.
