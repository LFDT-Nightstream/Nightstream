# Nightstream F′ package instructions

These instructions apply to `formal/nightstream-fprime`. The architecture
contract is `/FPRIME_LEAN_ARCHITECTURE_SPEC.md`; the goal is
`/FPRIME_STAGE1_GOAL.md`. Both outrank this file.

## Layers

Imports flow downward only. `scripts/check-boundaries.sh` enforces it.

```text
Spec       paper semantics (field, ring, Poseidon2, sumcheck, PiCCS/PiRLC/PiDEC
           verifiers, HyperNova F′ relation)
Circuit    the DSL: expressions, offset variables, operations, circuit monad,
           one FormalCircuit record, the circuit_norm simp set
Gadgets    one directory per gadget; exports only its FormalCircuit
Lifecycle  concrete F′ phase order, carried state, public layout, XOut
Layout     physical lowering and layout-preservation theorems
Export     serializer, witness IR, relation identifier, lake exe emitter
tests      axiom gate (explicit imports, `#audit_axioms` per theorem)
```

## Rules

- No import from `formal/nightstream-lean`. Copy the smallest audited
  definition and put a provenance comment (source path, commit) at the top.
- No generated modules, no embedded artifact data, no `native_decide`, no
  `sorry`/`admit`/`axiom`/`unsafe`, no file at or above 1,500 lines, no glob
  in `lakefile.toml`, one profile (`b = 2`, `k_rho = 16`).
- No `maxRecDepth`/`maxHeartbeats` override, except a per-declaration
  `set_option … in -- fixed-size: …` whose cost axis is a protocol constant
  (ring degree 54, Poseidon2 width) and never emitted data. Each such site is
  a recorded debt to replace by a structural proof. The boundary script
  rejects every other form.
- Subcircuits are opaque to parents: a parent proof that unfolds a child's
  operations is a bug. Proof cost must not grow with rows, columns, or
  schedule length.
- Every exported theorem is listed in `tests/Axioms.lean` with
  `#audit_axioms`. Allowed axioms: `propext`, `Classical.choice`,
  `Quot.sound`.
- Lean commands only through `scripts/validate.sh` (`static`, `build
  [target]`, `axioms`, `file <path>`, `all`), each under the 1,500 s cap.
  One Lean or Rust build process at a time.
- Before each command or edit: one active acceptance criterion and its
  closing evidence. Three rounds without closure: stop and report.
