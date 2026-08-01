# WASM application module proof

```text
property_id: WASM-APPLICATION-MODULE-PROOF
deployment: wasm-benchmark-42x6
model_level: complete for the application module relation
artifact_checked: exact Lean JSON fixtures checked
rust_conformant: direct Spartan and WHIR lifecycle checked
security_reduced: conditional assumptions stated; not complete
recursive_f_prime_terminal: excluded
```

## Scope

This property is the first concrete use of the program-parametric WASM
deployment boundary. The protocol code does not contain the 42-times-6
constants. Lean defines one module instance and exports two deterministic
manifests:

1. exact WASM bytes and parser-visible module facts; and
2. the proof relation, physical columns, rows, cost, and density data.

Rust validates and consumes those manifests. The test executes the exact
module in Wasmtime, obtains 252, proves the Lean-owned relation with Spartan,
opens it with WHIR, and verifies against the verifier-owned module bytes and
output.

This property does not prove the complete recursive F-prime terminal
relation. That relation remains a separate backend problem.

## Exact physical census

| Quantity | Exact value |
|---|---:|
| WASM bytes | 61 |
| Native CCS rows | 63 |
| Terminal R1CS rows | 63 |
| Committed columns | 0 |
| Lean public columns, including constant one | 63 |
| Private auxiliary columns | 1 |
| Total physical columns | 64 |
| Spartan public values, excluding implicit one | 62 |
| Spartan private witnesses | 1 |
| R1CS nonzero coefficients | 173 |
| Native CCS nonzero coefficients | 236 |
| Maximum R1CS row density | 3 |
| Maximum native CCS row density | 4 |
| Poseidon2 calls in the relation | 0 |
| Maximum live witness columns | 1 |

The rows are 61 byte pins, one multiplication, and one output link. Every
native selector is the verifier-fixed constant-one column. The terminal
lowering therefore removes the selector without adding a row or column.

## Deployment-local M4 evidence

| Obligation | Lean evidence |
|---|---|
| Exact module bytes | `WasmBenchmark42x6.module_bytes_exact` |
| Module computes 252 | `WasmBenchmark42x6.module_computes_252` |
| Native CCS soundness | `ModuleProofProgram.soundness` |
| Honest completeness | `ModuleProofProgram.honest_satisfies` |
| Finite compiler acceptance | `ModuleProofProgram.finite_accepts_honest` |
| Unique physical columns | `ModuleProofProgram.program_columnIds_nodup` |
| Unique row occurrences | `ModuleProofProgram.program_rowIds_nodup` |
| Receipt-derived cost | `ModuleProofProgram.program_cost_exact` |
| Exact terminal rows | `ModuleProofR1csLowering.selected_rows_exact` |
| CCS/R1CS equivalence | `ModuleProofR1csLowering.satisfies_iff` |
| Terminal soundness | `ModuleProofR1csLowering.soundness` |
| Manifest uses the same rows | `ModuleProofR1csLowering.manifest_rows_exact` |
| Assembled local evidence | `ModuleProofEvidence.m4` |

The fail-closed axiom guard is
`tests/Axioms/WasmBenchmark42x6ModuleProof.lean`. Its model theorems use only
the measured sets `[propext, Quot.sound]` or
`[propext, Classical.choice, Quot.sound]`. No guarded theorem uses
`Lean.trustCompiler`.

## Deployment-local M5 evidence

`Nightstream.Checks.Rust` checks both JSON fixtures byte for byte against the
Lean renderers. `neo_wasm::WasmApplicationProofSystem` then:

1. validates the schema, module identifier, exact bytes, Goldilocks modulus,
   four-matrix polynomial, column roles, selectors, rows, cost, and density;
2. requires every module byte to have its exact public pin row;
3. maps the ordered Lean columns to Spartan's private, one, and public order;
4. builds canonical sparse A, B, and C matrices;
5. proves with the three-member lockstep Spartan protocol and WHIR; and
6. verifies against public values reconstructed by the verifier.

The Rust tests reject wrong witnesses, wrong outputs, selector changes,
module-byte changes, polynomial changes, role changes, cost changes, density
changes, and a corrupted proof. The accepted output comes from a real Rust
Wasmtime execution. It is not a hand-authored acceptance bit.

This is Rust conformance for the application module relation only. It is not
Rust conformance for the complete recursive F-prime relation.

## M6 assumptions and open boundary

The proof lifecycle still depends on these cryptographic assumptions:

- Poseidon2 Fiat--Shamir is a random oracle for the three lockstep sum-check
  members;
- WHIR is binding and sound for the selected parameters; and
- the stated algebraic error accounting is valid for the Goldilocks field.

These assumptions are explicit. This property does not claim a machine-checked
end-to-end security reduction.

The complete recursive terminal relation is not sent to this backend. Its
current resource census is 171,261,238 rows, 165,937,070 private columns, and
78,069,863,880 Ajtai coefficient slots. The small module proof is a sound
deployable leaf and a reusable backend method. It is not a substitute for
that recursive relation.
