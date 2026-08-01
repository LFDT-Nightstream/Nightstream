# CIR-CURRENT-LEAN-M4

```text
property_id: CIR-CURRENT-LEAN-M4
assurance_state: model-proved for the focused theorem and domain capacity;
                 MSIS security validation open
milestone_status: local deployment correspondence complete;
                  generic recursive fixed-point construction complete;
                  42-times-6 benchmark recursive M4 theorem complete;
                  historical/current domain coexistence explicit;
                  production deployment selection open
shipping_status: current Rust equality open
profile: fixed-one, plain carrier width 270, proof-carrying application
         deployment; reduced 42-times-6 WASM method fixture
```

## Claim

For every complete Lean-owned `Deployment`, the exact finite Step and
Terminal programs emitted by that deployment have one assembled local
correspondence theorem:

`CurrentM4.deployment_local_correspondence`.

The theorem constructs all evidence from the deployment and its emitted
programs. A caller does not supply a semantic execution, accepted
proposition, row count, owner map, or security event.

Full recursive M4 also needs `CurrentM4.RecursiveSystemCoherence`. This
condition states that the relation inside the selected NIFS key is exactly
the thirteen-matrix relation compiled from the complete Step rows.

`CurrentFixedPoint.Family.m4` now constructs this evidence for every
proof-carrying deployment family that proves:

1. the exact recursive row, logical-width, and matrix-count equations; and
2. that rebuilding the deployment after the seed-system replacement compiles
   to the same physical relation.

`WasmBenchmark42x6.CurrentM4Family.family` constructs such a family for the
Lean-owned 42-times-6 benchmark Step program.
`WasmBenchmark42x6.CurrentM4Family.m4Evidence` then constructs full current
recursive M4 evidence for its exact 19,859,562-row,
19,969,313-owned-column encoding. The application program is fixed. The
setup verifier key is the only explicit `CurrentM4Family.Template` field
because it is setup-owned authority. Domain coverage and row nonemptiness are
derived from the emitted encoding.

This benchmark uses the reduced test lifecycle profile. The theorem is a
model-level method fixture. It does not select a production deployment and
does not establish Rust equality or a security reduction.

The separate [WASM application module proof](wasm-application-module-proof.md)
now closes a 63-row deployment-local backend path for the exact module bytes
and result. It proves and verifies with Spartan and WHIR. It does not replace
the 19,859,562-row recursive M4 theorem described here, and it does not claim
that the full recursive terminal relation fits the current backend.

## Evidence matrix

| M4 obligation | Current Lean theorem |
|---|---|
| Step soundness | `CurrentStepPhysicalRefinement.deployment_step_refines_from_physical_rows` |
| Terminal soundness | `CurrentTerminalPhysicalRefinement.deployment_terminal_refines_from_physical_rows` |
| Step honest completeness | `CurrentDeployment.deployment_step_cir_complete` |
| Terminal honest completeness | `CurrentDeployment.deployment_terminal_cir_complete` |
| Exact row and receipt ownership | `CurrentCompiler.obligationTree` |
| Exact row and cost accounting | `CurrentCompiler.obligationTree` and `CurrentCompiler.evidence` |
| Canonical field manifest | `CurrentCompiler.manifestCanonical` |
| Canonical Step and Terminal claims | `CurrentDeployment.deployment_structural_evidence` |
| Application-input codec recovery | `Deployment.applicationCodecRecovery` |
| Assembled local result | `CurrentM4.deployment_local_correspondence` |
| Recursive same-system boundary | `CurrentM4.RecursiveSystemCoherence` |
| Exact relation compiler | `CurrentCompiler.compiledSystem` |
| Generic recursive fixed point | `CurrentFixedPoint.Family.recursiveSystemCoherence` |
| Full recursive result | `CurrentFixedPoint.Family.m4` |
| Benchmark physical column stability | `WasmBenchmark42x6.CurrentM4PhysicalStability.columnIds_eq_of_constraintPolynomial_eq` |
| Benchmark complete Step row stability | `WasmBenchmark42x6.CurrentM4StepRowsAggregate.rows_eq_of_constraintPolynomial_eq` |
| Exact benchmark Step cost | `WasmBenchmark42x6.CurrentM4Cost.stepCost_exact` |
| Exact emitted rows and columns | `WasmBenchmark42x6.CurrentM4Cost.encodingRows_exact` and `encodingColumns_exact` |
| Exact carrier and block census | `WasmBenchmark42x6.CurrentM4Domain.carrierWidth_exact` and `liveBlockCount_exact` |
| Current domain capacity | `WasmBenchmark42x6.CurrentM4Domain.currentNc_covers` and `currentFe_covers` |
| Minimal block-variable count | `WasmBenchmark42x6.CurrentM4Domain.blockVariables_minimal` |
| Benchmark recursive family | `WasmBenchmark42x6.CurrentM4Family.family` |
| Benchmark current M4 evidence | `WasmBenchmark42x6.CurrentM4Family.m4Evidence` |

Step soundness is branch-complete. The base branch reaches the frozen base
transition. The recursive branch reaches the selected NIFS output and either
the paper relation or the unchanged occurrence-bound event.

## Application boundary

Application-specific constraints belong to the proof-carrying
`deployment.step` recipe. The generic F-prime program does not add a second
application program.

The selected 21-row canonical-opening gadget is therefore not a universal
F-prime row family. A deployment compiler that emits that gadget must prove
its own activation and can use
`selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent`.

## Shipping-application audit

Artifact check on 2026-07-29 found that the public application path is
program-parametric WASM, not the Fibonacci fixture:

```text
WasmProgramArtifacts
  -> neo_wasm::preprocess
  -> NebulaApplication
  -> NebulaFPrimePreprocessing
  -> neo_wasm::prove / neo_wasm::verify
```

`neo_wasm::preprocess` takes one program's parsed artifacts, initial locals,
and exported entry point. No non-test repository consumer selects one concrete
program for a shipping deployment.

The Fibonacci F-prime frontend is below `neo-fold-clean/tests/support` and its
paper-layer documentation names it as a test fixture. It is not the shipping
application.

The available WASM/Metal benchmark is now the first current recursive M4
method fixture. Its WAT
program loads `42` from linear memory, multiplies it by `6`, and returns
`252`. However, the benchmark calls `WasmNebulaProfile.test_profile()` and
`preprocess_seeded_reduced_memory_test_only`. It therefore cannot establish a
production deployment selection.

Consequently current-shipping M4 has one genuine selection boundary: a
deployment must select concrete WASM bytes, an exported entry point, initial
locals, and the production WASM profile. A benchmark deployment may be
constructed and checked without promoting it to shipping evidence. The
benchmark proof also does not validate the MSIS security shape of the
`25/19/6` transcript domain. That check remains required before production
selection.

## Corrected footprint and domain capacity

The benchmark footprint uses SuperNeo commitment rank `κ = 18`. It does not
use the expansion factor `T = 216` as a commitment width. An earlier model
passed `216` at that boundary and reported 68,002,318 rows and 68,411,505
columns. Those figures are withdrawn. They represented a twelve-fold
commitment-width expansion, not the selected verifier.

The corrected receipt-derived values are:

| Quantity | Exact value |
|---|---:|
| Selected NIFS static rows | 19,773,612 |
| Complete Step rows | 19,859,562 |
| Complete Step owned columns | 19,969,313 |
| Completed Phi81 carrier width | 19,969,362 |
| Live Phi81 blocks | 369,803 |
| `2^18` block capacity | 262,144 |
| `2^19` block capacity | 524,288 |

Thus 18 block variables are insufficient and 19 are minimal. The selected
current domain is `25/19/6`: 25 row and flat-column variables, 19 block
variables, and 6 lane variables.

The captured historical correspondence uses
`fixedPointProduction = 24/19/6`. The captured domain and pending-family
modules select that name directly. Current Lean-owned deployment modules use
`currentLeanProduction = 25/19/6`. The global `production` alias denotes the
current domain.

This separation is a typed ownership rule. It does not prove that the
`25/19/6` shape satisfies the production MSIS security parameters.

## Five M4 scopes

1. The historical captured-artifact M4 remains complete only for its exact
   4,193,134-row snapshot.
2. The current Lean-owned local Step and Terminal correspondence is complete
   for every proof-carrying `Deployment`.
3. The generic current Lean-owned recursive fixed-point theorem is complete.
   It starts with a shape-only seed, compiles the exact Step rows, installs the
   compiled relation, and proves same-system coherence for every
   `CurrentFixedPoint.Family`.
4. The reduced 42-times-6 benchmark has a focused current recursive M4 theorem
   for its exact Lean-owned 19,859,562-row program. Its `25/19/6` domain
   capacity and minimal 19-variable block axis are proved. This is a method
   fixture, not a production selection.
5. The current shipping M4 also needs one production deployment and exact equality
   between generated Rust and its closed Lean manifest.

The historical row-range obstruction blocks promotion of the stale artifact.
It does not block the new Lean-owned compiler route.

## Exclusions

The current benchmark property does not:

- select the reduced benchmark as a production deployment;
- construct the setup-owned verifier key;
- validate the production MSIS security shape for the `25/19/6` domain;
- prove that Rust emits the Lean manifest;
- treat Rust row counts or tests as protocol authority;
- add application-specific canonical-opening rows;
- bound the probability of a named security event.

## Trust boundary

The generic fail-closed guard is
`tests/Axioms/FPrimeFullHistorySelectiveCcsCurrentM4.lean`.
The benchmark-specific guard is
`tests/Axioms/WasmBenchmark42x6CurrentM4.lean`.
The measured axiom sets for
`CurrentM4.deployment_local_correspondence` and
`CurrentM4.deployment_m4`, and for
`CurrentFixedPoint.Family.recursiveSystemCoherence` and
`CurrentFixedPoint.Family.m4`, plus the benchmark family and
`WasmBenchmark42x6.CurrentM4Family.m4Evidence`, are:

```text
[propext, Classical.choice, Quot.sound]
```

It does not contain `Lean.trustCompiler`.

`CurrentM4Domain.currentNc_covers` depends on no axioms.
`CurrentM4Domain.blockVariables_minimal` has the measured set:

```text
[propext, Quot.sound]
```
