# Lean-owned fixed-one NIFS deployment closure

Status: deployment closure is proved at the application-parametric boundary,
including the selected physical Split-NC refinement. This file does not claim
a ground application or a Rust-conformant program.

## Authority

The authority order is:

```text
SuperNeo and HyperNova relations
  -> Lean-owned selected key and operational profile
  -> Lean-owned canonical rows and receipts
  -> complete Lean manifest
  -> future Rust generation and equality proof
```

Rust rows, Rust row counts, generated artifacts, and diagnostic measurements
are not inputs to the selected profile or certification.

The selected NIFS operation order is `PiCCS -> PiRLC -> PiDEC`. The profile
uses:

- one fresh source;
- fourteen running sources;
- fifteen PiRLC coordinates;
- the complete 270-coordinate public carrier;
- decomposition base `b = 2`;
- Lean-owned Poseidon2 constants and transcript rules; and
- setup-owned relation data whose dimensions and matrix count remain explicit
  Lean values.

The shipping Rust matrix count of 13 is candidate configuration evidence only.
The Lean manifest gets its matrix count from `Dimensions`.

## Current theorem matrix

Every closed headline below has a fail-closed axiom guard. The measured axiom
set for the complete deployment path is:

```text
[propext, Classical.choice, Quot.sound]
```

It does not contain `Lean.trustCompiler` or `native_decide`.

| Obligation | Lean-owned construction or theorem | Status |
|---|---|---|
| Goldilocks primality | `GoldilocksField.goldilocks_natPrime` from the exact Lucas certificate | closed |
| Project divisor law | `GoldilocksField.goldilocks_euclidPrime` | closed |
| Field inverse | `GoldilocksField.goldilocksFieldInverse` | closed |
| Selected NIFS key | `ConcreteNifsCanonicalKey.selected` | closed, parameterized by setup relation data |
| Statement and output serialization | `ConcreteNifsCanonicalSerialization` | closed |
| Running carrier coverage | `ConcreteNifsCanonicalRunningCoverage` | closed for every physical codec coordinate |
| Proof serialization | `ConcreteNifsCanonicalProofCodec` | closed, including initial transcript state, cursor, polynomial, and PiDEC evaluation sizes |
| Operational profile | `ConcreteNifsCanonicalOperationalProfile.operational` | closed |
| Complete NIFS certification | `ConcreteNifsCanonicalCertification.nifs` | closed |
| Eleven-call Step and Terminal certification | `ConcreteNifsCanonicalCertification.complete` | closed |
| Occurrence-bound paper event transport | `ConcreteNifsCanonicalCertification.recursiveNifs_refinesPaper_or_boundEvent` | closed |
| Proof-free manifest | `ConcreteNifsCanonicalCertification.manifest` | closed |
| Exact cost split | `ConcreteNifsCanonicalCertification.manifest_stepCost_split` | closed |
| Exact application term | `ConcreteNifsCanonicalCertification.applicationStepCost_exact` | closed |
| Selected physical Split-NC coverage | `SelectedPhysicalRefinement.selectedSplitNc_covers_opening`; exact 41-digit/20-borrow support census and ownership | closed |
| Selected verifier rows plus canonical-opening rows | `SelectedVerifierRefinement.selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent`; exported by `ConcreteNifsCanonicalCertification.selectedCanonicalOpening_refines_or_securityEvent` | closed, with unchanged output-binding/algebraic security event |

## Deployment boundary

`ConcreteNifsCanonicalCertification.Deployment` contains only data that
HyperNova setup or the selected application must supply:

1. the application profile and its exact codecs;
2. a proof-carrying application `step` recipe;
3. admissibility of the setup-selected default running value; and
4. exact equality between the setup NIFS footprint slot and the footprint of
   the Lean-constructed operational profile.

The footprint field is a static program equality over every physical call
frame. It is not an acceptance assumption, challenge assumption, semantic
claim, or caller-supplied row count.

The application step remains a `CallRecipe`, not a supplied natural number.
Thus the exact total is:

```text
manifest.stepCost
  = manifest.fixedProtocolCost + manifest.applicationStepCost
```

and:

```text
manifest.applicationStepCost
  = signature.callCost Call.step
```

A ground numeric total is intentionally unavailable until the deployment
selects a real application and setup. This does not leave the protocol-owned
NIFS program conditional.

## Phase 6 interchange status

- `model-level` —
  `ConcreteNifsCanonicalRustExport.render` composes one proof-carrying
  `Deployment` directly with the certified manifest and the deterministic
  schema-v1 JSON encoder. `render_exact` proves that no Rust value or measured
  cost enters this export.
- `artifact-checked` — Rust decodes schema version 1 only through
  `LeanCanonicalManifest.from_json_slice`. The validator rejects unknown
  fields, noncanonical coefficients, allocation or receipt drift, an invalid
  fixed-one ABI, and cost or statistics drift. It recomputes the application
  cost from the canonical root Step receipt and the fixed cost from every
  other Step receipt.
- `artifact-checked` — `emit_step` and `emit_terminal` allocate the validated
  ownership classes and append each normalized Lean row without changing its
  coefficients. The focused release test compares every emitted A, B, and C
  sparse row with the manifest under the returned one-to-one column map.
- `model-level` — Matrix count remains setup-selected in Lean. Rust does not
  require the historical value 13.

This is not yet a `Rust-conformant` deployment. The repository has no ground
`Deployment` value and therefore no selected JSON artifact, numeric total, or
deployment witness constructor. The schema fixture in the Rust test is not a
production F-prime program and its counts are not protocol measurements.

The JSON file is generated structure, not protocol authority. A production
consumer must bind it to a selected Lean deployment by regeneration and exact
source equality. It must not accept an operator-selected runtime JSON file as
the verifier relation. The deployment must also construct every Step and
Terminal witness coordinate from authoritative call inputs and its selected
application recipe.

## Split-NC refinement

The model-level file
`CanonicalOpeningSplitNc.lean` proves the needed canonicality chain from
Split-NC coverage at `b = 2`. It covers all 41 digit coordinates and all 20
retained-borrow coordinates in its Lean-owned abstract production layout.

The selected physical refinement now:

1. names the selected Lean-owned physical production layout;
2. proves that every digit and retained-borrow column used by the emitted
   canonical-opening rows is in the selected Split-NC coverage relation;
3. derives the required ternary bound from those selected verifier rows; and
4. feeds that bound into the existing 21-row canonicality theorem.

`selectedPhysicalRows_encoded_lt_modulus` is the deterministic model theorem.
`selectedVerifierAndPhysicalRows_encoded_lt_modulus_or_securityEvent` is the
security-reduced end-to-end theorem. It keeps output-binding or algebraic
verifier failure as the existing named event. It does not convert that event
to success.

The canonical certification imports and exports this theorem as application
construction evidence. It does not claim that an arbitrary proof-carrying
`step` recipe emits these application-specific rows.

The generated-artifact refinement is not used because it is artifact
evidence, not Lean authority.

## Excluded shortcuts

- The standalone 23-row same-operands hash-separation program is not part of
  the complete manifest. The two real F-prime hash calls have different state
  and running operands. Each call uses its own per-call preimage recipe.
- A digest is never authority. The canonical serialization binds every
  dynamic field in a fixed order.
- A correct declared row or column count is not program equality.
- A caller-supplied acceptance proposition is not a certification.
- A Rust constant, row, measurement, or generated layout is not a Lean setup
  value.
- Artifact theorems that use `Lean.trustCompiler` remain quarantined and are
  not imported by the canonical deployment path.

## Validation

Validated on 2026-07-29:

- focused canonical deployment build: pass;
- aggregate axiom gate: pass, 4,964 jobs;
- full Lean build: pass, 5,557 jobs;
- executable check: pass;
- focused Rust schema and exact-row emission test: pass, 12 tests;
- `neo-fold-clean` Rust all-target check: pass;
- structural gate: red on the repository-wide pre-existing ownership-header
  backlog in generated, artifact, and paper-reference modules; no violation
  was reported in the canonical deployment files.

## Promotion condition

The Lean-owned Phase 5 deployment is complete at the application-parametric
boundary. A deployment must still select its application/setup to obtain a
ground manifest and numeric total.

Rust replacement starts only after that promotion and after a deployment
selects the application/setup needed to evaluate one ground manifest. Phase 6
then requires:

1. a ground Lean `Deployment` value;
2. a generated manifest bound to that exact Lean source;
3. a deployment witness constructor for all manifest columns;
4. integration of the emitted Step and Terminal relations into the lifecycle;
   and
5. an exact comparison against the replaced Rust path before that path is
   removed.
