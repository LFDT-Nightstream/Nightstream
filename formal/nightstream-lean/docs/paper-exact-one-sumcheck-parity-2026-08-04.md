# Paper-Exact One-SumCheck Parity Report

Date: 2026-08-04

The selected protocol follows corrected SuperNeo Section 7.3: one joint
SumCheck on one zero-padded row cube. Earlier two-SumCheck review material was
removed because it described a protocol that is not selected.

## Executive summary

The selected Lean `PaddedRowIdentity` path matches the corrected SuperNeo
protocol at the model-level and security-reduced tiers.

- Rectangular CCS support uses zero row padding.
- The row cube contains both the logical rows and all assignment columns.
- The verifier prepends `M_0 = [I; 0]`.
- `Pi_CCS` uses one joint polynomial and one 24-round SumCheck.
- There is no column point, column claim, or second SumCheck.
- The gamma blocks, absolute target, degree bound, `Pi_RLC`, and `Pi_DEC`
  follow the paper layout.
- The exact algebraic numerator is `10599 + 24 * 9 = 10815`.
- The Poseidon2 transcript absorbs the full statement identifier before the
  public NIFS input.
- The compact recursive verifier receives only the statement identifier and
  public NIFS input. Lean proves that it returns the same result as the full
  verifier.
- HyperNova Construction 2 uses the NIFS only in the recursive branch. The
  base and terminal branches do not add a SumCheck.

The result is not an unconditional security proof. Module-SIS, Poseidon2
random-oracle security, the bounded sampler event, and accepted-child
extraction remain named security boundaries. The corrected HyperNova H.2
application compiler also remains an explicit proof-carrying input.

## Paper sources

The review used these local sources:

- `docs/superneo-paper/07-7-neo-s-folding-scheme-for-ccs.md`
- `docs/superneo-paper/13-d-deferred-theorems-and-proofs.md`
- `docs/hypernova-paper/13_6_2_NIVC_Compatible_multi_folding_schemes.md`
- `docs/hypernova-paper/14_6_3_A_compiler_from_NIVC_compatible_folding_schemes_to_NIVC.md`
- `docs/hypernova-paper/39_H_2_Proof_of_Lemma_3_Folding_CCS_NIVC_compatibility.md`
- `docs/hypernova-paper/hypernova-paper-errata-v2.patch`

The SuperNeo source includes the corrected absolute-target convention. The
HyperNova Lean contract follows the corrected Definition 12 requirements.

## Protocol selection from first principles

The following facts are necessary:

1. The joint SumCheck domain must contain every logical constraint row.
2. `M_0 z` must expose the same padded assignment that the norm relation uses.
3. The row capacity must contain every assignment column.
4. Zero padding is valid because the selected constraint polynomial vanishes
   on an all-zero padded row.
5. The paper uses one joint polynomial `Q`, not separate FE and NC protocols.

The following items are implementation conventions, not mathematical changes:

- The selected row capacity is `2^24`.
- The assignment width is `11,437,038`.
- The application has 13 matrices. The joint relation has 14 matrices after
  the identity matrix is prepended.
- The selected profile has one fresh source and 14 running sources.

No second domain is necessary. The inequality
`11,437,038 <= 2^24` lets the padded identity matrix live on the same row cube.

## Equation-by-equation map

| Paper obligation | Lean owner | Result |
|---|---|---|
| One row cube for the padded relation | `PaddedRowIdentity.lean:104` | Model-level, exact |
| Canonical row/column embedding | `PaddedRowIdentity.lean:133` | Model-level, exact |
| `M_0 = [I; 0]` | `PaddedRowIdentity.lean:158` | Model-level, exact |
| Identity-first matrix family | `PaddedRowIdentity.lean:162` | Model-level, exact |
| Padded CCS iff logical CCS | `PaddedRowIdentity.lean:330` | Model-level, exact |
| `M_0 z` equals the padded assignment | `PaddedRowIdentity.lean:353` | Model-level, exact |
| Connected joint truth iff logical truth | `PaddedRowIdentity.lean:474` | Model-level, exact |
| One joint SumCheck verifier input | `PaddedRowIdentityConcreteNifs.lean:218` | Model-level, exact |
| Full output absorbed before `Pi_RLC` | `PaddedRowIdentityPoseidon2.lean:218` | Model-level, exact |
| Corrected mixing numerator `10599` | `PaddedRowIdentitySecurity.lean:234` | Security-reduced, exact |
| SumCheck numerator `24 * 9 = 216` | `PaddedRowIdentitySecurity.lean:237` | Security-reduced, exact |
| Total numerator `10815` | `PaddedRowIdentitySecurity.lean:240` | Security-reduced, exact |
| One-fold algebraic floor of 114 bits | `PaddedRowIdentitySecurity.lean:245` | Security-reduced, exact |
| 64-fold algebraic floor of 108 bits | `PaddedRowIdentitySecurity.lean:254` | Security-reduced, exact |

The generic `PaperJoint` modules own the exact gamma schedule, absolute target,
terminal equation, degree reasoning, transcript replay, and causal root-count
theorems. The selected `PaddedRowIdentity` modules instantiate that path. They
do not use `PaperRectangular` or `SplitNc` as protocol authority.

## Verifier data flow

```text
full parameters + running structure + fresh structure + verifier data
                              |
                              v
               canonical Poseidon2 statement identifier
                              |
                              v
              absorb identifier, then public NIFS input
                              |
                              v
                    one joint Pi_CCS SumCheck
                              |
                              v
                    absorb complete Pi_CCS output
                              |
                              v
                    verifier-derived Pi_RLC parent
                              |
                              v
                    operational Pi_DEC equations
                              |
                              v
                       new running product
```

The transcript identifier tag and initial absorption are in
`PaddedRowIdentityPoseidon2.lean:71-80`.

## Compact recursive verifier

Corrected HyperNova requires a compact recursive verifier projection and a
fixed-length identifier for the full statement. The generic interface is in
`HyperNova/NIVCCompatibility.lean:592`.

The concrete implementation uses an empty verifier-data projection and one
field element as the statement identifier. This is sound for executable
verification because Lean proves these facts:

- the joint SumCheck Boolean does not read Ajtai-key or matrix entries
  (`PaddedRowIdentityConcreteNifs.lean:302`);
- the computed `Pi_RLC` commitment, public input, point, and evaluations do
  not read those semantic values (`PaddedRowIdentityConcreteNifs.lean:323`);
- the `Pi_DEC` Boolean does not read those semantic values
  (`PaddedRowIdentityConcreteNifs.lean:446`);
- the full verifier equals the compact verifier
  (`PaddedRowIdentityConcreteNifs.lean:479`).

The full values are not discarded. The statement identifier binds them. The
outer verifier owns the full statement. The recursive circuit owns only the
identifier and the public NIFS input. Identifier collision remains an explicit
security event.

The concrete Definition 12 interface is in
`PaddedRowIdentityNIVCCompatibility.lean:397-467`. Construction 2 computes the
identifier from each slot's complete statement in
`PaddedRowIdentityNIVCCompatibility.lean:473-508`.

## HyperNova parity

The model-level HyperNova boundary now includes:

- canonical prefix-free codecs with both inverse directions;
- distinct running CE and fresh CCS types;
- decoding of each satisfying canonical tuple;
- monotonicity and rectangular capacity laws;
- a deterministic parameter-owned committed-zero default;
- a compact recursive verifier projection;
- a fixed-length full-statement identifier;
- Construction 2 base, recursive, and terminal branch separation.

The selected default and terminal relations are in
`PaddedRowIdentityHyperNova.lean:107-182` and
`PaddedRowIdentityHyperNova.lean:215-287`.

The repository does not contain a corrected generic H.2 compiler for every
application. `ApplicationCompiler` remains an explicit input at
`PaddedRowIdentityNIVCCompatibility.lean:568`. This is an honest model-level
boundary, not an omitted assumption hidden in a theorem.

## Rust conformance scope

The selected Rust proof variant is `PaddedRowIdentity` at
`crates/neo-reductions/src/engines/pi_ccs_protocol.rs:247`. Its canonical codec
encodes only the one joint SumCheck messages. The selected prover and verifier
reject a second SumCheck phase, column point, `s_col`, and `y_zcol`.

The independent PaperExact engine and optimized engine are byte-exact on
square shapes and on both rectangular directions. The parity test is
`crates/neo-reductions/tests/padded_row_identity_parity.rs:243`.

The recursive verifier now uses `pi_ccs_circuit`, the same padded-row
one-joint protocol. Focused native-versus-circuit tests synthesize and check the
selected production relation. CUDA and Metal do not define alternative
protocol messages.

This is Rust-conformant evidence for the selected `Pi_CCS` profile, its public
`Pi_RLC` and `Pi_DEC` cross-checks, and the live recursive circuit. It is not a
complete Lean artifact for every coefficient of the production matrix
payload.

## Remaining security and implementation boundaries

1. Prove or audit the Phi81 low-norm invertibility boundary.
2. Supply deployment bounds for Module-SIS, Poseidon2, sampler shortfall, and
   accepted-child extraction.
3. Instantiate the corrected H.2 application compiler with a real application.
4. Bind the complete production matrix payload to a reviewed Lean artifact if
   production certification requires coefficient-by-coefficient Lean replay.
5. Prove the downstream outer-proof and on-chain verifier refinements.

These items do not change the one-SumCheck protocol. They are the remaining
steps from model-level and security-reduced parity to production certification.

## Validation

The final focused validation results are:

| Gate | Result |
|---|---|
| Lean `PaddedRowIdentityConcreteNifs` build | Pass: 2,151 jobs |
| Lean `PaddedRowIdentityHyperNova` build | Pass: 2,161 jobs |
| Lean `PaddedRowIdentityNIVCCompatibility` build | Pass: 2,164 jobs |
| Focused padded-row axiom gates | Pass |
| Focused Rust-conformance and HyperNova axiom gates | Pass |
| Lean static layer, quarantine, ownership, trusted-hole, and assurance-data gates | Pass |
| Rust PiCCS parity | Pass: 13 of 13 tests |
| Rust PiCCS strictness | Pass: 7 of 7 tests |
| Rust complete-carrier checks | Pass: 3 of 3 tests |
| Rust PiRLC and PiDEC comparison | Pass: 11 of 11 tests |
| Rust output canonicalization | Pass: 2 of 2 tests |
| Rust public PiRLC boundary | Pass: 1 of 1 test |
| Selected recursive manifest | Pass: 8 of 8 tests |
| Production native fold | Pass |
| Recursive R1CS self-sufficient relation | Pass |
| Metal adapter parity | Pass: 3 of 3 tests |
| Workspace release compile | Pass |

The focused axiom reports contain only Lean's standard `propext`,
`Classical.choice`, and `Quot.sound` dependencies. The security assumptions
listed above remain explicit theorem inputs.
