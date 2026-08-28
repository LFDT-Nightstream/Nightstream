# Nightstream F′ Stage 1 constraint tree

This file is the concise audit map for the current Lean-authored package
prefix. It defines no relation and gives no digest authority. The audit path
is:

```text
paper formula
  → Lean predicate
  → leaf FormalCircuit
  → phase assembler
  → proved physical layout
  → canonical Lean package
  → exact Rust matrices and independent assignment check
```

Status on this cut:

- Pilot: validated candidate; status open on the current identity.
- PiCCS: all phase-local gates are green; formally **status open** under the
  owner decision until the final fixed-point edge and final-identity rerun.
- PiRLC: validated candidate; status open until an external reviewer approves
  this exact source and artifact cut.
- PiDEC: phase-local work is owner-authorized and all local gates below are
  green; status open until exact external review.
- Stage 1: open. The application, terminal assembly, fixed point, complete
  domain proof, security composition, and package-only production path do not
  exist yet.

The external review files read at 22:20 CDT predate the verifier-context and
accepted-fixture correction in this cut. They are evidence for prior cuts
only. No status below uses them as approval of the current cut.

## Fixed profile and semantic authority

Lean is the semantic authority for exact SuperNeo v1_1. The fixed Nightstream
Goldilocks profile is:

| Parameter | Value |
|---|---:|
| Base `b` | 2 |
| `k_rho` | 16 |
| Bound `B` | 65,536 |
| Fresh / running PiCCS inputs | 1 / 16 |
| PiRLC inputs | 17 |
| PiDEC children | 16 |
| Ring degree | 54 |
| CCS matrices | 14 |
| PiCCS rounds | 26 |
| PiCCS round coefficients | 10 |
| Public-input words | 270 |

`Eval_K` is the separate Pad family. `Eval_A` is the separate 14-matrix
family. No v1.0 Pad-as-matrix-zero encoding is present. Transcript, state,
package-identity, and verifier-context binding use Poseidon2 only.

The public digest encoding is exact `encHash`:

```text
[marker = 1, 256 little-endian digest bits, 13 zero cells]
```

Rust now uses this same 270-cell encoding. `decodeHash_encHash` recovers every
canonical four-word digest, and `encHash_injective_fixed` proves that two such
public inputs are equal only when their digests are equal. The canonical state
preimage is 45,933 Goldilocks words. Each running group contains 972
commitment words, 270 public-input words, and 1,620 evaluation words.

`Layout.ProductionRelation.Plan` is the key-facing matrix authority boundary.
It derives all 14 typed SuperNeo matrices from 13 meaningful sparse row forms
in the canonical Boolean-row order; matrix slot 13 and every padding row are
zero by construction. `Plan.matrixVectorAt_matrix` proves that each matrix
image equals its sparse row-form evaluation. `ProductionRelation.polynomial_zeroImages`
proves that the fixed 74-term polynomial accepts the all-zero padding rows.
The phase compiler still must construct the complete low-norm plan before
these matrices can replace the current package relation metadata.

## Logical phase hierarchy

```text
Lifecycle/Pilot.lean                         ✓ two hash children
Lifecycle/PiCCS/v1_1/Formal.lean             ✓ twelve-child assembler
Lifecycle/PiRLC/v1_1/Formal.lean             ✓ seven-child assembler
Lifecycle/PiDEC/v1_1/Formal.lean             ✓ six-child assembler
Lifecycle/Stage1/Formal.lean                 ○ full Stage 1 assembler
```

PiCCS leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Statement binding | 160 | 0 |
| Digest-only statement absorption | 192,400 | 192,400 |
| Challenge derivation | 47,952 | 47,952 |
| Round transcript | 138,528 | 138,528 |
| Initial claim | 116,631 | 116,631 |
| SumCheck chain | 393,959 | 393,907 |
| `Eval_K` terminal | 8,486 | 8,486 |
| `Eval_A` terminal | 109,574 | 109,574 |
| CCS terminal | 20,794 | 20,794 |
| Norm terminal | 752 | 752 |
| Final identity | 130,447 | 130,445 |
| Output binding | 4,076,512 | 4,076,512 |

PiRLC leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Input binding | 0 | 0 |
| Sampler chain | 1,008,848 | 1,007,199 |
| Commitment combination | 2,495,124 | 2,495,124 |
| Public-input combination | 693,090 | 693,090 |
| `Eval_K` combination | 277,236 | 277,236 |
| `Eval_A` combination | 3,881,304 | 3,881,304 |
| Output binding | 0 | 0 |

PiDEC leaf ownership:

| Leaf | Rows | Column delta |
|---|---:|---:|
| Input binding | 0 | 0 |
| Public-input split and range checks | 22,680 | 18,090 |
| Commitment recomposition | 972 | 0 |
| `Eval_K` recomposition | 108 | 0 |
| `Eval_A` recomposition | 1,512 | 0 |
| Output binding | 0 | 0 |

The PiDEC verifier first checks centered parent magnitude `< 2^16`. Its
accepted path then uses the exact 16 signed radix-two digits. The computable
decision agrees with the semantic predicate. The accepted path cannot use
`fallbackDigit` for an out-of-range parent.

## Proved cumulative layout through PiDEC

The authoritative ledgers are:

- `PilotProduction.physicalRowCountValue_eq` and
  `PilotProduction.physicalColumnCount_eq`;
- `PilotPiCCS.cumulativeFootprints_eq`;
- `PilotPiCCSPiRLC.cumulativeFootprints_eq`;
- `PilotPiCCSPiRLCPiDEC.cumulativeFootprints_eq`;
- `Export.Stage1.Package.circuitPackage_layout_values`;
- `Export.Stage1.Package.circuitPackage_jointDomain_le_twoPow26`.

| Endpoint | Rows | Source columns / joint domain |
|---|---:|---:|
| Pilot | 13,599,570 | 13,691,432 |
| PiCCS input ABI | 13,599,570 | 13,720,468 |
| PiCCS statement binding | 13,599,730 | 13,720,468 |
| PiCCS statement absorption | 13,792,130 | 13,912,868 |
| PiCCS challenge derivation | 13,840,082 | 13,960,820 |
| PiCCS round transcript | 13,978,610 | 14,099,348 |
| PiCCS initial claim | 14,095,241 | 14,215,979 |
| PiCCS SumCheck chain | 14,489,200 | 14,609,886 |
| PiCCS `Eval_K` terminal | 14,497,686 | 14,618,372 |
| PiCCS `Eval_A` terminal | 14,607,260 | 14,727,946 |
| PiCCS CCS terminal | 14,628,054 | 14,748,740 |
| PiCCS norm terminal | 14,628,806 | 14,749,492 |
| PiCCS final identity | 14,759,253 | 14,879,937 |
| PiCCS output binding | 18,835,765 | 18,956,449 |
| PiRLC sampler chain | 19,844,613 | 19,963,648 |
| PiRLC commitment combination | 22,339,737 | 22,458,772 |
| PiRLC public-input combination | 23,032,827 | 23,151,862 |
| PiRLC `Eval_K` combination | 23,310,063 | 23,429,098 |
| PiRLC `Eval_A` combination | 27,191,367 | 27,310,402 |
| PiDEC input ABI | 27,191,367 | 27,356,194 |
| PiDEC public split | 27,214,047 | 27,374,284 |
| PiDEC commitment | 27,215,019 | 27,374,284 |
| PiDEC `Eval_K` | 27,215,127 | 27,374,284 |
| PiDEC `Eval_A` / current endpoint | 27,216,639 | 27,374,284 |

The current joint domain is 27,374,284. It is below
`2^26 = 67,108,864` with exact headroom 39,734,580. This headroom still must
contain the accumulator, running-instance transition, application, output
hash, and terminal checks. Compact serialization does not reduce backend
rows.

## Canonical package cut

| Package value | Exact value |
|---|---:|
| Unpadded rows | 27,216,639 |
| Private columns / constant source index | 27,374,006 |
| Public columns | 278 |
| Total unpadded columns | 27,374,285 |
| Caller-owned private inputs | 166,690 |
| Witness instructions | 884,207 |
| Assertion rows | 144,079 |
| Ordinary compiled rows | 1,028,286 |
| Poseidon2 invocations | 7,679 |
| Compact templates | 326 |
| Compact invocations | 167,246 |
| Padded matrix rows | 67,108,864 |

The verifier-owned Poseidon2 relation identifier is:

```text
[12756407480944487176, 17097603764386178571,
 11791428871054057896, 14346937702828624285]
```

The external-review source-cut fingerprint is:

```text
4baeea044e83ad791f03376077ea3838bbc5e85bb48636737cc15ce81c88d257
```

It is the SHA-256 of the sorted `shasum -a 256` records for all files under
`formal/nightstream-fprime`, `crates/nightstream-fprime`,
`crates/neo-fold-clean/{src,tests}`, and `crates/neo-reductions/src`, plus the
three Rust `Cargo.toml` files. It excludes `.lake`, generated artifacts, and
this audit file. Paths are part of each inner record.

SHA-256 identifies bytes only:

| Artifact | Bytes | SHA-256 |
|---|---:|---|
| Compact schema-8 plan | 71,897,785 | `238c41d103ad88ade0f8d5321d6d88c5a8ce5f100b279bbf2029d368ce3164fb` |
| Expanded schema-7 package | 93,257,213 | `0bc67f9b4d37cdb0bf40c72edfca9688e926c714ba2e9fdd62f166b383bd7855` |
| Pilot parity | 657,628 | `9657505d5400ab752c4fbc0b6acf34eff063ceda042c9d415f7b6f4e184a4862` |
| PiCCS parity | 1,576,908 | `752538cfa28559aee1f0e5b6f4440d02996438bcbb7d8df77f68479b046d6422` |
| PiRLC parity | 1,141,393 | `fd7ed2839dc6b1a9ded83387599c08d23a7bf048ee5d59f434d1e33d1fbd7f85` |
| PiDEC parity | 1,132,082 | `772ab2adf11063c31ee4390c43bf52a34449e6ecdbfa6cd2ccf67d16f1dded25` |
| PiRLC sampler parity | 9,202 | `e1ee42037d7750725c9442d7693b93eb60dd56c5507577370d4f06e65aad88a3` |

## Exact conformance evidence

Lean evidence on this cut:

- `validate.sh static`: all boundary checks passed in 7.6 seconds.
- focused verifier-context compile: passed in 10 seconds.
- `validate.sh axioms`: 3,327 jobs passed in 13 seconds after the final
  26-round context and accepted-fixture correction. Audited theorems use only
  `propext`, `Classical.choice`, and `Quot.sound`.
- forced compact package emission: 11 seconds with `LEAN_NUM_THREADS=10`;
  the complete 71,897,785-byte output was byte-identical to the checked-in
  artifact.
- changing the context descriptor and accepted fixture did not change the
  package rows, package bytes, or package identity. All four dependent parity
  fixtures were regenerated from Lean; the independent sampler was
  byte-identical.

The speed path preserves the generic specifications. It uses proved direct
symbolic states, shared PiCCS operation packets, linear Horner construction,
and streamed row classification. The emitter uses `LEAN_NUM_THREADS=10` and
writes one canonical order.

Rust evidence on this cut:

- exact package matrix conformance: 1/1 in 41.90 seconds;
- exact final matrix nonzeros `A/B/C`:
  `[87,898,768, 36,805,391, 26,911,986]`;
- independent assignment evaluation: all 27,216,639 unpadded rows and the
  padded zero domain passed;
- row-owner mutations: 138;
- column/public-owner mutations: 71;
- semantic input mutations: 16;
- total exact-package mutations: 225;
- strict package loader: 14/14 in 109.13 seconds;
- compact-plan loader: 9/9 in 78.26 seconds;
- pilot parity and mutations: 3/3 in 20.55 seconds;
- complete PiCCS Lean / PaperExact / optimized parity: 4/4 in 4.07 seconds;
- complete indexed PiRLC parity and handoff: 3/3 in 2.11 seconds;
- complete PiDEC Lean / PaperExact / optimized parity: 3/3 in 28.42 seconds;
- PiRLC sampler parity and fail-closed decoding: 2/2;
- identity-bound complete typed package consumer: 1/1 in 256.37 seconds;
  it consumed PiCCS and PiDEC inputs and rejected a changed public input.
- `nifs_engine_crosscheck`: 10/10 in 262.90 seconds, including the 270-word
  state-preimage bridge, PaperExact/optimized equality, carried accumulator,
  and Nebula auxiliary commitments;

The matrix comparator expands the Lean plan independently and compares every
final A/B/C row, column index, canonical coefficient, constant placement, and
public mapping. The assignment evaluator does not call the matrix builder,
row expander, witness generator, or package constraint evaluator.

The package consumer is a test bridge. It is not the final production
lifecycle, and its direct package proof is not evidence that Rust implements
the Lean relation. The semantic, matrix, assignment, and parity gates above
provide that evidence.

Superseded native v1.0 and radix-four test targets were removed from Cargo and
the tree. They are recoverable from Git. Their required v1.1 properties are
covered by the exact phase and package gates above. Current tests that still
serve a v1.1 contract were migrated to separate `Eval_K` / `Eval_A` and the
canonical nonempty running accumulator.

## Open authority and assembly edges

PiCCS must remain status open until this exact edge exists:

```text
final canonical package
  → exact 14-matrix LogicalRelation
  → ProductionKey.key
  → exact application F
  → StepHolds
  → recursive fixed point
  → rerun every PiCCS gate on the final identity
```

The remaining owner-ordered work is:

1. obtain external approval of the exact PiRLC and PiDEC source and artifact
   cut;
2. add the accumulator and running-instance phase assemblers;
3. add the exact application, output-hash, and terminal circuits;
4. build `Lifecycle/Stage1/Formal.lean` and the matching Stage 1 layout;
5. prove cross-phase wiring, deterministic soundness, the recursive fixed
   point, the complete `2^26` bound, and the separate security composition;
6. make the validated package the only reachable Rust production relation
   and remove the alternate radix relation;
7. rerun all PiCCS, PiRLC, PiDEC, matrix, assignment, parity, and mutation
   gates on the final package identity;
8. after separate owner approval of a production backend, execute the final
   production `prove → verify` obligation.

No backend is authorized on this cut. Backend acceptance cannot replace any
semantic or conformance gate in this file.
