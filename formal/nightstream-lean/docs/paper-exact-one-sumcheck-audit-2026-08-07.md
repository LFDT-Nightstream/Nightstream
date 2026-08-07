# Paper-Exact One-SumCheck Audit

Date: 2026-08-07

## Result

The selected `PaddedRowIdentity` Lean path now implements the corrected
SuperNeo protocol with one SumCheck. Its only mathematical specialization is
the permitted rectangular relation shape:

- `m = 2^24` rows;
- `n_F = 11,437,038 <= m` assignment columns;
- `M_1 = [I; 0]` on the common row domain;
- zero padding after the `14,944,219` logical rows;
- one 24-round row SumCheck;
- no column point, column claim, column opening, or column SumCheck.

The selected HyperNova semantic and public-verifier path also matches the
corrected Definition 12 and Construction 2 branch syntax. In particular, the
outer base proof is the unique `bottom` value. It has no recursive payload.
The first prover step uses fixed dummy advice, and the verifier checks the
base proof before it accepts.

This result is paper-exact at the semantic-model tier. It is conditional at
the security-reduction tier and incomplete at the generated-artifact tier.
The open boundaries are listed below. The report does not turn a stated
assumption or a missing artifact into a proved result.

## Authority and audit scope

The corrected Markdown trees in `docs/` are the only paper authority for this
audit. PDF text and older parity reports are not authority.

The audit reviewed the full two Markdown collections and made a declaration-
by-declaration comparison for these protocol-critical parts:

- SuperNeo Definitions 11 through 14, `Pi_CCS`, `Pi_RLC`, `Pi_DEC`, the
  concrete parameter appendix, and the deferred security proofs;
- HyperNova Definition 12, Construction 2, Construction 3, Lemmas 3 and 4,
  and Appendices H.2 and H.3;
- selected Lean protocol models, concrete instantiations, transcript code,
  public verifiers, terminal verifiers, frozen contracts, generated Rust
  receipts, and conformance tests;
- the SuperNeo protocol contract and its source lock.

The reviewed working-tree source hashes are:

| Source | SHA-256 |
|---|---|
| `docs/superneo-paper/INDEX.md` | `dd6ac6832d8f933928806357a8157d26033448ad594e5838665275205fd3b128` |
| SuperNeo Section 7 | `46a68727c2abfb5b856517a831bfa8b6f625bf508ae9dd9694a9b33e2e49fbde` |
| SuperNeo Appendix B | `8d4f3dc3ab252bf7ee17bf383c1f23679124c356a8729f0592af140b703e3bf5` |
| SuperNeo Appendix D | `37131dd724623d9599ff222c7f143182e04fc73e8dae69cb954c7cb253dd24cf` |
| `docs/hypernova-paper/INDEX.md` | `865fe3b9d1d8009eb13f12f951cd00cbd9da1a13841eb10cf99a144d4b4e9f34` |
| HyperNova Definition 12 | `316b269cc75af8e5be66042adba625c8d539efd36f9accf4bd4132e10651630d` |
| HyperNova Construction 2 | `afc14fc03acbb613745733c3c1a12d33d61cd6174b7a6a93ee9a8efa4c0503fe` |
| HyperNova Construction 3 | `c13f80ce4b8503fe4aeac7c1f3bd2926aaaa717dc2f910feb40cde9b3a88a68f` |
| HyperNova Appendix H.2 | `8e03c3e1665b1fc0748d489e0029050f23e011df9bd7119d784c5d62821df5fb` |
| HyperNova Appendix H.3 | `ae1d9b8bcd2f7a69f8c900281f3ee66c6f70fa55df64c906888ad1c08e8c065f` |

The SuperNeo hashes agree with `protocol-contract/src/sources/lock.toml`.
The protocol contract is SuperNeo-specific. This report records the
HyperNova hashes because that source tree is not in that lock.

## Assurance terms

This report uses five separate assurance tiers:

1. **Paper syntax:** the local model states the corrected equations and data
   flow.
2. **Semantic model:** Lean proves equivalence between the selected relation,
   verifier, and the independent paper transition.
3. **Security reduced:** Lean reduces failure to explicit algebraic,
   commitment, transcript, sampler, or extraction events.
4. **Rust conformant:** generated values and selected Rust behavior agree with
   the reviewed Lean boundary.
5. **Artifact checked:** a generated production circuit or matrix artifact is
   decoded and proved equivalent coefficient by coefficient.

Passing an earlier tier does not imply a later tier.

## SuperNeo comparison

### Relation shape and normalization

Corrected Definition 11 permits rectangular matrices
`M_j in F^(m x n_F)` when `m` is a power of two and `n_F <= m`. It fixes the
first matrix as `M_1 = [I; 0]`. Corrected Section 7.3 lets an application choose
any power-of-two `m >= max(m', n_F)`. Therefore, the selected `m = 2^24` is a
valid application specialization. The `m = 2^30` value in Appendix B.2 is one
tabulated estimate. It is not a universal protocol constant.

Lean owner:
`Nightstream/Implementation/R1CS/Correspondence/FPrimeFullHistory/SelectiveCcs/PaddedRowIdentity.lean`.

The file proves these exact facts:

- all logical rows and assignment columns fit in the same row cube;
- the padded matrices are zero after the logical-row prefix;
- the prepended matrix is the canonical injection `[I; 0]`;
- its matrix-vector product is the padded assignment;
- padded constraint satisfaction is equivalent to logical constraint
  satisfaction;
- the constant ring coefficient used by the norm relation opens the same
  authoritative assignment.

The assignment width is divisible by the ring degree 54. No partial
coefficient block is needed.

### Joint polynomial and one SumCheck

The corrected paper uses

```text
Q(X) = eq(X, alpha) * (F(X) + gamma^K * NC(X))
       + gamma^(2K+k) * Eval(X).
```

Its absolute target is

```text
T_abs = gamma^(2K+k) * T_local.
```

Its degree bound is

```text
D_Q = max(D_f + 1, 2b, 2).
```

The selected values are `D_f = 8`, `b = 2`, and `D_Q = 9`. The fresh,
norm, and carried terms use the paper's disjoint gamma ranges. Lean owns these
facts in the generic `SuperNeo/Folding/PiCCS/PaperJoint` modules and selects
them in `PaddedRowIdentityConcreteNifs.lean`.

The verifier replays one joint polynomial on `Fin 24`. The proof has one
SumCheck message sequence and one terminal row point. The selected imports do
not use `PaperRectangular` or `SplitNc` as protocol authority. A repository
search finds no selected column challenge or second SumCheck field.

### Complete `Pi_CCS` output

The output contains all `(K+k) * t` ring-extension evaluations at one point
`r'`. The verifier absorbs the complete output before it derives `Pi_RLC`
challenges. `FullOutputCoordinates` fixes the coordinate order. The selected
Poseidon2 transcript and concrete NIFS replay use that order.

### `Pi_RLC`

The verifier derives one ring challenge for each of the `K+k` input claims.
It uses the same challenges to combine the commitments, public inputs,
evaluation values, and witness assignments. The concrete sampler has the
selected paper profile:

- 15 sources;
- 54 coefficients per source;
- three full-field Goldilocks candidates per coefficient;
- rejection of `q-1`;
- balanced reduction modulo 5 to coefficients in `[-2,2]`;
- fail-closed behavior if all three candidates reject.

The shortfall union bound is `810 / q^3 <= 2^-182`, under the explicit
uniformity and independence premise. The verifier never uses an internal
fallback coefficient as an accepted challenge.

### `Pi_DEC`

The selected profile uses base `b = 2` and `k = 14` signed pieces. The prover
sends all child commitments and evaluations. The verifier derives each child
public input and checks commitment, public-input, and evaluation
recomposition. Lean does not accept a prover-supplied parent point or parent
claim in place of the verifier-derived values.

### Concrete parameters and security accounting

The selected ring parameters are the corrected paper values:

- Goldilocks base field;
- extension degree 54;
- `Phi(X) = X^54 + X^27 + 1`;
- `kappa = 18`, `b = 2`, `k = 14`;
- challenge coefficients in `[-2,2]`;
- `T <= 216` and `K <= 61`.

The selected application has 13 application matrices and one prepended
identity matrix. Its constraint polynomial has total degree 8, so the joint
SumCheck degree is 9. The exact algebraic numerators are:

```text
mixing:   10599
sumcheck: 24 * 9 = 216
total:    10815
```

The security theorems are classical and conditional. They do not claim a
quantum-random-oracle proof. Phi81 low-norm invertibility, Module-SIS/Ajtai
binding, Poseidon2 random-oracle or collision security, accepted-child
extraction, and sampler distribution remain explicit inputs.

## HyperNova comparison

### Definition 12

The corrected definition has six properties. Lemma 3 proves only Properties
1 through 5 for the paper's CCS encoder. It does not prove recursive-size
closure. The instantiated Theorem 3 is withdrawn.

The generic Lean `NIVCCompatibility.Holds` record includes all semantic
properties. `definition12_holds` requires explicit `RecursiveSizeClosure` and
`ApplicationCompiler` values. It does not manufacture these values from the
fixed-circuit Lemma 3 result. This matches the corrected paper status.

### Construction 2

The selected fixed-one shell has this exact control flow:

- iteration zero checks `z_i = z_0`, checks canonical unused advice, sets the
  complete running vector to the deterministic default, and performs no NIFS
  call;
- a positive iteration validates the prior program counter before indexing,
  binds the complete prior public state, copies the complete running vector,
  and updates exactly the selected slot with one NIFS verifier call;
- the output hash contains verifier keys, the iteration, `z_0`, the next
  state, the complete running vector, and the next program counter;
- the terminal base proof is exactly `bottom` and is not parsed as a recursive
  payload;
- the recursive terminal proof checks all running relations and the selected
  fresh relation, and performs no NIFS fold.

The semantic owner is
`Nightstream/HyperNova/Construction2/Paper.lean`. The selected executable
owners are `PaddedRowIdentityConstruction2.lean`,
`CanonicalTerminalVerifier.lean`, `Frozen.lean`, and the generated Rust
conformance boundary.

### Construction 3 transcript and statement identifier

The selected transcript uses the corrected values:

```text
domain: HyperNova/MultiFold/Fiat-Shamir/v2
labels: statement-id, proof, prover-message, verifier-challenge
```

Its fixed 53-event schedule contains every `Pi_CCS` prover message and
challenge, the complete `Pi_CCS` output, every `Pi_RLC` challenge, and the
final `Pi_DEC` prover message. The canonical frames have the proved selected
lengths, including 22,700 fields for the complete folded output and 34,796
fields for the final `Pi_DEC` message.

The statement identifier hashes the exact Construction 3 statement preimage:
the domain, `statement-id` label, schedule, public parameters, both complete
structures, and the underlying verifier key. Its result is four Goldilocks
field elements, not one field element. The recursive verifier receives this
fixed-length identifier and the public NIFS input. The outer verifier retains
the complete statement. A statement-identifier collision is an explicit
security event.

The transcript uses a canonical typed and length-delimited field encoding.
The extra fixed descriptors inside that encoding are framing for
`trEnc`; they do not add a protocol message or verifier challenge.

## Repairs in this change

This change removes the identified divergences:

- it locks the corrected SuperNeo Markdown revision and splits the large v5
  errata patch so that each file stays below 1,500 lines;
- it changes the protocol contract from a square or split-domain account to
  rectangular `[I;0]` with one row SumCheck;
- it replaces biased modulo sampling with the exact bounded full-field
  sampler and makes verifier shortfall fail closed;
- it makes concrete completeness conditional on sampler availability;
- it checks the canonical HyperNova base advice and dummy proof;
- it adds the exact outer `bottom | recursive payload` terminal syntax;
- it updates the public verifier, frozen boundary, generated receipts, Rust
  conformance tests, and axiom gates for that syntax;
- it restores the corrected Construction 3 domain, labels, event schedule,
  full-output absorption, and final `Pi_DEC` message;
- it changes the statement identifier from one Goldilocks field to four
  Goldilocks fields and removes a non-paper local prefix from its preimage.

## Remaining boundaries found by the fresh audit

No additional equation-level mismatch was found in the selected one-SumCheck
semantic path. The following boundaries remain open and must not be described
as proved paper instantiation:

1. **No production Definition 12 compiler witness.** The repository has no
   production `ApplicationCompiler` and no proved `RecursiveSizeClosure`
   instance for the selected application. This is the same missing Property 6
   that causes the corrected paper to withdraw instantiated Theorem 3.
2. **The selected state hash is application-owned.** Lean has the exact typed
   state-hash preimage, but the selected Construction 2 shell takes the
   application transition and the canonical Poseidon2 state-hash encoder as
   parameters. No generated application artifact supplies them here.
3. **The physical terminal payload lowering is not the outer proof parser.**
   `Implementation/Lowering/FPrimeFixedOne/Terminal.lean` verifies the
   recursive payload and treats it as erased at iteration zero. The exact
   `bottom | recursive payload` envelope is enforced by the semantic, public,
   frozen, and Rust-conformance owners. There is no generated physical
   envelope artifact to refine.
4. **No complete production artifact certificate.** The repository does not
   yet prove one coefficient-by-coefficient equivalence chain from the
   selected Lean relation through a generated native/Rust/R1CS artifact and an
   on-chain verifier.
5. **Perfect completeness is conditional on bounded sampler availability.**
   The ideal paper challenge has no shortfall. The executable three-attempt
   sampler has an explicit failure event bounded by `2^-182` under its named
   probability premise.
6. **Complexity and cryptographic assumptions remain external.** Lean does
   not prove polynomial-time execution, random-oracle security, Module-SIS,
   or the required Poseidon2 collision bound.

The legacy `SplitNc` circuit remains outside the selected protocol and is
reported as blocked in the generated assurance status. Its presence is not
evidence for the selected protocol and does not add a second SumCheck to it.

## Validation

All required focused and full validation gates pass.

| Gate | Result |
| --- | --- |
| Focused `tests.PaddedRowIdentityConcrete` build | PASS, 2,168 jobs |
| Focused `tests.PaddedRowIdentityNIVCCompatibility` build | PASS, 2,165 jobs |
| Focused `tests.PaddedRowIdentityConstruction2` build | PASS, 2,170 jobs |
| Focused PiCCS Rust-conformance build | PASS, 194 jobs |
| Full Lean build | PASS, 3,419 jobs |
| Lean axiom gate | PASS, 2,781 jobs |
| Lean executable declaration check | PASS |
| Lean static policy gate | PASS |
| Native-step Rust receipt-drift test | PASS, 1 test |
| PiCCS Lean/Rust artifact test | PASS, 1 test |
| Protocol-contract derived refresh | PASS, no changes |
| Protocol-contract package check | PASS |
| Protocol-contract repository check | PASS, 107 declarations checked |

The static gate confirms that active Nightstream Lean sources contain no
`sorry`, `axiom`, `admit`, `postulate`, or `unsafe`. The protocol-contract
package still reports its declared assurance release as blocked. That status
is correct because the open boundaries above are not discharged.
