# RV64IM Side-Opening Relation, Side Final Verifier, and Spartan Linkage Specification

STATE: DESIGN TARGET

This document specifies the minimal side-opening theorem, the optional side
final verifier packaging, and the Spartan linkage rules for RV64IM in
`neo-fold-next`.

The theorem-level facts checked here are:

- each carried opened object is a valid opening under the commitment scheme,
- each carried payload is the evaluation of that opened object at the canonical
  Phase 0 point,
- every active side-opening target required by the published statement is
  covered exactly once and no extra target is admitted,
- and those facts are bound to the published Nightstream statement.

Everything else carried in a side proof container is transport, compression,
audit material, or outer linkage unless a separate owner proves it is
theorem-bearing.

Throughout this document, "this owner" means the specification boundary defined
here: the claims and rules that `R_side_opening` is responsible for.

## Artifact Breakdown

The table below is normative. It states every item in the minimal side-opening
artifact, why it exists, which verifier equation it participates in, what the
verifier must actually check, and how Spartan uses it in the two-verifier path.

| Artifact item | Kind | Why it is required | Verifier equation enforced | What this verifier actually checks | Direct cryptography required | Spartan / recursive wrapper role |
| --- | --- | --- | --- | --- | --- | --- |
| `nightstream_statement_core_digest` | theorem-facing public statement binding | Binds the published execution/program context that the side theorem is bound to | `d_stmt = Poseidon2(encode_canonical(statement_core))` as an external composition obligation | The verifier must either recompute it from the canonical `NightstreamStatement` or consume it inside a larger relation that proves that equality | Poseidon2 compression plus canonical encoding of the underlying statement | Must be part of the public tuple bound into Spartan whenever Spartan is used |
| `SideSurfacePublic` | public theorem data | Identifies the active side-opening targets that must be covered | `SideSurfacePublic = DeriveSideSurface_V1(statement_core)` as an external composition obligation | Canonical derivation from the published statement; this spec carries only `d_stmt`, so canonical derivation is a composition obligation | No standalone hardness assumption; deterministic canonical derivation | Must be public to the fixed side relation, either directly or through a canonical compressed public-instance digest |
| `OpenedObjectPublic` | public theorem data | Names one opened object and the commitment context it must satisfy | Appears as the public input to `OpenVerify(OpenedObjectPublic_k, OpenedObjectProof_k) = 1` | It is only a claim descriptor until paired with `OpenedObjectProof` or encoded into `SideVerifierPublic` | Canonical encoding; Poseidon2 only if a digest form is used | Public input to the fixed side relation; Spartan should not replace the opening proof with raw replay |
| `OpenedObjectProof` | proof object | Proves that the named opened object is a valid opening under the commitment scheme | `OpenVerify(OpenedObjectPublic_k, OpenedObjectProof_k) = 1` | The verifier checks `OpenVerify = 1` for that claim | Ajtai/module-lattice commitment opening proof plus norm/alphabet proof | In the two-verifier path, this may be internal to `SideFinalVerify`; Spartan should link to the resulting public tuple instead of replaying it |
| `EvalPublic` | public theorem data | Names one evaluation claim: object reference `obj(i)`, target key, point, payload | Appears as the public input to `EvalVerify(EvalPublic_i, EvalProof_i; r_i, OpenedObjectPublic_{obj(i)}) = 1` and exact-cover matching | It is only a claim descriptor until paired with `EvalProof` or encoded into `SideVerifierPublic` | Canonical encoding; Poseidon2 only if a digest form is used | Public input to the fixed side relation; digest-form point/payload fields are only valid if recomputed in the relation |
| `EvalProof` | proof object | Proves that the claimed payload is the evaluation of the named opened object at the canonical point | `EvalVerify(EvalPublic_i, EvalProof_i; r_i, OpenedObjectPublic_{obj(i)}) = 1` | The verifier checks `EvalVerify = 1` for that claim | Chosen evaluation/opening proof system for the side-opening protocol | In the two-verifier path, this may be internal to `SideFinalVerify`; Spartan should link to the resulting public tuple instead of replaying it |
| `SideVerifierPublic` | canonical verifier tuple | Gives one exact public tuple that the side final verifier and Spartan can both bind to | `EncodeSideVerifierPublic(d_stmt, SideSurfacePublic, {OpenedObjectPublic_k}, {EvalPublic_i}) = SideVerifierPublic` | It must be the canonical encoding of the exact side claims defined by this spec | Canonical encoding; Poseidon2 only if digest-compressed | Preferred shared linkage object between the side final verifier and Spartan |
| `SideFinalProof` | final side proof | Verifies the backend-specific side proof against `SideVerifierPublic` | `SideFinalVerify(SideVerifierPublic, SideFinalProof) = 1` | The verifier checks `SideFinalVerify = 1` against the same `SideVerifierPublic` tuple used for linkage | Chosen side proof backend | Preferred two-verifier path: verify it separately, then make Spartan link to the same public tuple |
| Proof container / side bundle | serialization container | Carries the public theorem data and proof objects together | No standalone theorem equation; only the verifier equations over its contained claims matter | The verifier only trusts the equations over the contained public inputs and proofs | None beyond the cryptography already used by its contained proofs | May be carried as wrapper witness. The container itself is not theorem-bearing in either mode |
| Proof container digest | optional compression handle | Lets recursive or outer verifiers bind one compact handle to the carried container | `d_container = Poseidon2(encode_canonical(container_contents))` only when used as compression | The verifier must recompute it from the canonical contents before using it | Poseidon2 compression only | Compression only; if used beneath Spartan, the relation must still bind or recompute the underlying public data it stands for |

Negative corollary:

- `OpenedObjectPublic`, `EvalPublic`, `SideVerifierPublic`, or any digest field alone is never enough.
- The theorem is established by verifier checks over `OpenedObjectProof` and `EvalProof`, or by `SideFinalVerify` when a backend packages them, not by self-consistent digests.
- `nightstream_statement_core_digest` is required in this spec; the full `NightstreamStatement` is not carried here.

Spartan clarification:

- Spartan is not where individual opened-object or evaluation claims are first checked.
- When a backend uses a final side sumcheck proof, Spartan should bind to the
  same `SideVerifierPublic` tuple, not replay the side sumcheck rounds.
- In the two-verifier deployment path, the side final verifier checks the side proof
  against `SideVerifierPublic`, and Spartan checks linkage against that same
  tuple or a digest recomputed from it.

### Current Spartan Constraint Cost Model (Diagnostic)

This table is diagnostic, not normative. It records the current measured
side-related costs in this repo and separates:

- raw Phase 0 replay costs, which are now audit-only diagnostics rather than
  Spartan work, and
- the current side Spartan circuit, which has already been cut down to a small
  public-input binding circuit.

Measurement context for the rows below:

- date: `2026-04-12 01:00:46 CDT`
- git SHA: `d8683e67`
- raw Phase 0 source:
  `cargo test -p neo-fold-next --release --test rv64im_side_relation_phase0 -- --ignored --nocapture rv64im_side_relation_phase0_component_counts`
- side-Spartan structural source:
  `crates/neo-fold-next/src/nightstream/rv64im/side_relation_spartan.rs`
- enforced `N=2` side-circuit redline:
  `RV64IM_N2_SIDE_CONSTRAINT_BUDGET = 2_048`

| Path or item | Current measured or enforced value | Basis | Interpretation |
| --- | --- | --- | --- |
| Raw Phase 0 commitment replay | `79,824` constraints per active schema in every measured case | `rv64im_side_relation_phase0_component_counts` on `Stage1Rows`, `Stage2RegisterWrites`, `Stage2TwistLinks`, `Stage3Continuity` | This is the dominant raw replay cost. It is no longer the intended Spartan boundary. |
| Raw Phase 0 eval replay | `3,766` to `4,236` constraints per active schema | Sum of `point + point_eq + payload + payload_eq` from the same diagnostic | This is the remaining eval-side replay after separating out commitment replay. |
| Raw Phase 0 schema total | `83,590` to `84,060` constraints per active schema | Same diagnostic | This is useful only as an audit/debug figure for the old replay path. |
| Current side Spartan circuit | No raw Phase 0 replay terms remain in the circuit; only the statement digest is allocated as public input | `measure_rv64im_side_spartan_circuit_constraints` and `side_relation_spartan.rs` synthesize only public inputs and no replay subrelations | The side-Spartan boundary is now small by construction. This is the current theorem-facing circuit shape. |
| Current side Spartan `N=2` redline | `<= 2,048` total constraints | `RV64IM_N2_SIDE_CONSTRAINT_BUDGET` canary in `rv64im_n2_canaries.rs` | This is the current enforced upper bound for the tiny-N side circuit. It is a budget, not a measured exact count. |
| Current end-to-end side timing | unavailable from the canonical perf snapshot at this SHA | `NS_DEBUG_N=2 cargo test -p neo-fold-next --release --test perf -- --ignored --nocapture rv64im_mixed_opcode_perf_snapshot` currently fails with `InvalidSumcheckProof` | Do not treat stale historical perf numbers as current until the perf snapshot is green again. |

Interpretation notes:

- The old `437,287/op` figure is no longer the current side-Spartan circuit. It
  was the cost of the pre-cut replay-heavy side relation.
- The raw Phase 0 rows above remain useful because they show exactly what should
  stay out of Spartan: commitment replay and payload replay.
- The current side-Spartan circuit is intentionally much smaller. Its role is to
  bind the compact public statement, not to replay raw side witnesses.
- The table should be updated only from fresh diagnostics. If the canonical perf
  snapshot is failing, timing rows must say so rather than reusing stale data.

## Attack Breakdown

This table is normative. It lists the attack families relevant to the minimal
side-opening theorem, what each attack tries to break, and the primary
mechanism that blocks it.

| Attack family | What it attacks | Primary blocking mechanism | Protection status |
| --- | --- | --- | --- |
| Statement substitution | Rebind a valid side proof to a different published execution/program context | `nightstream_statement_core_digest` plus the statement-digest composition obligation | Protected iff the digest is recomputed from the canonical published statement or a larger relation proves that equality |
| Opened-object forgery | Claim an opened object without a valid commitment opening behind it | Verification of `OpenedObjectProof` | Protected |
| Norm/alphabet forgery | Use an opening witness with invalid coefficient bounds or invalid opening-side norm conditions | The norm/alphabet subtheorem inside `OpenedObjectProof` | Protected iff the proof system includes the required norm/alphabet theorem |
| Commitment-root forgery | Present packed data that does not match the claimed commitment-root binding | Verification of `OpenedObjectProof` | Protected |
| Opened-object digest forgery | Present an object identifier that does not match the actual opened object | Verification of `OpenedObjectProof` | Protected |
| Evaluation forgery | Claim a payload that is not the evaluation of the opened object at the canonical point | Verification of `EvalProof` | Protected |
| Point-derivation confusion | Substitute the wrong point by changing schema, slot, binding inputs, or transcript inputs | `EvalProof` plus verifier-owned canonical transcript derivation | Protected iff the verifier derives the canonical point from the fixed Fiat-Shamir inputs owned by this relation |
| Point/payload digest-preimage injection | Provide digest-form point or payload fields that are never recomputed from canonical preimages | Digest recomputation inside the fixed relation plus `EvalProof` | Protected iff digest-form point/payload fields are recomputed before being trusted |
| Payload substitution | Keep the same point/object but swap the claimed output payload | Verification of `EvalProof` | Protected |
| Cross-object mix-and-match | Reuse an eval proof with the wrong opened object | Fixed-relation cross-link via `obj(i)` mapping and `ObjectKey` identity | Protected iff the verifier enforces exact opened-object identity linkage through `obj(i)` |
| Cross-slot replay | Reuse one proof to satisfy multiple slots or targets | Exact target coverage via `TargetKey` bijection and cross-link checks | Protected iff coverage and no-extra-target checks are exact |
| Omitted-target attack | Leave one active side-opening target completely unproven | Exact coverage and no-extra-target checks over `SideSurfacePublic` | Protected iff coverage is exact |
| Inactive-family baggage smuggling | Force inactive families or inactive slots to become theorem-bearing work or fake proof obligations | Active-set semantics of `SideSurfacePublic` | Protected iff inactive targets are genuinely inert |
| Side-final / Spartan splice | Present a valid side final proof and a valid Spartan proof for different side public tuples | Equality or digest-recompute linkage between `SideVerifierPublic` and Spartan public inputs | Protected iff both verifiers consume the same tuple |
| Digest-chain self-consistency | Mutate underlying witness/proof data and recompute digests upward so every carried digest still matches | Direct verification of `OpenedObjectProof` and `EvalProof` against canonical public data | Not protected by digests alone |
| Proof-container substitution | Swap the transport bundle while preserving some inner identifiers | Parse the container, recover the canonical public instance, and verify the contained proofs | Not protected by the container alone |
| Hash collision attack on compressed bindings | Find two different canonical public objects with the same digest | Poseidon2 collision resistance | Protected only under the compression assumption; a matching digest alone does not prove the theorem |
| Execution-semantics forgery | Claim the wrong program/state semantics while keeping side openings locally valid | Outer published statement and execution-semantics relations | Not protected by this owner alone |
| Main/side linkage forgery | Bind a valid side artifact to the wrong main proof or wrong outer linkage object | Outer linkage/public-statement relations | Not protected by this owner alone |

## Mathematical Core

This section is normative. It fixes, in order:

1. the theorem-bearing side-opening relation,
2. the optional packaged final verifier,
3. the optional Spartan linkage relation,
4. and the wrapped-backend verifier equations when a backend is internalized.

### Canonical keys and canonical vectors

Define

    ObjectKey_k := EncodeOpenedObjectKey_V1(OpenedObjectPublic_k)

and

    TargetKey_t := EncodeTargetKey_V1(
        schema_family_t,
        family_stage_anchor_t,
        schema_t,
        slot_t,
        binding_digest_t)

The public instance of the side theorem is

    (d_stmt, SideSurfacePublic,
     (OpenedObjectPublic_k)_{k=0..n_obj-1},
     (EvalPublic_i)_{i=0..n_eval-1})

where both lists are canonical, ordered, and duplicate-free.

Each `EvalPublic_i` MUST determine all of the following:

    obj(i) ∈ [0, n_obj),    TargetKey_i

together with the canonical payload, or a digest-cache for that payload, and
any canonical point cache fields the backend chooses to carry.

Matching is always done on `TargetKey`. Opened-object linkage is always done on
`obj(i)` together with `ObjectKey_{obj(i)}`.

### External composition obligations

This owner carries only `nightstream_statement_core_digest` and the derived
side surface, not the full statement preimage. Therefore the following
equalities are composition obligations:

    d_stmt = H_{stmt,V1}(statement_core)

and

    SideSurfacePublic = DeriveSideSurface_V1(statement_core)

A verifier accepting this owner MUST either recompute both equalities from the
canonical `NightstreamStatement` or be wrapped in a larger relation that proves
them.

### The theorem-level relation `R_side_opening`

`R_side_opening` accepts iff all of the following hold.

For every opened object `k`:

    OpenVerify(OpenedObjectPublic_k, OpenedObjectProof_k) = 1

For every evaluation claim `i`, the canonical Phase 0 point is
verifier-derived:

    r_i = FS_to_Point_V1(d_stmt, ObjectKey_{obj(i)}, TargetKey_i)

For every evaluation claim `i`, the verifier checks the evaluation proof
against the referenced opened object:

    EvalVerify(EvalPublic_i, EvalProof_i; r_i, OpenedObjectPublic_{obj(i)}) = 1

Equivalent APIs are allowed, but the meaning must be the same: the verified
evaluation must be for the same opened object validated by the corresponding
`OpenVerify`; matching only labels or digests is insufficient.

Exact target cover is enforced by a bijection between active targets and
verified evaluation claims:

    for every t in Targets(SideSurfacePublic):
        |{ i : TargetKey_i = TargetKey_t }| = 1

    for every i:
        |{ t : TargetKey_i = TargetKey_t }| = 1

No orphan opened object is allowed:

    for every k:
        |{ i : obj(i) = k }| >= 1

Any digest-form point, payload, or public subfield is a cache only. If used,
it is trusted only after recomputation from its canonical preimage:

    d_{point,i} = Poseidon2(encode_canonical(point_i))
    d_{payload,i} = Poseidon2(encode_canonical(payload_i))

No inactive target, inactive family, or inactive slot induces a
theorem-bearing proof obligation.

This is the side theorem. Everything below is packaging, backend
verification, or linkage.

### Optional packaged final verifier `V_side_final`

If a concrete backend packages `R_side_opening` into one final proof, define

    SideVerifierPublic := EncodeSideVerifierPublic_V1(
        d_stmt,
        SideSurfacePublic,
        (OpenedObjectPublic_k)_k,
        (EvalPublic_i)_i)

`V_side_final` accepts iff:

1. `SideVerifierPublic` is exactly the canonical encoding above,
2. `SideFinalVerify(SideVerifierPublic, SideFinalProof) = 1`,
3. and the packaged theorem is definitionally the same theorem as
   `R_side_opening`, not an alternate public statement.

Any concrete backend profile must be fixed either by versioned verifier code or
by a backend profile tag encoded into `SideVerifierPublic`. At minimum, that
profile must pin the field/ring pair, cyclotomic degree, commitment
homomorphism `L`, input projection `L_in`, challenge set `C`, norm alphabet and
bounds, the CCS structure used to encode `R_side_opening`, the
embedding/lifted-transform rule, and the Fiat-Shamir transcript order and
domain tags.

### SuperNeo-backed `SideFinalVerify`

When the backend is SuperNeo, `SideFinalVerify` checks the versioned backend
profile `RecursiveSuperNeoSide_V1`.

`RecursiveSuperNeoSide_V1` is not the side theorem itself. It is the backend
construction that realizes `R_side_opening` by:

1. encoding the theorem as a fixed CCS/CE family `CCS_side_opening_V1`, and
2. using the SuperNeo step relation

    Π_SuperNeo := Π_DEC ∘ Π_RLC ∘ Π_CCS

   over that encoding.

`RecursiveSuperNeoSide_V1` must therefore pin all of the following:

1. the backend structure `s_side = ({M_j}, f)` and commitment map `L_side`,
2. the public-input projection `L_in,side`,
3. the coefficient embedding and lifted transform used by SuperNeo,
4. the public-coin transcript and its Fiat-Shamirization,
5. the intermediate relaxed CE relation and shared projection `φ_side`,
6. and the exact meaning of `SideFinalProof`
   (single step, recursive accumulator proof, or compressed recursive verifier).

In this repo, `SideFinalProof` means a terminal proof for the recursive
verifier of `RecursiveSuperNeoSide_V1`, not a bare single fold step.

Only verifier-visible equations belong here. Prover-side helper-polynomial
construction, hypercube replay, extractors, relaxed-binding arguments, and
security reductions do not.

Define the following notation:

    ℓ := log_2(m)
    I(i, j, ν) := (i - (K+1)) + k(j - 1) + ktν

#### SuperNeo embedding and lifted transform

This follows the paper construction.

`RecursiveSuperNeoSide_V1` uses the paper's coefficient embedding and lifted
transform:

    z ∈ F^(d · n_R)
    z = [z_1, ..., z_(n_R)]         with z_i ∈ F^d
    z_i^ring := Σ_{u=1}^{d} z_(i,u) X^(u-1) ∈ R_F
    z^ring := [z_1^ring, ..., z_(n_R)^ring] ∈ R_F^(n_R)

For each field matrix `M_j ∈ F^(m × n_F)`, the profile uses the lifted
transform `\bar{M}_j` from the paper. Backend evaluation claims are ring
evaluations

    Y_(i,j) := \widehat(\bar{M}_j z_i^ring)(r) ∈ R_K

with the required constant-term recovery equation

    ct(Y_(i,j)) = \widehat(M_j z_i)(r).

This embedding/lifted-transform rule is not optional in the SuperNeo profile.
`EmbedInput_V1` must mean this coefficient embedding and lifted-transform
pipeline, not an arbitrary backend-local embedding.

#### CCS/CE backend bridge

This is repo-local typing, pinned by the backend profile.

`CCS_side_opening_V1` is the versioned CCS/CE encoding of `R_side_opening`
used by `RecursiveSuperNeoSide_V1`. It fixes:

    CCS_side_opening_V1 := (s_side, L_side, L_in,side, EmbedInput_V1, Transcript_V1)

The backend profile must guarantee that accepting `CCS_side_opening_V1`
instances means exactly that the theorem-side equations of `R_side_opening`
hold: opened-object checks, evaluation checks, exact target cover,
object/target linkage, digest recomputation, and the external composition
obligations.

The profile must also pin the composition boundary from the paper. The same
projection `φ_side` must be used by the strong `Π_CCS` step and the weak
`Π_RLC` step. Here `φ_side` is the commitment projection of the intermediate CE
instance for `CCS_side_opening_V1`, together with the relaxed intermediate CE
relation used between `Π_CCS` and `Π_RLC`.

Fix one finite norm alphabet `S_b` and its vanishing polynomial

    R_b(u) := Π_{a ∈ S_b} (u - a)

If the theorem-facing public tuple carries field-form inputs while the backend
uses ring-form input projection, define the canonical projection

    x^R_i := EmbedInput_V1(x_i)

and use `x^R_i` in all backend equations below.

#### `Π_CCS`

Let the incoming carried CE claims share one point `r`, with ring evaluations
`Y_{i,j}` for `i ∈ {K+1, ..., K+k}`.

The claimed hypercube sum is

    T := Σ_{i=K+1}^{K+k} Σ_{j=1}^{t} Σ_{ν=0}^{d-1}
         γ^{I(i,j,ν)} · cf(Y_{i,j})_ν

The sum-check transcript reduces

    T =? Σ_{x ∈ {0,1}^ℓ} Q(x)

to one final evaluation claim at `r'`.

If the round polynomials are

    g_s(U) = Σ_{q=0}^{D_sc} a_{s,q} U^q

the verifier equations are

    claim_0 := T

    g_s(0) + g_s(1) = claim_{s-1}     for s = 1, ..., ℓ

    claim_s = g_s(r'_s)               for s = 1, ..., ℓ

    v_sc = claim_ℓ

With outgoing evaluations `Y'_{i,j}` at `r'`, the verifier computes

    F := Σ_{i=1}^{K} γ^{i-1} · f(ct(Y'_{i,1}), ..., ct(Y'_{i,t}))

    N := Σ_{i=1}^{K+k} γ^{i-1} · R_b(ct(Y'_{i,1}))

    E := eq(r', r) · Σ_{i=K+1}^{K+k} Σ_{j=1}^{t} Σ_{ν=0}^{d-1}
         γ^{I(i,j,ν)} · cf(Y'_{i,j})_ν

The SuperNeo backend profile must choose exactly one of the following:

1. `M_1 = I`, so the norm term is read from `ct(Y'_{i,1})`, or
2. a versioned explicit norm channel `NormTerm_V1(...)` replacing the displayed
   `N` equation everywhere it appears.

The backend profile is incomplete unless one of those two choices is fixed.

The terminal verifier identity is

    v_sc = eq(r', α) · (F + γ^K · N) + γ^{2K+k} · E

#### `Π_RLC`

The verifier derives ring challenges `ρ_i` from the canonical Fiat-Shamir
transcript and enforces the canonical challenge derivation and membership
rules. Then it checks

    c_out     = Σ_i ρ_i c_i
    x^R_out   = Σ_i ρ_i x^R_i
    Y_{out,j} = Σ_i ρ_i Y_{i,j}    for all j
    r_out     = r

with every input claim sharing the same evaluation point `r`.

#### `Π_DEC`

Let `B = b^k`. The verifier checks

    c     = Σ_{i=1}^{k} b^{i-1} c_i
    x^R   = Σ_{i=1}^{k} b^{i-1} x^R_i
    Y_j   = Σ_{i=1}^{k} b^{i-1} Y_{i,j}    for all j
    r_i   = r                                for all i

When decomposed public inputs are materialized, they must be deterministically
derived from the public input by the backend's split rule:

    (x^R_1, ..., x^R_k) := split_b(x^R).

If the backend exposes decomposed digits or decomposed inputs, it must
additionally enforce digit membership through `R_b`.

### Spartan linkage relation `R_side_link`

`R_side_link` binds the same side public tuple and the same published roots.

If digest-compressed linkage is used, define

    d_side := Poseidon2(encode_canonical(SideVerifierPublic))

Two-verifier mode:

- the side-final verifier checks `V_side_final`,
- Spartan checks only linkage,
- `Π_CCS`, `Π_RLC`, and `Π_DEC` do not appear inside Spartan.

Spartan-wrap mode:

- Spartan proves `V_side_final ∧ R_side_link`,
- only the verifier equations above may be circuitized,
- raw Ajtai openings, raw Phase 0 witnesses, prover-side sum-check
  construction, and multilinear witness/table replay are forbidden.

In either mode, the outer verifier must compare exactly the public tuple that
Spartan consumes:

    spartan_stmt_root = d_stmt

and either

    spartan_side_input = SideVerifierPublic

or

    spartan_side_digest = Poseidon2(encode_canonical(SideVerifierPublic))

### Verifier-cost formulas

Let:

- `ℓ = log_2(m)`
- `D_sc` = round-polynomial degree in the sumcheck verifier
- `c_K` = base-field multiplication cost of one extension-field multiplication
- `M_γ` = number of `γ` powers materialized by the verifier
- `C_f` = extension multiplications needed to evaluate the fixed polynomial `f`
- `L_b` = size of the norm-alphabet product `R_b` (i.e. `|S_b|`)

Then:

#### Generic sumcheck verifier only

- extension-field multiplications:
  - `M_sumcheck = ℓ * D_sc`

- verifier rows:
  - `Rows_sumcheck ≈ ℓ * (D_sc + 1)`

- base-field multiplication constraints:
  - `BF_sumcheck ≈ c_K * ℓ * D_sc`

#### Full `Π_CCS` verifier

- extension-field multiplications:
  - `M_ΠCCS ≈ ℓD_sc + 4ℓ + M_γ + K(C_f + 1) + (K + k)(L_b + 1) + ktd + 4`

- base-field multiplication constraints:
  - `BF_ΠCCS ≈ c_K * M_ΠCCS + Poseidon_cost`

This is still verifier cost. It does not include prover work, raw opened-object
replay, or raw side witness replay.

## Representative Constraint Shapes

The verifier equations above should compile into verifier constraints that look
like verifier logic, not raw witness replay.

Representative constraint shapes for one R1CS-style encoding are:

- Equality:
  - `lhs - rhs = 0`

- Boolean selector:
  - `s * (s - 1) = 0`

- Conditional match:
  - `s * (lhs - rhs) = 0`

- Exact-one coverage for active target `t`:
  - `sum_i s_{t,i} = 1`

- Exact-one use for verified claim `i`:
  - `sum_t s_{t,i} = 1`

- Digest recomputation:
  - `h = Poseidon2(preimage)`

- Proof-verifier acceptance bit:
  - `b_open = 1`
  - `b_eval = 1`

- Chunk or limb recomposition, when a backend decomposes a query, point, or
  payload into fixed-width pieces:
  - `z = Σ_k 2^(w*k) * chunk_k`

- Sumcheck round consistency:
  - `g_s(0) + g_s(1) - claim_{s-1} = 0`

- Horner step for sumcheck round evaluation:
  - `m_{s,q} - r'_s * h_{s,q+1} = 0`
  - `h_{s,q} - a_{s,q} - m_{s,q} = 0`

- Terminal `Π_CCS` identity:
  - `v_sc - eq(r', α) * (F + γ^K * N) - γ^{2K+k} * E = 0`

- Coordinatewise `Π_RLC` recombination:
  - `out_coord - Σ_i ρ_i * in_coord_i = 0`

- Coordinatewise `Π_DEC` recomposition:
  - `base_coord - Σ_i b^{i-1} * digit_coord_i = 0`

These are representative verifier-constraint shapes, not a mandate that every
implementation use explicit selector matrices. A conforming verifier may use
canonical ordering, lookup-style matching, or another fixed encoding, provided
it enforces the same equations above.

## Scope

This owner defines `R_side_opening`: its public instance, its public and proof
objects, the role of any proof container, the fixed verifier relations, and
the negative rules that keep digest compression from being mistaken for proof.
It does not own RV64IM execution semantics, `R_main^SN`, outer Nightstream
statement/linkage policy, legacy bridge/export/package artifacts, or the
concrete compact opening/evaluation and recursive backends.

Delegated ownership:

| Area | Owner |
| --- | --- |
| RV64IM semantics and `R_main^SN` | `riscv-kernel.md`, `riscv-main-relation.md` |
| Main-lane row-local residual (`W = 38` semantic-row width, `29` canonical R1CS rows before CCS conversion) | `riscv-kernel.md` §3.1, §4.1, §11.3 and `riscv-main-relation.md` |
| Outer published statement and linkage policy | Outer Nightstream statement/linkage owner |
| Transitional witness-boundary owner | `riscv-witness-backed-side-bridge.md` |
| Recursive backend and recursive export policy | `riscv-recursive-proof.md`, `riscv-recursive-instantiation.md` |

`riscv-witness-backed-side-bridge.md` is the transitional owner on the path to
this end-state theorem. Both documents must preserve the same stable checked
projection where their ownership overlaps.

## Normative References

This document is constrained by:

- `../../docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
  - relation-first `CE(b, L)` and its reductions
- `../../docs/superneo-paper/06_6_Strong_and_weak_interactive_reductions.md`
  - composition requires a stable checked projection
- `../../docs/superneo-paper/01_1_Introduction.md`
  - standard compilers apply after the folding relation is fixed
- `../../docs/hypernova.pdf.md`
  - recursion compiles a fixed verifier relation with an a la carte cost profile
- `./riscv-witness-backed-side-bridge.md`
  - digest-consistency is not enough across the side bridge boundary
- `./riscv-recursive-proof.md`
  - exported recursion targets a fixed relation, not arbitrary proof blobs

This document is normative only where its requirements are implied by, or are a
conservative specialization of, those references plus the repo security rules.

## Terms

Canonical means a value, encoding, ordering, or derivation rule has one
byte-exact or field-exact form fixed by this spec or a cited owner, so all
honest implementations derive the same result from the same checked
inputs. Determinism alone is not enough.

## Boundary Rules

The side proof container may carry:

- the public instance of `R_side_opening`
- the `OpenedObjectProof` objects
- the `EvalProof` objects
- `SideVerifierPublic`
- `SideFinalProof`
- optional routing summaries derived from those theorem-bearing subobjects

The proof container is transport only. Its digest, if any, is compression only
and may serve as a serialization checksum, recursive wrapper input, or indexing
handle, but never as a substitute for proof verification.

At the `R_side_opening` layer, private witness is confined to:

- the witness of each `OpenedObjectProof`
- the witness of each `EvalProof`
- any batching witness internal to those proof systems

The outer verifier relation must not rebuild raw packed columns, raw
commitment vectors, raw opened-object witnesses, or raw Phase 0 claim
witnesses.

Once `R_side_opening` exists, raw `phase0_claim_witnesses`, raw
`commitment_vector`, raw packed-column witnesses, digest-only opening
artifacts, and current stage, package, export, root-execution, and
kernel-export
summaries leave this theorem boundary. They may remain audit-only,
implementation-local, or outer-linkage-owned, but they are not theorem-bearing
here unless another owner proves they are theorem-bearing.

Preferred on-chain / recursive boundary:

- in two-verifier mode, the chain verifies `SideFinalVerify` against
  `SideVerifierPublic` and verifies Spartan against the same tuple, or a digest
  recomputed from it
- in Spartan-wrap mode, Spartan proves `V_side_final ∧ R_side_link`
- in either mode, Spartan must stay verifier-shaped and must not replay raw
  Ajtai openings, raw Phase 0 witnesses, prover-side sumcheck construction, or
  multilinear witness/table replay

Forbidden constructions:

1. accepting a side proof container only because a digest chain is
   self-consistent
2. accepting the theorem because inner digests are self-consistent
3. replaying raw Ajtai openings while also claiming the proof container itself
   is the proof of the theorem
4. treating `opening_artifact_digest` or any current side-bundle digest as
   proof of opening correctness
5. letting inactive families or inactive slots impose theorem-bearing proof
   obligations
6. treating a proof-byte digest as theorem meaning rather than as compression

## Conforming Modes

Three end states are conforming:

1. direct verification of the proof container for `R_side_opening`,
2. two-verifier mode: `V_side_final` checked separately and `R_side_link`
   checked by Spartan, with both verifiers consuming the same canonical
   `SideVerifierPublic` tuple,
3. Spartan-wrap mode: Spartan proves `V_side_final ∧ R_side_link`, with
   only verifier equations from Mathematical Core circuitized.

Neither path may reintroduce legacy bridge/export artifacts as theorem meaning.

The main-lane row-local residual is not part of this theorem. In this repo,
that residual is the canonical `W = 38` semantic-row layout with `29`
row-local R1CS constraints before `r1cs_to_ccs(A, B, C)` conversion, owned by
the main lane.

This boundary exists to restore the intended recursion profile: verifier work
scales with active opened objects and active evaluation claims, not raw Ajtai
commitment coordinates, raw opened-object witness size, or inactive machine
families. A conforming recursive instantiation must not incur
`O(commitment_coordinate_count)` verifier work to replay raw opening
coordinates.
