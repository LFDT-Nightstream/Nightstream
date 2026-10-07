# Nebula memory phase on SuperNeo F′ — security note

Status: **Proposed.** Companion to [`nebula-superneo.md`](./nebula-superneo.md)
("the spec"). The author wrote the lemmas below. A production claim needs a
review by someone other than the author (§7).

This note states the security goal, the assumptions, six lemmas with proofs,
and the composed bound. It evaluates the bound at the current values.

## 1. Threat model and security goal

The prover controls all proof bytes, witnesses, memory records, and proposed
roots. The verifier controls the verifier key, the application package, the
plan, the setup seed, and the public statement.

**Current values.** The F′ Stage 1 system has these levels:

| Quantity | Level | Source |
|---|---|---|
| Statistical error of one fold | 114.76 bits | `NeoParams::padded_row_security_summary_for_shape`: 7 matrices, sum-check degree 9, `ℓ = 28` |
| Main Module-SIS | 110.685 quantum Core-SVP bits after an eight-target allowance; selected by the owner | `decisions/fprime-stage1-main-ajtai-setup.md` |
| Poseidon2 | generic collision ceiling near 128 bits | `decisions/piccs-prior-state-digest.md` |

The lowest level is the Module-SIS level, about 110.7 bits. The trunk treats
these values as planning counts, not as an end-to-end theorem. The rows have
two units. The fold row is a statistical error for one check. The other rows
are work estimates for a computational attack. A comparison across the two
units is a planning guide only.

**Security goal.** Stage 2 must stay at these levels, with one approved
exception:

- the memory fingerprint term is at least 109.91 bits. The owner approved this
  level on 2026-10-05. It is the value of one fingerprint challenge pair for
  the example geometry of §5;
- each new digest uses the Stage 1 Poseidon2 model.

Stage 2 adds no commitment map, so it adds no Module-SIS assumption. The
memory term is the weakest statistical term: it is 4.85 bits below the fold
term. The Fiat–Shamir step multiplies both statistical terms by a query count
(§5).

**Open, and shared with Stage 1.** The F′ trunk has no approved oracle-query
bound, lifetime composition, or end-to-end target. The `protocol-contract`
ledger has such values (NSD-SECURITY-001 and NSD-THREAT-MODEL-001: 96 bits,
`2^18` oracle queries, 64 folds). That ledger describes the legacy profile:
`k = 14`, 15 PiRLC inputs, width-8 Poseidon2, and ChaCha20.
`decisions/piccs-prior-state-digest.md` states that this profile is not
authority for the Stage 1 package. §5 evaluates the legacy values for
information only.

## 2. Assumptions

**A1 (SuperNeo knowledge soundness).** SuperNeo v1.2 Theorem 3 holds for the
Stage 1 profile and its commitment map `L_full`. Stage 2 adds rows to the step
relation and a block to the state, not a commitment component (spec §7). The
theorem is an interactive reduction of knowledge. Its statistical error for one
fold is `f_fold/|𝕂| + f_fork/|C|`, with `|C| = 5^54`. `f_fold` is the
numerator of v1.2's `ε_test` (sum-check and Schwartz–Zippel), and `f_fork` is
the PiRLC extraction term. The trunk computes both with
`NeoParams::padded_row_security_summary_for_shape` for the final relation
shape. v1.2's `ε_uniq` reduces to relaxed binding of `L_full` (A2).

**A2 (Module-SIS binding).** `L_full` is `(2B, C)`-relaxed binding at the main
level, as SuperNeo v1.2 §7 requires. Its key comes from the SHAKE128 expander,
modeled as a random oracle. This is the Stage 1 assumption without change.

**A3 (Poseidon2).** `H` is a random oracle for Fiat–Shamir. Its digest is
collision resistant and preimage resistant at the ceiling of the trunk
decision. `ε_H` is the error of this assumption. Stage 2 adds digests of the
same kind as Stage 1.

**A4 (Recursive extraction).** HyperNova Lemma 4, as corrected in
`docs/hypernova-paper/`, gives knowledge soundness of Construction 2 for a
fixed constant iteration bound. Its premises are:

- the canonical encoders satisfy Definition 12, including Property 6
  (recursive-size closure);
- `hash` is collision resistant or a random oracle;
- the non-interactive folding scheme is knowledge sound. HyperNova states
  this as Assumption 1 for its own scheme. For F′, it is SuperNeo folding
  under A1 and A6.

Here the bound is `S_max · N`. Stage 2 adds the carry block to the state
preimage. The claims do not change. So Stage 2 must prove Property 6 and the
fixed-point and domain theorems again for the larger state
(`FPRIME_LEAN_ARCHITECTURE_SPEC.md` §6). Definition 12 is stated for arity
`(1, 1)`, and F′ folds one fresh claim with 16 running children. So Stage 1
must also adapt the lemma to that arity. The lemma gives no concrete extractor
time. That bound is also open for Stage 1.

**A5 (Field facts).** `q` is prime, and 7 is a quadratic nonresidue modulo
`q`. Then `𝕂 = F[U]/(U² − 7)` is a field of size `q²`, and `F → 𝕂` is
injective. A release needs machine-checked certificates for both facts.

**A6 (Fiat–Shamir game transfer).** The F′ relation computes outputs of `H`
inside a recursively proven step. A random oracle cannot run inside a circuit,
so no random-oracle proof covers these outputs directly. Stage 1 states this
gap as an explicit premise, which the owner approved on 2026-09-11
(`docs/reviews/nightstream-fprime-requirements/FIAT_SHAMIR_MODEL.md`; Lean
`FiatShamirModel` in `Lifecycle/Nifs/FiatShamirTransfer.lean`). The premise
transfers success from the real Poseidon2 experiment to a translated
experiment, through external functions `g_d` and `delta_d` of the total
permutation-query count `Q`.

The Stage 1 premise moves success to an interactive game, which has no oracle
query log. Stage 2 needs more. A6 therefore adds a premise of the same kind:

- **Observable oracle for in-circuit hashes.** In the translated experiment,
  every output of `H` that the relation computes — the memory challenges `η`
  (spec §9.3) and the record chains (spec §9.2) — is the answer to an oracle
  query. The reduction sees the query log, including the verifier's setup
  queries for `D_init`. The chain inputs are private witness data, as the
  inputs of the Stage 1 state hash already are.

Lemmas 2 and 3 hold in that experiment, and A6 carries the §5 bound to the real
protocol. This premise needs its own owner approval. Like the Stage 1 boundary,
it selects no numerical value for `g_d` or `delta_d`.

**Fallback.** If the owner approves only an interactive-style transfer, the
argument still works with one more term. Lemma 3 then uses the uniqueness
argument of SuperNeo v1.2 Appendix B: rewind at `η`, and a second, different
record set under the same root is a Poseidon2 collision. The memory term
becomes `q_η · (ε_test + ε_coll(t_1))`, where `t_1` is about twice the time of
one run with extraction.

## 3. Deterministic obligations

These obligations have no probability term. A failure is a defect.

| # | Obligation |
|---|---|
| Ob1 | Canonical decoding of statements, claims, and the carry block; no alias of a noncanonical field element |
| Ob2 | Packing layout: the generated relation packs every lane exactly as spec §9.1 states, with at most 63 bits per element and lengths fixed by the plan |
| Ob3 | Transcript framing of the chains and of the `η` transcript exactly as spec §9.2–§9.3 |
| Ob4 | Every satisfying assignment of the generated relation refines to the actions of spec §8–§12 |
| Ob5 | Every honest execution has a satisfying assignment, including `S = S_max`, the largest counter values, and idle steps |
| Ob6 | Static hash-residency check (spec §6.4) |
| Ob7 | Port coverage: no application memory effect bypasses a port (spec §10) |
| Ob8 | Lifecycle (spec §11.2, §12): the arm follows from the authenticated carry; each step hashes its own lanes; `close` runs exactly when `step` sets `idx = N`; the terminal requires a closed carry |

## 4. Lemmas

The lemmas cover the four joints of spec §1.1. The author wrote every proof in
this note, and each proof needs an independent review (§7).

| Joint | Argument | Status |
|---|---|---|
| J1: records into chains | Lemmas 1 and 2; Ob2, Ob3, Ob6 | Proof in this note, in the translated experiment of A6 |
| J2: carry thread | A3, A4; Ob8 | Assumed. The arity adaptation and Property 6 are open |
| J3: commit, then test | Lemmas 2–5 | Proof in this note, in the translated experiment of A6 |
| J4: segment joins and terminal | Lemma 6 | Proof in this note |

### Lemma 1 — Packing is injective

**Statement.** Suppose rows O1 and S1 hold, so every lane coordinate is in
`{0, 1}`. Then for each lane `l`, `pack_l` (spec §9.1) maps distinct lane
contents of one step to distinct sequences of field elements. Every packed
element is the field value of an integer below `2^63`.

**Proof.** Each chunk holds at most 63 bits `b_k ∈ {0, 1}`, so
`Σ_k 2^k · b_k < 2^63 < q`. The field value of this sum is the integer itself,
with no reduction modulo `q`. Binary representation is unique, so distinct
chunks give distinct elements. The plan fixes the lane length (spec §6.3), so
the chunk boundaries are fixed. Equal element sequences therefore give equal
bit strings. The record encoding (slot-major, field order, little-endian bits)
maps records to bit strings one to one. ∎

Both hypotheses are necessary. Without O1 and S1, a coordinate can be `−1`, and
`Σ 2^k · b_k` over `{−1, 0, 1}` is not injective. With 64 bits in one chunk,
values at or above `q` alias values below it.

### Lemma 2 — Records fixed before `η`

**Statement.** Work in the translated experiment of A6. Suppose a segment
closes: `D_seen = (D_pre.ops, D_mem, D_pre.fs)`, where the relation computes
`D_seen` from the lanes of the segment's `N` steps, and the three roots
entered an `η` query (spec §9.3). Then, except with `ε_H`, the packed lanes of
all `N` steps were inputs of chain queries made before that `η` query. The
lanes in the extracted witnesses equal those query inputs.

**Proof.** In that experiment, every output of `H` is an oracle answer. Ob8
makes each step hash its own lanes, and the relation computes `D_seen.l` by `N`
chain evaluations. The last one equals the root in the `η` query. For that
equality to hold, one of three events occurs:

1. The adversary made that chain query before the `η` query. The query input
   contains the index `N − 1`, the previous chain value, and the packed lane of
   step `N − 1`. By induction down to the fixed header, every chain input of
   the segment was in a query before the `η` query.
2. Two different chain inputs give the same output: a collision.
3. At some level of the chain, the adversary used a value before any query
   returned it, and a later query returns that value: a preimage. The root in
   the `η` query is one case.

Events 2 and 3 are inside `ε_H`. In event 1, the oracle is a function, so the
chain inputs in each extracted witness equal the logged inputs, unless two
inputs give one output.

The extractor of A1 and A4 forks only at fold challenges. To extract the
witness of step `j` of a segment, it forks at the fold in `A[j+1]`. That fork
comes after the segment's `η` query, because the witness of `u_j` contains `η`.
So every run that the extractor uses shares the oracle answers to the
segment's chain queries. In each run, `u_j.x` fixes the output carry through
the state hash, and the chain output in that carry fixes the chain input,
except with a collision. So all runs extract the same records. No extra fork at
`η` is necessary. The query log includes the verifier's setup queries, so the
same argument covers `D_init`.

The fixed length `N`, the index in each chain input, the fixed lane lengths, and
the domain tags prevent reordering, truncation, and extension. The IS and FS
chains share formulas on purpose. The transcript gives each root its role
through its fixed position. For the IS lane, the root `D_mem` is the output of
the previous segment's FS chain, or the verifier's `D_init`. ∎

### Lemma 3 — Fingerprint test

**Statement.** Work in the translated experiment of A6. Let the adversary make
at most `q_η` queries to the `η` transcript. For each closed segment, let
`A = IS ∪ WS` and `B = RS ∪ FS` be the multisets that the extracted records
define, each of size at most `m_mem`. Then

```text
Pr[ some segment closes with A ≠ B ]  ≤  q_η · ε_test + ε_H,
ε_test = 2·m_mem / |𝕂|.
```

**Proof, part 1: fixed multisets.** Fix `A ≠ B` with the tuple ranges of
Lemma 4. Spec §4.2 rules 3 and 4 make `t`, `g`, and `v` integers below `q`, so
each has one canonical field value. Distinct tuples therefore give distinct
polynomials `p(η1) = g + η1·v + η1²·t` in `𝕂[η1]`. Each factor
`f = p(η1) − η2` has degree 1 in `η2` with a unit leading coefficient, so it
is prime in `𝕂[η1, η2]`, and distinct tuples give non-associate factors.
Unique factorization keeps multiplicity. So the two products are equal as
polynomials only if `A = B`. For `A ≠ B`, their difference `G` is a nonzero
polynomial of total degree at most `2·m_mem`, because each factor has total
degree at most 2. Schwartz–Zippel gives `Pr[G(η1, η2) = 0] ≤ 2·m_mem/|𝕂|`,
which is `ε_test`, for a uniform pair.

**Proof, part 2: the multisets are fixed before `η`.** By Lemma 2, the records
of a closed segment are the inputs of chain queries made before its `η` query,
except with `ε_H`. By Lemma 1, the packed inputs determine the records. By
Lemma 4 and Ob6, the tuples of `A` and `B` are functions of the records, of
`ts` and `plan_digest` (both in the `η` query input), and of the structural
index. So for each `η` query, the multisets that a later close can use are
fixed by the query input and the query log before it. The answer `η` is
uniform and independent of them. By part 1, a query whose multisets differ
passes the product equation with probability at most `ε_test`. A union bound
over the `q_η` queries gives the statement. ∎

Under the observable oracle of A6, no fork at `η` and no commitment-binding term
is necessary: the chains hash the records themselves. A6 states the fallback.

In one segment, `|IS| = |FS| = R + M` and `|RS| = |WS|` is the active count.
So `m_mem ≤ R + M + N · B_ops`.

### Lemma 4 — Rows to multisets

**Statement.** Suppose an assignment satisfies rows O1–O9, S1–S3, and the
boundary rows of spec §8, and the hash-residency property Ob6 holds. Then the
outgoing products equal the incoming products times the fingerprints of the
extracted tuples:

```text
h_rs_out = h_rs_in · Π_{active j} f(RS_j),   h_ws_out = h_ws_in · Π_{active j} f(WS_j),
h_is_out = h_is_in · Π_j f(IS_j),            h_fs_out = h_fs_in · Π_j f(FS_j).
```

The tuple contents depend only on hashed lane coordinates and the carry fields
`ts` and `idx`, which do not depend on `η`.

**Proof.** O1 makes `pad_j` a bit, so the O8 and O9 factor is `1` for a pad and
`f` for an active slot. Induction over the slots gives the products. S2 and S3
give the scan products in the same way. O1 bounds `rt`, `v`, and `addr` by
their widths. For RAM, `g = R + addr < R + M` because `addr < 2^μ = M`. For
ROM, O6 gives `addr < R`. The carry field `ts` is a `W_ts`-bit word, the
boundary row bounds `out.ts < 2^W_ts`, and `wt_j ≤ out.ts`. So every tuple
satisfies the ranges of Lemma 3. Ob6 gives the last sentence. ∎

### Lemma 5 — Sequential consistency of one segment

**Statement.** Assume the following for one segment:

1. the segment has exactly one IS tuple and one FS tuple for each cell `g`;
2. the write timestamps of the active operations are distinct;
3. each operation has `rt < wt`;
4. a read has `v_w = v_r`, and no operation writes to ROM;
5. `IS ∪ WS = RS ∪ FS` as multisets.

Then the operations on each cell form one sequential history. The first
operation reads the IS value. Each later operation reads the value of the
operation before it. The FS tuple is the result of the last operation, or the
IS tuple if the cell has no operation.

**Proof.** Fix a cell `g`. Tuples for different cells never match, because the
tuple contains `g`. Order the operations on `g` by write timestamp:
`o_1 < … < o_n`. Restricted to `g`, assumption 5 says
`{IS_g} ∪ {WS(o_1) … WS(o_n)} = {RS(o_1) … RS(o_n)} ∪ {FS_g}`.

Consider `WS(o_n)`, the tuple with the largest timestamp `wt_n`. It must equal
a tuple on the right side. It cannot equal `RS(o_i)`: if `i < n`, then
`rt_i < wt_i < wt_n`, and if `i = n`, then `rt_n < wt_n`. So
`WS(o_n) = FS_g`. Remove both tuples.

Now consider `WS(o_{n−1})`. It cannot equal `RS(o_i)` for `i ≤ n − 1`, by the
same argument, and `FS_g` is gone. So `WS(o_{n−1}) = RS(o_n)`. Repeat for each
`o_i` in decreasing order: `WS(o_i) = RS(o_{i+1})`. At the end, the left side
holds only `IS_g` and the right side holds only `RS(o_1)`, so
`IS_g = RS(o_1)`.

Equal tuples have equal values. So `o_1` reads the IS value, each `o_{i+1}`
reads `v_w` of `o_i`, and `FS_g` holds `v_w` of `o_n`. Assumption 4 makes a
read keep the value and keeps ROM unchanged. If `n = 0`, then `IS_g = FS_g`. ∎

The relation gives the assumptions. Exact cover and the structural index give
assumption 1. The write timestamps are `ts_open + 1, …, ts_close`, one per
active operation, because `cnt` grows by one per active slot and the carry
passes `ts` from step to step without a reset (assumption 2). O4 gives
assumption 3, because spec §4.2 rule 3 keeps it an integer relation. O3 and O5
give assumption 4. Lemmas 2–4 give assumption 5, except with the error of
Lemma 3.

This proof needs no range check on IS or FS timestamps.

### Lemma 6 — Lifecycle and segment composition

**Statement.** Assume A1–A6 and Ob1–Ob8. If the verifier accepts, then, except
with the bound of §5, there is one execution of the application relation with
these properties:

- it starts from the statement's initial state and the plan's initial memory;
- it has `T = S · N` steps and ends in the statement's final state;
- its memory accesses are sequentially consistent across all segments;
- its final memory has the FS root `final_memory_root`.

**Proof.**

1. **Witnesses.** By A1 and A4, every claim `u_0 … u_{T−1}` has an extracted
   witness that satisfies the F′ relation. The Stage 1 terminal opens the 16
   children and `u_{T−1}` directly.
2. **Carry thread.** The state link of Construction 2 (A3) makes the input
   carry of `A[i]` equal to the output carry of `A[i−1]`. The arm follows from
   that carry (Ob8). So the steps run in order, each step hashes its own lanes
   once, and a segment closes exactly when `idx` reaches `N`.
3. **One segment.** At each close, Lemma 2 fixes the records before `η`.
   Lemma 4 turns the rows into the four products. Lemma 3 gives
   `IS ∪ WS = RS ∪ FS`, except with the error of Lemma 3. Lemma 5 gives one
   sequential history per segment.
4. **Segment joins.** For segment 0, `D_seen.is = D_init`. The verifier
   computed `D_init` with the same chain and packing over the plan images, and
   the relation reads it as a package constant. So by A3 (collision
   resistance) and Lemma 1, the IS records equal the plan images with `t = 0`. For segment `k + 1`, `D_seen.is = D_mem`, which is the
   FS root of segment `k`. The IS and FS chains share formulas, so the IS
   records of segment `k + 1` equal the FS records of segment `k`, cell by
   cell, except with `ε_H`. The global timestamp never resets, so the write
   timestamps of all segments are distinct. Joining the segment histories gives
   one sequential execution.
5. **Application.** The port rows (Ob7) make these memory accesses the
   accesses of the application relation. An idle step changes neither memory
   nor the application state.
6. **End.** The verifier rejects the initial envelope. The terminal requires
   `idx = N` in the final carry, `1 ≤ S ≤ S_max`, `T = S · N`, and
   `seg_idx = S`. The statement fields equal the final state. ∎

## 5. Composition and evaluation

**Theorem.** Assume A1–A6 and Ob1–Ob8. Let the adversary make at most `q_η`
queries to the `η` transcript. There is an extractor `E`, built from the
extractors of A1 and A4, such that, in the translated experiment of A6,

```text
Pr[accept and E's output is not a valid execution]
   ≤  ε_Stage1                          (A1, A4: every fold and the terminal)
    + q_η · ε_test                      (Lemma 3)
    + ε_H                               (A3: every query of the adversary and of the extractor)
```

with `ε_test = 2·m_mem/|𝕂|` and `m_mem ≤ R + M + N · B_ops`. `ε_Stage1` is
the Stage 1 error for the Stage 2 relation shape, including SuperNeo's
`ε_uniq` for `L_full`. The proof is Lemma 6 with a union bound.

A6 carries this bound to the real protocol through `g_d` and `delta_d`.
`ε_H` comes from a work estimate. It becomes a probability only for a stated
query count.

**Setup check (spec §4.2 rule 6).** For the final relation shape and the plan,
setup MUST check that `ε_test ≤ 2^−109.91`, the owner-approved floor. This
holds exactly when `R + M + N · B_ops ≤ 139,509`. Setup MUST reject the plan
if the check fails. A larger memory needs a new owner decision. The check
applies to the per-check value. The theorem multiplies `ε_test` by `q_η`, as
Fiat–Shamir multiplies the fold term by its query count.

**Evaluation at the current values.** The values use the trunk formula with
`f_fold = 7,781` and `f_fork = 304`. The example geometry is `R = 2^12`,
`M = 2^16`, `N = 64`, and `B_ops = 1,088`, so `m_mem ≤ 139,264`. A release
MUST recompute this table for the generated relation with
`NeoParams::padded_row_security_summary_for_shape`.

| Term | Bits |
|---|---:|
| One fold, `f_fold/|𝕂| + f_fork/|C|` | 114.76 |
| Main Module-SIS level (Stage 1) | 110.685 |
| Poseidon2 ceiling | about 128 |
| `ε_test`, example geometry | 109.91 |

The example geometry is at the floor. The memory term is the weakest
statistical term: 109.91 bits is below the fold term of 114.76 bits. A second,
independent challenge pair would give 219.83 bits. This specification does not
use it.

These are per-check values, before any query factor. For 64 folds and one
segment, the sum of the per-check values is 108.23 bits; the 64 folds alone
give 108.76 bits.

**Legacy-ledger values, for information.** This table applies the legacy
query bound `q_H = 2^18` as a lift on every statistical term.

| Lift | Fold term | `ε_test` | One fold and one segment |
|---|---:|---:|---:|
| none | 114.76 | 109.91 | 109.86 |
| `q_H` | 96.76 | 91.91 | 91.86 |

Under the legacy lift, the memory term misses the legacy 96-bit target by 4.09
bits. If recursive extraction also costs a factor of 64 folds (A4), the fold
term alone falls to 90.76 bits. The correct lift is part of A4 and A6.

## 6. Oracle-query census

Stage 2 adds these prescribed oracle calls to the Stage 1 census:

- per fold: the three record chains of `step`. At the example geometry with
  `W_ts = 32`, they absorb 1,987 + 1,106 + 1,106 packed elements plus tags
  and framing, about 360–370 permutations;
- per segment: one challenge transcript with two extension squeezes.

The query unit and the total belong to A6. When the trunk fixes them, the
generated relation MUST supply the exact census.

## 7. Open obligations

1. **Non-author review** of Lemmas 2, 3, 5, and 6, and of §5.
2. **Stage 2 authorization** by the owner.
3. **A6 Stage 2 premise.** Owner approval of the observable-oracle premise for
   in-circuit `H` (`η` and the record chains), or of the fallback in A6.
4. **Generated relation** that fits the `2^28` domain, with its exact shape and
   census, and the recomputed §5 table.
5. **Lean formalization** of the Stage 2 phase: Lemma 1 (packing), Lemmas 2–6,
   and the extended composition, Property 6, fixed-point, and domain theorems
   for the larger state (A4; `FPRIME_LEAN_ARCHITECTURE_SPEC.md` §6).
6. **Shared with Stage 1:** useful values of `g_d` and `delta_d` (A6), the
   concrete extractor time (A4), the arity adaptation of HyperNova Lemma 4, the
   terminal decider, encoding, and implementation terms, and an approved F′
   threat model with an end-to-end target. The legacy
   `protocol-contract/security-reduction.md` §8 shows the form of these terms.
7. **Query model, shared with Stage 1.** The Module-SIS level is a quantum
   estimate, but this note counts classical oracle queries. A quantum search
   finds a bad one-check challenge in about the square root of the classical
   work: about `2^55` for a `2^−109.91` check. The fold challenges have the
   same exposure.
