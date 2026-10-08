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
decision. `ε_coll(t)` is the probability that an adversary with expected time
`t` finds a collision of `H`. Stage 2 adds digests of the same kind as Stage 1.

**A4 (Recursive extraction).** HyperNova Lemma 4, as corrected in
`docs/hypernova-paper/`, gives knowledge soundness of Construction 2 for a
fixed constant iteration bound. Its premises are:

- the canonical encoders satisfy Definition 12, including Property 6
  (recursive-size closure);
- `hash` is collision resistant or a random oracle;
- the non-interactive folding scheme is knowledge sound. HyperNova states
  this as Assumption 1 for its own scheme. For F′, it is SuperNeo folding
  under A1 and A6.

Here the bound is `S_max · N`. Stage 2 adds no field to the state preimage:
the carry is in the application-state digest (spec §11.1). So the Stage 1
Property 6, fixed-point, and domain theorems apply to the memory
application's package. Definition 12 is stated for arity
`(1, 1)`, and F′ folds one fresh claim with 16 running children. So Stage 1
must also adapt the lemma to that arity. The lemma gives no concrete extractor
time. That bound is also open for Stage 1.

**A5 (Field facts).** `q` is prime, and 7 is a quadratic nonresidue modulo
`q`. Then `𝕂 = F[U]/(U² − 7)` is a field of size `q²`, and `F → 𝕂` is
injective. Lean proves both facts (`GoldilocksPrime.goldilocks_natPrime`,
`GoldilocksExtension.extensionNoZeroDivisors`), so A5 is not an assumption.

**A6 (Fiat–Shamir game transfer).** The F′ relation computes outputs of `H`
inside a recursively proven step. A random oracle cannot run inside a circuit,
so no random-oracle proof covers these outputs directly. Stage 1 states this
gap as an explicit premise, which the owner approved on 2026-09-11
(`docs/reviews/nightstream-fprime-requirements/FIAT_SHAMIR_MODEL.md`; Lean
`FiatShamirModel` in `Lifecycle/Nifs/FiatShamirTransfer.lean`). The premise
transfers success from the real Poseidon2 experiment to a translated
experiment, through external functions `g_d` and `delta_d` of the total
permutation-query count `Q`.

A6 extends that premise to the memory round of each segment. In the translated
interactive game, the prover sends the three roots `(D_pre.ops, D_mem,
D_pre.fs)` and `ts`, and the verifier then samples `η` uniformly. The record
chains need only collision resistance (A3). Lemma 3 holds in that game, and A6
carries the §5 bound to the real protocol. One transfer covers the whole F′
transcript. The owner approved this extension on 2026-10-07. Like the Stage 1
boundary, it selects no numerical value for `g_d` or `delta_d`.

**Optional stronger premise, not used.** A stronger premise would treat every
in-circuit output of `H` as an observable oracle answer, with the query log
visible to the reduction. The query order would then fix the records before
`η`, and Lemma 3 would need no rewind. This note does not rely on it.

## 3. Deterministic obligations

These obligations have no probability term. A failure is a defect.

| # | Obligation |
|---|---|
| Ob1 | Canonical decoding of statements, claims, and the carry words; no alias of a noncanonical field element |
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
| J1: records into chains | Lemmas 1 and 2; Ob2, Ob3, Ob6 | Proof in this note, deterministic |
| J2: carry thread | A3, A4; Ob8 | Assumed. The arity adaptation is open; Property 6 is the Stage 1 one |
| J3: commit, then test | Lemmas 2–5 | Proof in this note, in the translated game of A6 |
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

### Lemma 2 — Chain binding

**Statement.** Fix a lane `l` and the chain formulas of spec §9.2. Let
`P[0..N)` and `P′[0..N)` be two sequences of packed lanes whose chains have the
same root. If `P ≠ P′`, then the two chain computations contain two different
inputs of `H` with the same output: a collision.

**Proof.** Compare the two chains from the root down. At index `N − 1`, the two
outputs are equal. If the two inputs `(tag, [N − 1, D[N − 1]], P[N − 1])`
differ, they form a collision. Otherwise `D[N − 1] = D′[N − 1]` and
`P[N − 1] = P′[N − 1]`, and the same argument applies at index `N − 2`. Both
chains start at the same header. So if no level gives a collision, then
`P[j] = P′[j]` for every `j`. ∎

The fixed length `N`, the index in each chain input, the fixed lane lengths, and
the domain tags prevent reordering, truncation, and extension. The IS and FS
chains share formulas on purpose. The transcript gives each root its role
through its fixed position. For the IS lane, the root `D_mem` is the output of
the previous segment's FS chain, or the verifier's `D_init`. ∎

### Lemma 3 — Fingerprint test

**Statement.** Work in the translated game of A6, for one segment. Let `in` be
everything fixed before `η`: the three roots, `ts`, and the adversary's state.
After `η`, the extractor of A1 and A4 returns the segment's records. Let
`A = IS ∪ WS` and `B = RS ∪ FS` be the multisets that these records define,
each of size at most `m_mem`. Then

```text
Pr[ the close product equation holds and A ≠ B ]  ≤  ε_test + ε_coll(t_1),
ε_test = 2·m_mem / |𝕂|,
```

where `t_1` is at most twice the time `t_E` of one run with extraction.

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

**Proof, part 2: records extracted after `η`.** The extractor returns the
records after `η`, so they can depend on `η`. We use the uniqueness argument of
SuperNeo v1.2 Appendix B, proof of (ii) for PiCCS. Here `η` takes the role of
`(α, γ)`, and the fingerprint identity takes the role of the sum-check test.

One oracle call runs the rest of the game from `in` with a fresh uniform `η`,
and then runs the extractor. `Succ` is the event that the segment closes and the
extraction is valid. `Err` is `Succ` together with `A ≠ B`, so `Err ⊆ Succ`.
The uniqueness adversary makes one call. If `Err` does not occur, it stops.
Otherwise it repeats calls until `Succ` occurs, and it returns both record
sets. By the calculation of SuperNeo v1.2 Appendix B, `t_in + (a_in/p_in)·t_in`
with `a_in ≤ p_in`, its expected time `t_1` is at most twice the time of one
call.

- If the two record sets agree, then the multisets `(A, B)` that the first
  call fixed pass the equation for a fresh, independent `η`. By Lemma 4 and
  Ob6, the tuples depend only on the records, on `ts` (in `in`), and on plan
  constants. By part 1 and the conditional-probability step of SuperNeo v1.2
  Appendix B, `Pr[Err and agree] ≤ ε_test`.
- If they differ, both calls closed the segment against the same roots, which
  `in` fixes. By Lemma 1, different records give different packed lanes. By
  Lemma 2, the two chain computations then contain a collision of `H`. So
  `Pr[Err and differ] ≤ ε_coll(t_1)`.

So `Pr[Err] ≤ ε_test + ε_coll(t_1)`. ∎

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
   state of `A[i]` equal to the output state of `A[i−1]`. Both are state
   digests (spec §11.1), so the input carry of `A[i]` equals the output carry
   of `A[i−1]`, except with a collision (A3). The arm follows from
   that carry (Ob8). So the steps run in order, each step hashes its own lanes
   once, and a segment closes exactly when `idx` reaches `N`.
3. **One segment.** At each close, Lemma 4 turns the rows into the four
   products. Lemma 3 gives `IS ∪ WS = RS ∪ FS`, except with the error of
   Lemma 3. Lemma 5 gives one sequential history per segment.
4. **Segment joins.** For segment 0, `D_seen.is = D_init`. The verifier
   computed `D_init` with the same chain and packing over the plan images, and
   the relation reads it as a package constant. So by Lemmas 1 and 2, the IS
   records equal the plan images with `t = 0`, except with a collision. For
   segment `k + 1`, `D_seen.is = D_mem`, which is the FS root of segment `k`.
   The IS and FS chains share formulas, so by Lemmas 1 and 2 the IS records of
   segment `k + 1` equal the FS records of segment `k`, cell by cell, except
   with a collision. The global timestamp never resets, so the write
   timestamps of all segments are distinct. Joining the segment histories gives
   one sequential execution.
5. **Application.** The port rows (Ob7) make these memory accesses the
   accesses of the application relation. An idle step changes neither memory
   nor the application state.
6. **End.** The verifier rejects the initial envelope. The terminal requires
   `idx = N` in the final carry, `1 ≤ S ≤ S_max`, `T = S · N`, and
   `seg_idx = S`. The statement fields equal the final state. ∎

## 5. Composition and evaluation

**Theorem.** Assume A1–A6 and Ob1–Ob8. There is an extractor `E`, built from
the extractors of A1 and A4, such that, in the translated game of A6,

```text
Pr[accept and E's output is not a valid execution]
   ≤  ε_Stage1                          (A1, A4: every fold and the terminal)
    + S_max · (ε_test + ε_coll(t_1))    (Lemma 3, once per segment)
    + ε_coll(t_E)                       (Lemma 2 at every segment join)
```

with `ε_test = 2·m_mem/|𝕂|` and `m_mem ≤ R + M + N · B_ops`. `ε_Stage1` is
the Stage 1 error for the Stage 2 relation shape, including SuperNeo's
`ε_uniq` for `L_full`. One reduction checks every segment join in one run. The
proof is Lemma 6 with a union bound.

In Lean, the memory terms of this bound in the translated game are
`Game.fails_frequency` and, over `𝕂` with Poseidon2, `memory_bound` (§7
item 5). A6 carries this bound to the real protocol through `g_d` and
`delta_d`. The
transfer accounts for the oracle-query count. `ε_coll` comes from a work
estimate (A3). It becomes a probability only for a stated adversary time.

**Setup check (spec §4.2 rule 6).** For the final relation shape and the plan,
setup MUST check that `ε_test ≤ 2^−109.91`, the owner-approved floor. This
holds exactly when `R + M + N · B_ops ≤ 139,509` (Lean `Plan.secure_iff`). Setup MUST reject the plan
if the check fails. A larger memory needs a new owner decision. The check
applies to the per-check value. The A6 transfer adds the query loss, as it
does for the fold term.

**Above the cap.** The recommended upgrade for a plan with more than 139,509
tuples per segment is a second, independent challenge pair. Then `ε_test`
becomes `(2·m_mem/|𝕂|)²`: 219.83 bits at the example geometry, for about +2%
coordinates. It needs an owner decision and a spec change before use. It is not
part of this specification.

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
- per segment: one challenge transcript with two extension squeezes;
- per fold: two state digests (spec §11.1), for the input and the output
  state.

The query unit and the total belong to A6. When the trunk fixes them, the
generated relation MUST supply the exact census.

## 7. Open obligations

1. **Non-author review** of Lemmas 2, 3, 5, and 6, and of §5.
2. **Stage 2 authorization.** The owner authorized Stage 2 on 2026-10-07.
3. **A6 Stage 2 extension.** The owner approved it on 2026-10-07 in the form
   that A6 states. Its numerical values stay open with item 6.
4. **Generated relation** that fits the `2^28` domain, with its exact shape and
   census, and the recomputed §5 table.
5. **Lean formalization.** The model is in
   `formal/nightstream-fprime/NightstreamFPrime/Spec/Nebula/`, the Poseidon2
   framing in `Lifecycle/Nebula/Framing.lean`, and each theorem is audited in
   `tests/AxiomsNebula.lean`. Done:
   - Lemma 1, and the length of each packed lane;
   - Lemma 2, with the collision among the inputs of the two chains;
   - Lemma 3 part 1, as a count, as a frequency, and on `𝕂` as `2·m/q²`;
   - Lemma 3 part 2 for one segment, also on `𝕂`, in counting form: for a
     fixed transcript input and a call function of uniform `(η, coins)`;
   - Lemma 4 in typed form. The typed rows contain the pad gate. The field
     form of rows O1, O4, and the O8 gate is proved separately for the
     generated relation to use;
   - Lemma 5 and its converse;
   - the lifecycle of the typed model (Ob8 for the model);
   - Lemma 6 as a deterministic theorem over an extracted typed run: an
     accepted run gives an execution, or a collision among the run's own chain
     inputs and the `D_init` chain, or a segment that passes the product test
     with unbalanced multisets. In the concrete context, the collision is a
     Poseidon2 transcript collision: two different absorbed chunk sequences
     with the same digest;
   - Ob3 for the record chains: the framing of spec §9.1–§9.2 separates
     canonical chain inputs;
   - rule 6 of spec §4.2 (`ε_test ≤ 2^−109.91` exactly when
     `m_mem ≤ 139,509`);
   - completeness of the typed model (Ob5 for the model), for every hash
     function and every challenge function;
   - the memory terms of §5 in the interactive game of A6
     (`Spec/Nebula/Game.lean`, `Lifecycle/Nebula/MemoryBound.lean`). An
     outcome is the prover's coins and one fresh uniform challenge pair per
     segment. The prover commits each segment's transcript input before its
     challenge, and the extractor returns the run after the last challenge.
     The frequency of an accepted, consistent run that is not an execution is
     at most the frequency of a run collision, plus `S_max · 2·m_mem/q²`, plus
     one retry term per segment. Both collision terms are Poseidon2 transcript
     collisions. The retry adversary makes at most 2 calls in expectation
     (`t_1 ≤ 2·t_E`, counted in calls);
   - the first memory application as a Stage 1 application program
     (`Lifecycle/Nebula/MemoryProgram.lean`) for a plan with two ports. Its
     circuit holds exactly when the output state is the state digest of the
     output words and the decoded witness satisfies the memory rows and the
     machine rows, and every row reads only supported variables;
   - the soundness refinement of that relation: canonical carry decoding
     (Ob1), the field-bit packing of the lanes (Ob2), one invocation's rows to
     the model's `invoke` with a reachable output carry (Ob4, and Ob6 because
     the chains and the products read the same decoded records), the port rows
     to one machine step (Ob7), and the arm, `close`, and terminal rules (Ob8
     for the relation);
   - the run-level join (`MemoryApp.chain_soundness`): a chain of states whose
     steps satisfy the program's predicate, with the spec §13 terminal checks,
     gives Lemma 6's three results, or a collision in a state digest. The
     relation reads `plan_digest` and the chain headers as constants, and the
     terminal computes the initial state from the start carry and `D_init`.

   Still open:
   - the A6 transfer from the real verifier to this game, as a premise in the
     form of the Stage 1 `FiatShamirModel`. Its real-side event needs the
     Stage 2 verifier;
   - an extracted run whose segment transcript inputs differ from the
     committed ones is an extraction failure. A1 and A4 must bound it; the
     game theorem does not;
   - `ε_coll` as a probability needs a stated adversary time (A3);
   - completeness of the relation (Ob5 for the relation): an honest model
     step has a witness that satisfies the program's predicate, with the
     counter widths `W_step`, `W_seg`, `W_cnt` and the O4 words `diff_j`;
   - the memory application's package, with its Stage 1 fixed-point and
     domain theorems for a selected plan;
   - the Stage 1 extraction that turns an accepted proof over that package
     into the chain of states that `MemoryApp.chain_soundness` reads (A1–A4).
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
