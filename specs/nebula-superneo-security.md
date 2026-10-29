# Nebula memory phase on SuperNeo F′ — security note

Status: **Proposed.** Companion to [`nebula-superneo.md`](./nebula-superneo.md)
("the spec"). The author wrote the lemmas below. A production claim needs a
review by someone other than the author (§7).

This note states the security goal, the assumptions, six lemmas with proofs,
and the composed bound. It evaluates the bound at the current values.

## 1. Threat model and security goal

The prover controls all proof bytes, witnesses, memory records, and proposed
roots. The verifier controls the verifier key, the application package, the
program, the setup seed, and the public statement.

This note uses *relation* for the circuit that a package proves and *program*
for the ROM and RAM images. A package is the relation of one application at
one geometry. The geometry is every plan field except the images (spec §4.1).
The key of a program is its `D_init`, which the verifier computes from the
images and puts in the start carry (spec §9.2, §11.1). The relation and
`plan_digest` read no image, so one package and one setup serve every program
of a geometry: setups are one per (application, geometry). `D_init` is the
only path by which the program enters a proof. Two programs with different
images have different initial snapshots, so they have different `D_init`,
except with a collision (Lemmas 1 and 2). In Lean, the relation
`MemoryApp.program` takes a `Geometry`, which has no image field, so two plans
with one geometry have one relation (`MemoryApp.program_eq_of_geometry`).

**Program choice.** Security holds for a program fixed before the adversary
runs: the plan is a parameter of the end-to-end statement (§5), and the
adversary does not choose it. An adversary that chooses the program after it
sees the setup is not covered. That case needs the claim and the `Experiment`
to carry the program, and it is an open item (§7).

**Current values.** The F′ Stage 1 system has these levels:

| Quantity | Level | Source |
|---|---|---|
| Statistical error of one fold | 115.22 bits (conservative) | `NeoParams::padded_row_security_summary_for_shape`: 4 matrices, CCS degree 7 (sum-check degree 8), `ℓ = 28`, at the PiRLC guard's largest fresh count `K = 287` |
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
memory term is the weakest statistical term: it is 5.31 bits below the fold
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
relation, but no state field (the carry is in the application-state digest,
spec §11.1) and no commitment component (spec §7). The theorem is an
interactive reduction of knowledge. Its statistical error for one fold is
`f_fold/|𝕂| + f_fork/|C|`, with `|C| = 5^54`. `f_fold` bounds the numerator of
v1.2's `ε_test` (sum-check and Schwartz–Zippel), and `f_fork` is the PiRLC
extraction term. The trunk computes both with
`NeoParams::padded_row_security_summary_for_shape` at the largest fresh count
that the PiRLC guard allows, `K = 287`: `f_fold = 5,161` and `f_fork = 304`.
F′ folds one fresh claim, and for `K = 1` Lean proves the numerator `4589`
(`VerifierErrorBudget.test_error_eq`), so 5,161 is conservative. The fork
numerator `K + k_rho + 1` of the Rust summary grows with `K` (18 at `K = 1`),
so 304 is conservative too. v1.2's `ε_uniq` reduces to relaxed binding of
`L_full` (A2).

**A2 (Module-SIS binding).** `L_full` is `(2B, C)`-relaxed binding at the main
level, as SuperNeo v1.2 §7 requires. Its key comes from the SHAKE128 expander,
modeled as a random oracle. This is the Stage 1 assumption without change.

**A3 (Poseidon2).** `H` is a random oracle for Fiat–Shamir. Its digest is
collision resistant and preimage resistant at the ceiling of the trunk
decision. `ε_coll(t)` is the probability that an adversary with expected time
`t` finds a collision of `H`. Stage 2 adds digests of the same kind as Stage 1.
The sponge has four capacity words and a four-word digest, so the generic value
under the trunk's random-permutation reading is about `ε_coll(t) = t²/q⁴`
(about `t²/2^256`), with `t` counted in permutation calls. The owner has set no
adversary time (decision 2026-10-12), so `t` stays symbolic. For a stage of the
reverse extractor, this value applies only under the human-ignorance reading
of Assumption 1 (`formal/nightstream-fprime/SECURITY_MODEL.md`, trust item 4):
an algorithm that contains a known collision has `ε_coll = 1`.

**A4 (Recursive extraction).** HyperNova Lemma 4, as corrected in
`docs/hypernova-paper/`, gives knowledge soundness of Construction 2 for a
fixed constant iteration bound. Its premises are:

- the canonical encoders satisfy Definition 12, including Property 6
  (recursive-size closure);
- `hash` is collision resistant or a random oracle;
- the non-interactive folding scheme is knowledge sound. HyperNova states
  this as Assumption 1 for its own scheme. For F′, it is HyperNova errata
  Assumption 1 for the Poseidon2 fold (A6, first paragraph); A1 and the
  one-fold random-oracle theorem of PR #155 motivate its error.

In Lean, the Stage 1 history is the output of the HyperNova reverse extractor
(Lemma 17) under HyperNova errata Assumption 1 in the form of Definition 7
(`HyperNovaVisitedSecurity.history_failure_le`, merged from PR #155; owner
decision 2026-10-12). It holds for every application that fits `2^28`. Its
failure event is that the terminal verifier accepts and the reverse program
returns no history of valid steps: each returned step satisfies the
application's validity predicate (`HyperNovaHistory.ValidAdvice`).

Here the bound is `S_max · N`: the theorem covers an adversary whose every
output, accepted or not, has at most `S_max · N` steps (Lean `depthBound`). An
adversary with longer outputs reduces to this case through its clamped form
(`NebulaMemorySecurity.clamp`): the same tape, with each longer output
replaced by the same output at iteration 0. The clamp meets the bound
(`clamp_depthBound`). On each tape, the clamped output passes the memory
verifier exactly when the adversary's output does, and the two are then equal
(`clampOutput_eq_of_checks`, `clampOutput_eq_of_input_checks`). So
`bad_le_clamp` bounds the bad mass of the adversary's own outputs by the bound
of `bad_le` for the clamped form at depth `S_max · N`. Its premise is that the
clamped form's start stage is admitted (`StageAdmitted (Stage.start …)`), as
for any adversary of `bad_le`. Stage 2 adds no field to the state preimage:
the carry is in the application-state digest (spec §11.1). So the Stage 1
Property 6, fixed-point, and domain theorems apply to the memory application's
package. Definition 12 is stated for arity `(1, 1)`, and F′ folds one fresh
claim with 16 running children. So Stage 1 must also adapt the lemma to that
arity. The lemma gives no concrete extractor time. That bound is also open for
Stage 1.

**A5 (Field facts).** `q` is prime, and 7 is a quadratic nonresidue modulo
`q`. Then `𝕂 = F[U]/(U² − 7)` is a field of size `q²`, and `F → 𝕂` is
injective. Lean proves both facts (`GoldilocksPrime.goldilocks_natPrime`,
`GoldilocksExtension.extensionNoZeroDivisors`), so A5 is not an assumption.

**A6 (Fiat–Shamir in the circuit).** The F′ relation computes outputs of `H`
inside a recursively proven step. A random oracle cannot run inside a circuit,
so no random-oracle proof covers these outputs directly. Stage 1 states this
gap as HyperNova errata Assumption 1 in the form of Definition 7: every
admitted fold adversary has an efficient extractor that fails after a real
success with probability at most an error term of that adversary
(`HyperNovaVisitedSecurity.Assumption1`, PR #155). A one-fold random-oracle
theorem motivates the value of that error. This premise replaced the earlier
transfer premise `FiatShamirModel` (owner approval 2026-09-11).

A6 extends that premise to the memory round. The owner approved the extension
on 2026-10-07 and chose on 2026-10-12 to state it in the form of HyperNova
errata Assumption 1, as Stage 1 states its history premise. An adversary is
a stage of the reverse extractor with a memory statement that its tape gives.
The stage must be admitted in the same sense as the stages of A4
(`StageAdmitted` of `Closed`). For every such adversary, the probability that
the extracted run is accepted and has a claimed segment that passes the
product test with unbalanced multisets is at most one error term `error_mem`
of that adversary (Lean `NebulaMemorySecurity.MemoryRound`). One term covers
all segment slots. In §5, the claim is the memory statement that the IVC
adversary gives from its own tape. In Lean, `MemoryRound` holds when
`error_mem` is the probability of the event itself. So the content of A6 is
that `error_mem` is at most its target value (§5). Assumption 1 of A4 puts
its quantitative content in its error in the same way, but it also asserts
that an efficient extractor exists. The record chains need only collision
resistance (A3).

No random-oracle model covers `η`. The relation computes `η` with Poseidon2
inside the proven step, and the extractor gets the Poseidon2 inputs and
outputs in the witness, not oracle queries. A random-oracle proof would need
one more step, "the Poseidon2 in the circuit acts as the oracle", which is
this assumption, not a proof of it.

A proved game bound motivates the value of `error_mem`, as the one-fold
random-oracle theorem motivates the Stage 1 error. In the interactive game for
slot `k`, the prover sends the three roots `(D_pre.ops, D_mem, D_pre.fs)` and
`ts` of segment `k`, and the verifier then samples `η` uniformly. Lemma 3
holds for each slot: slot `k` is bad with frequency at most `2·m_mem/q²` plus
a retry term (`badAt_bound`). An accepted run with an unbalanced segment has a
first unbalanced segment, and the segments before it are balanced, so they
close with any challenge. A straight-line translation that guesses which of
the `Q_η` evaluations of the §9.3 transcript answers the first unbalanced
segment (the relation evaluates it in every invocation, and off-circuit
evaluations count too), or that none does, has the factor `Q_η + 1` once for
all slots. The extractor's rewinds can make the factor larger. The §9.3
framing separates canonical inputs (Ob3), so the committed transcript input is
the guessed oracle input. Like the Stage 1 boundary, A6 selects no numerical
value.

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
| Ob6 | Hash residency (spec §6.4): every application obeys it, because its rows have no variable for `η`, a product, or a carry word (spec §10) |
| Ob7 | Port coverage: no application memory effect bypasses a port (spec §10) |
| Ob8 | Lifecycle (spec §11.2, §12): the arm follows from the authenticated carry; each step hashes its own lanes; `close` runs exactly when `step` sets `idx = N`; the terminal requires a closed carry |

## 4. Lemmas

The lemmas cover the four joints of spec §1.1. The author wrote every proof in
this note, and each proof needs an independent review (§7).

| Joint | Argument | Status |
|---|---|---|
| J1: records into chains | Lemmas 1 and 2; Ob2, Ob3, Ob6 | Proof in this note, deterministic |
| J2: carry thread | A3, A4; Ob8 | The relation-level thread is proved in Lean (`RunLink`), except with a state-digest collision. The Stage 1 history is the reverse extractor of A4 (`history_failure_le`); Property 6 is the Stage 1 one |
| J3: commit, then test | A6; Lemmas 2–5 | A6 is a premise; Lemma 3 holds in the interactive game that motivates A6 |
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

**Statement.** Work in the interactive game that motivates A6, for one
segment. Let `in` be everything fixed before `η`: the three roots, `ts`, and
the adversary's state. After `η`, the extractor of A1 and A4 returns the
segment's records. Let `A = IS ∪ WS` and `B = RS ∪ FS` be the multisets that
these records define, each of size at most `m_mem`. Then

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
boundary rows of spec §8, and the hash-residency property Ob6 holds (spec
§6.4). Then the outgoing products equal the incoming products times the
fingerprints of the extracted tuples:

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
2. the write timestamps of the active operations are `ts_open + 1,
   ts_open + 2, …` in slot order (the relation gives this through O2, and the
   Lean proof uses this exact form);
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
give assumption 4. Lemmas 2 and 4 with A6 give assumption 5, except with the
A6 error `error_mem`; Lemma 3 motivates its value.

This proof needs no range check on IS or FS timestamps.

### Lemma 6 — Lifecycle and segment composition

**Statement.** Assume A1–A6 and Ob1–Ob8. If the verifier accepts, then, except
with the bound of §5, there is one execution of the application relation with
these properties:

- it starts from the statement's initial state and the program's initial memory;
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
   products. A6 gives `IS ∪ WS = RS ∪ FS`, except with `error_mem`; Lemma 3
   motivates its value. Lemma 5 gives one sequential history per segment.
4. **Segment joins.** For segment 0, `D_seen.is = D_init`. The verifier
   computed `D_init` with the same chain and packing over the program's
   images, and it computes the initial state from the start carry, which holds
   `D_init`.
   By step 2, the input carry of `A[0]` is that carry, except with a
   state-digest collision. So by Lemmas 1 and 2, the IS
   records equal the program images with `t = 0`, except with a collision. For
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
   `seg_idx = S`. The statement fields equal the final state, because the
   words after the last step open the final state, except with a
   state-digest collision. ∎

## 5. Composition and evaluation

**Theorem.** Assume A1–A6 and Ob1–Ob8. Let `E` be the HyperNova reverse
extractor of A4, run for `depth` stages, where every statement that the
adversary can output, accepted or not, has at most `depth` steps
(`depthBound`). On the joint law of the memory statement that the adversary
claims from its own tape and `E`'s sample,

```text
Pr[the verifier accepts and E's output is not a valid execution of the claim]
   ≤  Σ_j (hash_j + error_j)   (A4 under A6 paragraph 1: one term per reverse stage)
    + coll_link                (Lemma 6 steps 2 and 6: state-digest links of E's run)
    + coll_chain               (Lemma 2: the chains of E's run and every join)
    + error_mem                (A6 paragraph 2: one term for all segment slots)
```

with `ε_test = 2·m_mem/|𝕂|` and `m_mem ≤ R + M + N · B_ops`. `hash_j` is the
state-hash collision mass of reverse stage `j`, and `error_j` is the
Assumption 1 error of its fold adversary. `coll_link` and `coll_chain` are
the probabilities of the two collision events of `E`'s own run, so they need
no transfer. `error_mem` is the A6 error of the last reverse stage with the
adversary's claim. The one-fold random-oracle theorem of PR #155 motivates
the value of `error_j`. The target value of `error_mem` is about
`(Q_η + 1) · (ε_test + ε_coll(t_1))` (A6). The proof is Lemma 6 on `E`'s run
with a union bound.

In Lean, this is `NebulaMemorySecurity.bad_le` (§7 item 5). It holds for
every application with a port interface whose memory relation fits `2^28`,
for every commitment setup and every valid plan, so for every program of the
package's geometry: the relation is `MemoryApp.program p.toGeometry app` for
the plan `p`. Its premises are A4
(`Assumption1`, `Closed`, and an admitted start stage), the step bound
(`depthBound`), and A6 (`MemoryRound`); the right side is `badBound`. For an
adversary whose outputs can be longer, `NebulaMemorySecurity.bad_le_clamp`
bounds the bad mass of the adversary's own outputs by `badBound` of its
clamped form at depth `S_max · N`, if that clamped form is admitted (A4).

**Symbolic values.** The owner has set no adversary time `t` and no query
count `Q` (decision 2026-10-12), so each term stays a formula:

- `hash_j ≤ ε_coll(t_j)`, where `t_j` is the time of reverse stage `j`, and
  `coll_link, coll_chain ≤ ε_coll(t_E)` (A3).
- `error_j` has the value that the one-fold random-oracle theorem of PR #155
  motivates: about `17·(Q_j + 17)·ε_sample + (Q_j + 74)·ε_fold + msis + 2^−190`,
  with `ε_fold = 4589/q²` and `ε_sample ≈ 2^−125.4`
  (`formal/nightstream-fprime/SECURITY_MODEL.md`, adversary 3). Lean proves
  `4589/q²` for the F′ shape with one fresh claim
  (`VerifierErrorBudget.test_error_eq`), and every application that fits
  `2^28` has this shape, so a memory relation that fits has it too. The value
  `f_fold = 5,161` of A1 and the table below is the Rust summary at the
  largest fresh count that the PiRLC guard allows (287), so it is the more
  conservative value.
  `Q_j` is the query count of stage `j`, and `msis` is the advantage of an
  MSIS solver.
- `error_mem` has the target value `(Q_η + 1)·(2·m_mem/q² + ε_coll(t_1))` (A6).

**Depth limit, shared with Stage 1.** The sum has one Assumption 1 term for
each reverse stage, and the extractor of stage `j + 1` can rerun stage `j`. If
it reruns its adversary as the random-oracle extractor does,
`Q_{j+1} ≈ 17·Q_j²`. Then `(Q_j + 74)·ε_fold` is above 1 at `j = 1` when
`Q_0 = 2^64`, and at `j = 2` when `Q_0 = 2^30` (`SECURITY_MODEL.md`, trust
item 4). The depth is the bound on the adversary's outputs: `S_max · N` with
the clamp (A4), whatever the length of the accepted run. So, with these
values, this bound gives no concrete security when the depth is 2 or more and
`Q_0 = 2^64`, or 3 or more and `Q_0 = 2^30`. The A6 term grows in the same
way: its adversary is the last reverse stage, so its `Q_η` counts the queries
of every rerun. The Lean statement is correct for every depth. A useful number
needs a Stage 1 argument whose error does not grow in this way with the depth.

**Setup check (spec §4.2 rule 6).** For the final relation shape and the
geometry, setup MUST check that `ε_test ≤ 2^−109.91`, the owner-approved
floor. This holds exactly when `R + M + N · B_ops ≤ 139,509` (Lean
`Geometry.secure_iff`). Setup MUST reject the geometry if the check fails. In
Lean, every memory package that `lake exe emit` writes gets its relation from
`NebulaMemoryPackage.application`, which takes the proofs of rules 1–6. A
larger memory needs a new owner decision. The check applies to the per-check
value. The A6 error adds the query loss, as the Stage 1 error does for the
fold term.

**Above the cap.** The recommended upgrade for a geometry with more than 139,509
tuples per segment is a second, independent challenge pair. Then `ε_test`
becomes `(2·m_mem/|𝕂|)²`: 219.83 bits at the example geometry, for about +2%
coordinates. It needs an owner decision and a spec change before use. It is not
part of this specification.

**Evaluation at the current values.** The values use the trunk formula with
`f_fold = 5,161` and `f_fork = 304`. The example geometry is `R = 2^12`,
`M = 2^16`, `N = 64`, and `B_ops = 1,088`, so `m_mem ≤ 139,264`. A release
MUST recompute this table for the generated relation with
`NeoParams::padded_row_security_summary_for_shape`.

| Term | Bits |
|---|---:|
| One fold, `f_fold/|𝕂| + f_fork/|C|` | 115.22 |
| Main Module-SIS level (Stage 1) | 110.685 |
| Poseidon2 ceiling | about 128 |
| `ε_test`, example geometry | 109.91 |

The example geometry is at the floor. The memory term is the weakest
statistical term: 109.91 bits is below the fold term of 115.22 bits. A second,
independent challenge pair would give 219.83 bits. This specification does not
use it.

These are per-check values, before any query factor. For depth
`S_max · N = 64` (`N = 64`, `S_max = 1`), the sum of the statistical
per-check values (64 fold terms and one `ε_test`) is 108.53 bits; the 64 fold
terms alone give 109.22 bits.

**Legacy-ledger values, for information.** This table applies the legacy
query bound `q_H = 2^18` as a lift on every statistical term.

| Lift | Fold term | `ε_test` | One fold and one segment |
|---|---:|---:|---:|
| none | 115.22 | 109.91 | 109.88 |
| `q_H` | 97.22 | 91.91 | 91.88 |

Under the legacy lift, the memory term misses the legacy 96-bit target by 4.09
bits. If recursive extraction also costs a factor of the depth 64 (A4), the fold
term alone falls to 91.22 bits. The correct lift is part of A4 and A6.

## 6. Oracle-query census

Stage 2 adds these prescribed oracle calls to the Stage 1 census:

- per fold: the three record chains of `step`. At the example geometry with
  `W_ts = 32`, they absorb 1,987 + 1,106 + 1,106 packed elements plus tags
  and framing, about 360–370 permutations;
- per fold: one challenge transcript with two extension squeezes, 10
  permutations: 7 absorbed chunks and 3 squeeze permutations (the
  permutation after the last read is not computed, because no value reads
  it). The relation evaluates it in every invocation, and only an opening
  invocation uses the result;
- per fold: two state digests (spec §11.1), for the input and the output
  state.

The query unit and the total belong to A6. When the trunk fixes them, the
generated relation MUST supply the exact census.

## 7. Open obligations

1. **Non-author review** of Lemmas 2, 3, 5, and 6, and of §5.
2. **Stage 2 authorization.** The owner authorized Stage 2 on 2026-10-07.
3. **A6 Stage 2 extension.** The owner approved the memory-round extension on
   2026-10-07 and chose its Assumption 1 form on 2026-10-12. Its numerical
   values stay open with item 6.
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
   - Lemma 4 in typed form. The typed rows contain the pad gate. The relation
     refinement proves the field form of rows O1, O4, and the O8 gate for
     every application (`RowList`, `StepRefinement`);
   - Lemma 5 and its converse, for consecutive write stamps;
   - the lifecycle of the typed model (Ob8 for the model);
   - Lemma 6 as a deterministic theorem over an extracted typed run: an
     accepted run gives an execution, or a collision among the run's own chain
     inputs and the `D_init` chain, or a segment that passes the product test
     with unbalanced multisets. In the concrete context, the collision is a
     Poseidon2 transcript collision: two different absorbed chunk sequences
     with the same digest;
   - Ob3 for the record chains: the framing of spec §9.1–§9.2 separates
     canonical chain inputs;
   - Ob3 for the §9.3 transcript: canonical inputs (`ts < q`) that differ
     absorb different padded chunk sequences (`etaInputBlocks_injective`);
   - rule 6 of spec §4.2 (`ε_test ≤ 2^−109.91` exactly when
     `m_mem ≤ 139,509`);
   - completeness of the typed model (Ob5 for the model), for every hash
     function and every challenge function;
   - Lemma 3 per segment slot in the interactive game that motivates A6
     (`Spec/Nebula/Game.lean`, `Lifecycle/Nebula/MemoryBound.lean`). An
     outcome is the prover's coins and one fresh uniform challenge pair per
     segment slot. Slot `k` is bad when segment `k` of an accepted run has the
     transcript input committed before slot `k`'s challenge and passes the
     product test with unbalanced multisets. Its frequency is at most
     `2·m_mem/q²` plus one retry term (`badAt_frequency`, `badAt_bound`), and
     each retry disagreement is a Poseidon2 transcript collision. Lean proves
     only the arithmetic step of the retry count, `1 + #Err/#Succ ≤ 2` when
     `#Err ≤ #Succ` (`retry_expected_calls`); the retry process is not
     modelled;
   - the application interface (`Lifecycle/Nebula/PortApp.lean`, spec §10):
     an application supplies a model over its state words, its private
     words, and rows over those words and the operation slots, with Ob7
     (`PortApp.sound`) and its half of Ob5 (`PortApp.complete`), both over the
     decoded ports. The application state is its state words, so the claim
     that soundness attests and the statement that completeness gives name the
     same states, and two different word vectors never name one state. Every
     result below holds for every such application. The first test
     application is the two-word machine (`Lifecycle/Nebula/MachineApp.lean`,
     `Machine.portApp`): one machine step between the states that its two
     words give; its Ob7 and its completeness are proved from the decoded
     ports;
   - a memory application as a Stage 1 relation (`Application.Program`,
     `Lifecycle/Nebula/MemoryProgram.lean`) for a geometry. Its circuit holds
     exactly when the output state is the state digest of the output words
     and the decoded witness satisfies the memory rows and the application's
     rows, and every row reads only supported variables;
   - the soundness refinement of that relation: canonical decoding of the
     carry words (the carry part of Ob1; Lean types the statement fields), the
     field-bit packing of the lanes (Ob2), one invocation's rows to the
     model's `invoke` with a reachable output carry (Ob4), the application's
     rows to one application step (Ob7), and the arm, `close`, and terminal
     rules (Ob8 for the relation);
   - Ob6 by construction: the memory rows are the same for every application,
     and an application's rows have no variable for `η`, a product, or a
     carry word (spec §6.4);
   - the run-level join (`accepts_or_collision`): a chain of states whose
     steps satisfy the relation's predicate (`MemoryApp.holds_of_valid`), with
     the spec §13 terminal checks, gives a run that the model accepts, or a
     collision between two state-digest inputs of the run: at a link of the
     chain, or between the words after the last step and the statement's
     final words. `EndToEnd.bad_subset` joins it with Lemma 6 for the
     Poseidon2 chains (`poseidon2_soundness`). Every collision event, here
     and in the end-to-end statement, names inputs of the one run that the
     extractor returns: an event over all histories of a statement holds for
     a compressing hash and says nothing. The relation reads `plan_digest`
     and the chain headers as constants, and the terminal computes the
     initial state from the start carry and `D_init`;
   - completeness of the relation (Ob5 for the relation,
     `Lifecycle/Nebula/HonestStep.lean`, `MemoryApp.complete`): for a
     reachable carry, an invocation that `invoke` accepts, and one
     application step on its ports, the honest witness words with the
     application's private words satisfy the relation's validity predicate,
     and the step function gives the state digest of the output state and the
     output carry. The witness writes the segment bits, the timestamp bits,
     and the O4 words `diff_j`, and `MemoryApp.decode` reads the words back as
     the same witness (`WitnessEncoding`);
   - run-level completeness of the relation (Ob5 for runs,
     `Lifecycle/Nebula/HonestRun.lean`): the model's `invoke` keeps a carry
     reachable (`MemoryApp.reach_invoke`), and every execution of `S · N`
     steps with `1 ≤ S ≤ S_max` has a Stage 1 step history of the memory
     relation whose statement passes the §13 checks
     (`MemoryApp.history_of_execution`). This includes `S = S_max`, idle
     steps, and every reachable counter value;
   - the end-to-end statement (`Lifecycle/Nebula/EndToEnd.lean`,
     `Export/Stage1/NebulaMemorySecurity.lean`). An `Experiment` records, per
     sample, the Stage 1 statement, the claimed memory statement, and the
     history that the Stage 1 extractor returns; every event reads that one
     history. `stage1Extraction` proves Stage 1 extraction for the HyperNova
     reverse extractor with no premise: on an accepted sample, the advice that
     `HyperNovaHistory.run` returns is a step history of valid steps, or the
     failure event of `history_failure_le` occurs. `MemoryRound` is A6 for
     the stages that A4 admits, with one term for all segment slots. With it,
     `bad_le` bounds the mass of samples that the memory verifier accepts
     while the extracted history does not attest the claim, as in §5. The
     claim reads only the adversary's tape (`adversaryTape`). The Stage 1
     statement comes from the same tape (`input_adversaryTape`), and that
     tape has the adversary's law (`tape_map_adversaryTape`).
     `bad_le_clamp` gives the bound for an adversary whose outputs can be
     longer than `S_max · N` steps, on its own outputs, if its clamped form
     (`clamp`) is admitted (A4);
   - Ob1 at the Rust boundary (`crates/nightstream/src/nebula/mod.rs`,
     `Context::terminal_states`): the verifier rejects an initial or final
     application word, or a `final_timestamp`, at or above `q` before it
     hashes the words (`TerminalError::NonCanonical`). The other statement
     fields and carry counters must equal values below `q`, and the proof
     decoder rejects other lengths and words at or above `q`. The test
     `terminal_rejects_statement_integers_at_or_above_q` failed on the earlier
     code for all five integers, and `memory_conformance` checks the same
     rejection through `nebula::Verifier::verify` with a real proof;
   - the first memory package (`Export/Stage1/NebulaMemoryV1Package.lean`)
     for the test geometry `FirstPlan.geometry` (ROM and RAM of four words,
     `N = 2`, `S_max = 4`) and the machine. Its row, column, and carrier
     bounds give the Stage 1 fixed point and the `2^28` domain
     (`plan_fixedPoint`, `jointDomain_le_twoPow28`). A second test package
     (`NebulaMemoryN1Package`, geometry `SecondPlan.geometry`: `N = 1`,
     `r < μ`) covers the spec §14 cases that the first geometry cannot reach.
     Both use the shared package bounds of `NebulaMemoryPackage`, and both get
     their relation from `NebulaMemoryPackage.application`, which takes the
     proofs of spec §4.2 rules 1–6 (`Geometry.Valid`, `Geometry.Secure`).
     `lake exe emit` writes only these two memory packages. The relation
     `MemoryApp.program` is defined for every geometry, so the setup rule
     binds the packages, not every Lean term. The Rust crate loads each
     emitted package with a pinned structural identifier. The Rust verifier
     takes the program (`nebula::Program`, `u32` words), whose constructor
     checks the image sizes against the geometry and computes `D_init` once.
     The Rust crate also builds a memory relation in Rust for any application
     of the `nebula::Application` contract (`nebula::Circuit::compile`). It
     rejects a geometry that breaks spec §4.2 rules 1–6, and its verifier
     compiles the relation itself, so it never takes a circuit from a prover.
     For the machine at the two package geometries, `tests/nebula/parity.rs`
     requires its rows to equal the Lean package rows;
   - one setup per geometry (`Spec/Nebula/Plan.lean`): a `Geometry` has the
     seven fields of `plan_digest`, and a `Plan` extends it with the images.
     `planDigest`, `PortApp`, `MemoryApp.program`, the packages, and every
     row-level definition take a `Geometry`, and the carry transitions take a
     `Setup` (geometry, hash, challenges, `plan_digest`). Only the initial
     memory, `D_init`, the start carry, and the statement of soundness read a
     `Plan`. So two plans with one geometry have one relation
     (`MemoryApp.program_eq_of_geometry`), and `NebulaMemorySecurity.bad_le`
     covers every program of a package. The Rust tests run a second program
     of the V1 geometry: its honest invocations through the package circuit
     (`first_package_executes_a_second_program`), and, ignored, a proof of
     each program that verifies with its own program and fails the Stage 1
     state link with the other (`each_program_verifies_only_its_own_proof_on_cpu`).
     `d_init_matches_lean` checks `D_init` against Lean for a program with a
     distinct nonzero word in every cell (`FirstPlan.otherPlan`);
   - the memory events of catalog applications (spec §10.1;
     `Spec/Nebula/MemoryTrace.lean`, `Spec/Nebula/EventBinding.lean`,
     `Lifecycle/Nebula/FreeApp.lean`). An execution whose active ports bind
     the events of a catalog trace gives `MemoryConsistent` for the Lean copy
     of `MemoryTrace.v` (`memoryConsistent_of_executes`). This discharges the
     memory-consistency assumption of the Rocq proof of the WASM relation
     (`formal/wasm-rocq` on `main`) for the bound memories, when the regions
     are disjoint, the events are in range, and the initial state holds the
     program images. The free application (`freeApp`) is a `PortApp` with no
     rows, so §5 applies to it when its memory relation fits `2^28`: outside
     the bad events of §5, an accepted proof of it attests such an execution.

   Still open:
   - the Rust-built relation for another geometry or another application:
     no Lean package pins its memory rows, and the application's own rows
     need their own Ob7 and Ob5 (`PortApp.sound`, `PortApp.complete`). Until
     a package for that geometry exists and the Rust rows match it, its
     memory rows are checked by tests only;
   - the binding rows of the Rust memory backend (`nebula::CatalogBinding`,
     spec §10.1 rules I1 to I6): that, when they hold, the active slots of
     each step bind the events of the step in order, each event is in range,
     and the regions are disjoint (the hypotheses of
     `memoryConsistent_of_executes`). The rows are Rust only, and no Lean
     proof covers them yet. Rust tests check that the honest rows of a test
     catalog bind a Rust copy of `eventsForRow` in order, and that each of
     twelve broken rows that keep the memory rows true fails the binding rows.
     A manual mutation check removed each of the ten kinds of binding row in
     turn, and each time one of the broken rows passed. The match between the
     Lean copy and `MemoryTrace.v` is checked by reading only;
   - adaptive program choice: §5 bounds an adversary against a program fixed
     before it runs (§1). For a program that the adversary chooses after it
     sees the setup, the claim and the `Experiment` must carry the program,
     and no theorem states that case;
   - values for the adversary time `t` and the query counts `Q` and `Q_η`.
     They stay symbolic until the owner sets them (decision 2026-10-12); §5
     gives each term as a formula in them;
   - concrete values for the end-to-end bound. With the error that PR #155
     motivates, the depth limit of §5 (shared with Stage 1) makes the bound
     useless when the depth `S_max · N` is more than one or two;
   - a package for a production geometry, deferred to the real application
     (decision 2026-10-12). It needs: the application with a `PortApp`
     instance (a model over its state words, private words, rows, `sound`,
     `complete`); the geometry fields `R`, `μ`, `M`, `W_ts`, `B_ops` (the
     application's ports), `B_scan`, `N`, and `S_max`, which must pass spec
     §4.2 (rule 6: `R + M + N·B_ops ≤ 139,509`); the package bounds and the
     fixed point in `2^28`; and the Rust native run and word layout of that
     application. The package needs no image: the verifier takes each
     program at verify time.
6. **Shared with Stage 1:** useful values of the Assumption 1 errors (A4,
   A6), a Stage 1 argument whose error does not grow with the depth (§5), the
   concrete extractor time (A4), the arity adaptation of HyperNova Lemma 4,
   the terminal decider, encoding, and implementation terms, and an approved
   F′ threat model with an end-to-end target. The legacy
   `protocol-contract/security-reduction.md` §8 shows the form of these terms.
7. **Query model, shared with Stage 1.** The Module-SIS level is a quantum
   estimate, but this note counts classical oracle queries. A quantum search
   finds a bad one-check challenge in about the square root of the classical
   work: about `2^55` for a `2^−109.91` check. The fold challenges have the
   same exposure.
