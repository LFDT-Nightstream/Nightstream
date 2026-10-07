# Nebula memory phase on SuperNeo F′ — security note

Status: **Proposed.** Companion to [`nebula-superneo.md`](./nebula-superneo.md)
("the spec"). The author wrote the lemmas below. A production claim needs a
review by someone other than the author (§7).

This note states the security goal, the assumptions, seven lemmas with proofs,
and the composed bound. It evaluates the bound at the current values.

## 1. Threat model and security goal

The prover controls all proof bytes, witnesses, memory records, lane
commitments, and proposed roots. The verifier controls the verifier key, the
application package, the plan, the setup seed, and the public statement.

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
- each lane map binds at a level of at least 110.685 bits;
- each new digest uses the Stage 1 Poseidon2 model.

So the memory term is the weakest statistical term: it is 4.85 bits below the
fold term. The Fiat–Shamir step multiplies both statistical terms by a query
count (§5).

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
Stage 1 profile with the linear map `L = L*` of spec §7.1. It is an
interactive reduction of knowledge. Its statistical error for one fold is
`f_fold/|𝕂| + f_fork/|C|`, with `|C| = 5^54`. `f_fold` is the numerator of
v1.2's `ε_test` (sum-check and Schwartz–Zippel), and `f_fork` is the PiRLC
extraction term. The trunk computes both with
`NeoParams::padded_row_security_summary_for_shape` for the final relation
shape. v1.2's `ε_uniq` reduces to relaxed binding of `L*`, and Lemma 1 reduces
that to A2.

**A2 (Module-SIS binding).** `L_full` is `(2B, C)`-relaxed binding at the main
level, as SuperNeo v1.2 §7 requires. `L_ops` and `L_mem` are binding for bit
vectors: an adversary with expected time `t` finds two different `{0, 1}`
vectors with the same image with probability at most `ε_lane(t)`. The
lane-map decision record (spec §7.2) gives `ε_lane` from a Module-SIS estimate
at infinity norm 1, with its own multi-target allowance. All keys come from
the SHAKE128 expander, modeled as a random oracle.

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

Here the bound is `S_max · N`. Stage 2 adds the carry to the state preimage and
makes each claim larger. So Stage 2 must prove Property 6 and the fixed-point
and domain theorems again (`FPRIME_LEAN_ARCHITECTURE_SPEC.md` §6).
Definition 12 is stated for arity `(1, 1)`, and F′ folds one fresh claim with
16 running children. So Stage 1 must also adapt the lemma to that arity. The
lemma gives no concrete extractor time. That bound is also open for Stage 1.

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

A6 extends that premise to the outputs of `H` that Stage 2 adds:

- the memory challenges `η` (spec §9.3);
- the chain digests (spec §9.2).

In the translated experiment, each of these outputs is the answer to an oracle
query that the adversary made. Lemmas 3 and 4 hold in that experiment, and A6
carries the §5 bound to the real protocol. One transfer covers the whole F′
transcript: the Stage 1 and the Stage 2 outputs together. The extension needs
the owner's approval, as the Stage 1 boundary did. Like that boundary, it
selects no numerical value for `g_d` or `delta_d`.

## 3. Deterministic obligations

These obligations have no probability term. A failure is a defect.

| # | Obligation |
|---|---|
| Ob1 | Canonical decoding of statements, claims, bundles, and the carry block; no alias of a noncanonical field element |
| Ob2 | L-ALIGN for all three lanes; zero alignment coordinates (spec §6.3) |
| Ob3 | Bundle algebra in native code and circuit: PiCCS copy, PiRLC combination, PiDEC reconstruction (spec §7.3) |
| Ob4 | Transcript rules 1–3 of spec §7.4 |
| Ob5 | Same-witness terminal openings for the 16 children and the trailing fresh claim (spec §7.5) |
| Ob6 | Every satisfying assignment of the generated relation refines to the actions of spec §8–§12 |
| Ob7 | Every honest execution has a satisfying assignment, including `S = S_max`, the largest counter values, and idle steps |
| Ob8 | Static lane-residency check (spec §6.4) |
| Ob9 | Port coverage: no application memory effect bypasses a port (spec §10) |
| Ob10 | Lifecycle (spec §12): the arm follows from the authenticated carry; `absorb` reads only the folded claim's bundle; the terminal absorbs `u_{T−1}` and closes |

## 4. Lemmas

The lemmas cover the six joints of spec §1.1. The author wrote every proof in
this note, and each proof needs an independent review (§7).

| Joint | Argument | Status |
|---|---|---|
| J1: lanes into `L*` | Lemma 1; Ob2, Ob8 | Proof in this note, under A1 and A2 |
| J2: bundle through the fold | Lemmas 1 and 2 | Proof in this note. The trunk preimage already binds every running child |
| J3: delayed absorb | Ob10 | Deterministic obligation |
| J4: carry thread | A3, A4 | Assumed. The arity adaptation and Property 6 are open |
| J5: commit, then test | Lemmas 3–6 | Proof in this note, in the translated experiment of A6 |
| J6: segment joins and terminal | Corollary 1.1; Lemma 7 | Proof in this note |

### Lemma 1 — The product map is a SuperNeo commitment

**Statement.** Under Ob2, `L*` is an `R_F`-module homomorphism. If `L_full` is
relaxed binding, then `L*` is relaxed binding. Under A1 and A2, every claim
that a fold absorbs has an extracted bounded witness `z`, and each of its four
bundle components equals the matching component of `L*(z)`.

**Proof.** A ring scalar acts on each 54-coefficient ring element of the
embedded assignment. A projection that selects whole ring elements commutes
with that action and with addition. Under Ob2, each of `P_ops`, `P_is`, and
`P_fs` selects whole ring elements. So each component of `L*` is a module
homomorphism, and so is `L*`. PiDEC recomposition `Σ_i b^i · child_i` uses
base-field scalars. It acts on each coordinate, so it commutes with any
projection.

Let `(z, Δ)` and `(z′, Δ′)` be two relaxed openings of one bundle with
`Δ′·z ≠ Δ·z′`. Their first components are relaxed openings of one `L_full`
commitment with the same inequality. This breaks relaxed binding of `L_full`.
So relaxed binding of `L*` reduces to that of `L_full`.

SuperNeo v1.2 Theorem 3 is parametric in the linear commitment map. With
`L = L*`, its extractor returns, for each absorbed claim, a bounded witness `z`
with `L*(z)` equal to the claim's bundle. The lane components are then the
lane commitments of the same `z`. ∎

**Corollary 1.1 (equal snapshots).** If two IS or FS lanes have equal `L_mem`
commitments, their contents are equal, except with `ε_lane`. Both lanes are bit
vectors (rule S1), so two different contents break A2.

### Lemma 2 — Running-statement binding

**Statement.** (a) Suppose PiCCS binds the running claims through a digest
whose preimage omits the children, for example through the PiRLC parent only.
Then a prover can make the verifier accept a fold with an unsatisfied fresh
claim. (b) Under rule 2 of spec §7.4, the PiCCS input is fixed before `α` and
`γ`, except with `ε_H`. Then A1 applies.

**Proof of (a).** Number the children `0 … 15`, with recomposition weight `b^i`
for child `i`. Fix one matrix `j`. For any vector `δ`, set

```text
y′_{0,j} = y_{0,j} + b·δ,      y′_{1,j} = y_{1,j} − δ.
```

Then `b^0·y′_{0,j} + b^1·y′_{1,j} = b^0·y_{0,j} + b^1·y_{1,j}`. PiDEC
verification still passes, and the parent is unchanged, so the transcript is
unchanged. The PiCCS claimed sum gives child `i` and matrix `j` the weight
`γ^{w(i,j)}` with distinct exponents. The claimed sum therefore moves by

```text
Δ_T = (b·γ^{w(0,j)} − γ^{w(1,j)}) · ⟨χ_α, δ⟩.
```

The polynomial `b·X^{w(0,j)} − X^{w(1,j)}` is nonzero, so its value is nonzero
for all but a negligible set of `γ`. The vector `χ_α` is not zero. The prover
makes the fresh claim unsatisfied, with zero-check error `e`. It sees `α` and
`γ`, and then it solves `Δ_T = e` for `δ`. The claimed sum now equals the true
sum. The prover runs the sum-check honestly, so the outputs at the new point
are true evaluations. The later folds and the terminal check never read the
old child values. They accept. The same kernel freedom applies to the bundle
components of the children after `ρ`.

**Proof of (b).** Under rule 2, the prior-state digest covers every running
child in full: all bundle components and all evaluation values. A change to a
child changes the digest, except with `ε_H`. The digest enters the transcript
before `α` and `γ`. So the full running input is fixed before the challenges,
and the Fiat–Shamir transcript compiles the interactive protocol of SuperNeo
v1.2 §7.3. ∎

The trunk preimage already lists every running child in full
(`crates/nightstream/src/lifecycle/inputs.rs`). Stage 2 must keep this property
when the commitment block becomes the bundle.

### Lemma 3 — Commitments fixed before `η`

**Statement.** Work in the translated experiment of A6. Suppose a segment
closes: `D_seen = (D_pre.ops, D_mem, D_pre.fs)`, where `D_seen` comes from the
absorbed bundles and the three roots entered the `η` transcript (spec §9.3).
Then, except with `ε_H`, the `N` lane commitments of each lane were fixed when
the `η` query was made.

**Proof.** In that experiment, every output of `H` is an oracle answer.
`D_seen.l` is the output of the last chain query for lane `l`. For that output to equal the root in the `η` query,
one of three events occurs:

1. The prover made that chain query before the `η` query. The query input
   contains the index `N − 1`, the previous chain value, and `encode(C)`. By
   induction down to the fixed header, every commitment of the sequence was in
   a query before the `η` query.
2. Two different chain inputs give the same output: a collision.
3. At some level of the chain, the prover used a value before any query
   returned it, and a later query returns that value: a preimage. The root in
   the `η` query is one case.

Events 2 and 3 are inside `ε_H`. The fixed length `N`, the index in each chain
input, and the domain tags prevent reordering, truncation, and extension. The
IS and FS chains share formulas on purpose. The transcript gives each root its
role through its fixed position. For the IS lane, the root `D_mem` is the
output of the previous segment's FS chain, or the verifier's `D_init`. ∎

### Lemma 4 — Commit-then-test

**Statement.** Work in the translated experiment of A6. Let the adversary make
at most `q_η` queries to the `η` transcript of spec §9.3. For an accepted proof, run the extractor of A1 and
A4. For each closed segment, let `A = IS ∪ WS` and `B = RS ∪ FS` be the
multisets that the extracted lane contents define, each of size at most
`m_mem`. Then

```text
Pr[ some segment closes with A ≠ B ]  ≤  q_η · (ε_test + ε_lane(t_1)) + ε_H,
ε_test = 2·m_mem / |𝕂|,
```

where `t_1` is at most twice the time `t_E` of one adversary run with
extraction.

**Proof, part 1: fixed multisets.** Fix `A ≠ B` with the tuple ranges of
Lemma 5. Spec §4.2 rule 3 makes `t`, `g`, and `v` integers below `q`, so each
has one canonical field value. Distinct tuples therefore give distinct
polynomials `p(η1) = g + η1·v + η1²·t` in `𝕂[η1]`. Each factor
`f = p(η1) − η2` has degree 1 in `η2` with a unit leading coefficient, so it
is prime in `𝕂[η1, η2]`, and distinct tuples give non-associate factors.
Unique factorization keeps multiplicity. So the two products are equal as
polynomials only if `A = B`. For `A ≠ B`, their difference `G` is a nonzero
polynomial of total degree at most `2·m_mem`, because each factor has total
degree at most 2. Schwartz–Zippel gives `Pr[G(η1, η2) = 0] ≤ 2·m_mem/|𝕂|`,
which is `ε_test`, for a uniform pair.

**Proof, part 2: contents extracted after `η`.** The extractor returns the
contents after `η`, so they can depend on `η`. The lane commitments bind only
computationally, so part 1 does not apply directly. We fork at the `η` query
and use the argument of SuperNeo v1.2 Appendix B, proof of (ii) for PiCCS.
Here `η` takes the role of `(α, γ)`, and the fingerprint identity takes the
role of the sum-check test.

Fix a query index `i ≤ q_η`. Let `in` be the adversary's state just before
query `i`, with the query input. One oracle call answers query `i` with a
fresh uniform `η`, runs the adversary to the end, and runs the extractor.
`Succ` is the event that the call ends with an accepted proof, a segment
closes with the answer to query `i`, and the extraction is valid. `Err` is
`Succ` together with `A ≠ B` for that segment, so `Err ⊆ Succ`. The
uniqueness adversary makes one call. If `Err` does not occur, it stops.
Otherwise it repeats calls until `Succ` occurs, and it returns both contents.
By the calculation of SuperNeo v1.2 Appendix B, `t_in + (a_in/p_in)·t_in` with
`a_in ≤ p_in`, its expected time `t_1` is at most twice the time of one call.

- If the two contents agree, then the multisets `(A, B)` that the first call
  fixed pass the equation for a fresh, independent `η`. By part 1 and the
  conditional-probability step of SuperNeo v1.2 Appendix B,
  `Pr[Err and agree] ≤ ε_test`.
- If the two contents differ, both calls closed a segment with the answer to
  query `i`, and the input of that query holds the three roots. By Lemma 3
  over the queries of both calls, both calls absorbed the same lane
  commitments, except with `ε_H`. Lemma 1 makes each extracted content open
  its commitment. So two different bit vectors open one lane commitment. This
  breaks A2, so `Pr[Err and differ] ≤ ε_lane(t_1)`.

So `Pr[Err] ≤ ε_test + ε_lane(t_1)` for index `i`, plus the hash events. A
union bound over the `q_η` indices gives the statement. ∎

In one segment, `|IS| = |FS| = R + M` and `|RS| = |WS|` is the active count.
So `m_mem ≤ R + M + N · B_ops`.

### Lemma 5 — Rows to multisets

**Statement.** Suppose an assignment satisfies rows O1–O9, S1–S3, and the
boundary rows of spec §8, and the lane-residency property Ob8 holds. Then the
outgoing products equal the incoming products times the fingerprints of the
extracted tuples:

```text
h_rs_out = h_rs_in · Π_{active j} f(RS_j),   h_ws_out = h_ws_in · Π_{active j} f(WS_j),
h_is_out = h_is_in · Π_j f(IS_j),            h_fs_out = h_fs_in · Π_j f(FS_j).
```

The tuple contents depend only on lane coordinates and the carry fields `ts`
and `idx`, which do not depend on `η`.

**Proof.** O1 makes `pad_j` a bit, so the O8 and O9 factor is `1` for a pad and
`f` for an active slot. Induction over the slots gives the products. S2 and S3
give the scan products in the same way. O1 bounds `rt`, `v`, and `addr` by
their widths. For RAM, `g = R + addr < R + M` because `addr < 2^μ = M`. For
ROM, O6 gives `addr < R`. The carry field `ts` is a `W_ts`-bit word, the
boundary row bounds `out.ts < 2^W_ts`, and `wt_j ≤ out.ts`. So every tuple
satisfies the ranges of Lemma 4. Ob8 gives the last sentence. ∎

### Lemma 6 — Sequential consistency of one segment

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
assumption 3, because spec §4.2 rule 3 keeps it an integer relation. O3 and O5 give assumption 4. Lemmas 3–5 give assumption 5,
except with the error of Lemma 4.

This proof needs no range check on IS or FS timestamps.

### Lemma 7 — Lifecycle and segment composition

**Statement.** Assume A1–A6 and Ob1–Ob10. If the verifier accepts, then,
except with the bound of §5, there is one execution of the application
relation with these properties:

- it starts from the statement's initial state and the plan's initial memory;
- it has `T = S · N` steps and ends in the statement's final state;
- its memory accesses are sequentially consistent across all segments;
- its final memory has the FS root `final_memory_root`.

**Proof.**

1. **Witnesses.** By A1, A4, and Lemma 1, every claim `u_0 … u_{T−1}` has an
   extracted witness that satisfies the F′ relation and opens all four bundle
   components. The terminal check opens the 16 children and `u_{T−1}` directly
   (Ob5). By Lemma 2(b), each fold's running input was fixed before its
   challenges.
2. **Carry thread.** The state link of Construction 2 (A3) makes the input
   carry of `A[i]` equal to the output carry of `A[i−1]`. The arm follows from
   that carry (Ob10). So the steps run in order, a segment closes exactly when
   `idx = N`, and each claim is absorbed exactly once: `u_{i−1}` in `A[i]`, and
   `u_{T−1}` at the terminal.
3. **One segment.** At each close, Lemma 3 fixes the commitment sequences
   before `η`. Lemma 5 turns the rows into the four products. Lemma 4 gives
   `IS ∪ WS = RS ∪ FS`, except with the error of Lemma 4. Lemma 6 gives one
   sequential history per segment.
4. **Segment joins.** For segment 0, `D_seen.is = D_init`. Lemma 3 and
   Corollary 1.1 make the IS lanes equal to the plan images. For segment
   `k + 1`, `D_seen.is = D_mem`, which is the FS root of segment `k`. So the IS
   of segment `k + 1` equals the FS of segment `k`, cell by cell. The global
   timestamp never resets, so the write timestamps of all segments are
   distinct. Joining the segment histories gives one sequential execution.
5. **Application.** The port rows (Ob9) make these memory accesses the
   accesses of the application relation. An idle step changes neither memory
   nor the application state.
6. **End.** The verifier rejects the initial envelope. The terminal requires
   `idx = N` before its close, `1 ≤ S ≤ S_max`, `T = S · N`, and
   `seg_idx = S`. The statement fields equal the final state. ∎

## 5. Composition and evaluation

**Theorem.** Assume A1–A6 and Ob1–Ob10. Let the adversary make at most `q_η`
queries to the `η` transcript. There is an extractor `E`, built from the
extractors of A1 and A4, such that, in the translated experiment of A6,

```text
Pr[accept and E's output is not a valid execution]
   ≤  ε_Stage1                          (A1, A4: every fold and the terminal)
    + q_η · (ε_test + ε_lane(t_1))      (Lemma 4)
    + ε_lane(t_E)                       (Corollary 1.1 at every segment join)
    + ε_H                               (A3: every query of the adversary and of the reductions)
```

with `ε_test = 2·m_mem/|𝕂|` and `m_mem ≤ R + M + N · B_ops`. `ε_Stage1` is
the Stage 1 error for the Stage 2 relation shape, including SuperNeo's
`ε_uniq` for `L_full`. `t_E` is the time of one adversary run with extraction,
and `t_1 ≤ 2·t_E`. One reduction checks every segment join in one run. The
proof is Lemma 7 with a union bound.

A6 carries this bound to the real protocol through `g_d` and `delta_d`.

A4 gives no concrete bound on `t_E`, so the `ε_lane` terms have no number yet.
`ε_lane` and `ε_H` come from work estimates. They become probabilities only
for a stated adversary time and query count.

**Setup check (spec §4.2 rule 6).** For the final relation shape and the plan,
setup MUST check:

1. `ε_test ≤ 2^−109.91`, the owner-approved floor. This holds exactly when
   `R + M + N · B_ops ≤ 139,509`;
2. the lane-map decision record gives `ε_lane` at a level of at least 110.685
   bits.

Setup MUST reject the plan if a check fails. A larger memory needs a new owner
decision. Both checks apply to per-check values. The theorem multiplies
`ε_test` by `q_η`, as Fiat–Shamir multiplies the fold term by its query count.

**Evaluation at the current values.** The values use the trunk formula with
`f_fold = 7,781` and `f_fork = 304`. The example geometry is `R = 2^12`,
`M = 2^16`, `N = 64`, and `B_ops = 1,088`, so `m_mem ≤ 139,264`. A release
MUST recompute this table for the generated relation with
`NeoParams::padded_row_security_summary_for_shape`.

| Term | Bits |
|---|---:|
| One fold, `f_fold/|𝕂| + f_fork/|C|` | 114.76 |
| Main Module-SIS level | 110.685 |
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

- per fold: the three chain digests of `absorb`;
- per segment: one challenge transcript with two extension squeezes.

The query unit and the total belong to A6. When the trunk fixes them, the
generated relation MUST supply the exact census.

## 7. Open obligations

1. **Non-author review** of Lemmas 3, 4, 6, and 7, and of §5.
2. **Stage 2 authorization** by the owner, and a decision record for the lane
   maps: setup IDs, expander input layout, `κ_lane`, and the Module-SIS
   estimate at infinity norm 1 (spec §7.2).
3. **A6 extension.** Owner approval of the A6 extension to `η` and the chain
   digests, in the form of the Stage 1 boundary.
4. **Generated relation** that fits the `2^28` domain, with its exact shape and
   census, and the recomputed §5 table.
5. **Lean formalization** of the Stage 2 phase, with the extended composition,
   Property 6, fixed-point, and domain theorems (A4;
   `FPRIME_LEAN_ARCHITECTURE_SPEC.md` §6).
6. **Shared with Stage 1:** useful values of `g_d` and `delta_d` (A6), the
   concrete extractor time `t_E` (A4), the arity adaptation of HyperNova Lemma 4, the
   terminal decider, encoding, and implementation terms, and an approved F′
   threat model with an end-to-end target. The legacy
   `protocol-contract/security-reduction.md` §8 shows the form of these terms.
7. **Query model, shared with Stage 1.** The Module-SIS level is a quantum
   estimate, but this note counts classical oracle queries. A quantum search
   finds a bad one-check challenge in about the square root of the classical
   work: about `2^55` for a `2^−109.91` check. The fold challenges have the
   same exposure.
