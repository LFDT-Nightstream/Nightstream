# Nebula on SuperNeo — Security Note

Status: companion to [`nebula-superneo-implementation.md`](./nebula-superneo-implementation.md)
(v3 + the v3.1 stacks amendment). This note states and proves the four
lemmas the spec's §5.3 and §9 defer to, records the claims ledger with
dispositions, and lists what remains open and why. Authored to be attacked: every proof names the exact
property it leans on, so a reviewer can go after the leans.

## 0. Interface assumptions

The lemmas are stated against the following interfaces. Each is either an
existing theorem of the codebase's papers or an explicitly tracked gate —
none is silently assumed.

- **A1 (Pipeline knowledge soundness).** The composed folding pipeline
  `Π_DEC ∘ Π_RLC ∘ Π_CCS` is a reduction of knowledge from
  `CCS(b)^K × CE(b)^k` to `CE(b)^k` (SuperNeo Theorem 1), with soundness
  error `ε_pipe` per fold. In particular, from any accepting fold
  transcript plus valid terminal openings, an extractor produces witnesses
  `z_i` (norm-bounded, satisfying `c_i = A·embed(z_i)`) for every input
  claim.
- **A2 (MSIS binding).** The Ajtai maps `A`, `A_ops`, `A_mem`
  (Goldilocks preset, κ = 18, d = 54; independent seeded matrices) are
  binding against openings of ∞-norm < b = 2: producing two distinct
  low-norm openings of any commitment implies an MSIS solution of norm
  ≤ 2, probability `ε_MSIS`. The preset's analysis covers width `2^30`
  ring columns ≫ any lane here.
- **A3 (Poseidon2 collision resistance).** The Poseidon2 instance behind
  `engine::transcript` and `paper/digest.rs` is collision-resistant
  (`ε_CR`) and its transcript is modeled as a random oracle for
  Fiat–Shamir (query bound `q_H`). For the prover-claimed `D_pre` path
  (spec §6.2 L0b) the leaf/link chain must additionally be
  **preimage/second-preimage resistant**: a `D_pre` claimed without a
  known preimage list obligates the prover to exhibit one at close
  (external-review fix). Both properties follow from the RO modeling and
  are budgeted inside the `ε_CR` line. This assumption is already
  load-bearing for the existing chain (`z_i`, `acc_digest`, NIFS
  challenges); nothing here widens its scope.
- **A4 (F′ enforcement).** The §6.3 `NebulaLane` transition is enforced on
  the same path as the rest of F′: natively by `lifecycle::extend`/verify
  today, by the F′ R1CS/decider when it lands (spec §13 step 9, the named
  production gate). Lemma 2 is stated conditionally on this enforcement;
  the condition is the same one the whole chain already carries.
- **A5 (Uniqueness of openings pre-challenge).** Under A2, at the moment
  Π_RLC samples ρ, each absorbed commitment has at most one low-norm
  opening obtainable by any efficient prover across accepting
  continuations (the "unique relaxed opening" property SuperNeo's weak
  interactive reduction argument for Π_RLC already establishes; we reuse
  it, not re-prove it).

Verified code anchors used below (read, not assumed):

- Π_RLC absorbs the full digest of every input claim — including all of
  `claim.c.data` — before sampling ρ
  (`bind_input_claims_for_rho` → `pi_ccs_outputs_digest` →
  `append_*_claim_public_fields`, `paper/reductions/pi_rlc.rs:180`,
  `paper/digest.rs:414`). Spec rule R1 adds `adv` to this digest, as its
  per-lane leaf digests (spec §6.1); the leaf hop is inside `ε_CR`.
  Note the F′ chunk digest (`f_prime_chunk_claim_digest`) is **not** an
  absorb site: it is shape-only by documented design (absorbs neither
  `claim.x` nor `claim.c.data`) and `adv` mirrors `c` there — excluded
  (spec §5.2 R1).
- Π_RLC's commitment update is a public linear map
  (`out.c = mix_commits(rhos, inputs_c)`, RotRho ring action).
- Π_DEC has no verifier coins; the verifier reconstructs
  `parent.c = Σ_j b^j · c_j` from prover-supplied children and rejects on
  mismatch (`paper/reductions/pi_dec.rs`). Spec rule R2 mirrors this
  reconstruction for `adv`.
- Witnesses embed as `D × cols` ring-coefficient matrices; the RotRho
  action and `split_b` act column-wise
  (`CcsInstance::from_low_norm_assignment`, `neo-ajtai`).

## 1. Notation

`z ∈ F^m` is a step witness, packed column-major into the ring matrix
`Z ∈ F^{d×(m/d)}` (each column = one ring element's coefficients). A
**lane** `L` is a set of whole columns of `Z`; `Z[L]` is the sub-matrix.
The spec's lane regions (`ops`, `is`, `fs`) satisfy:

> **L-ALIGN (normative, added to the spec by this note):** every lane
> region's offset and width in `z` are multiples of `d = 54`; lane regions
> are padded to column boundaries with constrained-zero filler bits.

`adv = (c_ops, c_is, c_fs)` with claimed `c_L = A_L · Z[L]`. For an input
claim `i`, define the **defect** `δ_{i,L} := adv_{i,L} − A_L · Z_i[L]`,
where `Z_i` is the (unique, by A5) low-norm opening of `c_i`.

## 2. Lemma 1 — lane binding through the pipeline

**Statement.** Consider an accepting chain of `n_f` folds whose terminal
decider checks, for every final child `j`: `c_j = A·Z_j` (existing) and
`adv_{j,L} = A_L·Z_j[L]` for each lane (spec §5.2 R3). Then, except with
probability

```text
ε_1 ≤ ε_pipe·n_f + ε_MSIS + n_f·(K+k)/|C|
```

(`|C|` = size of the strong ρ-sampling set; interactive statement — the
§5 composition theorem applies the Fiat–Shamir `q_H` lift to the mixing
term), **every** claim absorbed at
any fold satisfies `δ_{i,L} = 0` for all lanes — i.e., each instance's
published lane commitments open to the lane content of the same witness A1
extracts for its main commitment.

**Proof.**

*Step 1 (Π_DEC, deterministic — zero soundness cost).* The verifier
accepts only if `adv*_L = Σ_j b^j · adv_{j,L}` (mirrored reconstruction)
where `adv*` belongs to the RLC output (parent) claim. The terminal checks
give `adv_{j,L} = A_L·Z_j[L]` for the final children. The pipeline's own
extraction sets the parent witness `Z* := Σ_j b^j Z_j` (this is exactly how
Π_DEC's RoK extractor is defined; it satisfies `c* = A·Z*` by linearity
and has norm < B = b^k). Since digit recombination acts column-wise and
lanes are whole columns (L-ALIGN):

```text
adv*_L = Σ_j b^j A_L·Z_j[L] = A_L·(Σ_j b^j Z_j)[L] = A_L·Z*[L],
```

so the parent's defect is zero. Recursing over folds, the parent-level
binding is inherited by every fold's RLC output with no probability loss.
(For non-terminal folds the "terminal check" role is played by the next
fold's extraction, via A1's recursion — same as for `c`.)

*Step 2 (Π_RLC, the mixing argument).* Fix a fold. At the ρ-sampling
point, all input `c_i` and `adv_i` are absorbed (R1 + the verified anchor
above), and by A5 each `c_i` has a unique obtainable low-norm opening
`Z_i`; hence the defects `δ_{i,L}` are well-defined **before** ρ is
sampled (failure of uniqueness is the `ε_MSIS` term). The verifier
computes `adv*_L = Σ_i adv_{i,L}·ρ_i` (mirror of `mix_commits`), and A1's
extractor produces `Z_i` with `Z* = Σ_i Z_i·ρ_i`. Because the RotRho
action is a right ring-action on columns, it commutes with the column
map `A_L` on whole-column lanes (L-ALIGN):

```text
δ*_L = adv*_L − A_L·Z*[L] = Σ_i (adv_{i,L} − A_L·Z_i[L])·ρ_i = Σ_i δ_{i,L}·ρ_i.
```

Step 1 gives `δ*_L = 0`. Suppose some `δ_{i₀,L} ≠ 0`. Conditioned on the
other challenges, the equation pins `δ_{i₀,L}·ρ_{i₀}` to a fixed value;
for a fixed nonzero `δ`, the map `ρ ↦ δ·ρ` is injective on the strong
sampling set (differences of distinct set elements are invertible — the
same property Π_RLC's own extraction uses). Hence the event has
probability ≤ `1/|C|` per input claim per fold; union over `K+k` inputs
and `n_f` folds gives the third term. ∎

**Corollary 1.1 (cross-instance lane equality).** If two claims publish
equal lane commitments under the same matrix (`c_fs(k,j) = c_is(k+1,j)`
under `A_mem`), then except with `ε_1 + ε_MSIS` their lane contents are
equal bit-for-bit. (Both lanes open their commitments by Lemma 1; two
distinct low-norm openings would break A2. Requires the spec's
byte-identical IS/FS lane layout **and the lane-neutral mem-domain chain
formula** — shared leaf/link tags and header for the `is`/`fs` chains
(spec §6.1/§6.3, external-review fix): with lane-typed tags the boundary
digests would differ even for honest continuity, so the equality this
corollary consumes could never be produced.)

**Remark (why no norm subtleties).** The classic lattice-extraction trap —
inverting challenge differences blows up norms — does not arise here: we
never solve for openings from the δ-equations. Openings come from A1;
Lemma 1 only shows the *public* adv values agree with them. δ's are public
ring vectors; no norm claim about them is needed.

## 3. Lemma 2 — segment soundness of the carried lane (CC-IVC)

**Statement.** Assume A3, A4, Lemma 1. For any accepted chain, except with
probability `ε_2 ≤ ε_CR + ε_1`, for every segment `k` there exist lane
contents (those extracted by Lemma 1) such that:

1. the segment's `γ1, γ2` equal the §6.2 transcript output on the plan
   digest, segment counters, and the three per-lane chain digests over that
   segment's lane-commitment **leaf digests** (spec §6.1; all fixed before
   γ; two collision-resistance hops between γ and the raw commitment
   list — leaf and chain — both inside `ε_CR`'s union bound);
2. the instances folded during segment `k` carry exactly that ordered list
   (`D_seen = D_pre` at close);
3. the running products `h_*` thread unbroken through the segment's `x`
   slots, start at `1_K`, and the close check `h_is·h_ws = h_rs·h_fs` was
   evaluated on them;
4. `ts` is a single non-resetting counter across the chain;
5. segment k+1's IS lane commitments equal segment k's FS lane
   commitments position-wise (`D_is = D_mem` at close), and segment 0's
   equal the plan's `D_init`.

**Proof sketch.** Induction over chain steps, in the style of Nebula
Theorem 5. The F′ state hash (`state_x_out`) absorbs the `NebulaLane` every
step (A4), so an accepting chain fixes the entire sequence of lane values
up to Poseidon2 collisions (`ε_CR`). Per step, the §6.3 transition
enforces: `x`-slot continuity against the carried lane (items 3–4), the
three per-lane `D_seen` chain updates, and at close the three
equalities. For item 2: `D_seen = D_pre` with equal, plan-fixed counts and
the position-ordered chain construction implies the absorbed and folded
leaf sequences — hence, by leaf collision resistance, the commitment
sequences — are equal element-wise, else a Poseidon2 collision at a link
or a leaf; a `D_pre` claimed with no preimage list obligates a preimage
at close (A3's preimage-resistance clause).
For item 1: γ is the transcript output on `D_pre`'s preimage list; the
list is fixed at segment open, and by Lemma 1 each commitment binds its
lane content, so the multiset contributions of the segment are determined
before γ — the commit-then-challenge premise of Lemma 3. For item 5:
Corollary 1.1 applied per position; for segment 0, `D_init` is a public
function of the plan (spec §7). ∎

**Honest condition.** Until spec §13 step 9 lands, A4's enforcement for
items 1–5 is the native lifecycle path plus audit replay — identical in
kind to the enforcement status of NIFS transcript checks today. Lemma 2
does not claim otherwise.

## 4. Lemma 3 — packed fingerprint soundness over K

**Statement.** Tuples are `(t, g, v)` with `t < 2^44`,
`g < R + M + S·2^σ` (the global cell index
`g = addr + ram·R + Σ_s stk_s·(R + M + s·2^σ)`, v3.1 form — injective
onto `(namespace, addr)` because E6 bounds ROM addresses below `R`, RAM
bitness bounds `addr < M` under `r ≤ μ`, and E12+E13 bound stack
addresses below `2^σ`), `v < 2^32`, all range-enforced by lane bitness.
Let `packed(t,g) = t + 2^44·g`, which stays below `q` by the spec §2
plan-validity bound `44 + log2(R + M + S·2^σ) ≤ 62`, and
`f_{γ1,γ2}(e) = γ2 − (packed(e) + γ1·v(e)) ∈ K`. Let `S₁, S₂` be multisets
of tuples with `|S₁| + |S₂| = 2n`, and `γ1, γ2 ← K` sampled independently
of `S₁, S₂`. If `S₁ ≠ S₂` as multisets, then

```text
Pr[ ∏_{e∈S₁} f(e) = ∏_{e∈S₂} f(e) ] ≤ 2n / |K|,     |K| = q² ≈ 2^128.
```

**Proof.** Ranges make `e ↦ (packed(e), v(e))` injective, so distinct
tuples map to distinct degree-≤1 polynomials `p_e(γ1) = packed(e) + γ1·v(e)`
in `K[γ1]`. Consider `G(γ1, γ2) = ∏_{S₁}(γ2 − p_e) − ∏_{S₂}(γ2 − p_e)` in
`K[γ1][γ2]`. `K[γ1]` is an integral domain with unique factorization in
`K[γ1][γ2]`; the monic linear factors `(γ2 − p_e)` are prime, so the two
products are equal as polynomials iff the multisets `{p_e}` — hence
`{e}` — are equal. Thus `S₁ ≠ S₂ ⇒ G ≢ 0`, and `G` has total degree ≤ 2n
(≤ n in γ2, ≤ n in γ1). Schwartz–Zippel over `K²` gives the bound. ∎

**Corollary 4.1 (per-segment memory soundness).** With
`m_seg := |IS| + |WS| + |RS| + |FS| = 2·(N·B_ops + R + M)` (pads
contribute the identity by the gated product rows), a segment whose
products balance but whose extracted trace is not sequentially consistent
occurs with probability ≤ `m_seg/|K|` per Fiat–Shamir attempt — Lemma 3
with `2n = m_seg`; the earlier phrasing set its own `n` to the *total*
multiset size and then applied `2n/|K|`, a 2× double-count
(external-review fix). (Nebula Lemma 7 supplies the combinatorial
reduction; Lemma 2 supplies commit-before-challenge; grinding multiplies
by `q_H`.)

**Lemma 4 (stack discipline, v3.1).** Assume the spec §4.1 rows hold on
every step of a segment (in particular E10–E14), the segment's product
equation balances, and γ was sampled per Lemma 2. Then, except with
Corollary 4.1's probability, for every stack `s` the segment's stack-`s`
ops form a sequentially consistent LIFO history: each pop returns the
value of the most recent unmatched push, annotated with that push's true
write time, and the segment ends with every push matched.

*Proof sketch (reduction to Blum et al. / Coral App. E).*

1. *Restriction is well-defined.* Stack tuples occupy `g`-ranges disjoint
   from ROM/RAM and from every scan position (spec §3.1; E12+E13 bound
   stack addresses below `2^σ`, E10 makes the namespace unambiguous), and
   IS/FS contain no stack-`g` tuples (the scan sweeps `[0, R + M)` only).
   Lemma 3 turns product balance into multiset equality
   `IS ∪ WS = RS ∪ FS` w.h.p.; restricting to stack-`s` `g`-values gives
   `WS_s = RS_s`.
2. *Pushes are distinct and pops match them bijectively.* Push write
   times `wt = ts_in + cnt` are strictly increasing and never reset, so
   `WS_s` has pairwise-distinct timestamps; `WS_s = RS_s` forces a
   bijection pop ↦ push agreeing on `(t, g, v)`. E4 on pops
   (`rt < wt_pop`) orients it: every pop's matched push precedes it.
3. *The bijection is LIFO.* E13 forces the address touched by stack-`s`
   ops to track the E12 counter — pushes write at `sp`, pops read at
   `sp − 1` — so at any fixed address `a`, pushes and pops alternate in
   time (the counter must descend through `a` between two pushes at `a`).
   Alternation plus the timestamp-oriented bijection of step 2 forces
   adjacent matching — each pop at `a` matches the latest prior push at
   `a` — which is exactly Blum et al.'s stack lemma (Coral App. E
   "Stacks": with stack IS/FS empty, `IS ∪ WS = RS ∪ FS` restricted to
   stack segments *is* `WS = RS`, and sequential consistency follows from
   their Lemma 1 / Theorem 1).
4. *Completeness and the close check.* An honest segment pops every push
   (`sp = 0` at close, spec §6.3), so `WS_s = RS_s` holds exactly and the
   stack factors cancel identically — the product equation stays exact
   for honest traces. The deterministic `sp = 0` close check is a
   companion, not the authority: an unpopped push already unbalances the
   product w.h.p. ∎

No new error term: stack ops occupy the same `B_ops` slots and contribute
one tuple each instead of two, so `m_seg` (Cor. 4.1) is unchanged. The
segment-locality rule (spec §3.1) is what makes step 1's restriction
argument compatible with per-segment γ — a cross-segment push/pop pair
would place its two tuples under different challenges, where Lemma 3 says
nothing.

## 4b. Lemma 5 — projection-checked ring action (enc(F′) candidate E)

**Status: PROPOSED, not yet part of the protocol.** This is the
soundness case for `encoding.md` candidate E — replacing the folded
F′ regime's `D²`-product materialization of `RotRho(ρ)·c` with a
polynomial-identity test. The gadget and its measured costs exist
(`engine/r1cs_circuit/ring_action.rs::enforce_ring_action_projection_batch`,
`tests/system/ring_action_projection.rs`); adoption is gated on this
lemma surviving non-author review AND the enc(F′) regime decision
choosing the folded road. The milestone gate is
`ivc_invariants.rs::folded_f_prime_shell_must_adopt_projection_budget`.

**Setting.** One fold inside the F′ circuit. Input commitments
`c_1 .. c_n ∈ R_q^κ` (κ = 18 ring components each; `n = K + k` fold
arity), fold challenges `ρ_i` from the strong sampling set, claimed
output `c* ∈ R_q^κ`, and per-component quotients
`q_m ∈ F^{≤ d−2}[X]` (d − 1 coefficient wires each, degree bound
structural). `R_q = F[X]/Φ`, `Φ = X^54 + X^27 + 1` monic.

**Transcript schedule (normative for any adoption).** In order:

1. absorb all input claims — including every `c_i.data` and, for
   Nebula chains, the `adv` leaf digests (the existing R1 sites);
2. squeeze `ρ` (existing);
3. absorb `c*` and every `q_m` (**new absorbs — both are prover
   claims and both must precede β**; a `q` absorbed after β lets the
   prover solve `q_m(β)` pointwise and forge any `c*`);
4. squeeze **one** `β ∈ K` for the step;
5. enforce, per component `m ∈ [κ]`, the identity at β:
   `Σ_i ρ_i(β)·c_{i,m}(β) = q_m(β)·Φ(β) + c*_m(β)` over `K`.

In-circuit, β's wires are pinned to the transcript squeeze by the same
Poseidon2-trace replay that pins ρ and the sum-check challenges today
(`paper/nifs/circuit.rs` messages discipline) — the gadget enforces
only the algebra; the schedule is the soundness.

**Two counts, deliberately distinct.** Let `P` be the total number of
ring-action **product pairs** the F′ shell covers — `P` drives
committed width (the Phase 1.3d coverage gate reports
`P_total = 465` per step). Let `J` be the total number of **projection
identities** emitted after batching — `J` drives the Schwartz–Zippel
error. The batching rule: a projection identity may batch many product
pairs **only when the protocol consumer uses the batched aggregate as
the authoritative value**. A client that consumes only an aggregate
mix — `c* = Σ_i ρ_i·c_i` — contributes one identity per output ring
component (κ), not one per input pair. A client whose individual
product outputs are consumed separately contributes one identity per
consumed output, unless an additional reviewed aggregation argument is
introduced. **Until the adoption census proves a smaller `J`, this
lemma uses the conservative bound `J ≤ P_total = 465`.**

**Statement.** Under the schedule above, with all operand coefficient
wires bound as in the Integration obligations below, and all `J`
identities sharing the one β: if every identity holds at β, then
except with probability `≤ J·(2d − 2)/|K|` per fold (per Fiat–Shamir
attempt; ×`q_H` under grinding), every checked output equals its true
linear mix in `R_q`.

**Proof.** Fix one identity (index it by its client and component `m`;
the argument is identical for every client, so we write the
commitment-mix case). Define
`G_m(X) := Σ_i ρ_i(X)·c_{i,m}(X) − q_m(X)·Φ(X) − c*_m(X) ∈ F[X]`,
`deg G_m ≤ 2d − 2` (products of degree-≤ d−1 operands; `deg q_m·Φ ≤
(d−2) + d`). Every coefficient of `G_m` is fixed before β is sampled
(items 1–3 of the schedule; commitment binding is A2, transcript
determinism A3). Suppose `c*_m ≠ Σ_i ρ_i·c_{i,m} mod Φ`. Then
`G_m ≠ 0` as a polynomial: if `G_m = 0` identically, reducing
`Σ ρ_i c_{i,m} = q_m·Φ + c*_m` mod Φ gives
`Σ ρ_i c_{i,m} mod Φ = c*_m` (as `deg c*_m < d`), a contradiction. A
nonzero polynomial of degree ≤ 2d − 2 vanishes at a uniform `β ∈ K`
with probability ≤ `(2d − 2)/|K|`. Union over all `J` identities of
the fold (one shared β is sound because every identity's operands
precede the one squeeze). Completeness: the honest `q_m` is the unique
quotient from division by the monic polynomial Φ
(`projection_quotient`), for which each `G_m` is identically zero. ∎

Conservative worked bound:
`J_max·(2d − 2)/|K| = 465 · 106 / 2^128 ≈ 2^−112.4` per fold — below
the Lemma-3 term (≈ 2^−110 per segment), so the composition budget's
shape is unchanged even without any batching credit. If the adoption
census (audit item 4) proves the only projection clients are the
pipeline commitment mix and the Nebula `adv` mirror, then `J = 4κ =
72` and the tighter bound is ≈ `2^−115.1` per fold — **the tighter
number is not assumed until the census is reviewed.** β **must** live
in `K`: over `F` even the tighter term is ≈ 2^−57 per fold,
unacceptable across `n_f · q_H`.

**Composition with Lemma 1.** Lemma 1 Step 2 consumes the verifier's
computation `adv*_L = Σ_i adv_{i,L}·ρ_i` (and the pipeline's
`c* = mix_commits(ρ, c)`), executed natively today. Under candidate E
those computations become projection-checked claims: first apply this
lemma (the claimed mix equals the true linear mix except with the
bound above), then Lemma 1's argument proceeds on the true mix
verbatim. The error adds; no circularity — β is sampled after ρ, and
Lemma 1's mixing event concerns ρ only.

**Integration obligations (each one is an audit item at adoption,
same discipline as R1's absorb-site inventory):**

1. The `c_i` coefficient wires evaluated at β must be the *same wires*
   the F′ image binds to the absorbed claims — a projection over
   free-standing copies proves nothing about the folded instances.
2. The ρ wires must be the transcript-pinned challenge wires (existing
   NIFS.V-circuit obligation, reused).
3. `c*`'s wires must be the ones carried forward as the folded
   accumulator's commitment (what `D_seen`-style consumers and the
   next fold read).
4. Site exhaustiveness — **the adoption census**: every Phase 1.3d
   ring-action product pair is assigned to exactly one
   projection-client row. Each row records the client, its product-pair
   count `P_j`, its projection-identity count `J_j`, the output
   consumer, and why batching is sound for that consumer. The final
   bound uses `J = Σ_j J_j`; until the census is reviewed, Lemma 5's
   conservative `J ≤ 465` stands. Known rows:

   | client | pairs `P_j` | identities `J_j` | consumer | batching justification |
   |---|---:|---:|---|---|
   | Π_RLC commitment mix | `κ·n` | `κ` | folded `c*` | only the aggregate mix is consumed |
   | Nebula `adv` tuple mirror (spec §5.2 R2) | `3κ·n` | `3κ` | folded `adv*` tuple | only the aggregate lane tuple is consumed |
   | `y`-side RotRho sites | TBD | TBD | TBD | census required |
   | **total** | 465 | **≤ 465 until reviewed** | F′ shell | conservative bound |

   Π_DEC's recomposition is scalar `b`-powers — linear, **not** a
   projection client.
5. Limb canonicality: evaluation rows reduce mod q on both sides, so
   aliased bit-encodings agree everywhere compared (same argument as
   spec §4.4's note).

**Completeness caveat (author self-review, attack 9).** If β lands on
a root of Φ in K (possible only if K contains 81st roots of unity),
the `q(β)·Φ(β)` term vanishes and the identity degenerates to
`Σ ρ_i c_i(β) = c*(β)` — which an **honest** prover generally fails,
since `Σ ρ_i c_i` has degree up to 2d − 2 while `c*` is its reduction.
Probability ≤ `d/|K| ≈ 2^−122` per squeeze: negligible, but it is a
nonzero completeness error and any adoption should either accept it in
the completeness budget or re-squeeze β when `Φ(β) = 0` (one native
check; the circuit never sees the rejected β). Soundness is unaffected
(at such β the check is still a valid SZ test of `Σρc − c*`).

**Author adversarial self-review (2026-07-08, at Nico's direction —
does NOT discharge the non-author review, which remains open).**
Attacks attempted and their outcomes: (1) schedule ordering — absorb
set before each squeeze verified sufficient; grinding on ρ and β both
`q_H`-lifted. (2) adversarial quotient — the proof quantifies over
*all* committed `q` (for a wrong `c*`, no `q` makes `G ≡ 0`), so a
lying `q` buys nothing. (3) F-coefficients-evaluated-in-K — `G ≠ 0`
in `F[X] ⊂ K[X]`, SZ over K applies. (4) degree bounds — `deg q ≤
d − 2` is structural in the d − 1 wire slots; no range rows needed.
(5) shared-β union bound — no independence needed across the J
identities; all operands precede the one squeeze. (6) negative ρ
coefficients — enter as their canonical F representatives on both the
native and circuit sides; no norm claim is made anywhere (Lemma 1's
remark applies). (7) in-identity batching over inputs — sound because
the consumer is the aggregate (the batching rule). (8) Φ(β) = 0 —
found; recorded above as a completeness caveat. Verdict: no soundness
break found; one completeness caveat added.

**Optimization recorded, not adopted:** the κ identities could be
RLC-combined under a second challenge τ into one identity
(`Σ_m τ^m·G_m(β) = 0`), saving κ − 1 final rows for an extra
`(κ − 1)/|K|` error term. v1 of any adoption should keep per-component
identities — simpler review, negligible row savings.

## 5. Composition theorem and evaluated budget

**Theorem.** Under A1–A5, an accepted chain of `n_seg` segments with valid
terminal decider checks attests a sequentially consistent memory history
starting from the plan's initial memory, except with probability

```text
ε_total ≤ ε_pipe·n_f + ε_MSIS + q_H · n_f·n_in/|C|     (Lemma 1, FS-lifted)
        + ε_CR                                          (Lemma 2)
        + q_H · n_seg · m_seg / |K|                     (Lemma 3 / Cor 4.1)

ε_MSIS := ε_MSIS(A) + ε_MSIS(A_ops) + ε_MSIS(A_mem)     (A2's instances,
                                                         union-bounded)
n_in   := per-fold input-claim count — SuperNeo's fold arity "K + k";
          instance counts, unrelated to the field K in |K|
m_seg  := |IS| + |WS| + |RS| + |FS| = 2·(N·B_ops + R + M)   (Cor. 4.1)
```

FS lift: Lemma 1 is stated interactively; under Fiat–Shamir the mixing
event is per-transcript-attempt, exactly like Lemma 3's, so it carries the
same `q_H` factor rather than assuming `ε_pipe`'s accounting subsumes it
(conservative — `ε_MSIS(A)` may likewise already be counted inside
`ε_pipe`; it is counted separately here, also conservative).

Worked v3 targets (`N·B_ops = R+M ≈ 2^17.1` under exact cover,
`m_seg ≈ 2^18.1`, `|K| ≈ 2^128`): the Lemma-3 term is ≈ `2^-109.9` per
attempt — the same
soundness regime as the host pipeline's own sum-checks. The Lemma-1 mixing
term is `q_H·n_f·n_in/|C|`, per attempt the same order as Π_RLC's
existing per-fold error.

## 6. Claims ledger (dispositions)

| # | claim | class | disposition |
|---|---|---|---|
| C1 | Multiset equation ⟺ sequential consistency | inherited | Nebula Lemma 7/Cor. 8; Coral App. D |
| C2 | Packed fingerprint sound given range checks | proven here | Lemma 3 (full proof) |
| C3 | Commit-then-challenge suffices | proven here | Lemma 2 item 1 + Cor. 4.1 |
| C4 | Lane commitments bind through Π_RLC/Π_DEC | proven here | Lemma 1; leans on A1/A5 (SuperNeo Thm 1, weak-reduction uniqueness) |
| C5 | Cross-segment memory continuity | proven here | Cor. 1.1 + Lemma 2 item 5 |
| C6 | Global ts / product threading | proven here | Lemma 2 items 3–4 (conditional on A4) |
| C7 | γ trust = ρ trust | verified + proven | code anchors §0 + Lemma 2 item 1 |
| C8 | L-ALIGN: lanes must be whole ring columns | **new constraint found by proof** | added to spec §5.1; without it, Step 2's commutation fails |
| C9 | Lane-residency completeness (no free interpretation bits) | **statically enforced** | `audit_lane_residency` runs at every `S_mem` construction (spec §15 criterion 7): fingerprint-input matrices may only read lanes, public `x`, or E2-constrained `cnt` aux — violations fail the build; reference-model attacks remain as regression (spec §12) |
| C10 | Engine accepts `S_mem` shape | build-resolved | PR 3 |
| C11 | Cost model within 2× | **measurement-only** | PR 3 spike; cannot be closed by writing |
| C12 | F′-R1CS absorb budget | roadmap-dependent | pinned when F′-R1CS design lands (production gate) |
| C13 | Workload fit of `M`, segment granularity of incrementality | **product decision** | owner: Nico |
| C14 | SIS/Ajtai accumulation for the carried chains | **dispositioned (v4 review)** | sound (chained, binding-only, hash-then-FS — never a challenge source) but deferred to the `enc(F′)` milestone: only live in the folded regime, and the NC range check (`b = 2`) forbids digit bases `w > 1` — spec §6.5, §14. A hash-then-FS lemma is owed here if ever adopted. |
| C15 | IS/FS boundary chains must be formula-identical | **completeness bug, fixed (external review)** | lane-typed `"is"`/`"fs"` tags made honest cross-segment continuity impossible; is/fs now share one mem-domain leaf/link tag pair and header (spec §6.1/§6.3/§7); Cor. 1.1 states the dependency |
| C16 | Externally accepted proofs end at closed segments | **normative rule added (external review)** | spec §6.3 finalization rule (`idx == 0`, `γ == ⊥`, header chains); mid-segment `State` is prover-only resume material (spec §6.4) |
| C17 | Stack ops are LIFO-consistent under the v3.1 rows | proven here (v3.1) | Lemma 4 (reduction to Blum et al. / Coral App. E); segment locality forced by per-segment γ (spec §3.1) |
| C18 | Projection-checked ring action is sound for the folded F′ regime | **proposed here (candidate E)** — gated on non-author review AND the enc(F′) regime decision | Lemma 5; gadget + measured costs exist; adoption audit items enumerated in the lemma (wire identity, transcript schedule, site exhaustiveness) |

## 7. What this note does not close

Three things, by nature not by effort: **C11** (a number only the engine
can produce), **C12** (an assumption about an unfinished design — it
becomes checkable when that design is pinned), and **independent review**
of Lemmas 1–4 (and now the proposed Lemma 5) — the author checking the
author is circular, and the leans named in §0 (A1's exact extractor
shape, A5's uniqueness property) are precisely where a reviewer should
push. Suggested review order: Lemma 1 Step 2 first (the only place a
challenge-set property is invoked), then Lemma 2's induction against
`state.rs`'s actual absorb set, then Lemma 4 step 3's alternation
argument against the actual E12/E13 rows. Lemma 5 can be reviewed
independently of the others (it is candidate material, not yet
load-bearing): push hardest on the transcript schedule (why `q` must
precede β) and on integration obligation 1 (wire identity — the classic
way projection arguments silently break).
