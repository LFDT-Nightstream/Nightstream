# Nebula on SuperNeo — Security Note

Status: companion to [`nebula-superneo-implementation.md`](./nebula-superneo-implementation.md)
(v3 + the v3.1 stacks amendment). This note states and proves the six
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
  error `ε_pipe` per interactive fold. For the production relation,
  SuperNeo D.4 gives `ε_pipe ≤ f_D4/|K|`, where the exact executable
  numerator is `f_D4 = 1,439,664`; Fiat–Shamir contributes the explicit
  `q_H` lift in §5. In particular, from any accepting fold
  transcript plus valid terminal openings, an extractor produces witnesses
  `z_i` (norm-bounded, satisfying `c_i = A·embed(z_i)`) for every input
  claim.
- **A2 (MSIS binding).** The Ajtai maps `A`, `A_ops`, `A_mem`
  (Goldilocks preset, κ = 18, d = 54; independent seeded matrices) are
  `b`-binding for fresh openings and `(B,C)`-relaxed binding for accumulated
  claims as required by SuperNeo Theorem 2. The latter is the conservative
  requirement: hardness of `MSIS_{m,8TB}^{∞,κ,q}` at `n_F=2^30` scalar
  coefficients. The pinned estimator reproduces 100.7 rough / 129.1 full
  bits for that Appendix-B.2 instance; §5 uses a rounded 100-bit floor. The
  same maximum width dominates every lane map here. Let `ε_MSIS` denote the
  union of failures for these three independently seeded maps.
- **A3 (Poseidon2 collision resistance).** The Poseidon2 instance behind
  `engine::transcript` and `paper/digest.rs` is collision-resistant
  (`ε_CR`) and its transcript is modeled as a random oracle for
  Fiat–Shamir (global adversarial random-oracle query bound `q_H`). For the prover-claimed `D_pre` path
  (spec §6.2 L0b) the leaf/link chain must additionally be
  **preimage/second-preimage resistant**: a `D_pre` claimed without a
  known preimage list obligates the prover to exhibit one at close
  (external-review fix). Both properties follow from the RO modeling and
  are budgeted inside the `ε_CR` line. This assumption is already
  load-bearing for the existing chain (`z_i`, `acc_digest`, NIFS
  challenges); nothing here widens its scope.
- **A4 (F′ enforcement).** The §6.3 delayed `NebulaLane` transition is part
  of the same authoritative relation as NIFS.V and `S_mem`: base carries the
  canonical lane unchanged; recursive step `i+1` consumes the exact
  `(fresh_x, fresh_adv)` wires for prior claim `u_i`; terminal finalization
  consumes trailing `u_T` before checking closure and recomputing public
  `x_out`. All three cases are implemented by the fixed relation and its
  terminal-induction lifecycle; Lemma 2 relies on those constraints, never on
  native history replay. The active R5 acceptance/tamper test is implementation
  evidence for this assumption, not a replacement for reviewing the relation.
- **A5 (Uniqueness of openings pre-challenge).** Under A2, at the moment
  Π_RLC samples ρ, each absorbed commitment has at most one low-norm
  opening obtainable by any efficient prover across accepting
  continuations (the "unique relaxed opening" property SuperNeo's weak
  interactive reduction argument for Π_RLC already establishes; we reuse
  it, not re-prove it).
- **A6 (two-level binding compression).** The CCS-claim, CE-claim,
  Π_CCS-output, Π_RLC-projection, and Nebula-leaf roles use independent
  seeded rank-2 Ajtai maps over the exact authoritative centered-unit
  encodings. Their maximum input is 29,168 field words, or 22,147 ring
  columns. A shared, independently seeded rank-1 map compresses the 108-field
  rank-2 output before Poseidon2. Two distinct valid openings differ by
  coefficients of infinity norm at most 2. Let `ε_BIND` be the union of the
  five long-map MSIS events and the one short-map MSIS event. We assume the
  ChaCha8-expanded fixed public matrices are hard as random Module-SIS
  instances of the stated dimensions; this is a concrete-instance
  assumption, not a reduction from A2. With `malb/lattice-estimator` commit
  `3e48ef421ec256afddb3e7d2249a77eab6e9ba12`, the conservative rough estimates
  are 167.0 bits for the maximum rank-2 map and 223.1 bits for the short map;
  unioning all six leaves more than 164 bits. The rejected rank-1 long map
  estimates only 59.9 rough bits. §4c proves the two-level hash-then-FS
  reduction; `scripts/estimate_nebula_sis.sage` reproduces the estimates.

Verified code anchors used below (read, not assumed):

- Π_CCS absorbs authority-bearing input claim digests — including
  `claim.c.data` and the R1 `adv` leaves — before its challenges. Π_CCS.V
  then equality-forwards `c`/`adv` to every output. Before Π_RLC samples
  ρ, `pi_ccs_outputs_digest/v2` absorbs only the newly sent `y_ring` and
  `y_zcol`; forwarded fields are not a second authority source
  (`paper/reductions/pi_ccs_split_nc_circuit/digests.rs`,
  `paper/digest.rs`). The leaf hop is inside `ε_CR`.
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

**Proof sketch.** Induction over HyperNova's delayed claim schedule, in the
style of Nebula Theorem 5. Base fixes the canonical lane and consumes no
claim. Recursive F′ step `i+1` first verifies/folds `u_i`; A4 then advances
the lane from that verifier's exact public suffix and `adv` wires. The
terminal relation performs the same transition on trailing `u_T`, so the
induction has no unchecked last claim. `state_x_out` absorbs every resulting
lane (including the post-terminal lane), fixing the sequence up to Poseidon2
collisions (`ε_CR`). Each transition enforces `x`-slot continuity (items
3–4), three `D_seen` updates, and the close equalities. For item 2:
`D_seen = D_pre` with equal, plan-fixed counts and
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

**Implementation condition.** A4 is implemented for the authoritative fixed
relation: base, bootstrap-recursive, steady-recursive, interior segment steps,
and terminal delayed transition are exercised by the active R4 encoder and R5
terminal-induction tests, including link/suffix/lane and pre-final-running
tampering. The terminal-induction capability is
crate-private and set only by that relation's preprocessing constructor; legacy
and generic frontends remain fail-closed. This closes the implementation gate,
not the non-author proof review tracked below.

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

**Status: ADOPTED and implemented; non-author review remains open.** The
authoritative folded F′ relation replaces `D²` product materialization with
this polynomial-identity check at every c/adv/X/y ring-action site. Native and
in-circuit schedules live in `paper/reductions/pi_rlc.rs` and
`paper/reductions/pi_rlc_circuit.rs`; C12 and the active R7 gate measure the
resulting fixed relation rather than the retired projection shell.

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
3. canonically encode every claimed aggregate output (`c*`, X/y aggregates,
   and the three `adv*` commitments) and every matching `q_m`, recompute the
   domain-separated A6 Π_RLC-projection binding map and its Poseidon2 digest,
   and absorb that digest (**all prover claims are therefore bound before
   β**; omitting a `q`, or accepting a digest not recomputed from these wires,
   lets the prover solve `q_m(β)` pointwise and forge its output);
4. squeeze **one** `β ∈ K` for the step;
5. enforce the identity at β for every output ring component of every
   client; for the commitment client and `m ∈ [κ]` this is
   `Σ_i ρ_i(β)·c_{i,m}(β) = q_m(β)·Φ(β) + c*_m(β)` over `K`.

In-circuit, β's wires are pinned to the transcript squeeze by the same
Poseidon2-trace replay that pins ρ and the sum-check challenges today
(`paper/nifs/circuit.rs` messages discipline) — the gadget enforces
only the algebra; the schedule is the soundness.

**Two counts, deliberately distinct.** Let `P` be the total number of
ring-action **product pairs** the F′ relation covers — `P` drives
committed width. Let `J` be the total number of **projection
identities** emitted after batching — `J` drives the Schwartz–Zippel
error. The batching rule: a projection identity may batch many product
pairs **only when the protocol consumer uses the batched aggregate as
the authoritative value**. A client that consumes only an aggregate
mix — `c* = Σ_i ρ_i·c_i` — contributes one identity per output ring
component (κ), not one per input pair. A client whose individual
product outputs are consumed separately contributes one identity per
consumed output, unless an additional reviewed aggregation argument is
introduced. For fold input count `n`, commitment width `κ`, active X
columns `a_X`, and `t` y-ring rows, the complete target census is
`P = n·(4κ + a_X + 2t + 2)` and `J = 4κ + a_X + 2t + 2` after batching
each aggregate over its `n` inputs. **The rows are wired; until the census
receives non-author review, this lemma uses the conservative bound `J ≤ P`.**

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
(items 1–3 of the schedule; commitment binding is A2,
projection-preimage binding is A6, and transcript determinism is A3).
Suppose `c*_m ≠ Σ_i ρ_i·c_{i,m} mod Φ`. Then
`G_m ≠ 0` as a polynomial: if `G_m = 0` identically, reducing
`Σ ρ_i c_{i,m} = q_m·Φ + c*_m` mod Φ gives
`Σ ρ_i c_{i,m} mod Φ = c*_m` (as `deg c*_m < d`), a contradiction. A
nonzero polynomial of degree ≤ 2d − 2 vanishes at a uniform `β ∈ K`
with probability ≤ `(2d − 2)/|K|`. Union over all `J` identities of
the fold (one shared β is sound because every identity's operands
precede the one squeeze). Completeness: the honest `q_m` is the unique
quotient from division by the monic polynomial Φ
(`projection_quotient`), for which each `G_m` is identically zero. ∎

Conservative worked bound at maximum v3.1 production shape (`n=15`,
`κ=18`, `a_X=46`, initial folded relation `t=15`): `P=2,250`, so
`P·(2d − 2)/|K| = 2,250 · 106 / 2^128 ≈ 2^−110.14` per fold. The wired
aggregate-batching count is `J=150`, giving ≈ `2^−114.04` per fold. The
conservative term remains below the Lemma-3 term (≈ 2^−110 per segment),
but the tighter number is not assumed until the census wiring is reviewed.
β **must** live
in `K`: over `F` even the tighter term is ≈ 2^−57 per fold,
unacceptable across `n_f · q_H`.

**Composition with Lemma 1.** Lemma 1 Step 2 consumes the verifier's
computation `adv*_L = Σ_i adv_{i,L}·ρ_i` (and the pipeline's
`c* = mix_commits(ρ, c)`). In the adopted relation these are
projection-checked claims: first apply this
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
   bound uses `J = Σ_j J_j`; until the census wiring is reviewed, Lemma 5's
   conservative `J ≤ P` stands. Maximum v3.1 production rows (`n=15`,
   `κ=18`, `a_X=46`, initial folded relation `t=15`):

   | client | pairs `P_j` | identities `J_j` | consumer | batching justification |
   |---|---:|---:|---|---|
   | Π_RLC commitment mix | `κ·n = 270` | `κ = 18` | folded `c*` | only the aggregate mix is consumed |
   | X active columns | `a_X·n = 690` | `a_X = 46` | folded `X*` | one aggregate per active column |
   | y_ring rows, two K limbs | `2t·n = 450` | `2t = 30` | folded `y_ring*` | one aggregate per row and K limb |
   | y_zcol, two K limbs | `2n = 30` | `2` | folded `y_zcol*` | one aggregate per K limb |
   | Nebula `adv` tuple mirror (spec §5.2 R2) | `3κ·n = 810` | `3κ = 54` | folded `adv*` tuple | one aggregate per tuple component and ring lane |
   | **total** | **2,250** | **150 after reviewed batching; ≤2,250 before** | authoritative F′ relation | complete client census |

   Π_DEC's recomposition is scalar `b`-powers — linear, **not** a
   projection client.
5. Limb canonicality: evaluation rows reduce mod q on both sides, so
   aliased bit-encodings agree everywhere compared (same argument as
   spec §4.4's note).

**The `Φ(β) = 0` non-caveat (author self-review attack 9 — REFUTED by
external review; retained as a correction record).** The self-review
initially claimed a completeness error when β lands on a root of Φ.
That claim was algebraically backwards: for the honest pair,
`P − out = q·Φ` *identically*, so `P(β) = out(β)` holds **exactly** at
any root of Φ — the honest prover never fails there, and no
re-squeeze is needed. Soundness at such β is likewise already inside
the SZ bound (`G` is a fixed nonzero polynomial; its roots, wherever
they lie, are counted once). Moreover the event is unreachable at our
parameters: `q ≡ 4 (mod 81)` gives `81 ∤ q² − 1`, so `K = F_{q²}`
contains no 81st roots of unity and `Φ_81` has **no roots in K at
all** (its roots live in `F_{q^27}`; `Φ` splits mod q into two
degree-27 factors). Nothing to handle, in two independent ways.

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
the consumer is the aggregate (the batching rule). (8) evaluation-wire
shortcuts (`β^0` term folded linearly) — algebraically identical.
(9) Φ(β) = 0 — the self-review claimed a completeness caveat here;
**external review refuted it** (see the correction record above: the
honest identity holds identically, and Φ has no roots in K at these
parameters). Verdict: no soundness break found; and the refuted attack
9 is itself the demonstration of why this self-review does NOT
substitute for the non-author pass — the author's one novel "finding"
was wrong, and a fresh reader caught it in one reading.

**Optimization recorded, not adopted:** the κ identities could be
RLC-combined under a second challenge τ into one identity
(`Σ_m τ^m·G_m(β) = 0`), saving κ − 1 final rows for an extra
`(κ − 1)/|K|` error term. v1 of any adoption should keep per-component
identities — simpler review, negligible row savings.

## 4c. Lemma 6 — two-level SIS binding before Fiat–Shamir

**Status: ADOPTED and implemented.** This lemma is the security contract for
the five witness-proportional transcript compressions in A6. It does not claim
that Poseidon2 repairs an SIS collision; both SIS layers are explicit bad
events.

For role `r`, let `E_r` be the exact centered-unit message encoded by the
relation, including its enforced reconstruction to the source fields. Define

```text
C_r = A_r · E_r                         (rank 2, role-specific seed)
S_r = B · E_short(C_r.data)             (rank 1, independent shared seed)
h_r = Poseidon2(v3_tag, r, |fields|, 2, S_r.data).
```

The five `A_r` matrices have independent seeds and domains. `B` is shared
because its input shape is always one 108-field rank-2 commitment; the role in
the final Poseidon2 envelope prevents cross-role substitution. All matrix
outputs and the final digest are recomputed from relation wires before the
digest is absorbed and before the associated challenge is squeezed.

**Statement.** Under A3 and A6, if two distinct valid authoritative encodings
for the same role and field count produce the same `h_r`, then one of the
following occurs: a rank-2 Module-SIS collision for `A_r`, a rank-1
Module-SIS collision for `B`, or a Poseidon2 collision. Consequently

```text
Adv_bind(h_r) ≤ ε_MSIS(A_r) + ε_MSIS(B) + ε_CR.
```

Across all five roles, the SIS contribution is `ε_BIND`; the global
composition counts `ε_CR` once.

**Proof.** Let `(E,C,S)` and `(E',C',S')` produce equal final digests. If the
Poseidon2 envelopes differ, equality is a Poseidon2 collision. Otherwise the
envelopes are equal, hence `S = S'` and the role, field count, and primary rank
agree. If the short messages differ, `B·(E_short(C)−E_short(C')) = 0` is a
nonzero Module-SIS solution with infinity norm at most 2. If they do not
differ, they reconstruct the same `C.data`; if the long messages differ,
`A_r·(E−E') = 0` is the corresponding rank-2 Module-SIS solution, again with
infinity norm at most 2. The remaining case has the same authoritative
encoding and is not a binding violation. This case split is exhaustive. ∎

**Hash-then-Fiat–Shamir corollary.** The relation does not accept a digest as
authority: it recomputes both linear maps and Poseidon2 from the exact message
wires, then replays the transcript squeeze. Therefore an adversary that makes
a challenge depend on a different pre-squeeze message either triggers one of
the three binding events above or performs another random-oracle attempt. The
latter is exactly the explicit `q_H` factor on each challenge-dependent term
in §5. Multiple valid centered-unit encodings of one field value do not evade
the argument: the encoding wires themselves are committed relation witness;
choosing another valid encoding is another pre-squeeze message and another
query, while obtaining the same digest is covered by the case split.

**Concrete estimates.** R7 pins the largest long map at 22,147 ring columns
(1,195,938 scalar coefficients) and the short map at 82 ring columns (4,428
coefficients). The estimator uses the conservative Euclidean collision bound
`2·sqrt(m)` induced by coefficient infinity norm 2. At the pinned estimator
commit, the rough/full costs are 167.0/190.2 bits for rank 2 and 223.1/242.1
bits for the short rank-1 map. Five long roles union to about 164.7 rough bits;
§5 rounds this down to a 160-bit floor. These are heuristic concrete-security
estimates under A6's random-matrix model, not a proof of hardness for the
specific fixed seeds.

## 5. Composition theorem and evaluated budget

**Theorem.** Under A1–A6, an accepted chain of `n_seg` segments with valid
terminal decider checks attests a sequentially consistent memory history
starting from the plan's initial memory, except with probability

```text
ε_total ≤ q_H · n_f · ε_pipe                            (A1, FS-lifted)
        + ε_MSIS + q_H · n_f·n_in/|C|                  (Lemma 1)
        + ε_BIND + ε_CR                                  (Lemma 6 + A3)
        + q_H · n_seg · m_seg / |K|                     (Lemma 3 / Cor 4.1)
        + q_H · n_f · J_proj·(2d−2) / |K|               (Lemma 5)

ε_MSIS := ε_MSIS(A) + ε_MSIS(A_ops) + ε_MSIS(A_mem)     (A2's instances,
                                                         union-bounded)
ε_BIND := Σ ε_MSIS(map) over five rank-2 maps and one short rank-1 map
n_in   := per-fold input-claim count — SuperNeo's fold arity "K + k";
          instance counts, unrelated to the field K in |K|
m_seg  := |IS| + |WS| + |RS| + |FS| = 2·(N·B_ops + R + M)   (Cor. 4.1)
J_proj := projection identities per fold; use conservative J_proj ≤ P
          until the maximum-geometry census receives non-author review
```

The same global `q_H` cap is applied to every public-coin failure term. This
is conservative: a single adversary has at most `q_H` total random-oracle
queries, while the displayed union gives every fold and segment the whole
allowance. `ε_MSIS` may already be represented inside the inherited pipeline
reduction; it is nevertheless counted separately.

**Declared production target.** The maximum supported geometry is
`n_seg = SEG_MAX = 2^16`, `N = 1,088`, hence `n_f = 71,303,168`. The profile
declares a **64-bit end-to-end floor** for adversaries making at most
`q_H = 2^16` random-oracle queries. R7 evaluates the conservative census
`J_proj = P = 2,250`, not the reviewed-batching candidate `J=150`, and uses
SuperNeo D.4's exact maximum-fresh numerator `f_D4 = 1,439,664` for the final
15,958,404-coordinate, 14-matrix, degree-8 relation:

| term | maximum-chain bits |
|---|---:|
| `q_H·n_f·f_D4/|K|` | 65.46 |
| conservative projection | 68.05 |
| Nebula fingerprint | 77.91 |
| strong-set mixing | 79.39 |
| A2 Module-SIS floor | 100.00 |
| A6 two-level binding floor | 160.00 |
| Poseidon2 collision/preimage floor | 128.00 |
| **union of all displayed terms** | **65.23** |

Thus the profile clears the declared target by about 1.23 bits; it does
**not** support a 100-bit maximum-chain claim. Because the margin is narrow,
changing `SEG_MAX`, `N`, `q_H`, the D.4 shape, or the conservative projection
census reopens this budget. The active R7 test computes and pins the formula;
the estimator script pins the two Module-SIS floors. Concrete hardness and
Poseidon2 remain assumptions A2/A3/A6 rather than facts established by tests.

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
| C11 | Cost model within 2× | **implemented and actively measured** | R7 constructs the authoritative production fixed point and pins both the selective census and final relation; C12 records the exact values. |
| C12 | F′-R1CS absorb budget | **implemented at production parameters** | The authoritative three-arm relation includes current `S_mem`, delayed lane transition, and c/adv/X/y projection. The two-level SIS bindings reuse the same 41 centered unit digits as the folded witness; compact seeded matrices, private Poseidon-output substitution, five-product direct CCS rows, telescoping evaluation accumulators, exact Karatsuba K-dot traces, and canonical-bit reuse elsewhere produce a reduced-profile square fixed point of 9,959,328 coordinates / rows. At Appendix B.2 parameters and maximum v3.1 geometry, the first selective census is 15,730,104 and the verifier-shape fixed point is **15,958,404 coordinates / rows, 14 matrices, degree 8**, with `M0 = I`. The active R7 gate pins those values below the unchanged 16M ceiling, validates the full D.4 factor against the final relation's actual shape, and evaluates the §5 maximum-chain budget. |
| C13 | Workload fit of `M`, segment granularity of incrementality | **product decision** | owner: Nico |
| C14 | SIS/Ajtai accumulation for carried chains or remaining claim hashes | **implemented and concretely estimated for five R2 binding roles** | Each role first uses an independent rank-2 map over its authoritative balanced-trit encoding; one independent short rank-1 map compresses the 108-field output, and a domain-separated Poseidon2 envelope enters Fiat–Shamir. `CscWithSeededPhi81` preserves both maps structurally through CCS/SuperNeo consumers; native/circuit parity, stage-tamper rejection, and width pins cover the adopted path. R7 pins 36 rank-2 plus 36 short rank-1 blocks, with maxima of 29,168 words / 22,147 ring columns and 108 words / 82 ring columns. A6 records the exact fixed-matrix assumption and estimator commit; Lemma 6 gives the hash-then-FS reduction. Carried `D` chains remain Poseidon2. Independent cryptographic review remains required, but the former unstated κ=1 assumption is gone. |
| C19 | Parent-authority accumulator handle | implemented; reduction argument needs non-author review | native and in-circuit NIFS.V both verify strict Π_DEC(parent, children) before the compact handle is derived or consumed, and Π_CCS already uses `ce_claim_digest(parent)` as the running Fiat–Shamir authority. The handle is `Poseidon2(tag, child_count, parent_present, ce_claim_digest(parent))`; malformed empty/non-empty shapes remain domain-separated. Red-team tests mutate input/output children, rebuild the handle and visible state, and still fail the Π_DEC rows. The proof obligation is that replacing exact-child hashing by the verified weak-reduction representative composes with A5/SuperNeo Π_DEC knowledge soundness. |
| C15 | IS/FS boundary chains must be formula-identical | **completeness bug, fixed (external review)** | lane-typed `"is"`/`"fs"` tags made honest cross-segment continuity impossible; is/fs now share one mem-domain leaf/link tag pair and header (spec §6.1/§6.3/§7); Cor. 1.1 states the dependency |
| C16 | Externally accepted proofs end at closed segments | **normative rule added (external review)** | terminal first consumes trailing `u_T`, then requires `idx == 0`, `γ == ⊥`, and header chains; checking the pre-terminal lane is insufficient (spec §6.3) |
| C17 | Stack ops are LIFO-consistent under the v3.1 rows | proven here (v3.1) | Lemma 4 (reduction to Blum et al. / Coral App. E); segment locality forced by per-segment γ (spec §3.1) |
| C18 | Projection-checked ring action is sound for the folded F′ regime | **implemented; non-author review remains open** | Lemma 5; authoritative NIFS.V uses the transcript-bound `q`/β wires at every c/adv/X/y site, and C12 measures the resulting fixed point. Adoption audit items remain the review checklist. |
| C20 | Terminal-only folded induction consumes every memory claim | **implemented; active R5 acceptance and tamper gate** | `NebulaFPrimeChainBuilder` deposits only the fixed relation with `K=1`; recursive steps consume prior `latest`, terminal finalization consumes trailing `latest`, and `verify_uncompressed` accepts the chain without `steps`/`public_batches`. `multi_chunk_f_prime_chain_must_verify_terminal_only` is active and rejects changed prior-link bits, delayed suffix bits, pre-final lane state, and the pre-final running commitment that carries earlier folded history. Capability ownership keeps non-authoritative image frontends fail-closed; the plain authoritative R1CS path separately checks HyperNova's running accumulator and latest F′ relation. |
| C21 | Shipped encoder fills the live lowered relation | **implemented; active plain and memory gates** | The memory encoder derives live q/β transcript advice, accumulator state, current suffix, and `adv`, normalizes the exact field assignment, then `MultiBranchLowNormR1cs::encode` fills the selected low-norm arm and checks it before commitment. The active memory gate covers three one-step segments and all three relation arms; focused suffix/relation tests cover absent `D_pre` on interior steps. The active stateful Fibonacci gate covers four plain F′ steps. |

## 7. What this note does not close

The Rust implementation and measured cost gates do not discharge
**independent review** of Lemmas 1–6, C19's parent-authority reduction, or
A6's fixed-matrix Module-SIS assumption and estimator methodology. The author
checking the author is circular, and the leans named in §0 (A1's exact
extractor shape, A5's uniqueness property, and A6's map hardness) are
precisely where a reviewer should push. Suggested review order: Lemma 1 Step 2 first (the only place a
challenge-set property is invoked), then Lemma 2's induction against
`state.rs`'s actual absorb set, then Lemma 4 step 3's alternation
argument against the actual E12/E13 rows. Lemma 5 can be reviewed
independently of the others; it is now load-bearing at Nico's direction,
with non-author review still open. Push hardest on the transcript schedule
(why the recomputed A6 binding of every `q` must precede β) and on integration
obligation 1 (wire identity — the classic
way projection arguments silently break). For Lemma 6, reproduce the pinned
estimator run first, then challenge the random-matrix model for the fixed
ChaCha8-expanded seeds and the three-event collision case split.
