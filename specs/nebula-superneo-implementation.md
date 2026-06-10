# Nebula on SuperNeo: Memory-Checked IVC

Date: 2026-06-10 (v3 — supersedes the 2026-06-09 v2. Review fixes:
pair-unit timestamp accounting, normative `c_adv` lane lifecycle,
genesis address-canonicity induction, ε-budget `M` term, canonical-u64
timestamp lanes, audit-path-only consequence of D8.)
Status: architecture spec for `neo-fold-clean`
Scope: Nebula-style offline read/write memory checking inside the
SuperNeo/F′ lifecycle, with per-segment finalization so proving can
continue across long-running executions.

References. Nebula paper: `docs/Nebula-2024-1605.pdf.md` (ePrint
2024/1605, Arun–Setty), cited below as "Nebula §…". Reference
implementation: `external/zkEngine_dev/src/wasm_snark/` (cited by file
below) — a complete non-lattice Nebula implementation over the
Nova/Arecibo group-based substrate. We take its schedule, rw-pair
discipline, padding policy, and verifier checks; its proof substrate
does not transfer. Security framework: SuperNeo strong/weak interactive
reductions (`docs/SuperNeo.pdf.md` §6) and HyperNova §6
(`docs/hypernova-paper/`).

---

## 0. Decisions

| # | Decision | Choice | Rationale (detail in section) |
|---|----------|--------|-------------------------------|
| D1 | Fingerprint challenge field | `K = F_{q²}` (~2^128), products computed in K | Goldilocks-native 64-bit challenges give ~2^-34 soundness on real traces — disqualifying at λ=100+. §3 |
| D2 | Tuple transport exec→ops | Per-step advice-slice Ajtai commitment `c_adv`, equality checked at audit/decider replay | In-circuit row hashing costs ~100k+ image bits/row; commitment equality is native and free. §5 |
| D3 | Chain topology | One lifecycle chain, one universal relation `S_nebula` with four branches | One accumulator, one preprocessing key, one terminal authority; F′ shell cost amortizes branches. Fallback: split chains if the union-structure probe exceeds budget. §4 |
| D4 | Challenge derivation | Dedicated challenge step; Poseidon2 transcript absorbing exec advice commitments, pre-committed FS commitments, IS binding, segment context | Absorbing FS before sampling is mandatory; zkEngine check #3 (`wasm_snark/mod.rs:489-500`). §6 |
| D5 | Memory state at segment boundary | Canonical address-ordered chunk list of FS advice commitments; `FS_k ≡ IS_{k+1}` by per-chunk commitment equality | MCC has no Merkle root; the commitment list *is* the boundary object. §7 |
| D6 | Padding | Real read/write pairs against reserved address 0 with correct timestamps; per-step op count enforced in-circuit | Padding rows are legitimate multiset members; nothing is skipped. Matches `multiset_ops.rs:356-360`. §2 |
| D7 | Timestamps | Global 64-bit counter advancing **once per read-write pair** (Nebula F_ops step 2c), canonical-u64 lanes with bit-decomposed comparisons in-circuit, watermark threaded through state and across segments, declared bound `N_max` in the public statement | zkEngine's 32-bit `enforce_lt` does not transfer; wraparound and non-canonical Goldilocks lanes are forgery vectors. §2, §3 |
| D8 | Per-pair commitment equality accumulation | Replay-level checks now (audit + full-history decider); O(1)-terminal accumulation is an open design item | Matches where accumulator authority lives today; until it lands, Nebula chains are audit-path-only (§5). Flagged for council review. §5, §10 |

---

## 1. What we port and what we replace

Nebula contributes three things:

1. **Commitment-carrying IVC**: the proof carries binding commitments to
   per-step non-deterministic advice, so a later phase can consume the
   same advice without any in-circuit hashing of it.
2. **Offline memory checking** (Spice lineage): read/write consistency
   as a multiset equation over `(addr, val, ts)` tuples, checked with
   randomized fingerprint grand products after the tuples are committed.
3. **Switchboard circuits**: per-instruction branches inside the
   execution step whose inactive branches carry zero witnesses, so the
   *commitment* cost is paid only for the active instruction.

We port (1) and (2) natively onto Ajtai/SuperNeo. For (3): the paper's
switchboard is instruction-level and its "inactive branches are free"
claim is a statement about MSM commitment costs. Under SuperNeo,
pay-per-bit Ajtai commitment preserves that property, but Π_CCS
sum-check cost scales with the *union structure*, not the active
nonzeros. We still adopt a (machine-level) switchboard for the four
Nebula machines (D3), because in this codebase the per-step F′ shell —
state hashing, NIFS payload encoding, recursive link — dominates the
branch bodies, and a single chain keeps one accumulator and one terminal
authority. An instruction-level switchboard inside `F_exec` is a later,
independent decision.

What does **not** transfer from zkEngine: its three separate
`PublicParams`/IVC chains glued by a native Layer-1 verifier
(`wasm_snark/mod.rs:452-514`) and its Layer-2 sharding SNARK. In our
production path every check that zkEngine's host verifier performs must
be either an in-circuit constraint of some branch or a replayed check of
the audit/decider. There is no host-checked memory proof in production.

**Paper fidelity map.** Where each Nebula component lands here, and the
status of every deviation:

| Nebula component | Here | Fidelity |
|---|---|---|
| Offline checker, rw pairs, padding (Nebula §4.2 codeboxes, Lemma 7, Corollary 8) | §2 MCC core | Faithful. One `ts += 1` per pair, same `rt < ts` / `wt = ts` rules; fingerprint challenges renamed `(γ₁, γ₂) → (γ, α)`. |
| CC-IVC / split-committed witness (Nebula §3, Construction 2) | Existing Construction-2 F′ shell unchanged; `c_adv` rides the instance (§5) | Faithful shell; transport adapted (next row). |
| Incremental commitment `C_i = H(C_{i−1}, C_ω)` (Nebula Construction 1) | **Replaced** by the folded `c_adv` lane + replay equality + terminal slice opening (§5) | Deliberate deviation: an Ajtai `c_adv` is 972 field elements — hashing it per step in-circuit is the cost profile Nebula exists to avoid. The binding argument mirrors Nebula Theorem 5 (§5). |
| `F_ops` / `F_scan` (Nebula §4.3 codeboxes) | `F_ops` / `F_scan` branches (§6) | Faithful, except the paper's per-row `a = a′ = i` becomes `a_IS == a_FS` plus genesis induction (§6, zkEngine-style). |
| Host verifier checks 1–4 (Nebula §4.3; `wasm_snark/mod.rs:452-514`) | `F_segment` in-circuit checks + audit/decider replay checks (§6) | Faithful relocation: every host check becomes a circuit row or a replayed verifier check; no host-checked memory proof in production. |
| Layer-2 `F_final` (Nebula §4.3, Figure 1) | One chain, per-segment phases, §7 boundary threading | Restructured, check-for-check equivalent: `z`/`ts` continuity → state chain; `C_mem = C_IS` → §7; challenge re-derivation → `F_chal` replay; product equation → `F_segment`. Discard-the-history incrementality deferred to D8 (§5). |
| Switchboard (Nebula Construction 3, Lemma 10, Theorem 11) | Machine-level branch selectors in `S_nebula` (§4) | Adapted: zero-witness inactive branches keep pay-per-use *commitment* cost under Ajtai; Theorem 11's cross-term trick is Nova-specific and unused; Π_CCS sum-check pays union structure — hence the D3 cost gate. |

---

## 2. The MCC core (normative)

Memory is a fixed address space `[0, M)` (data + stack + globals;
frontend-defined map). Every memory operation is a **read-write pair**
against untrusted memory, as in Spice/zkEngine (`multiset_ops.rs`):

- `read(a)`: `ts += 1`; emit `RS += (a, v, t_last)`, `WS += (a, v, ts)`
  — a read writes back the same value at the new timestamp.
- `write(a, v')`: `ts += 1`; emit `RS += (a, v_old, t_last)`,
  `WS += (a, v', ts)`.

`IS` is the initial memory multiset `{(a, v_a, t_a)}` over all `M`
addresses in **canonical address order** (segment 1: `t_a = 0` and
`v_a` from the public program image; segment k>1: threaded, §7). `FS`
is the final multiset in the same order. The protocol claims:

```text
IS ∪ WS == RS ∪ FS        (multiset equality)
```

checked via grand products of fingerprints with challenges
`α, γ ∈ K = F_{q²}` sampled after all tuples are committed (D1, D4):

```text
fp(a, v, t) = a + γ·v + γ²·t − α            (arithmetic in K)
h_IS · h_WS == h_RS · h_FS                   (products of fp over each set)
```

Per-tuple in-circuit checks (ops phase): read timestamp strictly less
than the incremented global counter, write timestamp equal to it
(Nebula F_ops steps 2c–2e), both via canonical-u64 bit-decomposed
comparisons (D7); each fingerprint proven nonzero by the inverse trick
(this is also what makes tuples countable — `gadgets/mcc.rs:64-92`);
per-step tuple counts equal to the declared per-chunk op budget `B_ops`.

**Unit (normative).** `B_ops` counts **read-write pairs**, and the
global counter advances by exactly one per pair — a chunk advances `ts`
by exactly `B_ops`. zkEngine's `MEMORY_OPS_PER_STEP` counts RS/WS
*entries* (two per pair; it pads up to `MEMORY_OPS_PER_STEP / 2`
pairs); do not import that unit.

**Padding (D6).** Execution chunks with fewer than `B_ops` memory ops
emit real `read(0)` pairs. Address 0 is reserved by the frontend memory
map. Padded VM steps (chunk tail) are no-op transitions that also emit
`read(0)` pairs. Every emitted pair participates in the multisets; the
in-circuit count check makes under- or over-emission unsatisfiable.
There is no "skipped row" anywhere in the argument.

---

## 3. Parameters and soundness budget

- Challenges `α, γ ∈ K` (~2^128). The fingerprint identity test has
  error ≤ `2·(M + N) / |K|` per segment for `M` memory cells and `N`
  read-write pairs: each side of the product equation is `M + N`
  fingerprint factors of total degree 2, and the scan contributes the
  `2·M` IS/FS factors — the `M` term must not be dropped (Nebula
  Corollary 8 / Theorem 9 state `O(M + N)/|F|`). With `M, N ≤ 2^40`
  this is < 2^-85 per segment. Per-segment errors and the
  nonzero-inverse gadget terms enter the protocol-level
  `eps_sumcheck_total` union bound (TODO.md). The spec for that budget
  must list: fingerprint identity error per segment × segment count,
  Ajtai binding (MSIS) for the advice commitments, and the existing
  Π_CCS/Π_RLC/Π_DEC terms. **Council sign-off required on the combined
  statement before implementation.**
- `γ²` is computed once per segment in the challenge step and carried
  as state lanes, so per-tuple cost is 3 K-muls (γ·v, γ²·t, product
  update) plus the nonzero check.
- Timestamps: one global 64-bit counter across the whole execution
  (all segments), advancing once per read-write pair (§2 unit). Public
  statement declares `N_max` (max total pairs). The per-pair
  comparisons are over full 64-bit decompositions under the repo's
  **canonical-u64 contract** (the `enforce_counter_*_no_wrap` gadget
  family): Goldilocks bit patterns in `[q, 2^64)` alias to small field
  values, so canonicality of the decomposition is normative —
  64-bit-ness alone is forgeable. The watermark (§7) makes
  cross-segment continuation explicit. Goldilocks order > 2^63 means
  no wraparound below `N_max < 2^63`.
- Tuples in the committed witness are bit-backed: `a` (log M bits),
  `v` (64 bits), `t` (64 bits), matching the existing F′ low-norm
  contract (b = 2). Branches decode bits to lanes where K-arithmetic
  needs them, using the existing lane/K-mul gadgets.

---

## 4. One chain, one universal relation

`S_nebula` is a single CCS structure with four branches and a branch
selector derived in-circuit from phase counters (the same discipline as
the existing `is_base ⇔ chunk-counter` link — selectors are never free
witness bits):

- `F_exec` — proves one VM/application chunk transition; emits `B_ops`
  tuple pairs bound to the semantics (§6); owns the advice slice.
- `F_chal` — one step per segment: derives `α, γ` (D4) and seeds the
  ops/scan phase state. (Thin; may be merged into the first `F_ops`
  step only if the **entire** §6 absorption completes before that
  step's first `h_RS`/`h_WS` product update in constraint order — a
  dedicated step is the structurally safer default.)
- `F_ops` — recomputes `h_RS`, `h_WS` over re-supplied tuple advice.
- `F_scan` — recomputes `h_IS`, `h_FS` over address-ordered chunks;
  enforces `a_IS == a_FS` per position (`mcc/mod.rs:202-208`).
- `F_segment` — closes the segment: checks the product equation and
  boundary handoff (§6), advances the long-running state.

Phase order within segment k, enforced by counter-derived selectors:
`F_exec × n_k → F_chal → F_ops × n_k → F_scan × ⌈M/B_scan⌉ →
F_segment`. The scan chunk budget `B_scan` is a separate plan constant
from the exec chunk shape — the address space and the trace have
unrelated sizes (zkEngine's `StepSize { execution, memory }` split).

All steps are instances of the one structure, folded by the existing
lifecycle (`Π_CCS → Π_RLC → Π_DEC`), accumulated into the one running
accumulator, closed by the existing terminal CE / decider authority.
The recursive F′ link, x_out digest chain, and `vk_fs` binding apply
unchanged; `S_nebula` is a new canonical plan, so preprocessing and
`vk_fs` change — version the plan. **Programs that do not use memory
checking are untouched**: they keep their existing plans — no `c_adv`
lane, no Nebula branches, no cost. Isolation is by canonical-plan
versioning, not a runtime flag.

**State extensions** (new `StateIn`/`StateOut` lanes, bit-backed):
branch/phase tag and intra-phase counters; segment index; global ts
watermark; `α, γ, γ²` (2 lanes each, K); running `h_RS, h_WS, h_IS,
h_FS` (2 lanes each, K); running advice-transport digests (§5); memory
boundary digest (§7). Order of 30–40 extra 64-bit lanes ≈ 2–3k image
bits plus their hash absorption — small against the ~134k-bit shell.

**Cost gate for D3.** Before building, probe the union-structure cost:
extend the layout-budget/shape tests with stub branches at target
budgets (`B_ops`, `B_scan`) and measure rows, image width, and prover
time per step vs. an exec-only structure. If the union overhead on
exec steps exceeds ~2× prover time, revisit D3 (fallback: separate
chains per machine, cross-bound at the decider — more terminal-path
surface, cheaper steps).

---

## 5. Commitment carrying: tuple transport without row hashing

This replaces v1's "Poseidon2 row-root inside F′", which is rejected on
cost: one bit-backed Poseidon2 absorption costs ≈ 3.3k image bits per
lane (the 40-lane `state_x_out` trace is 131,328 bits), so hashing each
3-lane tuple in-circuit costs ~10k bits/row before products — the
Merkle-cost profile Nebula exists to eliminate. Nebula's transport is
commitment equality, not hashing (`IC_exec == IC_ops`,
`wasm_snark/mod.rs:484-487`); we port that.

**Mechanism.** The tuple advice region occupies a fixed column range
(identical layout) in the `F_exec` and `F_ops` branch witnesses, and
the FS region likewise in `F_scan`. Each step's instance carries, next
to its main commitment `c`, an **advice-slice commitment**
`c_adv = A_adv · z_adv` under a dedicated Ajtai key `A_adv` over only
those columns (κ = 18, d = 54 ⇒ `c_adv` is 972 field elements, the same
encoding class as the existing `c_data_entries = 972` NIFS payload).
`A_adv` is verifier-owned preprocessing, registered next to `A`; proofs
never choose it.

**Lane lifecycle (normative).** `c_adv` is a first-class claim lane;
every surface that binds or folds `c` must cover it, or the binding
checks below consume unanchored values:

1. **Claim shape.** The step/CE claim gains a `c_adv` field. Claims
   without the lane are malformed for `S_nebula` programs.
2. **Verifier-side folding.** Π_RLC's verifier recompute
   (`rlc_public` / `rlc_public_matches`) mixes `c_adv` with the same
   rot(ρ) action as `c`; Π_DEC's `verify_dec_public` checks the same
   b-power recomposition over the children's `c_adv`. Both maps are
   ring-linear, so the folding rules are identical to `c`'s — but it is
   the *verifier-side* recompute that anchors the lane; prover-side
   mixing alone proves nothing.
3. **Digest surfaces.** `c_adv` is absorbed everywhere the claim is
   transcript- or accumulator-bound: the Π_CCS me-input projection
   digest, the pre-ρ `pi_ccs_outputs_digest`, and the
   `AccumulatorHandle`. These digests enumerate fields explicitly
   (`neo-reductions/src/engines/utils.rs`), so the new lane is
   invisible to them until added — omission leaves `c_adv` replay-bound
   only.
4. **Terminal opening.** The decider opens the folded `z*` once and
   checks `c* = A·z*` **and** `c_adv* = A_adv · slice(z*)`.

Together 1–4 give the Nebula-Theorem-5-style extraction: each step's
`c_adv` is a binding commitment to its slice, the lane folds under
verifier-recomputed ring-linear maps, and the terminal opening ties the
folded lane to the same opened witness the main commitment authority
already covers.

**Why no new reduction proof is needed (normative soundness story).**
Formally, the lane is specified as *enlarging the commitment
homomorphism*: define `ℒ′(z) := (A·z, A_adv·slice(z))`. Column
projection is an `R_F`-module homomorphism and `A_adv` is one, so `ℒ′`
is one — and SuperNeo's Definitions 12–14, Theorem 5 (evaluation
homomorphism), Lemma 3 (Π_CCS strong), Lemma 4 (Π_RLC weak), and
Theorem 7 (Π_DEC) are all stated for an **arbitrary** `R_F`-module
homomorphism `ℒ`. Theorem 1's composition therefore applies verbatim
to `CCS(b, ℒ′)` / `CE(b, ℒ′)`: `c_adv` is not a new mechanism, it is
the second coordinate of the one commitment the existing proofs already
cover. Relaxed binding of `ℒ′` follows a fortiori from `A`'s (a
composite collision is in particular an `A` collision). `A_adv`'s own
MSIS binding is needed only where two *different* steps' slices are
equated through their commitments — the §5 pairing and §7 boundary
checks — and is already a line item in the §3 ε-budget. Items 1–4
above are exactly the implementation obligations of "treat `(c, c_adv)`
as one commitment everywhere the proofs treat `ℒ`."

**Binding checks** (replay-level, D8):

1. Pairing: for segment k with exec steps `e_1..e_n` and ops steps
   `o_1..o_n`, check `c_adv(e_j) == c_adv(o_j)` for all j. Identical
   slice layout + Ajtai binding (low-norm openings) ⇒ identical tuples.
2. FS pre-commitment: the prover supplies the scan-phase FS slice
   commitments `c_adv(s_1..s_m)` *to the challenge step* (§6); after
   the scan steps run, check each scan step's actual `c_adv(s_i)`
   equals the pre-committed value.
3. Cross-segment: `c_adv^FS(k, i) == c_adv^IS(k+1, i)` per aligned
   chunk i (§7).

These are field equalities over carried instance data. The uncompressed
audit checks them during replay; the full-history decider adds them as
linear constraints over the same replayed data.

**Consequence: until the O(1)-terminal accumulation lands, Nebula
chains are audit-path-only.** Pause/resume works, but the per-step
trail must be retained for *all* segments and the full-history decider
replays it — Nebula's discard-the-history incrementality (paper
Figure 1) arrives only with the D8 follow-up. No new gate is needed on
the terminal-only verifier: every Nebula segment spans multiple
chunks, and `verify_uncompressed` already rejects `chunk_count > 1`.

**Open item (council):** the future O(1) terminal needs these
accumulated (e.g. a challenge-weighted running combination in state)
instead of replayed; that design is out of scope here and must not
block the audit-mode implementation.

Why a separate key `A_adv` instead of reusing columns of `A`: slice
commitments under `A`'s columns are not extractable independently of
the rest of `z`; an independent key makes `c_adv` a binding commitment
to the slice alone, and the terminal slice-check ties it to the same
opened `z*` the main commitment authority already covers.

---

## 6. Branch relations (checks, normative)

**F_exec.** Proves the chunk's VM transition. The binding rule: the
address and value lanes used by the transition semantics are the same
wires that form the tuple slice — no copy gadget between "semantic
memory access" and "emitted tuple" (this is the load-bearing constraint
from RAM_PROBLEM.md). Emits exactly `B_ops` pairs (padding per D6).
Advances the ts watermark by exactly `B_ops` — one increment per pair
(§2 unit; padding pairs increment like real ones). Threads the running
exec advice count.

**F_chal.** Derives, in one Poseidon2 one-shot trace with a new domain
tag (`neo.fold.clean/nebula_mcc_challenges/v1`):

```text
(α, γ) = H(tag, segment_index, ts_watermark,
           acc_digest_at_close,            # binds all folded exec steps
           D_exec_adv,                     # digest of c_adv(e_1..e_n) sequence
           D_FS_adv,                       # digest of pre-committed scan c_adv list
           D_IS_bind)                      # genesis digest or segment k-1 boundary digest
```

`D_exec_adv` and `D_FS_adv` are computed natively over instance data
and entered as public inputs of this step; the audit/decider recomputes
them during replay (same authority pattern as `acc_digest`). The
in-circuit work of `F_chal` is only the final absorption producing
`α, γ` into state. FS must be inside the absorption — if the prover
can choose FS after seeing `α, γ`, the scheme is broken (zkEngine
absorbs `IC_FS` for exactly this reason).

Why `D_exec_adv` exists at all: **a folded commitment is not a
transcript.** The RLC accumulator is a random linear combination of
history, not a binding digest of it — it binds the per-step advice only
through the extraction argument, which is the wrong tool for
Fiat-Shamir. Challenge derivation must absorb a *digest of the
per-step commitment sequence* (`D_exec_adv`), never the folded value
alone. `acc_digest` is absorbed too, but note it is already a Poseidon2
digest of the accumulator claims + Π_RLC parent (the
`AccumulatorHandle`), not the folded commitment itself.

Naming note: the MCC challenges `(α, γ)` collide with Π_CCS's own
transcript challenges (`ch.alpha`, `ch.gamma`). They are unrelated
values from unrelated domains — Π_CCS samples per fold step from the
Poseidon2 fold transcript; the MCC pair is derived once per segment by
this branch's one-shot trace and lives in state lanes. Implementation
identifiers must namespace them (`mcc_alpha` / `mcc_gamma`).

**F_ops.** Per pair: decode tuples from bits; enforce the timestamp
rules (D7); update `h_RS`, `h_WS` in K; enforce nonzero fingerprints;
count pairs. Carries the same advice slice layout as `F_exec` (its
`c_adv` is what binding check 1 consumes).

**F_scan.** Per address-ordered chunk: enforce `a_IS == a_FS`
positionwise; update `h_IS`, `h_FS`; count. IS advice for segment 1 is
bound to the public program image digest; for k>1 to the carried
boundary (§7). FS advice is the slice pre-committed at `F_chal`.

Address canonicity is inherited, not enforced per row. Nebula's
codebox asserts `a = a′ = i`; we, like zkEngine (`mcc/mod.rs:202-208`),
check only `a_IS == a_FS`, so canonicity rests on induction: the
genesis digest **must commit the full `(a, v, t)` tuples in canonical
address order — addresses included**. A value-only genesis digest
would admit an IS with one address duplicated and another missing
(consistently mirrored in FS), breaking sequential consistency for the
missing address. With the genesis object canonical, positionwise
equality makes FS canonical, and §7's chunk equality carries
canonicity to every later segment's IS.

**F_segment.** The segment close. Enforces, in-circuit over state:

1. `h_IS · h_WS == h_RS · h_FS` (K-arithmetic).
2. Phase counters consistent: `n_exec == n_ops`, scan chunks
   `== ⌈M/B_scan⌉`, all product lanes were seeded to 1 at `F_chal`.
3. ts watermark advanced exactly `B_ops · n_exec` since segment open
   (§2 unit: one increment per read-write pair); new watermark written
   to state.
4. Boundary advance: memory boundary digest ← `D_FS_adv` — the FS
   chunk commitment list is hashed **once, natively**, absorbed at
   `F_chal`, and replay-recomputed; `F_segment` copies that state lane
   into the boundary-digest lane (one equality row, no in-circuit
   re-hash of the list). Segment index += 1; product/challenge lanes
   cleared for the next segment. The next segment's `D_IS_bind` is
   this boundary digest (§6 `F_chal`, §7).

Replay-level (audit/decider) checks for the segment: binding checks
1–3 of §5 and recomputation of `D_exec_adv`, `D_FS_adv`, and the
`F_chal` transcript. Together these are the SuperNeo equivalent of
zkEngine's four Layer-1 verifier checks plus its Layer-2 shard binding.

---

## 7. Segment boundary and long-running state

There is no Merkle root. The memory state at a segment boundary **is**
the canonical address-ordered list of FS chunk advice commitments
`[c_adv^FS(k, 1..⌈M/B_scan⌉)]`, compressed for state-carrying into one
Poseidon2 digest (the "memory boundary digest"). Threading:

- `IS_{k+1}` advice must satisfy `c_adv^IS(k+1, i) == c_adv^FS(k, i)`
  per chunk (replay-level check; the identical slice layout makes this
  meaningful). Timestamps inside FS tuples are last-write times under
  the global counter, so continuation needs no re-timestamping.
- Segment 1: `IS` bound to the public program image (genesis digest in
  the public statement).
- The boundary digest, segment index, and ts watermark ride in
  `StateIn/StateOut` and are absorbed by the existing `state_x_out`
  public binding, so segment replay/reordering is excluded by the same
  chain discipline that orders ordinary F′ steps.

The segment public image (what `F_segment` exposes via boundary bits):
segment index; step range `[n_start, n_end)`; ts watermark in/out;
boundary digest in/out; `α, γ`; the four product lanes; `D_exec_adv`;
`D_FS_adv`. Field layout is part of the canonical plan, pinned by
layout-budget tests like every other F′ region — it is not an
implementation detail.

---

## 8. Attacks → mechanisms → tests

Repo discipline: each row lands with a red-team test that fails if the
mechanism is removed.

| Attack | Mechanism | Test (new, `neo-fold-clean/tests/system/`) |
|---|---|---|
| Mutate tuples between exec and ops | §5 check 1 (c_adv pairing) | flip one tuple bit in ops advice; audit must reject |
| Choose FS after seeing α, γ | FS pre-commitment absorbed at `F_chal` | re-derive challenges without `D_FS_adv`; forge FS; must reject |
| Prover-supplied challenges | `F_chal` transcript replayed by audit/decider | tamper α lanes in state; must reject |
| Hide real ops in padding | D6 + in-circuit pair count | drop one pad pair / emit B_ops+1; structure unsatisfiable |
| Timestamp wraparound / reuse | 64-bit decomposed lt/eq + watermark + `N_max` | rt ≥ gts tuple; must trip the comparison row |
| Memory reset between segments | §7 chunk equality + boundary digest in x_out chain | thread mismatched IS; replay must reject |
| Segment replay / reorder | segment index + watermark in state chain | duplicate a finalized segment; chain digest mismatch |
| Zero fingerprint zeroing a product | nonzero-inverse rows | craft tuple with fp = 0; row unsatisfiable |
| Carry data through inactive branches | counter-derived selectors gate every branch row | activate two branches at once; selector link must trip |
| Self-consistent re-digest of boundary | digests recomputed at replay, never trusted (repo security rule) | mutate boundary digest + re-digest; audit must reject |
| Drop `c_adv` from a digest surface | §5 lane lifecycle item 3 (projection digest, pre-ρ outputs digest, `AccumulatorHandle`) | omit the lane from one digest surface; the transcript/handle value must change and replay must reject |
| Genesis IS with duplicated/missing address | §6 address-canonicity induction (genesis digest commits address-ordered `(a, v, t)` tuples) | craft segment-1 IS with address `a` twice and `a′` absent, mirrored in FS; genesis binding must reject |

---

## 9. Cost probes (before building)

Using existing tooling (layout-budget tests, shape probes,
`perf_fibonacci_bits` harness):

1. Union-structure probe (D3 gate, §4): rows/width/prover-time of
   `S_nebula` stubs vs exec-only at candidate `B_ops ∈ {4, 8, 16}`,
   `B_scan ∈ {32, 64}`.
2. Per-op row cost of the `F_ops` body (target: ≤ ~4 K-muls + 2
   comparisons + bit rows per pair ≈ 3–4k image bits/op).
3. `F_chal` absorption budget — confirm the §6 preimage fits one
   one-shot trace at acceptable cost.
4. Scan cost per segment: `⌈M/B_scan⌉` steps; pick a maximum supported
   `M` for v1 and state it in the plan (large `M` makes per-segment
   scans the dominant cost; address-space sparsification is future
   work). Note the scan's real cost driver: each `F_scan` step pays the
   full per-step F′ shell (~134k image bits) regardless of how cheap
   the branch body is, and the step count scales with `M`, not the
   trace. The probe must measure scan-phase prover time at the v1
   maximum `M`; if it dominates, the D3 fallback (separate scan chain,
   cross-bound at the decider) is the relief valve.

## 10. Build sequence

1. Close the two council items: the ε-budget statement (§3) and any
   objection to replay-level binding checks (D8, §5). One council run,
   both questions.
2. Land the cost probes (§9); confirm or revise D3.
3. Plan/layout: extend the canonical plan with the new state lanes and
   branch regions; layout-budget + row-shape tests first (they pin the
   ABI).
4. `F_exec` with tuple emission, padding, ts watermark + its red-team
   rows; advice-slice commitment lane through Π_RLC/Π_DEC and the
   terminal slice check (audit mode).
5. `F_chal` + `F_ops` + the §8 tests that exist at this layer; segment
   of one chunk proves end-to-end in audit mode.
6. `F_scan` + `F_segment` + boundary threading; multi-segment chains in
   audit mode.
7. Decider parity: every replay-level check of §5–§7 added to the
   decider's constraint surface; red-team: each check removed from the
   decider must be caught by its test.
8. Only then: compact/on-chain exposure and the O(1)-terminal
   accumulation design (D8 follow-up).

Native host verification exists only inside tests as bring-up scaffolding;
it is never a production code path.

---

## 11. v1 critical questions — closed

| Question (v1) | Answer |
|---|---|
| Is the memory check portable, or are there hidden group-op assumptions? | Portable. The check is field/K arithmetic (§2). The group machinery lives in Nova's transport, replaced by D2/§5. |
| Is the hard part the second layer for continuing IVC? | The hard parts are transport binding (§5) and challenge discipline (§6). Layer 2 itself collapses to "more steps in the same chain" under D3 — audit-path-only until the D8 follow-up lands (§5). |
| Is the F′ accumulator already commitment-carrying? | No. The folded accumulator binds history only through extraction; transport needs the dedicated `c_adv` lane (§5). v1's conclusion was right; its row-root mechanism was wrong. |
| What exactly is committed? | The tuple/FS advice slices only, under a dedicated key `A_adv` — never the whole CCS witness (§5). |
| How are rows fixed before α, γ? | All exec `c_adv` are folded and digested, and FS slices pre-committed, before `F_chal` samples (§6). |
| Can Layer 2 be avoided by extending the input circuit? | No — challenge timing forces finalization regardless of where the boundary sits. Unchanged from v1; now with the mechanism specified. Single-shot designs (e.g. Coral, ePrint 2025/1420) accept the consequence: commit the whole trace up front, derive challenges, fold once, never resume. Our segments (§7) exist precisely to keep resuming. |
| What architecture shares one accumulator? | The universal branch relation `S_nebula` (§4), with a measured fallback (D3). |
| How is low-norm preserved? | Bit-backed tuples/lanes, K-products via existing kmul gadgets (§3). |
| What chunking model? | Fixed `B_ops` per exec chunk, separate `B_scan` for scans, Nightstream-owned globally; short calls pad (D6). |
| What transfers from zkEngine? | The schedule, the rw-pair discipline, padding policy, the four verifier checks, sharding's `C_FS == C_IS` idea. Not the proof substrate. |
| How does the flow yield CE claims? | Every step is an `S_nebula` CCS instance through the unchanged lifecycle; products/challenges ride in state, not in side proofs (§4). |
| Where does final authority come from? | The existing terminal CE/decider over the one accumulator, extended with the slice opening and replay checks of §5–§7. |

---

## 12. v2 review questions — answered

**Q1: Does this implement Nebula's Layer 2, so we follow the
protocol's best practices?**

Yes by function, not by shape — every obligation of the paper's
`F_final` (Nebula §4.3) has a designated home, and the two purposes
Layer 2 serves in the paper are split as follows.

| `F_final` obligation (paper) | Where it lives here | In-circuit or replay |
|---|---|---|
| 1. Parse finalized proofs' public IO | Segment public image (§7), exposed via boundary bits | in-circuit (state lanes) |
| 2. `z′_i = z_i`, `ts′_i = ts_i` (continuity) | x_out state chain: each step's state-in is hash-bound to the prior x_out; the ts watermark is a state lane (§4, §7) | in-circuit |
| 3. `C_mem_i = C_IS` (memory continuity) | Boundary digest threaded in state (§7) + per-chunk `c_adv^FS(k) == c_adv^IS(k+1)` equality (§5 check 3) | digest threading in-circuit; chunk equalities replay (D8) |
| 4. `γ`s derived by hashing the four set commitments | `F_chal` in-circuit absorption of (`acc_digest`, `D_exec_adv`, `D_FS_adv`, `D_IS_bind`); the digests are replay-recomputed public inputs (§6) | hybrid |
| 5. `h_IS · h_WS == h_RS · h_FS` (and seeds = 1) | `F_segment` checks 1–2 (§6) | in-circuit |
| 6. Fold `(U_F, U_ops, U_scan)` into running `U` | Continuous: every step of every phase is one `S_nebula` instance folded by the unchanged `Π_CCS → Π_RLC → Π_DEC` chain into the one accumulator. The paper needs a dedicated fold circuit because its three machines live in three IVC chains with separate keys; under D3 there is one structure and one chain, so "fold the segment in" is the chain continuing | in-circuit (the existing NIFS) |
| 7. Forward `(i+n, z_0, z, ts, C_mem = C_FS, U)` | x_out chain + segment public image; boundary digest ← FS commitment list (§7) | in-circuit |

The deeper purpose of Layer 2 — *containing challenge-dependence so
the chain stays incrementally updateable* — is preserved: ops/scan
phases are challenge-dependent only inside their segment, `F_segment`
clears the challenge/product lanes, and the next segment opens fresh.
Two scope limits, both deliberate: (a) until D8 lands, the replay-level
rows in the table make Nebula chains audit-path-only — pause/resume
works, history discard (the paper's Figure-1 compression) arrives with
the D8 follow-up; (b) zkEngine's Layer-2 *sharding* (parallel segment
proving glued by `C_FS == C_IS`) is explicitly not ported — segments
here are sequential chain phases. Neither limit weakens soundness; both
are stated where they bind (§1, §5).

**Q2: How does the main Π_CCS mix with the new CE surface, and why is
that correct?**

Three separate mechanisms, each with its own correctness argument —
and it matters that they are not conflated:

1. **`S_nebula` instances fold because Π_CCS is structure-generic.**
   SuperNeo Theorem 1 (`Π_DEC ∘ Π_RLC ∘ Π_CCS : CCS(b,ℒ)^K ×
   CE(b,ℒ)^k → CE(b,ℒ)^k`) is proven for an arbitrary structure `s`
   (Definition 11); branch selectors, K-mul rows, fingerprint products
   are just rows and monomials of `f` and `M_j`. The one typing
   requirement is HyperNova's `compat(s₁, s₂) ⇔ s₁ = s₂`: one
   accumulator demands one structure. That is what D3's universal
   relation provides — D3 is a *well-typedness* requirement for
   single-accumulator folding, not merely a cost amortization.
2. **`c_adv` mixes as the second coordinate of one commitment.** The
   normative story is §5's `ℒ′(z) := (A·z, A_adv·slice(z))`: the
   paper's relations and all three reduction proofs are stated for an
   arbitrary `R_F`-module homomorphism, so they apply to `ℒ′`
   verbatim. Π_CCS itself never computes over commitments — it binds
   them in the transcript and passes them through (`out.c_adv ==
   in.c_adv`, the same passthrough check as `c`); Π_RLC mixes both
   coordinates with the same rot(ρ) action; Π_DEC recomposes both with
   the same b-powers (slicing commutes with digit splitting, so
   `Σ b^{i−1}·A_adv·slice(z_i) = A_adv·slice(z)`); the terminal opens
   both against the same `z*`. The §5 lane-lifecycle items are exactly
   these obligations.
3. **The MCC state never mixes through Π_CCS at all.** Challenges,
   product lanes, watermark, and counters are bit-backed *witness
   lanes* with step-to-step continuity enforced by the x_out hash
   chain — not CE claims. Π_RLC's random combination acts on whole
   step witnesses to produce an *evaluation* claim; it never averages
   two steps' challenges or products into a semantic value. Semantic
   meaning lives only in (a) each step's CCS satisfaction, delivered
   by extraction through the fold chain, and (b) the public x_out
   ordering — the same discipline that already carries `chunk_count`
   and `acc_digest`. This is why no second CE family, no special
   "memory claim" type, and no change to the Π_CCS sum-check are
   needed.
