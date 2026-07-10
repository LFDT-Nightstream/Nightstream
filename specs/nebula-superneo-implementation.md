# Nebula on SuperNeo: Memory-Checking Integration Spec (v3)

Status: normative architecture and implementation spec for the
`neo-fold-clean` Nebula memory-checking integration.

Trust status: **architecture-sound and testable under native F′
enforcement; production security is gated on the F′ R1CS / compressed
decider enforcing the §6.3 `NebulaLane` transition** (§13 step 9 — the
same gate, and the same closure, as F′ itself). Until a VM frontend
attaches program semantics (§4), external claims must be scoped to "this
advice stream is a consistent RAM/ROM history under this plan", never
"this program executed".

Security companion: [`nebula-superneo-security-note.md`](./nebula-superneo-security-note.md)
— states and proves the lemmas §5.3 and §9 rely on, and holds the claims
ledger with dispositions.

Scope: one complete protocol target — a RAM sideline for the SuperNeo folding
chain, paper-faithful to Nebula's offline memory checking (`IS ∪ WS == RS ∪
FS` with public-coin fingerprints), carried as **commitment-carrying IVC
state inside F′** so the proof stays constant-size and incrementally
updatable. Native replay may exist as a test oracle; it is never verifier
authority.

Papers, local copies:
- Nebula (`docs/nebula-paper/`): §3 CC-IVC, §4 memory checking, §4.4 costs.
- SuperNeo (`docs/superneo-paper/`): §2 pipeline Π_CCS ∘ Π_RLC ∘ Π_DEC, D6
  recursion-overhead discussion.
- **Coral (`docs/Coral.pdf.md`): §5 segmented/foldable memory, Appendix D
  (fingerprint soundness, packing), Appendix E.1 (split-witness relaxed
  R1CS).** Coral is an independent, implemented instantiation of Nebula's
  memory checker inside Nova; v3 adopts three of its constructions
  explicitly (interleaved scan, split-witness commitments, packed
  fingerprints) and cites it per section.

---

## 0. Changelog

### v2 → v3 (review: Enzo)

| # | review finding | v3 resolution |
|---|---|---|
| 1 | v2 anchored all cross-step binding (continuity, γ derivation, product equation, commitment pairing) in "verifier rules" over the per-instance public data — i.e., a linear replay verifier. That defers the actual problem: the IVC version needs the carried commitment folded/hashed along the chain, or we are back to a non-succinct, non-incremental proof. | The memory carry is now **commitment-carrying IVC state inside F′** (§6): a constant-size `NebulaLane` in the F′ `State`, absorbed into `state_x_out` every step, with the incremental-commitment hashing (Nebula Construction 2's `C_i ← H(C_{i−1}, c_ωᵢ)`) realized as F′-side Poseidon2 digest chains over the per-step lane commitments. Same trust path as NIFS challenges: enforced natively by the lifecycle today, by the F′ R1CS/decider as that lands — but the *state* is constant-size by construction, so incrementality and succinctness are structural, not deferred. §6.5 spells out the per-step absorb inventory and its cost class. |
| 2 | Memory-tuple commitments and the main witness commitment should be **disjoint** — v2 committed the lane region twice (once inside `c` over the full `z`, once under `A_ops/A_mem`). The decider can put disjoint pieces together. | Adopted at the level that matters for binding and boundary equality (§5): each memory lane is bound **once under its own matrix** (`c_ops`, `c_is`, `c_fs` — IS/FS share `A_mem`, which is load-bearing for the boundary check), and the terminal decider "puts them together" by opening each against its slice of `z_fin`. v3 deliberately keeps the engine's full-`z` commitment `c` unchanged beside them, so lane coordinates are covered twice — a bounded, lane-only overhead (≈ 30% extra native pay-per-bit commit work at targets, zero circuit rows) chosen over reshaping the engine's committed relation, because the binding lemma's extraction anchor (security-note A1) is defined over full-`z` `c`. The two zero-duplication variants are recorded as deferred upgrades with named costs (§5.1). |
| 3 | Single universal circuit: how do padding and step sizes work, especially for scan? Wants trade-off analysis vs two circuits. | v3 drops the v2 switchboard entirely and adopts **Coral's interleaved scan** (Coral §5, Fig. 6 caption): every step is one uniform circuit containing `B_ops` memory-op slots *and* `B_scan` scan slots; the scan of `IS/FS` is spread across the segment's steps. One relation, no phase tags, no switch constraints, matches the engine's ℓ = 1 reality. §8 gives the requested trade-off analysis (interleaved vs switchboard vs two relations) with formulas. |
| 4 | Public ROM is required — the verifier needs a handle to the program table ("the trace comes from this program"), and the handle must not force the verifier to recompute per-segment fingerprints or mix with private RAM. | §7: memory is **segmented** (Coral §5.2) into public-ROM and RAM namespaces via a segment bit in the tuple and fingerprint. The verifier's ROM handle is the **plan-bound, γ-independent** digest `D_init` over the initial-memory lane commitments, recomputable by anyone from the public ROM image — no per-segment fingerprint work (that alternative, Coral's verifier-computed `h_IS_pub`, is documented with its trade-off and rejected as default precisely because of the multiple-γ-per-chain issue). Private initial memory is deferred; initial state is fully public (ROM + zeros). |

Terminology note: "D8" in older discussion was the v1 spec's decision-ID for
a positioned SIS/Ajtai sequence accumulator. That construction is retired
(see v1 → v2 below); the term does not appear in the normative text.

### v3 amendments (post-initial-draft, for re-reviewers)

| wave | what changed |
|---|---|
| Security note landed | Lemmas 1–3 written; **L-ALIGN** added to §5.1 (lanes must be whole ring columns — surfaced by Lemma 1's proof); §9 formula aligned to the note. |
| Fresh-eyes pass | Exact cover `N·B_scan = R+M` (the draft's targets violated its own `≥` constraint, and "structural scan pads" were impossible in a step-uniform structure); fingerprint switched to the global cell index `g = addr + seg·R` (kills namespace branching); §5 rewritten honestly — `c` stays full-`z`, lanes are a parallel `adv` tuple (≈30% lane-only double-commit), `c_app` removed; §6.3 gained `base`/`open_segment`; `x` total corrected to 1,400. |
| Cost pass (review: Enzo) | Per-lane `D_seen` chains (each commitment absorbed once — the draft double-absorbed 4,860 elems/step vs §6.5's claimed 2,916); γ transcript now absorbs the three `D_pre` digests (12 elements) instead of the raw list; §6.5 gained measured-anchored F′-R1CS numbers (≈135–180k/step) and the considered-and-rejected SIS-accumulator analysis. |
| Hardening pass (second review) | Trust-status banner + scope rule; §5.2 R1 absorb-site inventory with per-site negative tests; §13 step 9 obligations enumerated; §15 criteria 6 (named non-author reviews) and 7 (lane-residency statically enforced at `S_mem` build). |
| SIS/cost review (review: Enzo, v4 proposal + evaluation) | **L0a leaf digests** adopted (§6.1/§6.3/§6.5): each lane commitment crosses Poseidon2 exactly once per step; chains and transcript sites absorb 4-element leaves. **L0b** made explicit (§6.2): `D_pre` is prover-claimed at open, authority is the close equality. §6.5 re-priced **dual-regime** (bit-backed folded vs Spartan field-native — gated on `enc(F′)`, `paper/f_prime/encoding.md`); the SIS-accumulator analysis corrected in both directions and re-dispositioned to §14 (sound, but digit bases `w > 1` are unbuildable under the `b = 2` NC range check; only live in the folded regime). §5.2 R1 site list corrected (the F′ chunk digest is shape-only, absorbs no commitments — removed with rationale). §9 aligned to the note's FS-lifted formula. §4.4 limb-canonicality note; §13 step 3 cross-crate scope note. |
| External review pass (second reviewer) | **Completeness bug fixed:** the IS/FS chains were lane-typed (`"is"`/`"fs"` tags), so honest cross-segment continuity `D_seen[is] == D_mem` could never hold — is/fs now share one mem-domain leaf/link tag pair and header (§6.1/§6.3), `D_init` uses the identical formula (§7), and the §12 swap row's rejection point moved to Lemma 1 slice binding. **Finalization rule added** (§6.3): externally accepted proofs must end at a closed segment. Lifecycle guards (`advance_nebula` requires open segment); `D_pre` thin-air claims named as preimage-resistance (not just CR); plan rule `r ≤ μ` (§2); `n_in`/`m_seg` notation fixes in §9 (fold-arity vs field `K`; the `2n` double-count); prover-vs-verifier resume bundles (§6.4); `adv` shape invariant (§5.1); acceptance tests hardened (§13 steps 3–5). |
| **v3.1: stacks** (§14 → normative) | Coral §5.2's `check_push`/`check_pop` port: up to `S ≤ 2` **segment-local** stacks as extra memory namespaces. Namespace selection becomes **one-hot selector bits** (`ram`, `stk_0`, `stk_1`; ROM = none set) so the global index `g` stays *linear in lane bits* and v3 is the exact `S = 0` special case (byte-identical lanes and `x`). Pushes emit WS only (no `rt` check), pops emit RS only (with the E4 timestamp check); stacks never enter IS/FS or the scan; the per-segment product equation then forces stack-WS = stack-RS. **Per-segment γ forces segment locality** (§3.1): a push under segment k's γ cannot cancel a pop under segment k+1's γ, so every push is popped in its segment and `sp = 0` at every close (new §6.3 check). Stack bounds are bitness-pure: the running `sp` is a σ-bit word, so pop-at-empty and push-at-full are unrepresentable (capacity `2^σ − 1`). New rows E10–E14, `x` gains per-stack `sp_in`/`sp_out`, `NebulaLane` carries `sp`, plan v3.1 binds `S`/`σ`. Security note gains Lemma 4 (stack discipline, reduction to Blum et al. / Coral App. E). |
| Road A relation audit (2026-07-09) | The 14M-bit projection shell was reclassified as a cost prototype: shipped encoders do not fill it, and its K-mul slots lack semantic rows. The authoritative NIFS.V circuit now projection-checks the complete `c + adv` product commitment on transcript-bound `q`/`β` wires through PiCCS/PiRLC/PiDEC, with terminal slice openings. Paper re-read fixed the timing: as in HyperNova Construction 2 step 4(d) and Nebula Construction 2 step 4, F′ consumes the **previous** fresh claim's public memory suffix and witness commitment, not the claim it is currently producing. The authoritative recursive relation now enforces that delayed `NebulaLane` transition. Remaining production work is the current `S_mem` application relation, fixed-shape low-norm lowering, and the terminal delayed transition; no second verifier shell. Lemma 5's full production census is `P=1,275`, batched `J=85`. |

### v1 → v2 (condensed)

- Positioned SIS sequence accumulator removed: position-dependent matrices
  cannot be constants of the single folded structure; in-circuit Ajtai and
  bit-limbed 972-coefficient handles are 1–2 orders of magnitude too
  expensive. Replaced by lane commitments checked through the folding
  pipeline (now, in v3, the SW-CCS tuple of §5).
- Global timestamp counter across segments (Nebula `F_final`'s `ts'_i =
  ts_i` check) restored; per-segment reset breaks completeness.
- Pad gating, lane-residency rule, product initialization, fingerprint
  packing widths all made explicit.
- Poseidon2 remains the only hash family; Ajtai commitments are the
  scheme's native commitment family; no exceptions needed.

---

## 1. Protocol shape

Nebula's memory theorem (paper §4.2, Lemma 7 / Corollary 8): a batch of
memory operations is sequentially consistent iff there exists a final state
`FS` with

```text
IS ∪ WS == RS ∪ FS        (multisets of (namespace, timestamp, addr, value) tuples)
```

checked via fingerprints over `K = F_q²` with challenges sampled **after**
the prover is committed to all four multisets:

```text
h_IS · h_WS == h_RS · h_FS      where h_S = ∏_{e ∈ S} f_γ(e)
```

The integration proves this per **segment window** of the folding chain:

```text
segment k = N consecutive steps of one uniform circuit S_mem
  step i:  B_ops memory-op slots      (emit RS/WS, fingerprint them)
         + B_scan scan slots           (sweep IS/FS cells, fingerprint them)
  with N · B_scan = R + M   (exact cover: full memory swept exactly once)
```

Trust anchors:

1. **Folding + terminal decider** (existing pipeline): every step instance
   satisfies `S_mem`; the split-witness commitment components open against
   their slices of the final folded witness (§5).
2. **Lane commitments** (§5): MSIS-binding commitments to the op and scan
   lanes under dedicated matrices, fixed before γ; the engine's full-`z`
   commitment `c` stays unchanged beside them.
3. **F′ commitment-carrying lane** (§6): constant-size carried state —
   timestamps, running products, challenges, and Poseidon2 digest chains
   over the lane commitments (via per-commitment leaf digests, §6.1) —
   updated and checked by the F′ step function
   every step and absorbed into the F′ state hash. This is Nebula
   Construction 2 with the incremental commitment realized on the same
   path that carries `acc_digest` today.

Prover flow per segment (two passes, exactly Coral §5.1's discipline):
execute natively → build all lanes (ops + interleaved IS/FS scan) →
compute lane commitments → absorb them into the chain transcript → squeeze
`γ1, γ2` → build instances (products need γ) → fold via `extend`.

There is no in-circuit hashing of commitments inside `S_mem`, no in-circuit
Ajtai anywhere, and no linear memory-specific replay at verification: the
memory verifier work is (a) the F′ lane transitions, constant per step, and
(b) three decider opening checks, once per proof.

---

## 2. Parameters and constants

All normative; the plan artifact (§11) records them; changing any changes
the plan digest.

| Symbol | v3 target | Meaning / constraint |
|---|---|---|
| `F` | Goldilocks, `q = 2^64 − 2^32 + 1` | base field (`neo_math::F`) |
| `K` | `F_q²` (`neo_math::K`) | fingerprint/challenge field, `|K| ≈ 2^128` — the extension degree SuperNeo itself uses for sum-check soundness |
| `b` | 2 | witness ∞-norm bound (`neo-params` Goldilocks preset). Every committed coordinate in this design is a bit; words exist only as row-side linear recombinations. |
| `κ, d` | 18, 54 | Ajtai module dims (Goldilocks preset) |
| selectors | `1 + S` bits | memory namespace, **one-hot** (v3.1): `ram`, then one `stk_s` bit per stack; public ROM = no selector set. One-hot (not binary SEG) is what keeps the global index `g` linear in lane bits (§4.3) with no aux bit in the packed prefix; Coral §5.2's segment descriptor, adapted. `S = 0` reproduces v3 exactly (the lone `ram` bit is v3's `seg`). |
| `R` | `2^r`, target `2^12` | public-ROM cells (namespace addresses `[0, R)`) |
| `M` | `2^μ`, target `2^16` | RAM cells (namespace addresses `[0, M)`) |
| `S` | 0–2, target 2 | number of stacks (v3.1). Each stack is its own namespace; stacks are **segment-local** (§3.1) and never scanned. |
| `σ` | target 12 | stack-pointer width; stack capacity is `2^σ − 1` cells (bitness-pure bounds, §4.1 E12). Plan validity: `1 ≤ σ ≤ μ` when `S > 0`. |
| `VAL_BITS` | 32 | one cell = one 32-bit word (`ℓ = 1` value lane) |
| `TS_BITS` | 44 | timestamp width; plan enforces `SEG_MAX · N · B_ops < 2^TS_BITS` |
| `B_ops` | 64 | op slots per step |
| `B_scan` | 64 | scan slots per step |
| `N` | 1,088 | steps per segment; `N = (R + M) / B_scan` **exactly** (exact cover; divisibility is automatic for powers of two with `B_scan ≤ min(R, M)`). Longer segments are obtained by enlarging `M` with untouched cells — never by padding the scan (§3.3). |
| `SEG_MAX` | `2^16` | max segments per chain |
| `DEG_MAX` | 4 | max CCS degree of `S_mem` (product-update rows; engine already runs degree-7 structures) |

Packing (§4.3): fingerprints use the global cell index
`g = addr + ram·R + Σ_s stk_s·(R + M + s·2^σ)` — for `S = 0` this is v3's
`g = addr + seg·R < R + M ≤ 2^17`; `packed(t, g) = t + 2^TS_BITS · g`,
width `44 + 17 = 61 < 63` bits at v3 targets — no Goldilocks overflow.
Plan validity requires `TS_BITS + log2(R + M + S·2^σ) ≤ 62` (the address
space now includes the stack namespaces; the *scan* domain stays `R + M`),
and **`r ≤ μ`**
(external-review fix): the ops-lane `addr` is `max(r, μ)` bits and only
ROM addresses are range-gated (E6), so `r ≤ μ` is what makes RAM bitness
alone bound `addr < M`. A future plan wanting `r > μ` must add the
symmetric RAM gate (`ram_j · addr_bit_k = 0` for `k ∈ [μ, max)`) — not
part of v3.

Test profile (5-minute cap): `r = 4, μ = 8, B_ops = B_scan = 8, N = 34`
(`S = 0`); stack variant adds `S = 2, σ = 4`.

---

## 3. Memory model and lane encodings

### 3.1 Address space and tuples

A memory cell is identified by `(namespace, addr)`: public ROM
(`addr < R`), RAM (`addr < M`), or stack `s ∈ [0, S)` (`addr < 2^σ`,
v3.1). A memory tuple is `(namespace, t, addr, v)`. The **global cell
index**

```text
g = addr + ram·R + Σ_s stk_s·(R + M + s·2^σ)
```

linearizes ROM first, then RAM, then the stack namespaces, giving each a
disjoint `g`-range. Scan position `p ∈ [0, R + M)` sweeps ROM and RAM
only — **stacks are never scanned** (Coral §5.2 / Blum et al.: a stack's
IS and FS are structurally empty). Fingerprints use `g` directly (§4.3):
on the exec side `g` is linear in lane bits (one-hot selectors — this is
why the encoding is one-hot); on the scan side `g = p` is linear in the
public step index — no in-circuit branching on the namespace anywhere.

**Stack discipline is segment-local (normative, v3.1).** Every push is
popped within its own segment; `sp = 0` at every segment open and close
(§6.3). This is forced by per-segment γ, not chosen: with stacks excluded
from IS/FS, the segment-close product equation can only balance if the
segment's stack-WS equals its stack-RS as multisets, and a push
fingerprinted under segment k's γ can never cancel a pop fingerprinted
under segment k+1's γ. (Coral's stacks are execution-global only because
its challenge is execution-global; our γ is per-segment by design —
streaming with bounded prover memory.) A VM frontend wanting a stack that
survives a segment boundary spills it to RAM at the boundary, or places
segment boundaries at stack-empty points.

Initial state (chain start): every cell has `t = 0`; ROM cells hold the
plan's ROM image; RAM cells hold 0; stacks are empty (`sp = 0`,
represented by *no* tuples anywhere — not by zero-valued cells). The
initial state is fully public (private preloads: deferred, §14).

### 3.2 Ops lane

`B_ops` slots per step. Slot fields (all committed bits, bitness-checked):

| field | bits | meaning |
|---|---|---|
| `pad` | 1 | 1 = no memory op in this slot |
| `is_write` | 1 | 0 = read, 1 = write; on a stack namespace: 1 = push, 0 = pop |
| `ram` | 1 | RAM selector (v3's `seg` bit) |
| `stk_s` | S | one selector bit per stack (v3.1); one-hot with `ram` (E10), ROM = none set |
| `addr` | max(r, μ) = 16 | address within namespace; on a stack: constrained to the running `sp` (E13) |
| `v_r` | 32 | value read (writes: old value; pops: popped value; pushes: constrained 0) |
| `v_w` | 32 | value written back (reads/pops: constrained `v_w = v_r`; pushes: pushed value) |
| `rt` | 44 | prover-supplied timestamp of the previous access; pops: the push time; pushes: constrained 0 |

`OP_BITS = 127 + S`. Derived tuples (row-side expressions, never
materialized) — a RAM/ROM op emits both, a **push emits WS only, a pop
emits RS only** (Coral Fig. 7):

```text
RS_j = (g_j, rt_j, v_r_j)      emitted unless push        (skip_rs = pad + Σ_s sw_s)
WS_j = (g_j, wt_j, v_w_j)      emitted unless pop         (skip_ws = pad + Σ_s (stk_s − sw_s))
wt_j = ts_in + cnt_j
cnt_j = Σ_{i ≤ j} (1 − pad_i)             (running non-pad count; stack ops count)
sw_s_j = stk_s_j · is_write_j             (aux "push to stack s" bit, E11)
sp_s_j = sp_s_{j−1} + 2·sw_s_j − stk_s_j  (running stack pointer, σ-bit aux word,
                                           E12; sp_s_{−1} = x.sp_s_in)
```

**Lane-residency rule (normative).** Every witness bit that determines a
slot's contribution to any multiset (`pad`, `is_write`, the selector
bits, `addr`, `v_r`, `v_w`, `rt`) must live inside the committed lane.
Auxiliary witnesses (`cnt`, `sw_s`, `sp_s`, comparison borrows,
running-product bits) may live outside it only if constrained as
deterministic functions of lane bits and public `x` — `sw_s` (which
gates the E8/E9 skip terms) is pinned by E11 exactly as `cnt` is by E2.
A free interpretation bit outside the lane re-opens the post-challenge
selection attack (§12 has the test that documents it).

### 3.3 Scan lanes (IS and FS)

Each step also carries two lanes of `B_scan` slots (identical layout):

| field | bits | meaning |
|---|---|---|
| `v` | 32 | cell value |
| `t` | 44 | cell timestamp |

`CELL_BITS = 76`. Slot `j` of step `i` (within segment k) refers to scan
position `p = i · B_scan + j`; its global index `g = p` — and hence the
fingerprint's packed prefix — is a linear expression in public `x.idx` plus
the slot constant, which makes the sweep canonical-by-address with zero
comparison rows and no namespace branching. There are **no scan pads**:
exact cover (`N·B_scan = R + M`, §2) makes every scan slot a real cell. A
step-uniform structure could not host position-dependent pad rows anyway —
pad-ness would depend on `x.idx`, not on the structure — which is why exact
cover is normative and "more room per segment" is achieved by enlarging `M`
with untouched cells, whose IS and FS factors cancel.

The IS lane of a step in segment k and the FS lane of the corresponding
step in segment k describe the same positions; segment-boundary continuity
is commitment equality between segment k's FS lanes and segment k+1's IS
lanes (§6.4).

---

## 4. The step circuit `S_mem`

One uniform circuit; no phases, no switches. It contains the op block, the
scan block, and the boundary rows. A VM frontend later attaches its program
semantics over the same witness, reading the ops lane as its memory port
(the port contract is §3.2 + the rows below; a frontend adds rows, never
relaxes these). v1 ships portless: the chain proves "this advice stream is
a consistent RAM/ROM history".

### 4.1 Op block (per slot j)

With `rom_j = 1 − ram_j − Σ_s stk_s_j` (the ROM selector: linear, and a
bit given E1 + E10), `skip_rs_j = pad_j + Σ_s sw_s_j`, and
`skip_ws_j = pad_j + Σ_s (stk_s_j − sw_s_j)` (each a bit: `pad` and the
stack selectors are mutually exclusive via E7). For `S = 0` every gate
below reduces literally to its v3 form.

| # | constraint | deg | rows |
|---|---|---|---|
| E1 | bitness of lane bits, aux bits (`diff`, `cnt`, `sp_s`), and per-slot running-product bits (2 × 128) | 2 | ≈ OP_BITS + 51 + S·σ + 256 |
| E2 | `cnt_j = cnt_{j−1} + (1 − pad_j)` (7-bit recombinations, `cnt_{−1} = 0`) | 1 | 1 |
| E3 | read consistency: `(1 − is_write_j)·(v_w_j − v_r_j) = 0` (covers pops: `v_w = v_r =` popped value) | 2 | 1 |
| E4 | timestamp order, gated: `(1 − skip_rs_j)·(wt_j − rt_j − 1 − diff_j) = 0`, `diff_j` a fresh 44-bit recombination. Applies to every RS-emitting op — RAM/ROM ops *and pops* (Coral's `assert(push_time < ts)`); pushes are exempt (Coral §5.2: no `rt` check on push). | 2 | 1 |
| E5 | ROM write ban: `is_write_j · rom_j = 0` (one row — the selector replaces v2's address-range AND-tree; Coral §5.2's "without costly range checks") | 2 | 1 |
| E6 | ROM address bound, gated: `rom_j · addr_bit_k = 0` for `k ∈ [r, 16)` | 2 | 16 − r |
| E7 | pad canonicality: `pad_j · w = 0` for `w ∈ {is_write, ram, stk_0.., addr, v_r, v_w, rt}` (word-level) | 2 | 6 + S |
| E8 | RS product update: `h_rs_j = h_rs_{j−1} · g_rs_j` in K (two component rows), `g_j = skip_rs_j + (1 − skip_rs_j) · f_γ(RS_j)` expanded inline — pushes multiply by 1 | 4 | 2 |
| E9 | WS product update: same with `WS_j` and `skip_ws_j` — pops multiply by 1 | 4 | 2 |
| E10 | selector exclusivity (v3.1): pairwise products of `{ram, stk_0, stk_1}` are zero | 2 | C(S+1, 2) |
| E11 | push-bit binding (v3.1): `stk_s_j · is_write_j − sw_s_j = 0` (product + linear families in one row) | 2 | S |
| E12 | stack-pointer update (v3.1): `sp_s_j = sp_s_{j−1} + 2·sw_s_j − stk_s_j`, `sp_s_j` a σ-bit word. Bitness of `sp_s_j` **is** the bounds check: pop-at-empty forces `sp = −1` and push-at-full forces `sp = 2^σ`, both unrepresentable — no comparison rows, same trick as RAM addressing. | 1 | S |
| E13 | stack address binding (v3.1): `stk_s_j · (addr_j − sp_s_{j−1} + 1 − is_write_j) = 0` — push writes at `sp` (then `sp += 1`), pop reads at `sp − 1` (Coral Fig. 7). With E12's bitness this also bounds `addr < 2^σ` on stack ops, so no separate stack address-range gate is needed. | 2 | S |
| E14 | push canonicality (v3.1): `(Σ_s sw_s_j) · rt_j = 0` and `(Σ_s sw_s_j) · v_r_j = 0` — a push's RS-side fields are dead and pinned to zero (commitment canonicality, same rationale as E7) | 2 | 2 |

### 4.2 Scan block (per slot j)

| # | constraint | deg | rows |
|---|---|---|---|
| S1 | bitness of IS/FS lane bits + product bits | 2 | ≈ 152 + 256 |
| S2 | IS product update `h_is_j = h_is_{j−1} · f_γ(t_is_j, g_p, v_is_j)`, with `g_p = idx·B_scan + j` linear in `x` | 3 | 2 |
| S3 | FS product update, same shape | 3 | 2 |

### 4.3 Fingerprint (packed, Coral App. D form)

```text
g            = addr + ram · R + Σ_s stk_s · (R + M + s·2^σ)
                                          (global cell index, < R + M + S·2^σ)
packed(t, g) = t + 2^TS_BITS · g          (≤ 62 bits by plan validity)
f_γ(t, g, v) = γ2 − (packed + γ1 · v)     over K, γ1, γ2 ∈ K
```

`g` is linear in lane bits on the exec side (the one-hot selectors make
the namespace bases plain coefficients) and in `x.idx` on the scan side,
so the namespace never branches in-circuit; `g ↦ (namespace, addr)` is
injective because E6 forces `addr < R` whenever no selector is set, RAM
bitness bounds `addr < M` (`r ≤ μ`), and E12+E13 bound stack addresses
below `2^σ` — the namespace `g`-ranges are disjoint.
Packing is sound because every packed component is range-checked (bits) —
Coral App. D "Packing" paragraph; its ROM caveat (no-ts-check ROM forbids
packing) does not apply because v3 keeps the `rt < wt` check for ROM reads
(Coral's cheaper no-ts ROM variant: deferred, §14). One K-scale and one
K-subtraction per tuple; the product update `h · g` is the only K×K
multiply (Karatsuba components, degree ≤ 4 with the pad gate inline; K
convention: `neo_math::K`'s binomial constant, referenced, never
hard-coded).

### 4.4 Boundary rows and public input

Chunk-level rows chain `h_*_0` from `x.h_*_in` and pin `x.h_*_out`,
`x.ts_out = ts_in + cnt_{B_ops−1}`, and (v3.1) per stack
`x.sp_s_out = sp_s_{B_ops−1}` — slot 0's E12 reads `x.sp_s_in` directly
(linear, 5 + S rows).

`x` layout (all bit slots; F′ — not circuit rows — checks their
well-formedness and continuity, §6):

| slot | bits |
|---|---|
| `seg_idx` (segment k) | 16 |
| `idx` (step within segment) | 16 |
| `ts_in`, `ts_out` | 44 + 44 |
| `γ1`, `γ2` | 2 × 128 |
| `h_rs`, `h_ws`, `h_is`, `h_fs` (in/out each) | 8 × 128 |
| `sp_s_in`, `sp_s_out` per stack (v3.1, appended) | 2·S·σ |
| **total** | 1,400 + 2·S·σ |

Limb-canonicality note: the `γ` and `h` slots encode 64-bit limbs as bits,
and bit patterns ≥ q alias their mod-q reductions. No canonicality
constraint is required: the circuit's recombination rows and the F′ decode
(§6.3) both reduce mod q, so aliased encodings agree everywhere they are
compared and never reach a fingerprint input un-reduced. Honest encoders
emit canonical limbs.

Circuit totals at v3 targets: op block `64 × ≈455 ≈ 29k` rows, scan block
`64 × ≈420 ≈ 27k` rows, `x` region ≈ 1.3k → **`S_mem` ≈ 58k rows** before
any VM frontend.

---

## 5. Split-witness committed CCS (SW-CCS)

The SuperNeo adaptation of Coral's SW-R1CS (Coral App. E.1), replacing
v2's under-specified "advice-lane" scheme.

### 5.1 Relation

The witness `z = [x ‖ w_app ‖ lane_ops ‖ lane_is ‖ lane_fs]` keeps the
engine's existing full-`z` commitment `c = A · embed(z)` **unchanged**; the
claim additionally publishes one commitment per memory lane, each under its
own matrix:

```text
adv   = (c_ops, c_is, c_fs)
c_ops = A_ops · embed(lane_ops region)       A_ops: fresh seeded Ajtai matrix
c_is  = A_mem · embed(lane_is region)        A_mem: fresh seeded Ajtai matrix
c_fs  = A_mem · embed(lane_fs region)        (same A_mem — load-bearing)
```

`embed` is the SuperNeo field→ring packing already used by
`CcsInstance::from_low_norm_assignment`; an all-zero lane's component is
the zero commitment (free, recognizable as a constant). Seeds and labels
(`"nebula/A_ops|A_mem/v3"`) come from the plan.

**Shape invariant (normative, external-review fix).** A Nebula claim
carries exactly the three components `(ops, is, fs)`; when the F′
`NebulaLane` is active, a missing/empty/partial `adv` is rejected at
deposit. Non-Nebula claims carry an empty `adv` — never a partial tuple —
so nothing silently passes through Nebula state.

**Honest accounting (review #2).** Each lane is bound *once under its own
matrix*, and the decider puts the pieces together (§5.2) — that is the
disjointness that matters for binding and boundary equality. But lane
coordinates also still sit inside `c`, because the engine's committed
relation — and the extraction anchor of security-note Lemma 1 (A1) — is
defined over the full `z`. The overhead is bounded and lane-only: ≈ 18k of
≈ 57k committed bits per step are committed twice, ≈ 30% extra native
pay-per-bit commit work, zero extra circuit rows. The two zero-duplication
variants are deferred with named costs: (i) a true Coral-style tuple
relation (`c_app` over the app slice only) — touches the engine's committed
relation and requires re-proving A1 for tuple commitments; (ii) reusing
`A`'s own lane columns with a repeated `A_mem` block — zero extra commit
work, but `c` alone then stops binding `z` and A1's extraction must be
restated against the tuple. Neither is worth the frozen-surface risk in v1.

**L-ALIGN (normative; required by security-note Lemma 1).** Every lane
region's offset and width in `z` are multiples of `d = 54`, padding each
lane to whole ring columns of the embedding with constrained-zero filler
bits. Without column alignment, the RotRho fold action and `split_b` digit
recombination do not commute with lane slicing, and the mirroring argument
of Lemma 1 fails. `layout.rs` owns these paddings; the lane-width formulas
of §3 are rounded up accordingly.

`A_mem` shared between IS and FS lanes is what makes cross-instance
commitment equality meaningful (`c_fs` of segment k vs `c_is` of segment
k+1 commit the same content under the same matrix). This is the same
alignment requirement Coral states as "follow-up proofs need to use the
same chunk size" (Coral §5.3, persistent memory).

### 5.2 Folding and decider

- **Folding (R2):** every public linear update the pipeline applies to a
  single-commitment claim is applied component-wise to the tuple — Π_RLC's
  ρ-combination, Π_DEC's digit recomposition. This mirrors Coral
  Construction E.1 step 3 (`CM_{w_i,new} = CM_{w_i} + r · CM_{w'_i}`), costs
  three extra ring RLCs per fold, and does not touch Π_DEC/`y_zcol`
  semantics — same public arithmetic, three more ring vectors.
- **Transcript (R1, absorb-site inventory — exhaustive):** the tuple is
  bound — as its three 4-element leaf digests (§6.1) — wherever an
  authority-bearing input claim is first absorbed: (i) Π_CCS's input
  instance digest inside the NIFS transcript (`ccs_claim_digest` for fresh
  claims and `ce_claim_digest` for the running parent authority), before
  Π_CCS challenges; (ii) any terminal/decider transcript that absorbs
  claims. Π_CCS.V constrains every forwarded output `c`/`adv` coordinate
  equal to that already-bound input. Consequently the later pre-ρ
  `pi_ccs_outputs_digest/v2` absorbs only the newly sent `y_ring` and
  `y_zcol` messages; reabsorbing forwarded `c`/`adv` there would add cost
  without adding authority. The F′ chunk-digest
  preimage is **deliberately not on this list** (v4-review correction):
  `f_prime_chunk_claim_digest` is a shape-only digest that absorbs neither
  `claim.x` nor `claim.c.data` (fixed-point rationale documented in
  `paper/digest.rs`), and `adv` mirrors `c` there — excluded. Safe because
  the F′-side binding of `adv` is the per-step `D_seen` chain (§6.3),
  which is content-binding, strictly stronger than the chunk digest's
  domain-separation role. Tests pin both halves: changing input `adv`
  changes the Π_CCS authority digest, while changing only an equality-bound
  forwarded copy leaves the v2 output digest unchanged and still fails the
  verifier's forwarding equality.
- **Decider (R3, "put them together", review #2):** the terminal decider,
  which already opens the final folded witness `z_fin`, checks each folded
  component against its slice:

```text
A_ops · embed(z_fin[ops])  == c_ops_fin       (and likewise is, fs under A_mem)
```

  Three native ring products over the low-norm post-DEC witness, beside the
  existing full-witness opening against `c`. Because each lane has its own
  matrix and all openings are slices of the *same* `z_fin` that `c` opens,
  Coral's overlap attack (App. E.2's `a_0 + x` example) has no channel here
  and no non-overlap sum-check is needed — that patch is specific to
  committing padded vectors under one shared generator set.

### 5.3 Security claim

Let `δ_i = tuple_i − (A_• · embed(slice_i))_•` per instance (fixed once
instance i is absorbed). Component-wise mirroring plus the terminal opening
give `Σ_i coeff_i · δ_i = 0` with the pipeline's own post-absorption fold
coefficients; a nonzero `δ` survives with at most the pipeline's existing
RLC/DEC soundness error. MSIS binding of each matrix (collision
`‖Δ‖∞ ≤ 1`, bit lanes; preset supports width `2^30` ≫ any lane) then binds
each component to its slice content. The lemma is stated and proven as
Lemma 1 (with Corollary 1.1 for cross-instance lane equality) in the
[security note](./nebula-superneo-security-note.md), against the verified
code anchors of its §0; L-ALIGN above is the constraint that proof
surfaced.

---

## 6. The F′ commitment-carrying lane (the IVC part)

This section is the answer to review finding #1: how the memory argument
is carried **by the IVC itself**, in constant state, rather than by a
replay verifier.

### 6.1 Carried state

`paper/construction2/state.rs` `State` gains a `NebulaLane` (absorbed into
`state_x_out_digest` and the F′ step transcript context every step, like
`acc_digest` and `semantic_state_digest` today):

```text
NebulaLane {
  seg_idx:  u64,           // current segment k
  idx:      u64,           // step within segment
  ts:       u64,           // global timestamp (never resets)
  gamma:    (K, K),        // (γ1, γ2) of the open segment; ⊥ before squeeze
  h:        [K; 4],        // running (h_rs, h_ws, h_is, h_fs)
  sp:       [u64; S],      // running stack pointers (v3.1); 0 at every
                           // segment boundary (§3.1 segment locality)
  D_pre:    [[F; 4]; 3],   // per-lane (ops, is, fs) chain digests, computed
                           // natively at segment open over the segment's
                           // lane-commitment leaves and absorbed by the γ
                           // transcript
  D_seen:   [[F; 4]; 3],   // per-lane running chains over the folded claims'
                           // adv leaves — each commitment crosses Poseidon2
                           // exactly once per step (its leaf)
  D_mem:    [F; 4],        // boundary handle: previous segment's final fs chain
}
```

Constant size. Nebula Construction 2's carried commitment `C_i` is realized
by the `D_*` Poseidon2 digest chains over per-commitment **leaf digests**
(v4-review L0a). Each lane commitment is hashed exactly once per step into

```text
leaf_ops = Poseidon2("nebula/leaf/ops", c_ops)
leaf_is  = Poseidon2("nebula/leaf/mem", c_is)    // lane-NEUTRAL memory tag
leaf_fs  = Poseidon2("nebula/leaf/mem", c_fs)    // same tag — load-bearing
```

and every consumer — the `D_seen`/`D_pre` chains (`D ← Poseidon2(D_prev,
link_tag, leaf)`, the paper's `C_i ← hash(C_{i−1}, C_ω)` with one extra
collision-resistance hop, security-note Lemma 2) and every R1 transcript
site (§5.2) — absorbs the 4-element leaf, never the raw 972-element
commitment. **Tag discipline (external-review fix):** the `is` and `fs`
leaves and links share one *memory-domain* tag pair
(`"nebula/leaf/mem"`, link tag `"mem"`) and one shared chain header;
`ops` has its own domain. This is what makes segment k's FS chain and
segment k+1's IS chain *formula-identical*, so the §6.4 boundary equality
`D_seen[is] == D_mem` can hold for honest continuity — lane-typed tags
here would make honest two-segment proofs fail. Lane identity is enforced
by tuple position (which `adv` component, which decider slice, which
product-row family), not by the hash tag. One chain per lane (review:
Enzo, cost pass): 2,916 field elements cross Poseidon2 per step (the
three leaves) and no commitment is hashed twice — the segment-close
equalities compare per-lane chains directly. An all-zero lane's leaf is a
constant tag (§6.5, pay-per-use).

### 6.2 Transcript events (segment open)

At segment open (before any of the segment's instances fold), the prover
derives γ from a per-segment transcript. There is no persistent chain
transcript in this codebase: like the per-step F′ transcript
(`F_PRIME_STEP_TRANSCRIPT_LABEL` in `paper/f_prime/native.rs`), the γ
transcript is a **fresh Poseidon2 transcript seeded from the F′ carried
state at segment open** — the state digest fields bind prior history at
the chain's own authority level (per `state.rs`: `z_i` is shape-only
domain separation; content binding rides `acc_digest` and, for memory, the
`NebulaLane`'s own `D_*` fields, all under A4's enforcement status), and
the resulting `D_pre` carried digest binds the absorbed commitment list
into every subsequent step's state hash:

```text
label   "neo.fold.clean/nebula/gamma/v3"
seed    state-digest fields at segment open (vk_fs, z_i, acc_digest, nebula lane)
absorb  plan_digest, seg_idx, ts
absorb  D_pre[ops], D_pre[is], D_pre[fs]     // 12 field elements: the per-lane
        // chain digests over the segment's lane-commitment leaves, computed
        // natively by the prover at open (same chain formula as D_seen)
squeeze γ1 = K(cf, cf);  γ2 = K(cf, cf)                       // cf = challenge_field
```

Absorbing the three chain digests instead of the raw commitment list keeps
the per-segment transcript replay constant-size in the eventual F′ R1CS
(the binding chain runs per step in `D_seen` anyway; between γ and the raw
commitments sit the leaf and chain collision-resistance hops, security-note
Lemma 2). Stated once, normatively
(v4-review L0b): `D_pre` is a **prover-claimed** value at open — no
open-time wide absorb exists in any realization — and its authority is
retroactive, via the close equality `D_seen == D_pre`.
Commit-then-challenge needs only that the pre-γ object is *binding*, which
the leaf/link chain over MSIS-binding commitments provides; a false
`D_pre` claim fails the close check except with the hash chain's binding
error — collision resistance for a known-preimage deviation, **preimage
resistance for a thin-air claim** (both under A3's random-oracle
modeling; security-note Lemma 2, external-review fix). The honest native
path computes `D_pre` from the precommitment list (§6.3 `open_segment`);
"claimed" names its authority status and how the eventual F′ R1CS treats
it — a witness input verified at close.

Commit-then-challenge is inherited from the transcript discipline: γ is a
deterministic Poseidon2 function of MSIS-binding commitments to every
multiset contribution of the segment — the same state-seeded-transcript
mechanism that makes the fold challenges ρ sound (Π_CCS first absorbs
authority-bearing input claim digests, including `c.data` and `adv`;
Π_CCS.V equality-forwards those coordinates, then
`pi_ccs_outputs_digest/v2` absorbs the new evaluation messages before ρ),
at the same trust status,
replayed by the same future decider machinery.

### 6.3 Per-step transition (pseudocode, normative)

The transition follows HyperNova's one-step-delayed induction. A produced
claim has public input

```text
u_i.x = [1 || enc_inst(x_out_{i-1}) || S_mem.x_i || open_i || bits(D_pre_i)]
```

and product commitment `L+(Z_i) = (c_i, adv_i)`, where
`adv_i = (c_ops,i, c_is,i, c_fs,i)` commits to the three lane slices of
the **same** witness. Base F′ produces `u_1` but has no previous claim to
consume, so its lane remains the canonical base lane. Recursive F′ step
`i+1` runs `NIFS.V(..., u_i, ...)`, then consumes the verifier-bound pair
`(u_i.x suffix, u_i.adv)` to advance the lane while the current
application witness produces `u_{i+1}`. This is Nebula Construction 2's
`C_i <- H(C_{i-1}, u_i.C_W)` schedule, specialized to the three lane
commitments.

**Fold arity (normative for this version):** Nebula-enabled F′ uses
`K = 1`. SuperNeo supports larger fresh batches, but batching makes segment
open/close timing and the per-claim application transition a second
compiler obligation. It is outside this version; an implementation must
reject `K != 1` rather than silently reuse one state seed across a batch.

```text
base(plan):                                   // chain start (State::base)
  lane ← { seg_idx: 0, idx: 0, ts: 0, gamma: ⊥,
           h: [1_K; 4], D_*: headers, D_mem: plan.D_init }
  // no advance: there is no prior fresh claim

recursive_f_prime(U_i, u_i, lane, current_app_witness):
  U_{i+1} ← NIFS.V(U_i, u_i)
  (step_x, open, D_pre) ← decode(u_i.x suffix)
  lane ← consume_delayed(lane, step_x, open, D_pre, u_i.adv)
  produce u_{i+1} from this same F′ + S_mem relation

open_segment(lane, D_pre):                    // selected by prior claim's open bit
  require lane.idx == 0  and  lane.gamma == ⊥
  lane.D_pre ← D_pre                          // claimed at production time;
                                              // authorized retroactively at close
  lane.gamma  ← γ1, γ2 per §6.2 (fresh transcript seeded from state,
                                 absorbs plan, counters, D_pre digests)

advance_nebula(lane, u):
  require lane.gamma != ⊥  and  lane.idx < N         // segment must be open
  assert u.x.seg_idx == lane.seg_idx  and  u.x.idx == lane.idx
  assert u.x.ts_in   == lane.ts
  assert u.x.γ1 == lane.gamma.0  and  u.x.γ2 == lane.gamma.1
  assert u.x.h_*_in  == lane.h                       // 4 K-equalities
  assert u.x.sp_*_in == lane.sp                      // v3.1: S sp-equalities
  leaf_ops ← Poseidon2("nebula/leaf/ops", u.adv[ops])
  leaf_is  ← Poseidon2("nebula/leaf/mem", u.adv[is])
  leaf_fs  ← Poseidon2("nebula/leaf/mem", u.adv[fs])
  lane.D_seen[ops] ← Poseidon2(lane.D_seen[ops], "ops", leaf_ops)
  lane.D_seen[is]  ← Poseidon2(lane.D_seen[is],  "mem", leaf_is)
  lane.D_seen[fs]  ← Poseidon2(lane.D_seen[fs],  "mem", leaf_fs)
  lane.h  ← u.x.h_*_out
  lane.sp ← u.x.sp_*_out                             // v3.1
  lane.ts ← u.x.ts_out
  lane.idx += 1
  if lane.idx == N:                                  // segment close
    assert lane.sp == [0; S]                         // v3.1 segment locality (§3.1):
                                                     // deterministic companion to the
                                                     // product equation, which already
                                                     // rejects an unpopped push w.h.p.
    assert lane.D_seen == lane.D_pre                 // folded lanes are the pre-committed
                                                     // lanes, per lane (3 equalities)
    assert lane.h_is · lane.h_ws == lane.h_rs · lane.h_fs        // Nebula product equation
    assert lane.D_seen[is] == lane.D_mem             // memory continuity (see 6.4)
    lane.D_mem ← lane.D_seen[fs]
    reset: idx ← 0, seg_idx += 1, h ← [1_K; 4], gamma ← ⊥,
           D_pre/D_seen ← headers                    // ts is NOT reset; sp is already 0
```

Chain headers: the `is` and `fs` chains initialize from one shared
mem-domain header; `ops` has its own. Header symmetry plus the shared
`"mem"` tags (§6.1) make segment k's FS chain and segment k+1's IS chain
formula-identical — the §6.4 boundary equality compares
identically-computed digests.

**Finalization rule (normative).** The trailing latest claim `u_T` has not
been consumed by another F′ step. The terminal relation therefore performs
both operations in order: `(1)` verify the terminal NIFS fold of `u_T`,
`(2)` run `consume_delayed` on the exact `fresh_x` and `fresh_adv` wires
output by that verifier. It installs the resulting lane and post-fold
accumulator in the final state, recomputes the public `x_out` over both,
then requires a **closed segment**: `lane.idx == 0`, `lane.gamma == ⊥`, and
`D_pre`/`D_seen` at their headers. Checking the fold without consuming its
lane pair is incomplete; checking closure on the pre-terminal lane is the
same off-by-one error. Mid-segment state remains prover-only resume material.

Implementation status: the authoritative recursive F′ R1CS enforces the
delayed transition and binds its input/output lane digests. The current
direct `S_mem` lifecycle still uses the pre-migration deposit-time replay;
the terminal delayed transition and composed `S_mem + F′` lowering remain
the fail-closed production gate in §13 step 9. Memory adds obligations to
F′, not a new verifier class.

### 6.4 Segment boundary and the initial state

`D_mem` carries the previous segment's FS commitment-sequence digest;
segment k+1's IS sequence must chain to the identical digest
(`D_seen[is] == D_mem` at close). Since IS/FS lanes share `A_mem` and the
digest chains are
position-ordered with the same mem-domain formula (shared leaf/link tags
and header, §6.1/§6.3), this equality holds iff the lane commitment
sequences match pairwise — i.e., segment k+1 opens memory in exactly the
state segment k left it (values *and* timestamps; timestamps are global).
For k = 0, `D_mem` is initialized to the plan's `D_init` (§7).

Resume is two different bundles (external-review fix). The **verifier**
resume handle of a finished chain is `(State digest ⊇ NebulaLane{ts,
D_mem})` alone. The **prover**, when paused mid-segment, additionally
needs the open segment's precommitted lane-commitment list and remaining
witness plan — without them it cannot close against `D_pre`
(commit-before-challenge forces segment-level lookahead). A mid-segment
pause is internal prover state; it is never externally verifiable (§6.3
finalization rule).

Shipped realization (`frontends/nebula/prove.rs::resume_segment`): the
carried lane **is** the checkpoint — γ, `D_pre`, the step index, and the
`ts`/`h`/`sp` carry all live on it — so the prover re-supplies only the
segment's trace (the "remaining witness plan"), and the pair is
authenticated by recomputing the trace's lane chains against the lane's
`D_pre`: a wrong or mutated trace cannot pass, because γ was squeezed
over `D_pre`.

### 6.5 Cost class of the F′ additions (honest accounting, dual-regime)

Per step, F′ natively computes one **leaf digest** per lane commitment
(§6.1) — `3 · κ·d = 2,916` field elements cross Poseidon2 per step, each
commitment exactly once — then every chain link and transcript absorb
consumes 4-element leaves, plus 4 K-equality and counter checks; the
per-segment γ transcript absorbs only the three `D_pre` digests
(12 field elements). Natively this is microseconds. Steps with all-zero
lanes (pure compute chunks, once a VM frontend exists) absorb a constant
zero-commitment tag instead: pay-per-use is preserved.

The eventual in-circuit cost is **regime-dependent**, because the private
F′ witness encoding `enc(F′)` is an open design decision
(`paper/f_prime/encoding.md`; the same gate as §13 step 9):

- **Folded regime** (F′ becomes a low-norm CCS instance each step): every
  committed coordinate is a bit, so Poseidon2 must use the bit-backed
  builder (`engine/ccs_native/poseidon2.rs`, verified shape: RATE 4,
  21,888 committed bits ≈ 22k rows per permutation). One 972-element leaf
  = 244 permutations ≈ 5.3M committed bits; the memory carry is
  **3 leaves ≈ 732 permutations ≈ 16M committed bits per step**. Without
  leaves it would be ≈ 2× (each commitment crossing Poseidon2 in both its
  `D_seen` chain and the R1 transcript sites) — which is why leaves are
  normative. This rides the same curve as the in-circuit absorption of
  the claim's own `c.data` and `X` region that NIFS replay already
  requires (`append_ce_claim_public_fields`, computed in-circuit per
  child claim): the baseline dominates the carry.
- **Compressed-decider regime** (the F′ R1CS is discharged by the
  terminal Spartan proof): witness wires are native field elements; the
  field-native gadget (`engine/r1cs_circuit/poseidon2.rs`, 344 mults per
  permutation) prices a 972-element leaf at ≈ 60–90k constraints and the
  carry at ≈ 0.2–0.3M constraints per step, amortized into the one
  terminal proof.

The earlier "≈ 122 permutations ≈ 45–60k constraints per commitment"
figure was the field-native cost quoted for the folded regime — the wrong
gadget for that regime (v4-review correction). The authoritative shape audit
now provides the first whole-relation lower bound at reduced κ = 1: after
overlapping one-hot branch advice, steady F′ alone forces at least
30,083,645 committed bits before real bit widths, versus the 16M engineering
budget. C12 remains the production-κ measurement gate.

**Considered and deferred: SIS/Ajtai accumulators for the `D` chains**
(review question: Enzo; v4 proposal). Merkle–Damgård chaining over the
Ajtai map, `D_i = B₁ · G⁻¹(D_{i−1}) + B₂ · G⁻¹(input_i ‖ tag_i)`, is
*sound*: a chain collision is a low-norm MSIS solution for independent
seeded matrices; chaining (not summing) gives position binding; and SIS
output is never a challenge source — challenges stay Poseidon2 squeezes
over a binding compression ("hash-then-FS"). Two corrections to the
earlier rejection rationale (v4 review):

- (i) Its cost comparison used the field-native Poseidon2 figure. In the
  folded regime the honest comparison **favors** SIS on committed
  material — ≈ 62k bits + ≈ 62k rows per absorbed commitment versus
  ≈ 5.3M bits/rows for bit-backed Poseidon2 — at the price of ≈ 60M nnz
  of dense linear structure constants per absorb (sum-check work, plus
  GB-scale matrices unless the engine exploits the ring structure).
- (ii) Digit bases `w > 1` are **unbuildable** regardless of parameters:
  the NC range check forces every committed coordinate of a fresh
  instance into `{−1, 0, 1}` (`b = 2`), so only `w = 1` digits ride the
  scheme's norm enforcement for free. `B = 2^14` is the RLC-transient
  binding headroom, not an enforced bound on fresh witnesses.

Disposition: **reopened as a Road A blocker** by the authoritative shape
audit. The present Poseidon construction remains the normative, sound
fallback, but it cannot meet the committed-width budget. Adopting SIS still
requires a compiler/engine construction and hash-then-FS lemma. The first
native/circuit primitive now pins the trade: for three fields at κ=1,
Ajtai plus one final Poseidon2 digest costs 10,532 field columns, 10,537 rows,
and 93,425 nonzeros **before** low-norm lowering. The complete toy lowering
measures 661,445 committed bits and 671,981 CCS rows. The compiler now aliases
each recorded canonical decomposition child onto its source field's existing
64-bit slot (with direct, mismatch, and three-arm tests), so this measurement
does not double-commit those bits. Ajtai itself needs only `Dκ` equations, but
a naïve matrix still has Θ(`Dκ·64N`) coefficients for `N` canonical fields.
The remaining production construction must represent the seeded ring map
structurally; materializing a full accumulator as CSC is not acceptable.
Until that and the binding lemma land, the fixed-point compiler's budget guard
stays fail-closed. The two mem lanes merging into one component (heavier
boundary rule) remains the other recorded knob.

---

## 7. Public ROM (verifier's handle to the program)

Requirement (review #4): the verifier must be able to state "this trace
ran against *this* table" without recomputing challenge-dependent
fingerprints and without learning private RAM.

Mechanism:

1. ROM is the default namespace (no selector set, v3.1 — v3's `seg = 0`),
   swept by the same scan as RAM (§3).
   Writes to it are banned in-circuit (E5). Reads are ordinary timestamped
   reads (sequential consistency includes ROM).
2. The plan generator lays the public ROM image (and zero RAM) into the
   initial scan lanes, computes their `A_mem` commitments and the digest
   chain `D_init = fold_{j ∈ [0,N)} Poseidon2(·, "mem", Poseidon2("nebula/leaf/mem", c_init_j))`
   — one link per step over the N per-step initial-scan-lane commitments,
   starting from the shared mem-domain header: the identical formula as
   the IS/FS chains (§6.1; external-review fix) — and records
   `rom_image_digest` and `D_init` in the plan (§11). **Anyone can
   recompute `D_init` from the ROM image and public parameters** — that is
   the verifier's handle, and it is γ-independent (Nebula's observation
   that `C_FS`/`C_IS` depend only on state, not challenges), so it works
   unchanged across any number of segments, each with its own γ.
3. Chain start binds `D_mem ← D_init` (§6.4); every later segment chains
   from it. A different ROM ⇒ different `D_init` ⇒ different plan.

Alternative documented and **not** chosen as default: Coral's public-ROM
optimization (Coral §5.2) removes ROM cells from the in-circuit scan and
has the verifier compute `h_IS_pub` itself from the table and γ (the check
becomes `IS_pub ∪ IS_priv ∪ WS = RS ∪ FS`). It saves `R/B_scan · ≈420`
rows per segment but costs the verifier `O(R)` field work **per segment**
(one fingerprint per γ) — exactly the multiple-fingerprint burden review
#4 flags. Adopt it later only if ROM size dominates the scan and the
verifier profile can afford it; it composes cleanly (drop ROM scan slots,
add the verifier-side factor into V6's equation).

Privacy note: this repo does not currently target zero-knowledge; reads'
addresses are witness data (not revealed by the proof), but no formal ZK
claim is made (Coral's blinding treatment is the reference if that changes;
deferred, §14).

---

## 8. Single vs. multiple circuits (review #3 analysis)

Options for hosting op-checking and scan in the chain:

| option | structure(s) | per-step sum-check load | padding/step-size behavior | engine fit |
|---|---|---|---|---|
| **(a) Interleaved uniform (chosen; Coral §5)** | one, `≈ 58k` rows | full structure every step | `B_ops`/`B_scan` fixed per plan; segment length `N = (R+M)/B_scan` (exact cover); op pads cheap (gated); no scan pads | matches ℓ = 1 engine today |
| (b) v2 switchboard (exec-phase + scan-phase arms) | one, `≈ 60k` rows (sum of arms + switches) | full structure every step — an exec step still pays the scan arm's rows in sum-check (zeros are free to *commit*, not to *sum-check*) | phase counts per segment are extra plan constants; pads in both arms; phase-order checks needed | frontend-only, but strictly more machinery than (a) for the same load |
| (c) two relations (separate exec and scan structures) | two: `≈ 31k` + `≈ 28k` | each step pays only its own structure (~2× saving on scan-only or exec-only steps) | chunk sizes independent; big-`M`/small-`M` plans retune scan without touching exec | requires heterogeneous folding (two running accumulators or SuperNova-style indexing) — **not supported by the ℓ = 1 engine**; a core change with its own spec |

Decision: (a). It is the smallest correct thing on the current engine, it
eliminates the entire phase apparatus (option b's only advantage over (a)
is none — same sum-check load, more moving parts), and Coral demonstrates
the shape at scale. Option (c) is the real long-term contender — it
is what "pay-per-use across heterogeneous step types" ultimately wants —
but it is gated on multi-relation folding in the engine and is deferred
with that dependency named (§14). The trade-off a plan author controls
today: `B_scan` ties segment length to memory size (`N = (R+M)/B_scan`);
large memories want large `B_scan` or long segments, small memories that
want long segments enlarge `M` with untouched cells; all are plan
constants compiled into the structure.

---

## 9. Soundness argument and error budget

1. **Folding/decider** (existing pipeline): every instance satisfies
   `S_mem`.
2. **SW-CCS binding** (§5.3): each instance's lanes open its commitment
   tuple; MSIS makes tuples binding.
3. **CC-IVC carry** (§6): `D_pre = D_seen` at close means the folded lanes
   are the pre-absorbed ones; γ was squeezed from the transcript after
   absorbing them (commit-then-challenge); `h` products thread unbroken
   through `x` (F′ equalities); `ts` is global; `D_seen[is] = D_mem` chains
   memory across segments; all bound step-by-step by the state hash.
4. **In-circuit discipline** (§4): `rt < wt`, `wt = ts_in + cnt`, read
   value consistency, ROM ban, pad gating, structural scan addressing.
5. **Corollary 8 / Coral App. D**: the product equation with
   post-commitment γ implies sequential consistency except with the SZ
   error below.
6. **Stacks (v3.1, Lemma 4 / Coral App. E / Blum et al.)**: stack
   namespaces have disjoint `g`-ranges and empty IS/FS, so the balanced
   product equation forces stack-WS = stack-RS as multisets (w.h.p.);
   together with E12/E13's `sp` discipline and E4 on pops, each pop
   returns the value of its matching (LIFO) push. `m_seg` is unchanged:
   stack ops occupy the same op slots and each contributes one tuple
   instead of two.

```text
ε_total ≤ ε_pipe·n_f + ε_MSIS + q_H · n_f·n_in/|C|                      (Lemma 1, FS-lifted)
        + ε_CR                                                          (Lemma 2)
        + q_H · Σ_seg m_seg / |K|                                       (Lemma 3 / Cor. 4.1)

ε_MSIS := ε_MSIS(A) + ε_MSIS(A_ops) + ε_MSIS(A_mem)
n_in   := per-fold input-claim count (SuperNeo's fold arity "K + k" —
          instance counts, unrelated to the extension field K in |K|)
m_seg  := |IS| + |WS| + |RS| + |FS| = 2·(N·B_ops + R + M)
```

This is, term for term, the composition theorem of the
[security note](./nebula-superneo-security-note.md) §5 — one canonical
formula, defined there. (`ε_pipe·n_f` is the host pipeline's own folding
budget over `n_f` folds; `q_H·n_f·n_in/|C|` is Lemma 1's mixing term,
FS-lifted with the same `q_H` as every other challenge-dependent event;
`ε_CR` is Poseidon2 collision/preimage resistance over all leaf, link,
and state hops.)

Worked v3 target per segment (`m_seg = 2·(N·B_ops + R+M) ≈ 2^18.1`):
`m_seg/|K| ≈ 2^-110` per Fiat–Shamir attempt — the same soundness regime
SuperNeo accepts for its own sum-checks. (The earlier `2n/|K|` phrasing
double-counted: Lemma 3's `2n` *is* the total multiset size, i.e.
`m_seg` — external-review fix.)

---

## 10. Cost budget

| item | formula | v3 targets |
|---|---|---|
| `S_mem` rows | `B_ops·≈455 + B_scan·≈420 + |x| + boundary` | ≈ 58k |
| committed bits per step (active) | ≈ same as rows (bitness-dominated) | ≈ 57k bits |
| steps per segment | `N = (R+M)/B_scan` | 1,088 (≈ 2^17.1 ops capacity) |
| per-op amortized rows | `≈455 + B_scan·420/B_ops` | ≈ 875 at `N_ops = R+M`; → 455 as ops ≫ memory |
| F′ native per step | 3 leaf digests (2,916 F elems hashed) + compact chain/transcript absorbs + checks | µs-scale |
| F′-R1CS (future) per step | 3 leaf absorbs; regime-dependent on `enc(F′)` (§6.5) | ≈ 732 perms ≈ 16M bits folded; ≈ 0.2–0.3M constraints Spartan |
| decider | + 3 lane openings (§5.2) beside the existing witness opening, once | negligible |
| prover per segment, extra passes | 1 native execution pre-pass + lane commits (`O(N·(B_ops+B_scan))` ring ops; includes §5.1's ≈ 30% lane double-commit) | seconds-scale, outside circuits |
| stacks delta (v3.1, `S = 2, σ = 12`) | per op slot: `+S` lane bits, `+S·(1 + σ)` aux, E10–E14 ≈ 11 rows + bitness | ≈ +2.5k rows (≈ +4%); `x` +48 bits |

Off-by-2× on any line at implementation time reopens the spec (v1's D3
discipline).

The dominant in-circuit cost is the running-product intermediates (2 × 128
bits per op slot, 2 × 128 per scan slot) — Nebula Fig. 2's "6 arb" under
lattice bit-encoding. Nebula-Opt (grand products outside the circuit,
paper §4.4) is the lever if it dominates; deferred.

---

## 11. Plan artifact

```text
NebulaPlan {
  plan_version: "nebula-superneo/v3.1",
  params_digest, structure_digest,
  a_ops_label, a_mem_label, plan_seed,
  selector bits, r, mu, S, sigma, VAL_BITS, TS_BITS,
  B_ops, B_scan, N, SEG_MAX,
  rom_image_digest,
  D_init,                    // §7: verifier's ROM handle (γ-independent)
  error_budget,              // §9 evaluated
}
```

`plan_digest = Poseidon2(canonical serialization)`; absorbed at every
segment open (§6.2). All hashing is Poseidon2; the only algebraic binding
objects are Ajtai commitments (the scheme's native family).

---

## 12. Red-team tests (required)

Assertions target the specific circuit row, F′ check, or decider opening —
never a host replay comparison.

| attack | required rejection |
|---|---|
| Flip a lane bit after committing (lane ≠ tuple) | §5.2 decider opening fails (via mirrored folds) |
| Recompute the tuple for the flipped lane instead | `D_seen ≠ D_pre` at segment close (F′), because the pre-absorbed list fixed γ |
| Move `pad`/`is_write` outside the lane (hacked layout) and re-select ops after γ | documented-attack test: hacked layout accepts the cheat, normative layout rejects at `D_seen == D_pre` — this test exists to pin the lane-residency rule |
| Stale read (classic memory lie) | product equation at F′ segment close fails except w.p. §9 |
| `rt ≥ wt` on a live slot | E4 |
| Use a pad slot as a real op | E7 |
| Write to ROM | E5 |
| ROM address out of range | E6 |
| Fresh memory at segment start | `D_seen[is] ≠ D_mem` at close |
| Reset timestamps between segments | F′ `ts_in` equality fails |
| Tamper any carried `x` slot between steps | F′ per-step equalities fail |
| Squeeze γ before absorbing all lane commitments (omit one) | `D_pre` mismatch ⇒ `D_seen ≠ D_pre`; transcript replay diverges |
| Forge a leaf (absorb `leaf ≠ Poseidon2(tag, adv component)`) | F′ recomputes leaves from the deposited claim's `adv`; a forged leaf diverges `D_seen` from `D_pre` (or the transcript replay) at close |
| Products not initialized to `1_K` | F′ segment-open reset + first-step equality fail |
| Swap IS and FS lanes of one step (adv components and `D_pre` list swapped consistently) | decider/mirrored-fold slice binding: `adv[is]` must open against the IS lane region of the same `z_fin` (Lemma 1); if lane *contents* are also swapped, the product identity and next segment's boundary check fail (w.p. §9) |
| End the proof mid-segment (stale read folded, stop before `idx == N`) | §6.3 finalization rule: terminal verifier requires `idx == 0`, `gamma == ⊥`, header chains |
| Fold a step before `open_segment` (γ still ⊥/default) or past `idx == N` | `advance_nebula`'s `require lane.gamma != ⊥ and lane.idx < N` |
| Wrong ROM image with correct shape | `D_init` mismatch (plan binding) |
| Change any §2 constant | plan digest / structure digest binding |
| Pop a different value than was pushed (stack lie, v3.1) | product equation at close — the RS tuple matches no WS push tuple, w.p. §9 |
| Pop claiming a wrong `push_time` | product equation (tuple mismatch); a future time additionally fails E4 |
| Push without popping (segment ends with a live cell) | `sp ≠ 0` at close (§6.3, deterministic) — and the product equation independently, w.p. §9 |
| Pop from an empty stack / push past capacity | E12: the next `sp` word (`−1` / `2^σ`) is unrepresentable in σ bits |
| Stack op at `addr ≠ sp` (out-of-discipline access) | E13 |
| Cross-stack splice (push `stk_0`, pop the tuple from `stk_1`) | product equation: disjoint `g`-ranges make the tuples distinct |
| Two selector bits set on one slot | E10 |
| `sw` aux bit lies about push-ness | E11 |
| Nonzero `rt`/`v_r` smuggled on a push | E14 |
| Tamper `x.sp_*` between steps | F′ `sp` equality (§6.3) |

Where each row lands (implemented; all under `crates/neo-fold-clean/tests/`):

| Rows | Test |
|---|---|
| Honest 2-segment chain with memory continuity | `nebula/segment.rs::two_segment_chain_with_memory_continuity_verifies` |
| Stale read; fresh memory; IS/FS swap; ts reset; `D_pre` tamper; terminal flip; wrong ROM | `nebula/redteam.rs` (one test per row, named after it) |
| Row-level op forgeries (E3–E7) | `nebula/circuit.rs::forged_ops_are_rejected_by_rows` |
| Every §6.3 lane check + finalization rule | `nebula/lane.rs` (one rejection test per check) |
| `adv` binding through Π_CCS/Π_RLC/Π_DEC | `nebula/adv_fold.rs` |
| All v3.1 stack rows + honest e2e + the documented non-rejection (a value lie passes every row, dies at the product equation — Lemma 4 layering) | `nebula/stack.rs` |
| §6.4 prover resume (honest + fail-closed) | `nebula/segment.rs` resume tests |
| Step 9 gadgets ≡ native transition + tamper sweep | `f_prime/nebula_lane_circuit.rs` |

---

## 13. Implementation plan (repo-mapped, ordered)

Tests live in `crates/neo-fold-clean/tests/`, `FoldingMode::Optimized`
only, every invocation under the 5-minute cap (test profile of §2).

1. **`frontends/nebula/layout.rs`** — §2 constants, §3 lane layouts, §4.4
   `x` layout, recombination helpers, native encode/decode.
   *Accept:* round-trips; constants consumed by later steps.
2. **`frontends/nebula/circuit.rs`** — `S_mem` builder (mixed-gate style of
   `engine/ccs_native/poseidon2.rs`), rows E1–E9, S1–S3, boundary.
   *Accept:* honest traces satisfy; one mutation test per row family.
3. **Lane commitments (core, mirror-only)** — `adv: (c_ops, c_is, c_fs)`
   added beside the untouched `c` (empty for non-Nebula claims,
   non-breaking); per-lane leaf digests (§6.1) computed once and absorbed
   at every R1 site; component-wise folding mirroring (NIFS, Π_RLC, Π_DEC
   public updates); decider slice-openings. No Π_DEC/`y_zcol` semantic
   changes. Cross-crate scope: `CcsClaim`/`CeClaim` live in `neo-ccs`
   (`relations.rs`) and are serialized — `adv` lands there with a serde
   default (empty) so existing artifacts and non-Nebula callers are
   untouched; the change freezes another public surface.
   *Accept:* lane-bit flip ⇒ opening rejection; §5.1 `adv` shape
   invariant enforced (partial tuple rejected); existing tests green.
4. **`paper/construction2/state.rs` + F′ native** — `NebulaLane` (§6.1),
   absorb into `state_x_out` context, and mirror §6.3's delayed schedule:
   base carries the lane unchanged; recursive F′ consumes the previous
   latest's suffix/`adv`; terminal finalization consumes the trailing latest.
   *Accept:* per-check rejection tests (each assert and `require` in §6.3
   has one, including advance-before-open and open-twice); §6.3
   finalization rule enforced (mid-segment terminal state rejected).
5. **`frontends/nebula/prove.rs`** — segment prover (two-pass flow of §1).
   *Accept:* 2-segment honest chain with **nontrivial memory continuity**
   (segment 0 writes RAM cells and reads ROM; segment 1 reads segment 0's
   writes) folds and verifies end-to-end — this is the test that would
   have caught the lane-typed-tag continuity bug; pause/resume from
   `State` between segments.
6. **`frontends/nebula/plan.rs`** — §11 artifact, `D_init` generation.
   *Accept:* plan/ROM mutation tests.
7. **Red-team suite** — §12, `tests/nebula_redteam.rs`.
8. **Perf snapshot** (`--ignored`) — record §10 actuals at v3 targets.
9. **Authoritative folded F′ relation** — compose the current `S_mem`
   application relation, NIFS.V, and `NebulaLane` transition into one
   fixed-shape relation, then lower that relation through the low-norm R1CS
   compiler. The recursive lane rows are implemented and consume the
   previous fresh claim's verifier wires; the terminal relation must consume
   the trailing claim before final closure. Obligations: fixed `K = 1`;
   `seg_idx`/`idx`/`ts`/`γ`/`h`/`sp` continuity against the suffix; exact
   suffix/`adv` pairing from one NIFS output; three `D_seen` chain updates;
   segment-open `D_pre` binding and γ squeeze; and the close checks —
   `D_seen == D_pre` per lane, the product equation,
   `D_seen[is] == D_mem`, `D_mem ← D_seen[fs]`, and reset **without**
   resetting `ts`. Tracked as the single named gap between "testable" and
   "production-trusted" — the same gap, and the same closure, as F′ itself.
   **Executable milestone gates** (`tests/system/ivc_invariants.rs`):
   `multi_chunk_f_prime_chain_must_verify_terminal_only` and
   `nebula_chain_must_verify_terminal_only_with_memory` are `#[ignore]`d
   acceptance tests that fail today by design; step 9 is done when they
   pass un-ignored and their active fail-closed tripwire twins are
   removed in the same change.
10. **v3.1 stacks** — one pass across the existing owners, no new module:
    selector/`sp` layout + `MemSpace` domain type (`layout.rs`), native
    push/pop + segment-locality errors (`trace.rs`), rows E10–E14 and the
    reshaped E4/E5/E6/E7/E8/E9 gates + `sp` boundary rows (`circuit.rs` —
    the lane-residency audit extends its allow-list by the E11-pinned `sw`
    aux exactly as it allows `cnt`), `x`/`NebulaStepX` `sp` slots and the
    lane's `sp` carry + close check (`nebula_lane.rs`), plan v3.1 binding
    of `S`/`σ` (`plan.rs`). `S = 0` must reproduce v3 byte-identically
    (pinned by test). *Accept:* honest push/pop segment verifies
    end-to-end; every §12 stack row fails at its named check; the §10
    stacks-delta line is measured within 2×.

## 14. Deferred (explicitly out of scope for v3)

- **Two-relation folding** (§8 option c) — gated on engine support for
  heterogeneous structures; the long-term pay-per-use shape.
- **Stacks beyond v3.1's shape** — v3.1 made segment-local stacks
  normative (§2, §3.1, §4.1 E10–E14, §6.3). Still deferred: more than 2
  stacks (mechanical selector widening), per-stack heights, and
  **persistent stacks** — a stack surviving a segment boundary needs
  either RAM spill (frontend concern) or stack participation in FS/IS
  snapshots, which negates the Blum-optimization and re-opens the
  per-segment-γ analysis (§3.1).
- **Coral's public-ROM scan elision** (§7 alternative) and **no-ts ROM**
  (Coral App. D) — verifier-work vs. constraint-count trade.
- **Nebula-Opt** grand products (paper §4.4).
- **Private initial memory / persistent segment export** — Coral §5.3
  persistent memory; export `c_fs` sub-sequences as portable commitments.
- **Parallel / distributed segment proving** — v3 segments are sequential
  by design (`D_mem` hands FS lanes to the next segment). Proving
  segments out of order requires each worker to derive its opening
  memory independently — zkEngine (`external/zkEngine_dev`,
  `construct_IS`) does this by natively replaying all prior shards'
  execution, an O(history) cost per shard. If distributed proving ever
  matters, that replay-vs-forwarding trade needs its own design.
- **Multi-word values** (`ℓ > 1`): fingerprint extends with `γ1^j` powers
  per Coral's general form.
- **Zero-knowledge** — repo-wide non-goal today; Coral App. C is the
  blinding reference.

## 15. Completion definition

v3 is implemented when: (1) §13 steps 1–8 are merged with their acceptance
tests; (2) every §12 row fails for its stated reason; (3) the plan artifact
records §2 constants, `D_init`, and the evaluated §9 budget; (4) the F′
native path enforces every §6.3 assert (no memory proof verifies without
them); (5) §13 step 9 is tracked as the named production gate riding the
F′-R1CS milestone; (6) two named items are **independently reviewed by
someone other than the author**: security-note Lemma 1 Step 2 (challenge-set
injectivity and the RotRho/`A_L` commutation, against the engine's actual
sampling set) and Lemma 2's induction (against `state.rs`'s actual absorb
set); (7) the lane-residency rule is statically enforced: the `S_mem`
builder audits, at construction time, that every column read by the
fingerprint-input matrices lies in the committed lanes, public `x`, or the
E2-constrained `cnt` aux — a layout change reintroducing a free
interpretation bit must fail the build, not a review; (8) the terminal-only
multi-chunk gates pass only after the folded fixed-shape F′ relation enforces
NIFS.V, `adv` forwarding/RLC/DEC, and the §6.3 `NebulaLane` transition. A
projection cost shell or native replay is not evidence for this item.
