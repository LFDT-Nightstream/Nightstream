# Compression milestone 1: layer 0 + layer 1 v1 (synthesized design)

Date: 2026-10-07. Branch: `claude/compression-spartan-whir`, based on e402a6d99.
Inputs: `m1-arena/GROUNDING.md` and the three candidates (`m1-arena/candidate-{a,b,c}.md`).
Numbers marked "est." are estimates. Nothing here is measured yet.

## Problem

At the end of folding, the accumulator has 16 running CE(b) claims and 1 fresh CCS claim,
with 17 witnesses (about 221.7 MB). The task is a post-quantum proof of about 114 bits that
contains no witness.

- **Layer 0:** Π_CCS, then Π_RLC, with no Π_DEC. The result is one CE(B, L) claim. Paper
  Lemma 1 (strong-weak composition, Theorem 12) makes this a reduction of knowledge.
- **Layer 1:** an argument of knowledge for that one claim, covering all five conjuncts of
  Definition 21 over the full carrier W.
  - In v1 the verifier still reads the Ajtai key and the matrices.
  - Milestone 2 replaces those reads with setup-commitment openings.
  - Milestone 3 runs this verifier inside a shrink circuit.

## Usage (caller's view)

```rust
let prover = circuit.prover(Engine::Optimized, minimum_security_bits)?;
let mut proof = prover.prove(z0, &steps[0])?;
for step in &steps[1..] { proof = prover.extend(&proof, step)?; }
let finished = prover.finish_with_spartan(&proof)?;   // no witness; `proof` stays usable
let bytes = finished.to_bytes();

let verifier = Verifier::from_package(&circuit, Engine::Optimized, minimum_security_bits)?;
let finished = verifier.decode_final_proof(&bytes)?;
verifier.verify(&expected_state, &finished)?;          // same verb as for `Proof`
```

`finish_with_spartan` rejects an initial proof, because it has no fold to finish.
`Verifier::verify` takes `Proof` or `FinalProof` through a sealed trait, so existing call
sites do not change.

## Shape

### New crate `crates/neo-spartan`

This crate owns layer 1 and every Plonky3 0.8 type. It uses renamed dependencies
`p3-*-v08 = { package = "p3-*", version = "=0.8.0" }`. The workspace stays on 0.5.3.

| File | Owns |
| --- | --- |
| `lib.rs` | Contract header; `Relation`, `Proof`, `Error`, `prove`, `verify`; the layer-1 Fiat–Shamir schedule as one sequence |
| `field.rs` | `Gl`/`Ext` aliases (Ext = cubic trinomial x³ − x − 1), 0.5↔0.8 conversions, Re/Im projection, eq tables |
| `hash.rs` | Poseidon2 width 16 as a p3 0.8 permutation built from `neo_ccs` round constants; Merkle (MMCS) and challenger types; transcript handoff |
| `pcs.rs` | WHIR configuration from the security budget; commit, `open_at`, `verify_at`; point-order reversal |
| `ring.rs` | The single list of 37 ring rows; λ powers; block weights ω_b; targets Ȳ; quotient Q; the v1 streamed Ω̃(s_hi) |
| `gkr.rs` | logUp fractional-sum GKR (Papini–Haböck), prover and verifier |
| `norm.rs` | Public histogram over [−H, H], table sum Σ m_t/(β − t) |
| `sumcheck.rs` | Small round helpers: univariate evaluations, interpolation, linear product sum-check |
| `codec.rs` | Strict `Proof` bytes |

**Public surface:**
- `Relation::new(key, matrices, public_blocks, point_variables, norm_bound, security_bits)`
- `prove(&Relation, transcript, &CeClaim, &Mat<F>)`
- `verify(&Relation, transcript, &CeClaim, &Proof)`
- `Proof::{to_bytes, from_bytes}`
- `Error`

The transcript is passed by value (`Poseidon2Transcript`). Ownership therefore moves to
layer 1, and no other code writes to it after the handoff.

### Changes in `crates/nightstream`

- **`folding/compose.rs`:** `prove_parent_with_rows` (C, then R) and `verify_parent`. The
  existing fold becomes "parent + Π_DEC", so there is one code path.
- **`lifecycle/finish.rs` (new):**
  - `FinalProof {state, running claims, fresh claim, Π_CCS proof, neo_spartan::Proof}`
  - `finish_with_spartan` and `verify_final`
- **Codec:** magic `NS-FINAL-PROOF01`; reuses the claim writers in `encoding.rs`.
- **`circuit.rs`:** `Prover::finish_with_spartan`, `Verifier::decode_final_proof`, and the
  sealed verify trait.

## Protocol

### Layer 0

This is the unchanged prefix of a normal fold.

**Prover:**
1. `checked_prior_state`, `prepare_running`, `validate_running_parent_authority`.
2. Π_CCS: reset, then "fold/v2", statement, α/γ, 28 rounds, outputs.
3. Π_RLC: the parent claim plus the dense parent witness Z = Σ ρ_i Z_i (54 × n_R).

**Verifier:**
1. Statement checks (shapes, canonical children, state hash = fresh.x).
2. `prepare_running` with the recomputed digest.
3. `pi_ccs::verify`, then `pi_rlc` recomputation through `rlc_public`.

The parent is recomputed; it is never read from the proof.

### Layer 1 algebra

The witness table is z(64b + l) = Z[l, b] for b < n_R and l < 54. It is 0 elsewhere on the
2^26 cube: the lane is the low 6 bits and the block is the high 20 bits. Honest values satisfy
|z| ≤ H = sources·T·(b−1) = 17·216·1 = 3,672 (Def. 22 guard; T from Theorem 5).

Ring rows Σ_b g_{ρ,b} ⋆ z_b = y_ρ in R_F (37 rows, one ordered list):

| Rows | g_{ρ,b} | y_ρ |
| --- | --- | --- |
| 22 commitment rows | key element a_{i,b} | parent c, row i |
| Eval_K Re, Im | bar(Re/Im χ_r(54b + ·)) | Re/Im of Eval_K |
| Eval_A_j Re, Im (j < t) | bar(Re/Im (M_jᵀχ_r)_b) | Re/Im of Eval_A_j |
| 5 public rows | [b = j]·1 | X_j |

- **Batching:** λ batches the 37 rows into one: ω_b = Σ_ρ λ^ρ g_{ρ,b}. Because bar is
  F-linear, the Eval part is bar(u_b), with u(c) = Σ_ρ λ^ρ·(projected weight at c).
- **Quotient:** the prover sends Q = (Σ_b ω_b z_b − Ȳ) div Φ81, which has degree ≤ 52.
- **Check at ζ:** at a random ζ, Σ_{b,l} z_{b,l} Ω_b ζ^l = V, where Ω_b = ω_b(ζ) and
  V = Ȳ(ζ) + Q(ζ)·Φ81(ζ).
- **Linear sum-check:** the claim is Σ_x Ω̃(x_hi)·Λ(x_lo)·z(x) = V, with Λ(l) = ζ^l for
  l < 54 and 0 otherwise. Block bits are bound first, on u(b) = Σ_l Λ(l) z(64b + l). Lane
  bits are bound last. This ends at a point s.
- **Norm:** logUp with a public histogram m over [−H, H] that covers all 2^26 cells
  (padding counts as 0). The fractional sum Σ_x 1/(β − z_x) is proven by GKR with leaves
  (p, q) = (1, β − z). The child bit is the top bit. The leaf claim gives z̃(ρ) = β − q̃(ρ).
- **Opening:** WHIR `open_at` on the one committed column, at two points: ρ (GKR leaf) and
  s (linear sum-check).

### Layer 1 transcript

| # | Step |
| --- | --- |
| T0 | `tr` (by value): absorb the domain chunk "Nightstream/SuperNeo/compress/v1"; seed = 4-word squeeze |
| T1 | `ch = DuplexChallenger<Gl, Perm16, 16, 12>`; observe the seed, the profile words (shape, H, WHIR parameters) and the parent statement (c, X, r Re/Im, Eval_K Re/Im, Eval_A Re/Im) |
| T2 | WHIR commit of z (2^26); p3 observes the root |
| T3 | observe the histogram (2H + 1 words) |
| T4 | β ← Ext |
| T5 | GKR, 26 layers: observe the root (P₀, Q₀); per layer, μ, rounds (h(0), h(2), h(3)), child values, τ |
| T6 | λ ← Ext |
| T7 | observe Q (53 Ext) |
| T8 | ζ ← Ext |
| T9 | linear sum-check, 26 rounds (h(0), h(2)) |
| T10 | WHIR `open_at` / `verify_at` at [ρ, s]; p3 adds OOD, evaluations, PoW and queries |

**Verifier final checks:**
- root: P₀ = Q₀·Σ m_t/(β − t), and Q₀ ≠ 0;
- GKR leaf: the WHIR value at ρ equals β − q̃(ρ);
- linear: the last claim equals Ω̃(s_hi)·Λ̃(s_lo)·z̃(s).

In v1, Ω̃(s_hi) is streamed from the key (SHAKE), χ_r and the matrix rows.

## Soundness map

Ext ≈ 2^192. L = WHIR list size: p3 draws its OOD samples after our challenges, so every
outer term is multiplied by L.

| Step | Error |
| --- | --- |
| Layer 0 (Lemma 1) | exact census from `padded_row_security_summary_for_shape` |
| logUp at β | (2^26 + 2H + 1)·L/|Ext| |
| GKR (26 layers, degree 3) | (Σ 3k + 2)·L/|Ext| |
| λ (36 powers), ζ (degree 106), linear (26 × degree 2) | (36 + 106 + 52)·L/|Ext| |
| WHIR opening (Johnson bound, `prescribed_security`) | derived budget |
| Poseidon2 Merkle and sponge, 4-word digests | 2^-128 classical (same assumption as today) |

The WHIR security target is derived, not chosen:

    budget = 2^-minimum_security_bits − (layer-0 error) − (outer layer-1 terms)

`Relation::new` returns `Error::Budget` when p3 cannot reach the target. PoW bits = 0 until
the owner sets a value.

**Extraction:** WHIR gives z. logUp and GKR give every cell in [−H, H], so ‖z‖∞ ≤ H < B.
The 37 rows give conjuncts 1, 2, 4 and 5. Lemma 1 with Lemma 4 then gives CCS(b) × CE(b)^16,
and the state hash binds them to `State`.

## Synthesis decision

The base is candidate C. The cross-judge scored C 28, A 25 and B 22. C has the strongest
first slice, a rank-one quotient check, and a milestone-2 key path that needs one opening.

**Grafts:**
- **A (norm):** a public histogram over [−H, H] with one z-side GKR. This replaces C's
  committed 2^17 count column, its second GKR and its stacked two-table commitment.
  - It gives less code and a fixed proof shape for milestone 3.
  - It removes an unconfirmed p3 stacking path.
  - H comes from the parameters (Def. 22), not from a new constant.
- **A (layout):** one 2^26 column on the (block, lane) cube, opened at two points (ρ, s).
  This replaces C's 54 columns and θ-merge.
- **A/B:** the transcript is passed by value, so nightstream hands ownership over.
- **A/B:** the parent statement is observed in the layer-1 challenger, so neo-spartan is a
  standalone argument of knowledge.
- **A/B:** the WHIR level is derived from the budget, and `Error::Budget` signals failure.
- **B:** pow_bits = 0 until the owner sets a value.
- **B:** a semantic cross-check test of the ring rows against `eval_real_v1_1_openings`.
- **A/B:** `prove` does not refuse a false linear statement, so mutation tests reach the
  verifier.
- **A:** `finish_with_spartan` rejects an initial proof.
- **A:** the sealed verify trait (no public API break).

**Rejected:**
- **B's digit planes:** 16 times the committed data, about 30 GB peak (est.).
- **C's exact (−B, B) committed table:** more code and an unconfirmed stacking API, for a
  bound that H already implies.
- **C's crate-level ownership of the fold transcript:** split ownership.
- **B's public API break.**

## Tradeoffs accepted

- **Bound:** we prove |z| ≤ H = 3,672, which is stronger than < B, in exchange for a fixed
  29 KB histogram and no second commitment.
- **Two opening points:** in exchange for a smaller prover (no 2^26 eq table).
- **Two p3 generations in the build:** in exchange for unchanged fold bytes and pinned
  constants.
- **The statement (16 claims, about 275 KB) travels in clear:** the shrink layer absorbs it
  in milestone 3.
- **No zero knowledge:** the histogram, Q and the openings leak information about z.

## Open questions for the owner

1. **Fresh tail.** The terminal verifier checks that the fresh witness's 26 tail lanes are
   zero. CE(B), Lemma 1 and every intermediate fold lack this check, so the compressed
   verifier cannot keep it. Is it a required semantic?
2. **Johnson bound.** Is JohnsonBound acceptable? The SoK (2026/1367) states that soundness
   up to Johnson is proven, but a reviewer should confirm p3-security's constants.
3. **Bound H.** Is it acceptable to prove |z| ≤ H (from the parameters) instead of exactly
   < B?
4. **WHIR settings.** Do you approve PoW bits, rate and folding factor? v1 uses PoW 0,
   rate 1/2 and folding 4 (the WHIR paper's benchmark setting).
5. **Backends.** PaperExact and Crosscheck return `Unavailable` for finishing in v1. Layer 0
   runs on the CPU for every engine. Is that acceptable?
6. **Digest width.** 4-word Poseidon2 digests give about 85 bits under the BHT quantum
   attack, which is the same assumption the current system makes. Is that acceptable?

## Implementation slices

Each slice has its own success criteria.

1. **Spike.** Crate, dependencies, `hash.rs` and `pcs.rs`.
   - Test: the p3 0.8 Poseidon2 matches ours.
   - Test: WHIR commit and two-point `open_at`/`verify_at` on a 2^12 table, through a
     seeded challenger, with tampering rejected.
   - Test: the production-shape configuration (2^26, Ext, JB, PoW 0) reaches the required
     bits.
2. **Layer-1 math.**
   - `ring.rs`: rows matched against the repository commitment and `eval_real_v1_1_openings`.
   - `norm.rs` and `gkr.rs`: round trip plus tamper tests.
   - `sumcheck.rs`.
   - `prove`/`verify` on synthetic CE(B) claims with the real key prefix, plus mutation
     tests.
3. **nightstream.** `prove_parent`/`verify_parent`, `finish.rs`, the codec, the public API,
   a toy test on the parity fixtures, and an ignored production test.

## Implementation status (2026-10-07, branch `claude/compression-spartan-whir`)

All three slices are implemented. The design held; deviations:

- **Plonky3 Poseidon2.** `new_from_rng_128` with the workspace seed gives the
  workspace permutation exactly. The parity test is pinned, so no wrapper is needed.
- **Π_RLC parent on the wire.** The final proof carries `pi_rlc::Proof` and reuses
  `pi_rlc::verify`, which recomputes the parent and compares it. This avoids a new
  `derive_parent` function and costs 16 KB.
- **v1 verifier weights.** The v1 verifier rebuilds the block weights with the
  prover's function. There is one formula; milestone 2 replaces it with openings.
- **Point length.** It comes from the recomputed parent claim, so there is no second
  copy of the cube formula.

Measured (production Poseidon2 circuit, CPU, one fold, `poseidon_finish_with_spartan_verifies`):

| Item | Value |
| --- | --- |
| `finish_with_spartan` | 25.5 s |
| Final proof | 918,934 bytes (accumulator proof: 221,738,816 bytes) |
| Final verification | 9.8 s |
| Whole test (compile + base + fold + finish + verify) | 62 s, peak RSS 19.2 GB |

Synthetic layer 1 at a 2^16 cube: prove 33 ms, verify 9 ms, 156 KB proof at 100.5 bits.

Tests:
- `neo-spartan`: 11 tests.
  - Poseidon2 parity; WHIR two-point openings with tampering; the production-shape
    budget.
  - GKR leaf and root against direct sums; forged histograms.
  - Ring rows against the repository commitment and `eval_real_v1_1_openings`.
  - Prove/verify with mutations of every statement and proof part; false witnesses.
- `nightstream`: toy finish on both parity fixtures, with layer-0 and layer-1
  mutations; an ignored production test.
