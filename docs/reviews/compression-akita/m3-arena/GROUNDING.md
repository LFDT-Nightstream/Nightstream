# Grounding: compression milestone 3 (the shrink proof)

Date 2026-10-07. Repository: /Users/nicarq/starstream/develop/nightstream-clean-up/.claude/worktrees/gallant-hamilton-76061a,
branch `claude/compression-spartan-whir` (PR #171), milestone 2 at commit dd8a38805.
Read first: `docs/reviews/compression-akita/M2-DESIGN.md` (incl. its "Result" section),
`crates/neo-spartan/src/*.rs` (layer 1, ~3.6k lines), `crates/nightstream/src/lifecycle/{finish,verify}.rs`.

## 0. Task and owner decisions

M3 must turn a `FinalProof` (M2) into a **shrink proof of at most 200 KB, goal 100 KB or less, verified natively in about 100-400 ms** (owner, 2026-10-07; first stated as 150 KB) that a native verifier
on the owner's own chain checks against the same statement: the expected `State` and the trusted
compression key. Owner decisions:

- 2026-10-07: post-quantum, about 114 bits proven (Johnson bound, no capacity conjectures),
  existing published parts, no new cryptographic assumptions.
- 2026-10-07: **route = our own WHIR stack** (sum-check + p3-whir 0.8 over the cubic Goldilocks
  extension, as in `neo-spartan`). Plonky3-recursion was rejected: its Goldilocks challenge field is
  degree 2 (about 2^128), which leaves roughly 96-100 proven bits for a STARK of our size.
- 2026-10-07: **proof-of-work grinding of up to 20 bits is allowed on the last layer.**
- Standing: Poseidon2-only hashing in proof/transcript paths (no Keccak/Blake without approval);
  hard 64 GB machine memory cap, one heavy job at a time; every test <= 300 s (`--release`);
  files < 1,500 lines; tests in `tests/` directories; no new Rust features or env vars without
  approval; lean code; the folding protocol, b = 2, k_rho = 16 and the SHAKE128 key are unchanged.
- M2 may be retuned (WHIR rates, query counts, table layout) if that makes the shrink circuit
  smaller; M2's soundness argument must stay intact.

## 1. What the shrink proof must establish

`PreparedLifecycle::verify_final(expected_state, key, proof)` (crates/nightstream/src/lifecycle/finish.rs):

1. `check_statement` (lifecycle/verify.rs:192): shape checks; recompute the terminal state hash
   `pi_ccs_v1_1_state_hash(preimage)` over the 16 running claims (each commitment is κ = 22 ring
   elements × 54 coefficients) and the parent public input; compare with the fresh claim's public
   input.
2. `prepare_running` + `validate_running_parent_authority` (folding/compose.rs): PiDEC verify that
   the 16 running claims are the canonical child family of their parent authority (ring
   arithmetic over commitments and evaluation claims).
3. `pi_ccs::verify` (folding/pi_ccs.rs:89 -> neo-reductions/src/engines/pi_ccs_joint_protocol.rs:477,
   `verify_with_binding`, digest-only transcript binding): the PiCCS sum-check over K = F_p[u]/(u^2-7).
4. `pi_rlc::verify` (folding/pi_rlc.rs:84): the random linear combination of 17 commitments and
   claims by small ring challenges -> the parent CE(B) claim.
5. `neo_spartan::verify(relation(key), transcript, parent, layer1)`: M2 layer 1.

Every step is deterministic given the proof; loop bounds and shapes are fixed by the package and
the key. The only data-dependent "control" is which Merkle leaves/paths are opened (query indices
drawn from the transcript), the direction bits of paths, and the domain points `g·ω^index`.

Transcripts: layer 0 uses `neo_transcript::Poseidon2Transcript` (crates/neo-transcript/src/poseidon2.rs);
layer 1 starts with `hash::challenger` (absorb a domain chunk into the layer-0 transcript, squeeze a
4-word digest, seed a p3 `DuplexChallenger<Gl, Poseidon2Goldilocks<16>, 16, 12>`), then all p3-whir
transcript work goes through that challenger (`observe`, `sample_algebra_element`, `sample_bits`,
grinding check).

## 2. Measured facts (production package, CPU, this Mac)

M2 (dd8a38805): finish 39.7 s; final proof 1,600,157 B; verify 32 ms; finish-test peak RSS 37.3 GB;
setup 63-69 s / 27 GB disk; key 1,912 B.

**Poseidon2 permutations in the M2 layer-1 verifier** (counted, one honest verify, cumulative marks):

| Phase | Permutations |
| --- | --- |
| handoff, shape/key/profile/statement words | 201 |
| P0 root, histogram, early batched GKR (5 trees, 26 layers), early values | 1,068 |
| ring/quotient/linear sum-check, finals, partials | 45 |
| P1 root, 125 query indices | 12 |
| 125 setup leaves (26 oracles × 8 values = 208 words) + 24-level paths | 5,166 |
| late GKR + P1 opening (2^26 cells, rate 1/2, 129 batches) | 8,470 |
| P0 opening (2^28 cells, rate 1/2, 7 batches) | 8,627 |
| **total layer 1** | **23,589** |

Layer 0 (state hash over ~20k words, PiCCS/PiRLC transcript) is not counted; estimate 2k-4k.

**WHIR (p3-whir 0.8, one base-field column, 2 prescribed-point openings, 114 bits Johnson bound,
folding 4), proof bytes = commitment + opening (bincode):**

| log2 size | rate | PoW bits | bytes | prove time |
| --- | --- | --- | --- | --- |
| 20 | 1/2 | 0 | 188,413 | 0.08 s |
| 20 | 1/2 | 20 | 159,133 | 0.18 s |
| 20 | 1/16 | 20 | 81,557 | 0.60 s |
| 20 | 1/64 | 20 | 69,173 | 2.2 s |
| 20 | 1/256 | 20 | 61,941 | 8.1 s |
| 22 | 1/16 | 20 | 89,685 | 2.5 s |
| 22 | 1/64 | 20 | 74,965 | 8.5 s |
| 24 | 1/2 | 20 | 213,162 | 1.5 s |
| 24 | 1/16 | 20 | 109,002 | 8.7 s |

Consequence: one WHIR opening of a ~2^24-2^25 witness at rate 1/16 with 20-bit PoW fits the
150 KB budget with ~35 KB left; **two** WHIR openings plus M2-style honest-fold leaves (≈ 30
queries × ~1.4 KB at a 1/16 setup rate) do not fit. So the last layer must commit once (or the
second commitment must be tiny).

Poseidon2 (workspace, crates/neo-ccs/src/crypto/poseidon2_goldilocks.rs; neo-spartan/src/hash.rs):
width 16, rate 12, digest 4, S-box x^7, 8 full rounds (4 + 4) and 22 partial rounds = 150 S-boxes
per permutation; constants from `new_from_rng_128(ChaCha8Rng::from_seed(SEED))`. A rank-1 constraint
system needs about 4 product constraints per S-box (x2, x4, x6, x7) ≈ 600 per permutation; a CCS
with a degree-7 term needs 1 constraint per S-box.

## 3. Our stack (what exists to build on)

`crates/neo-spartan`: `sumcheck.rs` (rounds as h(0), h(2..d)); `gkr.rs` (batched fraction GKR,
product leaves); `pcs.rs` (p3-whir `PrescribedPointPcs` over several base-field tables, opened at
caller points; rate and PoW are constants today: rate 1/2, PoW 0, folding 4, Johnson bound, level
search to a target); `setup.rs` (honest setup oracles + fold); `matrix.rs` (two-level scatter);
`mle.rs` (closed-form MLEs); `field.rs` (Ext = cubic trinomial x^3 - x - 1, Kx = K ⊗ Ext);
`hash.rs` (workspace Poseidon2 in p3 0.8 types, Merkle hash/compress, challenger handoff).
`crates/neo-ccs` has a CCS structure type used by the folding scheme (the F′ application itself is
a CCS with Poseidon2 gadgets, but it is generated by the Lean pipeline; do not depend on Lean).

## 4. Open design questions (for the candidates)

1. Arithmetization of the verifier: R1CS + Spartan (2019/550), CCS + SuperSpartan (2023/552) with
   a degree-7 S-box term, an AIR, data-parallel GKR for the permutations (2023/1284 style), or a mix.
2. How the verifier of the shrink proof learns the circuit's matrix evaluations: holography
   (setup commitments, costs proof bytes), uniform/structured blocks (cheap closed forms), or the
   native verifier regenerating the circuit and evaluating it (costs verifier time, no key).
3. Number of shrink layers (one fat layer vs. a holographic inner layer + a small outer layer).
4. Whether to retune M2 first (higher P0/P1/setup rates, fewer queries) to cut the 23.6k
   permutations, at what prover memory/time cost.
5. How witness generation works (instrument the native verifier? a second "circuit" implementation
   of the same verifier that both computes and records constraints?) and how prover and circuit stay
   provably in sync (one source of truth).
6. The in-circuit p3-whir verifier: replicate p3-whir 0.8 `verify_at` exactly (stacked layout,
   sum-check, OOD, STIR queries, final polynomial, grinding) or replace M2's PCS with something
   simpler to verify in-circuit.
7. Native verifier budget on the chain: time, key size. M2 verify is 32 ms with a 1.9 KB key.
8. Prover budget: the production test must stay under 300 s per invocation (compile ~15 s + finish
   ~40-47 s today) and under 64 GB.
