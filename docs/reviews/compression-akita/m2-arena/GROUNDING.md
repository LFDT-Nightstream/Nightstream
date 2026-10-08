# Grounding: compression milestone 2 (succinct layer-1 verifier)

Date 2026-10-07. Repository: /Users/nicarq/starstream/develop/nightstream-clean-up/.claude/worktrees/gallant-hamilton-76061a,
branch `claude/compression-spartan-whir` (PR #171), milestone 1 at commit 3198aa960.
Read first: `docs/reviews/compression-akita/M1-DESIGN.md` (the M1 design and its implementation status),
`crates/neo-spartan/src/{lib,ring,gkr,norm,pcs,hash,field,sumcheck}.rs` (the M1 layer-1 code), and
`crates/nightstream/src/lifecycle/finish.rs`.

## 0. Task and owner decisions

Owner chose "Full M2 (succinct verifier)" on 2026-10-07: the layer-1 verifier must stop reading the
Ajtai key (967,203,072 SHAKE128 coefficients) and the CCS matrices (1,541,414,313 expanded entries).
Both must be handled by setup commitments opened in every proof (or another existing technique), so
that verifier work is polylogarithmic plus WHIR verification. Milestone 3 (next) will prove the
layer-1 verifier (plus the layer-0 replay) inside a "shrink" proof whose own verifier must also be
succinct; final proof target <= 150 KB on the owner's chain. Keep M3 in mind: every verifier step M2
adds will run inside the M3 circuit.

Standing constraints (CLAUDE.md, owner):
- Post-quantum, about 114 bits total; Poseidon2-only hashing in proof/transcript paths.
- Only paper-faithful protocol changes; existing published techniques, no new cryptographic
  assumptions. b = 2, k_rho = 16, B = 2^16 unchanged. The Ajtai key derivation (SHAKE128) and the
  folding protocol stay unchanged.
- Hard 64 GB machine memory cap; the prover process already peaks near 19 GB at finish (compile,
  prover and verifier in one process in the test). One heavy job at a time.
- Tests: tests/ directories, <= 300 s each, --release; files < 1,500 lines; no new features/env vars.

## 1. What the M1 verifier still reads

In M1, after the batched ring-row check at ζ and the 26-round linear sum-check (block bits first,
then lane bits; final point s = (s_lane: 6, s_block: 20)), the verifier needs

    Ω~(s_block) = Σ_b eq(s_block, b) · ω_b(ζ)

with ω_b(ζ) = T1_b + T2_b + T3_b + T4_b (λ = row-batching challenge, τ = bar(1, ζ, ..., ζ^53),
bar symmetric so Emb(bar(u))(ζ) = <u, τ>):

- T1 (key): Σ_{i<22} λ^i Σ_{l<54} ζ^l A[i, b, l]; A = coefficient_block(PRODUCTION_SEED, i, b).
- T2 (Eval_K, Pad): Σ_{l<54} τ_l · π_K(χ_r(54 b + l)); χ_r = eq over the 28 binary bits of the
  logical coordinate c = 54 b + l; π_K(v) = λ^{K,re} Re v + λ^{K,im} Im v.
- T3 (Eval_A): Σ_{l<54} τ_l · Σ_j Σ_i M_j[i, 54 b + l] · π_j(χ_r(i)).
- T4 (public): [b < 5] · λ^{x_b}.

So Ω~(s_block) = Σ T over all four. T4 is trivial. Today the verifier computes T1-T3 by one SHAKE pass
over the key and one pass over all matrix runs (verify 9.8 s CPU in the production test).

## 2. Measured sizes (production Poseidon2 package, e402a6d99 + M1)

- Rows n = 1,004,131; logical columns m = 43,963,750; carrier W = 43,963,776 = 54 × 814,144 blocks.
- Key: 22 rows × 814,144 blocks × 54 lanes = 967,203,072 coefficients (7.74 GB as u64).
- CCS matrix runs (from `visit_matrix_runs_until`): 38,533,993 runs:
  [A 4,377,403; B 3,818,087; C 11,652,969; SboxInput 18,685,534].
  37,572,008 runs have length 41 and ratio 3; 961,985 runs have length 1 and ratio 1.
  Every length-41 run starts at the same residue modulo 41: each is one physical field value
  expanded into 41 balanced-ternary digit coordinates with weights 3^k.
  Expanded entries: [176,542,323; 155,411,287; 477,085,809; 732,374,894] = 1,541,414,313.
- Layer-1 z commitment today: 2^26 cells (blocks 2^20 × 64 lanes), rate 1/2, WHIR in p3-whir 0.8.

## 3. Facts and ideas found so far (verify before relying on them)

- A 41-digit run starting at column c = 54 b + l (l < 54) touches only blocks b and b + 1 (41 < 54).
  So Σ_{k<41} 3^k g(c + k) with g(c) = eq(s, ⌊c/54⌋)·τ_{c mod 54} equals
  eq(s, b)·H0[l] + eq(s, b + 1)·H1[l], where H0[l] = Σ_{k<min(41, 54-l)} 3^k τ_{l+k} and
  H1[l] = Σ_{54-l <= k < 41} 3^k τ_{l+k-54}: two 54-entry tables the verifier computes in O(54·41).
  T3 then is a sparse sum over 38.5 M runs of val · E_row[i] · (E_blk[b]·H0[l] + E_blk[b+1]·H1[l]):
  a SPARK-style memory-checked sum whose lookup tables all have cheap closed-form MLEs
  (E_row = π_j(χ_r) over rows, a tensor; E_blk = eq(s_block, ·), a tensor; H0, H1 are 54-entry tables).
  Spartan's SPARK (offline memory checking) and logUp-GKR lookups are the published techniques; the
  crate already has a logUp fraction-sum GKR (`gkr.rs`).
- T2: χ_r is a tensor over the binary bits of c, but the sum runs over (b, l) with c = 54 b + l.
  Two known ways: (a) the verifier evaluates it with a digit-carry dynamic program over the bits of b
  (54 = 32 + 16 + 4 + 2; one DP per lane l, about 7k field operations each, about 360k in total), or
  (b) treat Pad as a fifth "matrix" (identity, W entries) inside the same memory-checked sum.
- T1: the only data with no structure. With the tensor layout (i: 5 bits, b: 20 bits, l: 6 bits),
  T1 = Π(1 + λ^{2^t}) Π(1 + ζ^{2^t}) · Ã(x(λ), s_block, x(ζ)) with x_t(c) = c^{2^t}/(1 + c^{2^t})
  ("scaled eq point" for power weights; check the formula), i.e. ONE opening of a committed 2^31-cell
  polynomial. Plain p3-whir needs the table, its rate-1/2 codeword and the Merkle tree in RAM:
  about 8-16 GB + 16-32 GB + 8 GB. That exceeds the memory budget together with the rest.
  Options seen: tighter flat layout (2^30 cells) plus a sum-check and a DP for the weight;
  splitting into several commitments opened sequentially (more WHIR proofs, bigger M3 circuit);
  setup data (codeword + tree) stored on disk with a custom streaming first WHIR round;
  per-proof regeneration from the SHAKE seed. None is decided.
- p3-whir 0.8 API (crate source in scratchpad/p3whir/p3-whir-0.8.0): `MultilinearPcs::{commit,
  observe_commitment, open, verify}`, `PrescribedPointPcs::{open_at, verify_at, prescribed_security}`;
  one commitment can hold several tables (`Layout::new_witness(Vec<Table>, folding)`); a table has
  many columns of equal arity; `OpeningProtocol` has per-table point schedules; the prover keeps the
  whole codeword and Merkle tree (`WhirProverData { layout, merkle_data }`). Our wrapper: `pcs.rs`.
  The p3-sumcheck crate (scratchpad/p3whir/p3-sumcheck-0.8.0) also has a jagged layout and a GKR /
  lookup layer (check what exists: `jagged`, `ring_switch`, `zk`, product/lookup utilities).
- Layer 1 already uses the cubic extension Ext (about 2^192); every new draw must be charged in
  `outer_terms` (lib.rs) over WHIR's candidate set; budget = caller minimum minus fold census error.

## 4. Questions the design must answer

1. How is T1 proven succinctly under the 64 GB cap (layout, commitment(s), prover memory and time
   per proof, setup storage, proof-size and M3-circuit cost)?
2. How is T3 proven (exact memory-checking protocol, committed setup columns, per-proof commitments,
   GKR/logUp instances, handling of the two run types and of the b + 1 block, costs)?
3. How is T2 handled (DP or lookup), with its M3-circuit cost?
4. How do these compose into one transcript and one WHIR batch where possible (number of WHIR
   openings per proof, their sizes, and the M3 circuit cost of verifying them)?
5. What does setup produce (a `Setup`/verifying key with roots), who computes it, how is it bound
   to the package identity, and how does the prover get the data it needs each proof?
6. Error budget and test plan (toy sizes under 300 s; production timing under 300 s).
