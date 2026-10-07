# Grounding: SuperNeo compression milestone 1 (layer 0 + layer 1 v1)

Date 2026-10-07. Repository: /Users/nicarq/starstream/develop/nightstream-clean-up/.claude/worktrees/gallant-hamilton-76061a
(branch `claude/compression-spartan-whir`, base e402a6d99 = tip of the open PR stack #163 → #168 → #169 → #170).
Paths below are relative to that root. "Derived" marks numbers computed, not measured.

## 0. The task and the owner decisions

Nightstream is a SuperNeo lattice folding IVC (Goldilocks). Today the final proof is the
whole accumulator plus 17 witnesses (~221.7 MB, derived) and the terminal verifier reruns
every check with the witnesses. There is no compression step. The owner approved building one.

Owner decisions (2026-10-07):
- Final proof must be post-quantum. About 114 bits of security is the target.
- Use existing, published parts. No new cryptography. Spartan-style sum-checks + WHIR
  (Plonky3 `p3-whir`) is the approved route (the SuperNeo paper names "Spartan with a FRI-based PCS").
- Later milestones (NOT this one): setup commitments for the Ajtai key and the matrices
  (so the verifier becomes small), then a recursion "shrink" layer that proves the layer-1
  verifier in a small circuit (final proof target <= 150 KB on the owner's own chain; EVM
  would need Keccak in the last layer, Midnight an in-circuit verifier). The design of this
  milestone must not block those.
- This milestone: **layer 0 + layer 1 v1**.
  - Layer 0: at the end of folding, run Pi_CCS + Pi_RLC on the final 16 running claims + 1
    fresh claim, and stop (no Pi_DEC). Output: one CE(B) claim + its witness.
  - Layer 1 v1: a Spartan/WHIR-style argument of knowledge for that one CE(B) claim, so no
    witness is sent. In v1 the verifier may read the Ajtai key and the matrices itself
    (O(key + nnz) work, seconds); the proof must not contain witnesses.
  - Public entry point named in the repo rules: `finish_with_spartan` (lifecycle names:
    prove, extend, finish_with_spartan, verify).

## 1. Paper authority (docs/superneo-paper-v1_2)

- Lemma 1 (02_technical_overview.md:87-89): Pi_RLC ∘ Pi_CCS is a **reduction of knowledge**
  from CCS(b, L)^K × CE(b, L)^k to CE(B, L) (Pi_CCS strong, Lemma 7; Pi_RLC weak, Lemma 8;
  strong-weak composition, Theorem 12). Theorem 3 adds Pi_DEC to get CE(b)^k. So stopping
  after Pi_RLC and proving CE(B) with an argument of knowledge is paper-supported (sequential
  composition, Lemma 4).
- Definition 21 (07_superneo_folding_scheme_for_ccs.md:15-24), relation CE(B, L), instance
  (c, x, r ∈ K^{log m}, y ∈ R_K, {y_j ∈ R_A}_{j∈[t]}), witness z:
  1. c = L(z)            (Ajtai commitment of the packed ring vector)
  2. x = L_in(z)         (public input = first n_{R,in} ring entries)
  3. ||z^b||_∞ < B       (B = 2^16)
  4. y = h~(r), h = Emb(Trans(Pad))·z          (Eval_K, ring-valued: 54 K coefficients)
  5. y_j = h~_j(r), h_j = Emb(Trans(M_j))·z    (Eval_A, j = 1..t, ring-valued)
  All conjuncts except the norm are linear in z. The CCS gate itself is NOT part of CE(B):
  Pi_CCS already reduced the fresh CCS claim into the Eval_A claims. So layer 1 is
  "linear claims + norm", no Spartan zero-check of the CCS polynomial is needed.
- Definition 22: strong sampling set with expansion T, (K + k)·T·(b − 1) < B; commitment
  (2B, C)-relaxed binding. Here K + k = 17, T = 216, so honest parent norm <= 17·216 = 3,672.
- The SuperNeo paper gives no decider construction. Neo §1.5 sketches Spartan + FRI.

## 2. Code facts (from three explorers)

### 2.1 Public API and lifecycle (crates/nightstream)
- Exports (src/lib.rs:12-13): Circuit, Prover, Verifier, Engine, EngineError, Error,
  Proof (= lifecycle::Stage1Envelope), State (= Stage1State {iteration, z0: [F;4], current: [F;4]}).
  `lifecycle` and `folding` are private modules; everything in them is pub(crate).
- circuit.rs: Circuit::{compile L67, write L85, load L92, identity L102, prover(engine, min_bits) L107};
  Prover::{prove L139, extend L145-160, encode_proof L163}; Verifier::{from_package L177,
  compile L186, decode_proof L205, verify(expected_state, proof) L210}. No finish/compress entry.
- ProofState::{Initial, Active{running: RunningInstance, latest: LatestInstance}} (lifecycle/mod.rs:131-136).
  RunningInstance {claims: Vec<CeClaim> (16), witnesses: Vec<Mat<F>> (16 signed-unit 54×814,144),
  parent_authority: Option<CeClaim>} (folding/claims.rs:14-18). LatestInstance holds one fresh CcsInstance.
- After the first `prove`, running claims are `canonical_zero` (zero commitments/evals, r = 0^28).
- extend (lifecycle/extend.rs:47-93): checked_prior_state (step_inputs.rs:271-309: recompute state
  hash, require fresh.x == encHash(digest)) → prepare_running (extend.rs:98-140: set every running
  fold_digest = recomputed digest, pad evals to 64, rebuild parent_authority) →
  validate_running_parent_authority → PreparedLifecycle::prove (prove.rs:21-77) → step_inputs →
  complete_proved_step (commit new fresh witness).
- Fold composition (folding/compose.rs:7-39): Pi_CCS (pi_ccs::prove_from_parts_with_rows,
  pi_ccs.rs:29-57) → Pi_RLC (pi_rlc::prove_refs, pi_rlc.rs:44-64 → Output{claim, witness: Mat<F>})
  → Pi_DEC (pi_dec::prove_with_production_key). Verify (compose.rs:40-55): pi_ccs::verify
  (returns 17 CeClaims) → pi_rlc::verify (recomputes the whole parent via rlc_public) → pi_dec::verify.
- **Layer 0 already exists in tests**: tests/lifecycle_native/staged_fold.rs::ccs (L214-313) and
  ::rlc (L314-369) run C then R without DEC and replay the verifier. staged.rs::load_claims
  (L274-292) shows the digest normalization (checked_prior_state + prepare_running).
- Backends (engine/mod.rs:44-50): Metal, PaperExact, Optimized, Crosscheck. Each implements C/R/D as
  ONE function (compose.rs, engine/metal.rs:15-107, paper_exact.rs:51-119, crosscheck.rs:13-82 with
  require_match on the whole NifsProof). Metal's Pi_RLC (split_rlc_witnesses) keeps the parent on
  the device and returns only the 16 signed-unit digit planes (Z_parent = Σ 2^i D_i, D_i ∈ {−1,0,1});
  there is no Metal API that returns the unsplit parent.
- Terminal verifier (lifecycle/verify.rs:47-247), [S] = statement only, [W] = needs witness:
  1 [S] state equality, iteration bounds; 2 [S] Initial case; 3 [S] 16 claims/16 witnesses;
  4 per running child: [S] no adv, commitment shape (54×22 = 1,188 words), m_in = 270, r.len = 28
  and shared, eval shapes/zero surplus, eval_a count t; [W] X == project(witness);
  5 [S] fresh: no adv, commitment shape, x len 270; 6 [S] check_pi_ccs_v1_1_canonical_children
  (inputs.rs:231-257); 7 [S] rebuild state preimage (inputs.rs:150-210), Poseidon2 state hash,
  encHash, require == fresh.x; 8 [W] fresh tail zero, fresh Z prefix == x; 9 [W] recommit all 17
  witnesses (the ONLY norm check: signed units); 10-11 [W] streamed pass over all matrix rows:
  recompute running Eval_K/Eval_A at r and check the fresh CCS row by row.
  The verifier never reads parent_authority or the carried fold_digests.
- Codec (lifecycle/encoding.rs): magic NS-STAGE1-PROOF1; kinds INITIAL = 0, ACTIVE = 1.
  ACTIVE = header + 17 claims (16,464 B each) + 16 witnesses (13,026,304 B each, ±1 lane masks)
  + fresh C, x + fresh witness ≈ 221,738,816 B (derived).
- Tests: toy CCS fixtures in tests/engines/parity.rs (Fixture::new L74-133, bit() L136 t=1,
  selected_polynomial() L155 t=4, m = 55, m_in = 54, 54×2 witness; run() L204-300 full harness;
  run in CI, not ignored). Every production lifecycle test in tests/circuit_lifecycle.rs is
  #[ignore] (full F′ compile exceeds the 300 s cap). Crate-private tests attach via
  `#[path = "../../tests/..."]` modules (engine/mod.rs:153-155, lifecycle/mod.rs:158-160).
  Lifecycle helpers hard-code production constants (16 / 270 / 28 / t = 4), so toy fixtures test
  layer 0 only at the folding level, not the state-binding level.

### 2.2 Reductions (crates/neo-reductions, crates/neo-transcript, crates/neo-math)
- F = Goldilocks; K = BinomialExtensionField<F, 2>, u² = 7 (neo-math/src/field.rs:10-13); ring
  Rq([F; 54]) over F[X]/Φ81, Φ81 = X^54 + X^27 + 1 (ring.rs:10-18); bar transform with
  ct(bar(a)·b) = <a, b> (ring.rs:164-232). K is the only extension field in the workspace.
- Params (neo-params/src/lib.rs:95-107): κ = 22, b = 2, k_rho = 16, B = 2^16, T = 216.
- Pi_CCS driver: neo-reductions/src/engines/optimized_engine/paper_joint.rs (prove_with_matrix_rows
  :827-862); transcript schedule pi_ccs_joint_protocol.rs (bind_and_sample_with_trace :174-228,
  verify_with_trace :426-471). Transcript: reset_v1_1 (zero state!) → 12-word domain chunk
  "Nightstream/SuperNeo/fold/v2" → one statement stream (prior digest 4 words, fresh commitment
  1,188, fresh x 270) → α/γ read from lane pairs → 28 rounds (absorb 9 K coefficients, coin =
  lanes 0,1) → absorb all outputs' Eval_K/Eval_A. Running claims are bound ONLY through the
  prior digest (= state hash); the caller must authenticate them (pi_ccs.rs:38-52). The PiCCS
  verifier is matrix-free.
- Pi_CCS outputs: 17 CeClaims (fresh first), shared r' ∈ K^28; c and X copied from inputs; new
  Eval_K (64 slots, 54 used) and Eval_A (4 × 64). Pi_CCS already checks |z| < b for all 17 sources
  via range products on Eval_K[0], and the CCS gate for the fresh source.
- Pi_RLC: sampler common.rs:742-839 (absorb [4, i], squeeze_digest_v1_1, decode mod 5^54 into
  coefficients in {−2..2}, ρ_i = rot(a_i), strong-set check); guard 17·216·(b−1) < 2^16.
  Parent: c = Σ ρ_i c_i (claims.rs ajtai_rlc_mixer :66-75), X, Eval_K, Eval_A = Σ rot(ρ_i)·(·).
  Parent witness Z = Σ ρ_i Z_i: dense Mat<F> 54 × 814,144 = 43,963,776 F (~352 MB) on CPU.
  Verifier rlc_public (api.rs:657-805) recomputes the parent from the inputs; the `combined`
  claim on the wire is redundant.
- Pi_DEC uses no transcript; skipping it = not calling it. Without Pi_DEC NOTHING enforces the
  B bound on the parent: layer 1 must.
- neo-transcript/src/poseidon2.rs: Poseidon2 Goldilocks WIDTH 16, RATE 12, CAPACITY 4;
  fold_domain_chunk_v1_1 (:14), reset_v1_1 (:48), absorb_v1_1 (:55, permutes after every chunk,
  partial included), read_pair_v1_1 (:68, one K coin from lanes 2p,2p+1 without permuting; 6 per
  state then the caller absorbs a zero chunk), squeeze_digest_v1_1 (:77), state_prefix_v1_1 (:86),
  from_state_and_absorbed (:227). A new protocol gets its own domain chunk string. Layer 1 can
  continue from the Pi_RLC end state.
- Sum-check helpers: neo-reductions/src/sumcheck.rs poly_eval_k (:107), interpolate_from_evals
  (:144); generic RoundOracle/run_sumcheck_prover (:298) use a legacy framed transcript (tests and
  PaperExact only). MLE helpers: tensor_point, EqualityWeights, SuperneoZBlocks, evaluate_terminal_rows
  (window_eval.rs:22), pad_openings (openings.rs:83). Nothing computes the column vector M_j^T χ_r.

### 2.3 Data the layer-1 argument touches
- Witness layout: Mat<F> 54 rows (lanes) × 814,144 columns (blocks); logical coordinate c = 54·block
  + lane (common.rs:893-1008). Logical width m = 43,963,750; carrier W = 43,963,776 (26 tail lanes).
  After Pi_RLC the tail lanes are NONZERO in general; Eval_K (Pad) covers the full carrier, so layer 1
  must cover all W coordinates. Public input X = first 5 ring columns (270 words).
- Point convention: r ∈ K^28, low bit first (r[0] = bit 0) (neo_ccs::utils::tensor_point). p3 Point
  is big-endian (x_1 = MSB): reverse at the boundary. The 28-variable cube is bigger than the data
  (n < 2^20, W < 2^26): eq over unused high bits contributes Π(1 − r_t).
- Eval semantics: Eval_K = Σ_{c<W} χ_r(c)·bar(e_{c mod 54}) ⋆ z_{⌊c/54⌋} ∈ R_K; Eval_A_j =
  Σ_{i<n} χ_r(i)·Σ_b bar(M_j[i, 54b..54b+53]) ⋆ z_b ∈ R_K. Both are Σ_b bar(w_b) ⋆ z_b with
  w = χ_r (Eval_K) or w = M_j^T χ_r (Eval_A_j); constant term = <w, z>. Each Eval is 54 K-linear
  functionals of z (coefficient k = <σ_k(w), z>, σ_k mixes lanes via bar/rot). Stored as 64 K
  (54 + 10 zero) in CeClaim.eval_k / eval_a[j] (neo-ccs/src/relations.rs:466).
- The 54-lane packing is not a power of two: c = 54b + l does not factor as eq(r_hi, b)·eq(r_lo, l)
  (pad_openings handles it with 32 phases over 64-wide low factors).
- Ajtai (crates/neo-ajtai/src/nightstream_fprime_setup.rs): SETUP_ID
  "nightstream-ajtai-shake128-wide256-v1"; key element (row, block) = SHAKE128(SETUP_ID ‖ seed ‖
  row_u32_le ‖ block_u64_le), 54 lanes of 32 bytes reduced mod p (coefficient_block :83 fast path;
  key.rs::split_pair parallel). Rows κ = 22, columns 814,144 (one public matrix, apps bind a prefix;
  MAX_MESSAGE_COLUMNS 4,708,530). Commitment c_i = Σ_j a_{i,j}·z_j mod Φ81 (ring rows), stored
  column-major 1,188 words (types.rs:55). Production commit functions accept ONLY signed units
  {−1,0,1} (signed_unit_prefix_blocks :253 → RangeViolation); the RLC parent (|z| <= 3,672) has no
  direct commit path (its c is computed homomorphically). No function evaluates the key or A^T·w
  at a point. Full key = 22 × 814,144 × 54 = 967,203,072 coefficients (7.74 GB as u64).
- CCS matrices: n = 1,004,131 rows, carrier W columns, t = 4 [A, B, C, SboxInput], f = M0·M1 − M2
  + M3^7 (degree bound 8). Rows come from a Lean-generated matrix program visited as geometric runs
  (MatrixRun{column, column_count, coefficient, ratio}, sealed.rs:498) through the MatrixRows trait
  (superneo_eval/matrix_rows.rs:41, PackageRows in lifecycle/evaluation.rs:13). Nonzeros at M5
  (from PR #169 body): 57,823,067 / 15,665,400 / 11,356,963 physical R1CS (≈ 84.8M); per-CCS-matrix
  counts are not pinned in the repo.
- Today's terminal verify cost: 12.88 s CPU, 3.27 s Metal (README); 1.58 s Metal at e402a6d99.

### 2.4 External parts (Plonky3 0.8, checked from crates.io sources)
- p3-whir 0.8.0 (2026-09-23, MIT/Apache, ~18.4k lines) depends on p3-{challenger, commit, dft,
  field, matrix, maybe-rayon, merkle-tree, multilinear-util, security, sumcheck, symmetric, util,
  zk-codes} 0.8.0. The workspace pins p3 =0.5.3 (field, goldilocks, matrix, symmetric, poseidon2,
  challenger). Two semver-incompatible versions can coexist via renamed dependencies (the deleted
  wip-spartan did this with a git rev: `p3-field-whir = { package = "p3-field", ... }`); values
  cross as canonical u64.
- `WhirProver<EF, F, Dft, MT, Challenger, L>` implements `p3_commit::MultilinearPcs<EF, Challenger>`
  (commit / observe_commitment / open / verify) with bounds F: Field + TranscriptField + Ord,
  EF: ExtensionField<F>, MT: Mmcs<F>, Challenger: FieldChallenger<F> + GrindingChallenger<Witness = F>
  + CanSampleUniformBits + CanObserve<commitment>. A prescribed-point API (open_at / verify_at) exists
  at the 6b6a3b4 rev. One commitment can stack several tables; OpeningProtocol allows several points
  per table. WhirConfig uses EF::bits() for every error term, rejects folded domains beyond
  F::TWO_ADICITY = 32, and has a PoW budget. SecurityAssumption includes Johnson bound.
- p3-goldilocks 0.8.0 extensions: BinomiallyExtendable<2> (W = 7, equal to our K) and <5> (W = 3).
  No degree 4. We cannot add one outside Plonky3 (orphan rule). 7 is a non-residue mod p and 5 is
  odd, so u² − 7 stays irreducible over F_{p^5}: K ⊗ F_{p^5} = F_{p^10} = F_{p^5}[u]/(u² − 7).
- Security note: with EF = K (128-bit field), proven Johnson-bound proximity errors at 2^26 are far
  below 114 bits (estimate 2^-50..2^-80). EF = F_{p^5} (320 bits) has ample margin. K-valued claims
  can be split into Re/Im F_p-weighted linear claims so the prover works only in F_{p^5}; the
  verifier evaluates eq(r, s) for r ∈ K, s ∈ F_{p^5} in F_{p^10} and projects.
- Deleted crates/wip-spartan (git show 557f13ef6^:crates/wip-spartan/...): 6,382 lines, Spartan2 fork.
  Reusable (~800 lines): provider/pcs/whir_pc.rs (629: WHIR config, Merkle PaddingFreeSponge<Perm,16,12,4>
  + TruncatedPermutation<Perm,2,4,16>, DuplexChallenger<Gl,Perm16,16,12>, outer→inner transcript bridge
  via 4 squeezed "anchors", prescribed-point open_at/verify_at), provider/poseidon2.rs (150),
  tests/whir_pcs.rs (cross-version parity of K arithmetic and Poseidon2). Tied to the old path:
  r1cs/, spartan.rs, parallel_repetition.rs, goldi.rs (ff field), sumcheck.rs (base-field challenges,
  3 repetitions), polys/, traits (elliptic-curve-shaped Engine). It used EF = cubic (does not contain K),
  UniqueDecoding, λ = 125, PoW budget accidentally = KAPPA 18, rate 1/2.
- Poseidon2 width-16 constants are generated identically in p3 0.5.3 and the 6b6a3b4 rev (source
  evidence; no test). The Merkle tree and challenger for WHIR can be built over OUR permutation
  (wrap it as a p3 0.8 CryptographicPermutation) so hashing stays our exact Poseidon2 instance.

## 3. Repository rules that bind the design (CLAUDE.md / AGENTS.md)
- Poseidon2-only hashing in proof, transcript and public-digest paths; no mixed hash families.
- b = 2, k_rho = 16, B = 2^16 (hard rule). Only paper-faithful protocol changes (owner rule).
- No new Rust features or ENVs without approval. Keep code lean; no file over 1,500 lines.
- Public protocol API uses lifecycle names (prove, extend, finish_with_spartan, verify); hide
  state-machine constructors; small public surfaces, private fields, domain types.
- Tests go in tests/ (never in implementation files); `cargo test --release`; every test run
  <= 300 s; FoldingMode::Optimized only; a test added for a bug must fail on that bug.
- Digests are compression, not authority: every carried digest across a trust boundary must be
  recomputed from authoritative inputs or replayed into the transcript.
- Lean (formal/**) is another session's lane: do not edit it in this milestone.
- 64 GB machine memory cap; one heavy job at a time.

## 4. Sizes and rough costs (derived/estimated, not measured)
- Statement after the last extend: 16 running claims (each C 1,188 words + X 270 + r 56 + evals
  540 words) + fresh C, x ≈ 270 KB. Layer-0 messages: 28 rounds × 9 K + 17 × 5 × 54 K outputs ≈ 78 KB.
- One WHIR opening of a 2^26 base-field polynomial (Johnson bound, rate 1/2, 128 bits, cubic EF):
  317 KiB in the WHIR paper; F_{p^5} elements are larger (40 B vs 24 B).
- Parent witness in the (block: 20 bits, lane padded 54→64: 6 bits) layout = 2^26 entries.
- v1 verifier: O(key + nnz + W): one SHAKE expansion of 967M coefficients (seconds on CPU) + one
  pass over the matrix runs + O(W).

## 5. Open design questions (the candidates decide)
1. Where the argument lives (crate boundaries, who owns p3 0.8 types, what nightstream exposes).
2. Field for WHIR and challenges (F_{p^5} with Re/Im split, or something else) and how K claims map.
3. What is committed: the parent z (one polynomial) or its 16 signed digit planes, or other.
4. How the norm bound ||z||_∞ < B (or the tighter honest 2^12) is proven.
5. How C = A z over R = F[X]/Φ81 is checked (quotient-lift at a random point, coefficient-wise
   random combination, ...), and how the verifier gets the weight vector's MLE at the final point.
6. How the 54 + 4·54 ring-valued evaluation claims and x = L_in(z) are batched into sum-checks.
7. Transcript: one Poseidon2 duplex for layer 0 + layer 1 + WHIR, and domain separation given that
   Pi_CCS resets the transcript.
8. Proof object, codec and the verifier entry (CompressedProof? Verifier::verify on it?).
9. Backends: CPU first; how Metal/PaperExact/Crosscheck relate (Metal has no unsplit parent API).
10. Test plan within 300 s (toy fixtures + real key prefix), including mutation tests.
