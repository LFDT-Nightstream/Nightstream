# Compression milestone 2: a succinct layer-1 verifier (synthesized design)

Date: 2026-10-07. Branch `claude/compression-spartan-whir` (PR #171), on top of M1.
Inputs: `m2-arena/GROUNDING.md` and the candidates `m2-arena/candidate-{a,b,c}.md`.
"est." marks an estimate; nothing here is measured yet.

## Owner decisions (2026-10-07)

- **Full M2.** The layer-1 verifier stops reading the Ajtai key (967,203,072 coefficients) and
  the CCS matrices (38,533,993 runs, 1,541,414,313 expanded entries).
- **Honest fold approved.** Setup data is opened by folding it against an honestly computed
  setup codeword (see the lemma below), not by a WHIR proximity proof.
- **Setup files on the prover's disk** (est. 29 GB, of which 23.6 GB is the key).
  Verifiers need only a small key.

## Base and grafts

The base is candidate C (`m2-arena/candidate-c.md`). The cross-judge ranked C and B equal
(22 points, A 21) and recommended C for its lowest M3 hashing and prover time with p3-whir
unchanged. C's design holds:

- **Two-level scatter.**
  - Per-run row values `e = χ_r(i)` are committed early, because r is known before layer 1.
  - Runs are summed into slots early (`U`).
  - Slots are summed into blocks late. The only late matrix data is 2^20 values (`Ω^A`).
- **One setup tree.** It holds the 22 key rows, the run columns (row, slot, coefficient) and
  the slot block. λ merges the key rows into one oracle.
- **Two per-proof WHIR commitments.**
  - P0 (early): z, e, U, b̂ (copy of the slot block), mult̂.
  - P1 (late): the two folded setup groups and `Ω^A`.
- **Eval_K (Pad)** by a long-division DP: 3,626 transitions at production size, verified
  against brute force.
- **The 41-digit runs** spill into block b+1 through the denominator b̂ + 1 of the block
  scatter. Two 54-entry lane tables per run class give the lane weights.

Corrections and grafts from the judge:

1. **OOD timing.** p3 0.8 draws the out-of-domain samples when it opens, not when it commits.
   C's "fact 3" is wrong as stated. The query count t = ⌈β' + 1 + log2 L1⌉ already
   charges the candidate list L1 of P1, so soundness holds. The lemma below states it
   correctly.
2. **Table widths.** P0 tables must have power-of-two widths for p3 `Table`. Split them into
   tables of width 1, 2 and 8. Total P0 stays ≤ 2^28 cells.
3. **Size estimate.** The layer-1 proof is about 0.9 MB, not 0.73 MB: a 2^28 opening is at
   least the M1 2^26 opening. M3 is about 22–23k Poseidon2 permutations.
4. **One source for the run tables.** After setup, the setup directory is the only source of
   the run and slot tables (from B). The prover's layer 1 needs no `MatrixRows`. Layer 0
   still uses the package rows.
5. **Spike first.** Measure p3 0.8 commit and open peak memory at 2^28 cells. Print the run
   census: distinct slots (must be ≤ 2^21; the judge's bound is 2.03M), per-lane counts, and
   the largest run count per (matrix, row).
6. **Tests from B.** Check that the succinct weights equal M1's `block_weights`. Mutating
   any single setup cell must be rejected under the honest key.
7. **Public API.**
   - `verify` keeps its sealed form for `Proof`.
   - `FinalProof` is checked with `Verifier::verify_final(state, key, proof)`, because the
     key is an explicit trusted input. M1's sealed `FinalProof` impl goes away, so there is
     one way to verify each proof type.
   - The key comes from `Circuit::compression_key()` (streaming, no disk) or from a pinned
     copy. A prover never supplies it.

## Honest fold (lemma)

**Setting.**
- O is a multilinear polynomial in n variables with monomial coefficients c_S.
- Its univariate form is Ô(Y) = Σ_S c_S·Y^{Σ_{i∈S} 2^i}, so Õ(y, y², y⁴, …) = Ô(y).
- The setup tree commits the exact codeword of Ô on a coset L, |L| = 2^{n+1} (rate 1/2).
- Leaf j holds the 8 values Ô(y_j·ω_8^v). Its root R_S is computed by the verifier (or
  pinned) from public inputs.

**Protocol.** The goal is to prove the claim Õ(x) = v at a point x ∈ Ext^n.
1. The prover sends w_u = Õ(u, x_{3..}) for u ∈ {0,1}^3. The verifier checks
   Σ_u eq(x_{..3}, u)·w_u = v.
2. Draw α ∈ Ext^3. The fold is g(X') = Õ(α, X'), with ĝ(Z) = Σ_u α^[u]·Ô_u(Z), where
   Ô(Y) = Σ_u Y^u·Ô_u(Y^8).
3. The prover commits g in P1 (WHIR). The P1 openings must give
   g̃(x_{3..}) = Σ_u eq(α, u)·w_u.
4. Draw t leaf indices j. For each, the verifier checks the Merkle path against R_S and
   computes Ô_u(q_j) = 8^{-1}·y_j^{-u}·Σ_v ω_8^{-uv}·Ô(y_j·ω_8^v), with q_j = y_j^8.
   It requires the P1 opening of g at the power point (q_j, q_j², …) to equal
   Σ_u α^[u]·Ô_u(q_j).

**Soundness.** Suppose the verifier accepts. P1's WHIR proof binds g to one of at most L1
candidate multilinear polynomials g''. The OOD samples are drawn at opening, after our
challenges, so we take a union bound over the list.

- **Wrong candidate.** Suppose g'' ≠ g. Then ĝ'' − ĝ is a nonzero univariate of degree
  < 2^{n−3}. The q_j range over 2^{n−2} distinct values (one per leaf), so each query passes
  with probability < 1/2. Over t queries and L1 candidates the error is ≤ L1·2^{−t}.
- **Wrong partials.** Suppose g'' = g. Then g̃(x_{3..}) = Σ_u eq(α, u)·w*_u, where w* are
  the true partials. If w ≠ w*, the check in step 3 passes with probability ≤ 3/|Ext|,
  because α is drawn after w.
- **Result.** Otherwise w = w*, and step 1 gives v = Õ(x).

The total is ≤ L1·2^{−t} + 3/|Ext| + ε_WHIR(P1). The verifier draws α after w, and the query
indices after the P1 root. These are the only ordering requirements.

Several oracles share one tree, one α and one set of queries. Each group (key rows with
weights λ^i; run columns with weights 1, μ, μ², μ³) is a linear combination, so the same
argument applies to the combined oracle. The error rises by ≤ (group size)/|Ext|.

**Precedent.**
- Holographic IOPs (Marlin, Fractal) query honestly generated index oracles without a
  low-degree test.
- FRI and WHIR use the same fold-consistency check (WHIR, ePrint 2024/1586).
- Akita (ePrint 2026/1983) calls the idea "setup offloading".

## Costs (est.)

| Item | Estimate |
| --- | --- |
| Setup | about 1–1.5 min per circuit, about 29 GB on disk |
| Finish | about 35–55 s CPU |
| Prover peak | about 35–44 GB (cap 64 GB) |
| Layer-1 proof | about 0.9 MB |
| Verifier | about 50 ms |
| M3 | about 22–23k Poseidon2 permutations plus about 0.65M base multiplications |

## Implementation slices (each with its own tests, ≤ 300 s each)

0. **Spike.** Measure the run census facts and the p3 0.8 commit/open peak at 2^28.
1. **Formulas.** Scaled-eq point, lane split H0/H1, Eval_K DP, step and interval MLEs,
   embedding, coset fold. Port these as numeric tests against brute force, with negative
   controls.
2. **`setup.rs`.** Oracle encoding (Möbius + NTT on a coset), the Merkle tree, the disk
   store, the streaming root, partials, fold, query read and check. Test round trips and
   tampering at toy sizes.
3. **Batched GKR.** Several trees of different depths, shared randomness, product leaves.
4. **`matrix.rs`.** Run and slot tables, e, U and Ω^A, leaf formulas, the verifier's step
   MLEs and H tables. Test that the results equal M1's weights.
5. **`lib.rs`.** The new transcript schedule, `Key`, `Setup`, `prove` and `verify`. Mutate
   every proof and statement part.
6. **nightstream.**
   - `Circuit::compression_setup(dir)` and `compression_key()`.
   - `finish_with_spartan(proof, setup)` and `verify_final(state, key, proof)`.
   - Toy tests and ignored production tests with peak RSS.

## Result (2026-10-07, measured on this Mac, CPU)

All six slices are implemented on the branch. Production package, one fold:

| Item | Measured |
| --- | --- |
| Setup build (`Prover::compression_setup`) | 63–69 s, 27 GB on disk |
| Key derivation without files (`Verifier::compression_key`) | 50–56 s; key 1,912 bytes |
| Setup peak RSS (build, then derive, one process) | 21.1–21.2 GB in three runs; the first run reported 55.1 GB and did not repeat (cause unknown) |
| Finish (`finish_with_spartan`) | 39.7 s |
| Final proof | 1,600,157 bytes |
| Final verification (layer 0 replay + layer 1) | 32 ms (M1: 9.8 s) |
| Finish test peak RSS | 37.3 GB |
| Setup queries t | 125 (P0 118.7 bits, P1 118.1 bits at a 117-bit layer-1 share) |

Deviations from the text above:

1. **Run tables on reopen.** `Setup::open` rebuilds the run and slot tables from the
   package rows and checks that their structure equals the key's. Graft 4 asked for the
   setup directory as the only source. The prover holds the rows for layer 0 anyway, and
   a mismatch can only break completeness: the verifier uses the key's root.
2. **API placement.** `Prover::compression_setup`, `Prover::open_compression_setup`,
   `Verifier::compression_key` and `Verifier::verify_final(state, key, proof)`.
   `Verifier::verify` now takes only an accumulator proof.
3. **Budget.** Each commitment report targets β + 1. Every draw after P0 is charged over
   both candidate lists (P0's by Plonky3, P1's by subtracting log2 L1). The query count is
   derived, not fixed: t = ⌈(β + 1) + 2 + log2 L1⌉, so the query term is at most half of
   P1's share.
4. **P1 rate** is 1/2, as for P0.
5. **Table widths.** Plonky3 requires power-of-two rows per column, not power-of-two
   column counts, so P0 has four tables (cube, runs, slots, rows) of widths 1, 2,
   2·matrices + 1 and 1. Correction 2 above does not apply.
6. **Blocks.** The block variable count is at least one, so a one-block relation still
   has a block tree.
