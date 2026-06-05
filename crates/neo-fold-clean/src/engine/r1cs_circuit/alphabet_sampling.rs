//! Rejection sampling from a small alphabet — in-circuit parity to
//! `neo_reductions::common::draw_alphabet_vector`.
//!
//! Owns the byte-level rejection sampler used by Π_RLC.V's ρ derivation.
//! Each call to [`enforce_alphabet_sample_5_d`] mirrors one native invocation
//! `draw_alphabet_vector(tr, D, &[-2,-1,0,1,2], seed)` — same sponge
//! absorbs, same digest squeeze, same accept/reject filter, same first-D
//! selection semantics. [`enforce_pi_rlc_rhos_from_transcript`] wraps it
//! with the per-ρ `append_fields_raw([0, i])` outer domain separator,
//! matching `sample_rot_rhos_n`.
//!
//! ## Why hardcoded to size-5 alphabet
//!
//! Production (`neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET`) uses
//! `[-2, -1, 0, 1, 2]`. This is the only alphabet the strict HyperNova
//! soundness story applies to today; we hardcode size and indexing to keep
//! gadget complexity (range check + mod-5 + symbol) flat. Generalizing to
//! arbitrary alphabets is a follow-up if the production set ever changes.
//!
//! ## Native algorithm summary
//!
//! ```text
//!   ctr = seed
//!   while out.len() < need:
//!     tr.append_fields_raw([1, ctr])
//!     dig = tr.digest32()                      // 32 LE bytes
//!     for w in dig.chunks_exact(2):
//!       x = u16::from_le_bytes(w) as u32       // 16-bit chunk
//!       if x < 65535:                          // bucket = (2^16 / 5) * 5 = 65535
//!         idx = x mod 5
//!         out.push(alphabet[idx])              // alphabet[i] = i - 2 for size-5
//!     ctr += 1
//! ```
//!
//! Per `digest32()` we get 16 chunks (4 lanes × 4 chunks per lane); each
//! chunk has accept rate `65535/65536 ≈ 0.99998`. The honest path almost
//! never rejects; the gadget still has to handle reject for soundness.
//!
//! ## Constraint structure
//!
//! Iterations: `MAX_ITER = 4` gives `64` chunks. Probability that fewer
//! than `need = 54` chunks accept across 64 trials is bounded by
//! `Binom(64, 1/65536) ≥ 11)` ~ astronomically small. We never need more
//! than 4 iterations in practice.
//!
//! Per chunk:
//!   - 64-bit canonical bit decomposition of the digest lane (~70 cons).
//!   - 16-bit chunk-value as a free `Lc` (no allocation).
//!   - Range check `chunk != 65535` via inverse trick (3 cons).
//!   - `chunk = 5·q + idx`, `idx ∈ {0..4}`, `q < 13107` (~20 cons).
//!   - Symbol = idx − 2 (linear, 1 cons).
//!
//! Selection (first-N-accepts): per output position `j ∈ [0, need)`, a
//! one-hot vector over the `TOTAL_CHUNKS` chunks picks the unique chunk
//! whose prefix-sum-of-accepts equals `j` and that accepts.

use neo_math::ring::D;
use neo_math::F;
use p3_field::{Field, PrimeCharacteristicRing};

use crate::engine::r1cs_circuit::boolean::enforce_bit;
use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::transcript::TranscriptGadget;
use crate::engine::r1cs_circuit::u64_arith::decompose_var_to_u64_bits;

/// Per-call digest iterations: 4 iterations × 16 chunks = 64 chunks. With a
/// per-chunk reject rate of `1/65536`, P(≥ 11 rejects out of 64) is bounded
/// by `C(64,11) · (1/65536)^11 ≈ 2^32.75 · 2^-176 ≈ 2^-143`. Including
/// higher-k terms tightens this slightly but it's astronomically rare in
/// either form — the honest path consumes ~1 iteration in practice.
const MAX_ITER: usize = 4;
const CHUNKS_PER_ITER: usize = 16;
const TOTAL_CHUNKS: usize = MAX_ITER * CHUNKS_PER_ITER;
const BUCKET: u32 = 65535;
const ALPHABET_SIZE: u64 = 5;
/// `q = chunk / 5` for chunks in `[0, 65535)` fits in 14 bits (max `13106`).
const Q_BITS: usize = 14;
/// Slack `s = cum_after_last - D` lies in `[0, TOTAL_CHUNKS - D]`. For
/// `TOTAL_CHUNKS = 64, D = 54`: `s ∈ [0, 10]`, fits in 4 bits.
const SLACK_BITS: usize = 4;

/// One processed chunk: `(accept_bit, symbol_wire, prefix_sum_after_this_chunk)`.
struct ChunkRecord {
    accept: Var,
    symbol: Var,
    /// Running cumulative accept count *after* this chunk's contribution.
    cum_after: Var,
}

/// Sample `D = 54` alphabet symbols from `[-2, -1, 0, 1, 2]` via Poseidon2
/// rejection sampling — the production ρ-coefficient derivation for Π_RLC.
///
/// **Parity contract** — for any prior transcript state, the witness values
/// of the returned wires equal the native output of
/// `draw_alphabet_vector(&mut native_tr, D, &[-2,-1,0,1,2], seed)`. The
/// in-circuit transcript advances by exactly `MAX_ITER = 4` digest
/// iterations regardless of how many native iterations the honest path
/// would have taken; production callers always need `D = 54` symbols where
/// 4 iterations is the expected count.
///
/// **Completeness boundary**: the gadget assumes ≥ `D` accepts among 64
/// chunks. With per-chunk reject rate `1/65536`, P(≥ 11 rejects) is
/// bounded by ≈ `2^-143` (see `MAX_ITER` constant). An explicit
/// `cum_after_last = D + slack`
/// constraint (with `slack ∈ [0, 10]`) makes "enough accepts" a circuit
/// gate, so even an adversarial witness violating this is rejected by the
/// verifier rather than silently producing nonsense.
pub fn enforce_alphabet_sample_5_d(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    seed: u64,
) -> [Var; D] {
    // 1. Drive the transcript through MAX_ITER iterations, processing each
    //    16-bit chunk as it lands.
    let mut chunks: Vec<ChunkRecord> = Vec::with_capacity(TOTAL_CHUNKS);
    let mut cum_prev = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(cum_prev), &Lc::zero());

    for iter in 0..MAX_ITER {
        let ctr = seed.wrapping_add(iter as u64);
        transcript.append_fields_raw_const(builder, &[F::ONE, F::from_u64(ctr)]);
        let dig = transcript.digest_fields(builder);

        for &lane_var in dig.iter() {
            let bits = decompose_var_to_u64_bits(builder, lane_var);
            for chunk_idx in 0..4 {
                let chunk_bits = &bits[chunk_idx * 16..(chunk_idx + 1) * 16];
                let record = process_chunk(builder, chunk_bits, cum_prev);
                cum_prev = record.cum_after;
                chunks.push(record);
            }
        }
    }
    debug_assert_eq!(chunks.len(), TOTAL_CHUNKS);

    // 2. Enforce `cum_after_last = D + slack`, slack ∈ [0, TOTAL_CHUNKS - D].
    //    This is the "enough accepts" gate — independent of selection.
    let cum_after_last = chunks.last().expect("at least one chunk").cum_after;
    let cum_val = builder.witness()[cum_after_last.col()];
    let d_f = F::from_u64(D as u64);
    let slack_val = cum_val - d_f;
    let slack = builder.alloc(slack_val);
    // Bit-decompose slack to bound its range to [0, 2^SLACK_BITS).
    use p3_field::PrimeField64;
    let slack_u64 = slack_val.as_canonical_u64();
    let mut slack_lc = Lc::zero();
    let mut pow2 = F::ONE;
    for i in 0..SLACK_BITS {
        let bit_val = (slack_u64 >> i) & 1;
        let b = builder.alloc(F::from_u64(bit_val));
        enforce_bit(builder, b);
        slack_lc.add_term(b, pow2);
        pow2 = pow2 + pow2;
    }
    builder.enforce_eq(&Lc::from_var(slack), &slack_lc);
    // cum_after_last == D + slack.
    let mut sum = Lc::from_var(slack);
    sum.add_constant(d_f);
    builder.enforce_eq(&Lc::from_var(cum_after_last), &sum);

    // 3. Select the first D accepted symbols via per-output one-hot vectors.
    let v = select_first_n_accepts(builder, &chunks, D);
    let mut out = [Var::ONE; D];
    out.copy_from_slice(&v);
    out
}

/// Derive the Π_RLC `count` ρ-coefficient vectors from the transcript.
///
/// For each `i ∈ [0, count)`:
/// 1. Absorb the per-ρ domain separator `append_fields_raw([0, i])`.
/// 2. Call [`enforce_alphabet_sample_5_d`] with `seed = i`.
///
/// Returns `count` length-`D` coefficient vectors, each the first column
/// of the rotation matrix `rot(a_i)` Π_RLC.V's existing
/// [`crate::paper::reductions::pi_rlc_circuit`] gadgets consume.
pub fn enforce_pi_rlc_rhos_from_transcript(
    builder: &mut R1csBuilder,
    transcript: &mut TranscriptGadget,
    count: usize,
) -> Vec<[Var; D]> {
    let mut rhos = Vec::with_capacity(count);
    for i in 0..count {
        // Outer per-ρ domain separator (matches `sample_rot_rhos_n`).
        transcript.append_fields_raw_const(builder, &[F::ZERO, F::from_u64(i as u64)]);
        let coeffs = enforce_alphabet_sample_5_d(builder, transcript, i as u64);
        rhos.push(coeffs);
    }
    rhos
}

// ── Chunk processing (range + mod-5 + symbol) ───────────────────────────

fn process_chunk(builder: &mut R1csBuilder, chunk_bits: &[Var], cum_prev: Var) -> ChunkRecord {
    debug_assert_eq!(chunk_bits.len(), 16);

    // chunk_value as a free Lc (no allocation).
    let mut chunk_lc = Lc::zero();
    let mut pow2 = F::ONE;
    for &b in chunk_bits {
        chunk_lc.add_term(b, pow2);
        pow2 = pow2 + pow2;
    }
    let chunk_val_f = builder.eval(&chunk_lc);
    let chunk_val_u64 = canonical_u64(chunk_val_f);

    // accept = (chunk != 65535).
    let bucket_f = F::from_u64(BUCKET as u64);
    let diff_val = chunk_val_f - bucket_f;
    let accept_val = if diff_val == F::ZERO { F::ZERO } else { F::ONE };
    let accept = builder.alloc(accept_val);
    enforce_bit(builder, accept);

    // Inverse witness for the "accept = 1" branch.
    let inv_val = if diff_val == F::ZERO {
        F::ZERO
    } else {
        diff_val.inverse()
    };
    let inv = builder.alloc(inv_val);
    let diff_lc = chunk_lc
        .clone()
        .add_scaled(&Lc::from_const(bucket_f), -F::ONE);

    // We want `accept = (chunk != BUCKET)`. Using the "δ = (a == b)" pattern
    // with `δ = 1 - accept`:
    //   A: (1 - accept) · (chunk - BUCKET) = 0    (if chunk != BUCKET, accept = 1)
    //   B: (chunk - BUCKET) · inv         = accept (if chunk != BUCKET, inv supplies 1)
    let mut one_minus_accept = Lc::from_const(F::ONE);
    one_minus_accept.add_term(accept, -F::ONE);
    builder.enforce(&one_minus_accept, &diff_lc, &Lc::zero());
    builder.enforce(&diff_lc, &Lc::from_var(inv), &Lc::from_var(accept));

    // mod-5 decomposition: chunk = 5*q + idx, idx ∈ {0..4}, q in [0, Q_MAX].
    let idx_val = chunk_val_u64 % ALPHABET_SIZE;
    let q_val = chunk_val_u64 / ALPHABET_SIZE;
    let idx = builder.alloc(F::from_u64(idx_val));
    let q = builder.alloc(F::from_u64(q_val));

    // idx ∈ {0, 1, 2, 3, 4}. This is an unsigned residue, not a centered
    // low-norm value: allowing negative residues would let the prover encode
    // the same transcript chunk as `5 * (q + 1) + (idx - 5)` and sample
    // symbols outside the native alphabet.
    enforce_mod5_index(builder, idx);

    // q ∈ [0, 2^14) — bit-decomposition (Q_MAX < 2^14 = 16384).
    let mut q_lc = Lc::zero();
    let mut q_pow2 = F::ONE;
    for i in 0..Q_BITS {
        let bit_val = (q_val >> i) & 1;
        let b = builder.alloc(F::from_u64(bit_val));
        enforce_bit(builder, b);
        q_lc.add_term(b, q_pow2);
        q_pow2 = q_pow2 + q_pow2;
    }
    builder.enforce_eq(&Lc::from_var(q), &q_lc);

    // Constraint: chunk = 5*q + idx.
    let mut rhs = Lc::zero();
    rhs.add_term(q, F::from_u64(ALPHABET_SIZE));
    rhs.add_term(idx, F::ONE);
    builder.enforce_eq(&chunk_lc, &rhs);

    // symbol = idx - 2 (alphabet = [-2, -1, 0, 1, 2]).
    let symbol_val = idx_val.wrapping_sub(2);
    let symbol_f = if (symbol_val as i64) < 0 {
        // i.e., idx_val ∈ {0, 1} → symbol_val ∈ {-2, -1} in 𝔽
        let neg = 2 - idx_val;
        -F::from_u64(neg)
    } else {
        F::from_u64(symbol_val)
    };
    let symbol = builder.alloc(symbol_f);
    let mut sym_rhs = Lc::from_var(idx);
    sym_rhs.add_constant(-F::from_u64(2));
    builder.enforce_eq(&Lc::from_var(symbol), &sym_rhs);

    // cum_after = cum_prev + accept.
    let cum_after_val = builder.eval(&Lc::from_var(cum_prev)) + accept_val;
    let cum_after = builder.alloc(cum_after_val);
    let mut cum_rhs = Lc::from_var(cum_prev);
    cum_rhs.add_term(accept, F::ONE);
    builder.enforce_eq(&Lc::from_var(cum_after), &cum_rhs);

    ChunkRecord {
        accept,
        symbol,
        cum_after,
    }
}

fn enforce_mod5_index(builder: &mut R1csBuilder, idx: Var) {
    let mut acc = Lc::from_var(idx);
    for a in 1..=4 {
        let mut factor = Lc::from_var(idx);
        factor.add_constant(-F::from_u64(a));
        if a == 4 {
            builder.enforce(&acc, &factor, &Lc::zero());
        } else {
            let next = builder.alloc_mul(&acc, &factor);
            acc = Lc::from_var(next);
        }
    }
}

// ── Selection (first-N-accepts via per-output one-hot) ──────────────────

fn select_first_n_accepts(builder: &mut R1csBuilder, chunks: &[ChunkRecord], need: usize) -> Vec<Var> {
    // Compute "cum_before[k]" wires: cum_before[0] = 0, cum_before[k] = chunks[k-1].cum_after.
    let zero_cum = builder.alloc(F::ZERO);
    builder.enforce_eq(&Lc::from_var(zero_cum), &Lc::zero());
    let cum_before: Vec<Var> = std::iter::once(zero_cum)
        .chain(chunks[..chunks.len() - 1].iter().map(|c| c.cum_after))
        .collect();
    debug_assert_eq!(cum_before.len(), chunks.len());

    // For each output position j, find the unique chunk k with
    // cum_before[k] = j AND chunks[k].accept = 1.
    let mut out: Vec<Var> = Vec::with_capacity(need);
    for j in 0..need {
        let target_j = F::from_u64(j as u64);

        // Determine the one-hot bit positions from the witness: find the
        // unique k with cum_before[k] = j and chunks[k].accept = 1.
        let mut one_hot_pos: Option<usize> = None;
        for (k, c) in chunks.iter().enumerate() {
            let cum_b = builder.witness()[cum_before[k].col()];
            let acc = builder.witness()[c.accept.col()];
            if cum_b == target_j && acc == F::ONE {
                one_hot_pos = Some(k);
                break;
            }
        }
        let pos = one_hot_pos.unwrap_or_else(|| {
            panic!(
                "alphabet sampling: no chunk satisfies (cum_before = {}, accept = 1); \
                 not enough accepts in {TOTAL_CHUNKS} chunks for need = {need}",
                j
            )
        });

        // Allocate one-hot vector.
        let one_hot: Vec<Var> = (0..chunks.len())
            .map(|k| {
                let v = if k == pos { F::ONE } else { F::ZERO };
                let var = builder.alloc(v);
                enforce_bit(builder, var);
                var
            })
            .collect();

        // Sum to 1: Σ one_hot[k] == 1.
        let mut sum_lc = Lc::zero();
        for &b in &one_hot {
            sum_lc.add_term(b, F::ONE);
        }
        builder.enforce_eq(&sum_lc, &Lc::from_const(F::ONE));

        // Extract symbol_j = Σ one_hot[k] · chunks[k].symbol. Quadratic in
        // (one_hot, symbol), so allocate via alloc_mul of pair sums and sum
        // the products. For R1CS, the standard trick: pair-wise products
        // and accumulate.
        let mut sym_acc = Lc::zero();
        let mut acc_acc = Lc::zero();
        let mut cum_acc = Lc::zero();
        for (k, c) in chunks.iter().enumerate() {
            let mul_sym = builder.alloc_mul(&Lc::from_var(one_hot[k]), &Lc::from_var(c.symbol));
            let mul_acc = builder.alloc_mul(&Lc::from_var(one_hot[k]), &Lc::from_var(c.accept));
            let mul_cum = builder.alloc_mul(&Lc::from_var(one_hot[k]), &Lc::from_var(cum_before[k]));
            sym_acc.add_term(mul_sym, F::ONE);
            acc_acc.add_term(mul_acc, F::ONE);
            cum_acc.add_term(mul_cum, F::ONE);
        }

        // Pin: accept_j = 1, cum_prev_j = j.
        builder.enforce_eq(&acc_acc, &Lc::from_const(F::ONE));
        builder.enforce_eq(&cum_acc, &Lc::from_const(target_j));

        // Allocate the output symbol and bind to the extracted Lc.
        let out_sym = builder.alloc(builder.eval(&sym_acc));
        builder.enforce_eq(&Lc::from_var(out_sym), &sym_acc);
        out.push(out_sym);
    }
    out
}

// ── Helpers ─────────────────────────────────────────────────────────────

#[inline]
fn canonical_u64(v: F) -> u64 {
    use p3_field::PrimeField64;
    v.as_canonical_u64()
}
