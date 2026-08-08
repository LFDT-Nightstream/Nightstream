//! In-circuit alphabet sampling — parity against
//! `neo_reductions::common::draw_alphabet_vector` (inlined here since the
//! native version is private to the reductions crate).
//!
//! Coverage:
//!   - Production-sized parity (need = D = 54) for empty and pre-absorbed
//!     transcripts.
//!   - Outer-domain-separator wrapper `enforce_pi_rlc_rhos_from_transcript`
//!     mirrors `sample_rot_rhos_n`'s `append_fields_raw([0, i])` prefix.
//!   - Forced-rejection path: a seed chosen so an exact candidate is 65535 at a
//!     known position, exercising the accept = 0 branch and the
//!     "skip-this-chunk" selection logic.
//!   - Tamper rejection: changing a sampled symbol breaks the constraint.

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::{
    enforce_alphabet_sample_5_d, enforce_pi_rlc_rhos_from_transcript,
};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_math::ring::D;
use neo_math::F;
use neo_params::goldilocks_paper_b2::PI_RLC_SAMPLER_DIGEST_ROUNDS;
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};
use p3_field::PrimeCharacteristicRing;

const APP: &[u8] = b"neo.test.alphabet_sampling/v1";
const ALPHABET: [i8; 5] = [-2, -1, 0, 1, 2];

/// Inlined replica of the production fixed-round sampler.
fn native_draw_alphabet_vector(tr: &mut Poseidon2Transcript, need: usize, alphabet: &[i8], seed: u64) -> Vec<i8> {
    let alphabet_len = alphabet.len() as u32;
    let bucket = 65_535 / alphabet_len * alphabet_len;
    let mut out = Vec::with_capacity(need);
    let mut ctr = seed;
    for _ in 0..PI_RLC_SAMPLER_DIGEST_ROUNDS {
        tr.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
        let digest = tr.digest32();
        for lane in digest.chunks_exact(8) {
            let value = u64::from_le_bytes(lane.try_into().expect("digest lane"));
            for offset in [0, 16] {
                let raw = ((value >> offset) & 0xffff) as u16;
                let candidate = (!raw) as u32;
                if candidate < bucket && out.len() < need {
                    out.push(alphabet[(candidate % alphabet_len) as usize]);
                }
            }
        }
        ctr = ctr.wrapping_add(1);
    }
    assert_eq!(out.len(), need, "fixed production sampler shortfall");
    out
}

/// Native mirror of the full Π_RLC ρ-derivation: per-i `[0, i]` outer
/// separator followed by the inner alphabet sampler.
fn native_pi_rlc_rhos(tr: &mut Poseidon2Transcript, count: usize) -> Vec<Vec<i8>> {
    let mut rhos = Vec::with_capacity(count);
    for i in 0..count {
        tr.append_fields_raw(&[F::ZERO, F::from_u64(i as u64)]);
        rhos.push(native_draw_alphabet_vector(tr, D, &ALPHABET, i as u64));
    }
    rhos
}

fn symbol_to_f(s: i8) -> F {
    if s >= 0 {
        F::from_u64(s as u64)
    } else {
        -F::from_u64((-s) as u64)
    }
}

fn f_from_i64(v: i64) -> F {
    if v >= 0 {
        F::from_u64(v as u64)
    } else {
        -F::from_u64((-v) as u64)
    }
}

#[test]
fn alphabet_sampling_d_matches_native_empty_session() {
    let mut native = Poseidon2Transcript::new(APP);
    let native_syms = native_draw_alphabet_vector(&mut native, D, &ALPHABET, 0xDEADBEEF);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let circ_syms = enforce_alphabet_sample_5_d(&mut b, &mut tr, 0xDEADBEEF);

    assert!(
        b.is_satisfied(),
        "circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    for (i, (sym_var, &native_sym)) in circ_syms.iter().zip(native_syms.iter()).enumerate() {
        let circ_val = b.witness()[sym_var.col()];
        let native_val = symbol_to_f(native_sym);
        assert_eq!(circ_val, native_val, "symbol {i} divergence");
    }
}

#[test]
fn alphabet_sampling_d_matches_native_after_absorbs() {
    let mut native = Poseidon2Transcript::new(APP);
    native.append_fields(b"prior", &[F::from_u64(7), F::from_u64(11)]);
    let native_syms = native_draw_alphabet_vector(&mut native, D, &ALPHABET, 0x1234);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let prior = vec![b.alloc(F::from_u64(7)), b.alloc(F::from_u64(11))];
    tr.append_fields(&mut b, b"prior", &prior);
    let circ_syms = enforce_alphabet_sample_5_d(&mut b, &mut tr, 0x1234);

    assert!(
        b.is_satisfied(),
        "post-absorb circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    for (i, (sym_var, &native_sym)) in circ_syms.iter().zip(native_syms.iter()).enumerate() {
        let circ_val = b.witness()[sym_var.col()];
        let native_val = symbol_to_f(native_sym);
        assert_eq!(circ_val, native_val, "symbol {i} post-absorb divergence");
    }
}

#[test]
fn pi_rlc_rhos_wrapper_matches_native() {
    // count = 3 to keep the test small but exercise the per-i outer separator.
    const COUNT: usize = 3;
    let mut native = Poseidon2Transcript::new(APP);
    let native_rhos = native_pi_rlc_rhos(&mut native, COUNT);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let circ_rhos = enforce_pi_rlc_rhos_from_transcript(&mut b, &mut tr, COUNT);

    assert!(
        b.is_satisfied(),
        "pi_rlc_rhos wrapper unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    assert_eq!(circ_rhos.len(), COUNT);
    for (i, (circ_rho, native_rho)) in circ_rhos.iter().zip(native_rhos.iter()).enumerate() {
        for (j, (sym_var, &native_sym)) in circ_rho.iter().zip(native_rho.iter()).enumerate() {
            let circ_val = b.witness()[sym_var.col()];
            let native_val = symbol_to_f(native_sym);
            assert_eq!(circ_val, native_val, "ρ_{i}[{j}] divergence");
        }
    }
}

#[test]
fn alphabet_sampling_rejects_tampered_output() {
    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let syms = enforce_alphabet_sample_5_d(&mut b, &mut tr, 0x5555);
    assert!(b.is_satisfied(), "baseline");

    let target = syms[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(
        !b.is_satisfied(),
        "tampering a sampled symbol must break the constraint"
    );
}

#[test]
fn alphabet_sampling_rejects_negative_mod5_residue_forgery() {
    // Hacker model: keep the transcript digest chunk fixed, but represent a
    // native residue r ∈ {1,2,3,4} as the alternate field equation
    // `chunk = 5 * (q + 1) + (r - 5)`. Before the sampler constrained
    // `idx` as an unsigned mod-5 residue, the centered low-norm check allowed
    // `idx = r - 5`, producing a sampled symbol outside [-2,2].
    let seed = (0u64..)
        .find(|&seed| {
            let (chunk, idx, _) = first_chunk_mod5(APP, seed);
            chunk < 65535 && idx > 0
        })
        .expect("small seed search should find an accepted first chunk with nonzero residue");
    let (_chunk, idx_native, q_native) = first_chunk_mod5(APP, seed);
    assert!(
        (1..=4).contains(&idx_native),
        "test setup must use a residue with negative alternate"
    );

    let baseline = build_sampler(seed);
    assert!(baseline.builder.is_satisfied(), "baseline sampler must satisfy");

    let idx_alt = idx_native as i64 - 5;
    let q_alt = q_native + 1;
    let symbol_alt = idx_alt - 2;
    assert!(
        !(-2..=2).contains(&symbol_alt),
        "alternate residue should leave the production alphabet"
    );

    let candidates = negative_residue_candidate_cols(&baseline.builder, idx_native, q_native);
    assert!(
        !candidates.is_empty(),
        "test must locate at least one native mod-5 decomposition candidate"
    );

    let mut accepted_forgery = None;
    for candidate in candidates {
        let mut run = build_sampler(seed);
        apply_negative_residue_forgery(&mut run, candidate, idx_alt, q_alt, symbol_alt);
        if run.builder.is_satisfied() {
            accepted_forgery = Some(candidate.idx_col);
            break;
        }
    }

    assert!(
        accepted_forgery.is_none(),
        "alphabet sampler accepted a negative mod-5 residue forgery at idx column {:?}; \
         Π_RLC rho sampling must be transcript-derived over unsigned residues 0..4",
        accepted_forgery
    );
}

/// Confirm that `seed` causes at least one candidate == 65535 within the fixed
/// iterations, given the supplied app label and prior absorbs. Returns the
/// number of rejection events observed (≥ 1 for a true rejection seed).
fn count_rejection_chunks(app: &'static [u8], seed: u64) -> usize {
    let mut native = Poseidon2Transcript::new(app);
    let mut ctr = seed;
    let mut count = 0;
    for _ in 0..PI_RLC_SAMPLER_DIGEST_ROUNDS {
        native.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
        let digest = native.digest32();
        for lane in digest.chunks_exact(8) {
            let value = u64::from_le_bytes(lane.try_into().expect("digest lane"));
            for offset in [0, 16] {
                let raw = ((value >> offset) & 0xffff) as u16;
                if !raw == 65_535 {
                    count += 1;
                }
            }
        }
        ctr = ctr.wrapping_add(1);
    }
    count
}

#[test]
fn alphabet_sampling_handles_forced_rejection_chunk() {
    let rejection_seed = (0u64..)
        .find(|&seed| count_rejection_chunks(APP, seed) > 0)
        .expect("deterministic transcript search must find a rejected candidate");

    let mut native = Poseidon2Transcript::new(APP);
    let native_syms = native_draw_alphabet_vector(&mut native, D, &ALPHABET, rejection_seed);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let circ_syms = enforce_alphabet_sample_5_d(&mut b, &mut tr, rejection_seed);

    assert!(
        b.is_satisfied(),
        "rejection-path circuit unsatisfied at seed {rejection_seed:#x} (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    for (i, (sym_var, &native_sym)) in circ_syms.iter().zip(native_syms.iter()).enumerate() {
        let circ_val = b.witness()[sym_var.col()];
        let native_val = symbol_to_f(native_sym);
        assert_eq!(
            circ_val, native_val,
            "rejection-path symbol {i} divergence at seed {rejection_seed:#x}"
        );
    }
}

struct SamplerRun {
    builder: R1csBuilder,
    syms: [neo_fold_clean::engine::r1cs_circuit::Var; D],
}

fn build_sampler(seed: u64) -> SamplerRun {
    let mut builder = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut builder, APP);
    let syms = enforce_alphabet_sample_5_d(&mut builder, &mut tr, seed);
    SamplerRun { builder, syms }
}

fn first_chunk_mod5(app: &'static [u8], seed: u64) -> (u32, u64, u64) {
    let mut native = Poseidon2Transcript::new(app);
    native.append_fields_raw(&[F::from_u64(1), F::from_u64(seed)]);
    let dig = native.digest32();
    let raw = u16::from_le_bytes([dig[0], dig[1]]);
    let chunk = (!raw) as u32;
    let idx = (chunk as u64) % 5;
    let q = (chunk as u64) / 5;
    (chunk, idx, q)
}

#[derive(Clone, Copy)]
struct Mod5Candidate {
    idx_col: usize,
    q_bits_start: usize,
    symbol_col: usize,
    fixed_mod5_intermediates: bool,
}

fn negative_residue_candidate_cols(builder: &R1csBuilder, idx_native: u64, q_native: u64) -> Vec<Mod5Candidate> {
    let witness = builder.witness();
    let idx_f = F::from_u64(idx_native);
    let q_f = F::from_u64(q_native);
    let symbol_f = f_from_i64(idx_native as i64 - 2);
    let mut out = Vec::new();
    for idx_col in 1..witness.len().saturating_sub(20) {
        if witness[idx_col] != idx_f || witness[idx_col + 1] != q_f {
            continue;
        }
        for (q_bits_start, symbol_off, fixed_mod5_intermediates) in [(idx_col + 2, 16, false), (idx_col + 5, 19, true)]
        {
            if witness[idx_col + symbol_off] != symbol_f {
                continue;
            }
            let q_bits_ok = (0..14).all(|bit| {
                let want = F::from_u64((q_native >> bit) & 1);
                witness[q_bits_start + bit] == want
            });
            if q_bits_ok {
                out.push(Mod5Candidate {
                    idx_col,
                    q_bits_start,
                    symbol_col: idx_col + symbol_off,
                    fixed_mod5_intermediates,
                });
            }
        }
    }
    out
}

fn apply_negative_residue_forgery(
    run: &mut SamplerRun,
    candidate: Mod5Candidate,
    idx_alt: i64,
    q_alt: u64,
    symbol_alt: i64,
) {
    let idx_col = candidate.idx_col;
    let idx_alt_f = f_from_i64(idx_alt);
    let q_alt_f = F::from_u64(q_alt);
    let symbol_alt_f = f_from_i64(symbol_alt);
    run.builder.tamper_witness(idx_col, idx_alt_f);
    run.builder.tamper_witness(idx_col + 1, q_alt_f);
    if candidate.fixed_mod5_intermediates {
        let i1 = idx_alt_f * (idx_alt_f - F::ONE);
        let i2 = i1 * (idx_alt_f - F::from_u64(2));
        let i3 = i2 * (idx_alt_f - F::from_u64(3));
        run.builder.tamper_witness(idx_col + 2, i1);
        run.builder.tamper_witness(idx_col + 3, i2);
        run.builder.tamper_witness(idx_col + 4, i3);
    }
    for bit in 0..14 {
        run.builder
            .tamper_witness(candidate.q_bits_start + bit, F::from_u64((q_alt >> bit) & 1));
    }
    run.builder
        .tamper_witness(candidate.symbol_col, symbol_alt_f);

    // Output 0 selects chunk 0 for the seeds chosen above. The selection
    // gadget allocates, in order: 64 one-hot bits, 64 triples
    // `(one_hot * symbol, one_hot * accept, one_hot * cum_before)`, then the
    // output symbol. Update the selected product and output symbol so the
    // only remaining question is whether the chunk's residue constraints
    // accept the negative representative.
    let out0_col = run.syms[0].col();
    let selected_mul_sym_col = out0_col - 1 - (3 * 64);
    run.builder
        .tamper_witness(selected_mul_sym_col, symbol_alt_f);
    run.builder.tamper_witness(out0_col, symbol_alt_f);
}
