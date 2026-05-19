//! In-circuit alphabet sampling — parity against
//! `neo_reductions::common::draw_alphabet_vector` (inlined here since the
//! native version is private to the reductions crate).
//!
//! Coverage:
//!   - Production-sized parity (need = D = 54) for empty and pre-absorbed
//!     transcripts.
//!   - Outer-domain-separator wrapper `enforce_pi_rlc_rhos_from_transcript`
//!     mirrors `sample_rot_rhos_n`'s `append_fields_raw([0, i])` prefix.
//!   - Forced-rejection path: a seed chosen so chunk == 65535 occurs at a
//!     known position, exercising the accept = 0 branch and the
//!     "skip-this-chunk" selection logic.
//!   - Tamper rejection: changing a sampled symbol breaks the constraint.

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::{
    enforce_alphabet_sample_5_d, enforce_pi_rlc_rhos_from_transcript,
};
use neo_fold_clean::engine::r1cs_circuit::{R1csBuilder, TranscriptGadget};
use neo_math::ring::D;
use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};
use p3_field::PrimeCharacteristicRing;

const APP: &[u8] = b"neo.test.alphabet_sampling/v1";
const ALPHABET: [i8; 5] = [-2, -1, 0, 1, 2];

/// Inlined replica of `neo_reductions::common::draw_alphabet_vector` so this
/// test does not depend on a private function.
fn native_draw_alphabet_vector(tr: &mut Poseidon2Transcript, need: usize, alphabet: &[i8], seed: u64) -> Vec<i8> {
    let m = alphabet.len() as u32;
    let bucket = (1u32 << 16) / m * m;
    let mut out = Vec::with_capacity(need);
    let mut ctr = seed;
    while out.len() < need {
        tr.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
        let dig = tr.digest32();
        for w in dig.chunks_exact(2) {
            let x = u16::from_le_bytes([w[0], w[1]]) as u32;
            if x < bucket {
                let idx = (x % m) as usize;
                out.push(alphabet[idx]);
                if out.len() == need {
                    break;
                }
            }
        }
        ctr = ctr.wrapping_add(1);
    }
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

/// Confirm that `seed` causes at least one chunk == 65535 within MAX_ITER
/// iterations, given the supplied app label and prior absorbs. Returns the
/// number of rejection events observed (≥ 1 for a true rejection seed).
fn count_rejection_chunks(app: &'static [u8], seed: u64) -> usize {
    let mut native = Poseidon2Transcript::new(app);
    let mut ctr = seed;
    let mut count = 0;
    for _ in 0..4 {
        native.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
        let dig = native.digest32();
        for w in dig.chunks_exact(2) {
            let x = u16::from_le_bytes([w[0], w[1]]);
            if x == 65535 {
                count += 1;
            }
        }
        ctr = ctr.wrapping_add(1);
    }
    count
}

/// Hard-coded seed that triggers a rejection chunk under [`APP`] starting
/// from an empty Poseidon2 transcript. Discovered via brute-force search
/// (see git history); the test below sanity-checks that this seed still
/// hits a rejection so we catch silent drift.
const REJECTION_SEED: u64 = 0x2b1;

#[test]
fn alphabet_sampling_handles_forced_rejection_chunk() {
    let rejects = count_rejection_chunks(APP, REJECTION_SEED);
    assert!(
        rejects >= 1,
        "hard-coded REJECTION_SEED = {REJECTION_SEED:#x} no longer triggers a rejection; \
         re-run a brute-force search and update the constant"
    );

    let mut native = Poseidon2Transcript::new(APP);
    let native_syms = native_draw_alphabet_vector(&mut native, D, &ALPHABET, REJECTION_SEED);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, APP);
    let circ_syms = enforce_alphabet_sample_5_d(&mut b, &mut tr, REJECTION_SEED);

    assert!(
        b.is_satisfied(),
        "rejection-path circuit unsatisfied at seed {REJECTION_SEED:#x} (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    for (i, (sym_var, &native_sym)) in circ_syms.iter().zip(native_syms.iter()).enumerate() {
        let circ_val = b.witness()[sym_var.col()];
        let native_val = symbol_to_f(native_sym);
        assert_eq!(
            circ_val, native_val,
            "rejection-path symbol {i} divergence at seed {REJECTION_SEED:#x}"
        );
    }
}
