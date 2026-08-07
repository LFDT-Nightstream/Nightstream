//! Π_RLC.V circuit — native vs gadget parity for the full §7.4 reduction.
//!
//! Coverage:
//!   - Commitment combination `c = Σ ρ_i · c_i` (synthetic ρ).
//!   - X combination `X = Σ ρ_i · X_i` (synthetic ρ).
//!   - y_ring row combination `y_j = Σ ρ_i · y_{i,j}` (synthetic ρ).
//!   - Transcript-driven ρ: full c+X+y combination uses ρ derived
//!     in-circuit via `enforce_pi_rlc_rhos_from_transcript`, mirroring
//!     native `sample_rot_rhos_n`.
//!   - Tamper rejection across input/combined/ρ wires.

#[path = "../support/mod.rs"]
mod support;

use neo_ajtai::{s_mul_add, Commitment};
use neo_ccs::{CcsStructure, Mat, SparsePoly};
use neo_fold_clean::engine::optimized;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::engine::transcript::Transcript as PaperTranscript;
use neo_fold_clean::paper::construction2::RunningInstance;
use neo_fold_clean::paper::digest::pi_ccs_outputs_digest;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_commitment_inputs, alloc_rlc_x_inputs, alloc_rlc_y_row_inputs, enforce_rlc_commitment_combination,
    enforce_rlc_x_combination, enforce_rlc_y_row_combination,
};
use neo_fold_clean::paper::relations::CcsClaim;
use neo_fold_clean::paper::{nifs, pi_ccs, pi_rlc};
use neo_fold_clean::{config, preprocess, CcsInstance, Preprocessing};
use neo_math::ring::{cf, rot_apply_vec, Rq, D};
use neo_math::{KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const KAPPA: usize = 4; // small for fast tests; the gadget is κ-generic

fn deterministic_rq(seed: u64) -> Rq {
    let mut coeffs = [F::ZERO; D];
    let mut s = seed;
    for slot in coeffs.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *slot = F::from_u64(s & 0xFF);
    }
    Rq(coeffs)
}

fn deterministic_commitment(seed: u64) -> Commitment {
    let mut data = Vec::with_capacity(D * KAPPA);
    let mut s = seed.wrapping_mul(0x9E3779B97F4A7C15);
    for _ in 0..D * KAPPA {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        data.push(F::from_u64(s & 0xFFFF));
    }
    Commitment {
        d: D,
        kappa: KAPPA,
        data,
    }
}

fn native_combine(rhos: &[Rq], cs: &[Commitment]) -> Commitment {
    assert_eq!(rhos.len(), cs.len());
    let mut acc = Commitment::zeros(D, KAPPA);
    for (rho, c) in rhos.iter().zip(cs.iter()) {
        s_mul_add(&mut acc, rho, c);
    }
    acc
}

#[test]
fn rlc_commitment_combination_accepts_honest_combination() {
    let rhos = vec![deterministic_rq(1), deterministic_rq(2), deterministic_rq(3)];
    let cs = vec![
        deterministic_commitment(11),
        deterministic_commitment(13),
        deterministic_commitment(17),
    ];
    let combined = native_combine(&rhos, &cs);

    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &rho_cols, &cs, &combined).expect("alloc rlc inputs");
    enforce_rlc_commitment_combination(&mut b, &wires);

    assert!(
        b.is_satisfied(),
        "circuit must accept native (rhos, cs, combined) — first bad row: {:?}",
        b.first_unsatisfied_row()
    );
}

#[test]
fn rlc_commitment_combination_rejects_tampered_combined() {
    let rhos = vec![deterministic_rq(21), deterministic_rq(22)];
    let cs = vec![deterministic_commitment(31), deterministic_commitment(32)];
    let combined = native_combine(&rhos, &cs);

    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &rho_cols, &cs, &combined).expect("alloc rlc inputs");
    enforce_rlc_commitment_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.combined_c_data[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "circuit accepted a tampered combined commitment");
}

#[test]
fn rlc_commitment_combination_rejects_tampered_input_commitment() {
    let rhos = vec![deterministic_rq(41), deterministic_rq(42)];
    let cs = vec![deterministic_commitment(51), deterministic_commitment(52)];
    let combined = native_combine(&rhos, &cs);

    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &rho_cols, &cs, &combined).expect("alloc rlc inputs");
    enforce_rlc_commitment_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.inputs[0].c_data[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "circuit accepted a tampered input commitment lane");
}

#[test]
fn rlc_commitment_combination_rejects_tampered_rho() {
    let rhos = vec![deterministic_rq(61), deterministic_rq(62)];
    let cs = vec![deterministic_commitment(71), deterministic_commitment(72)];
    let combined = native_combine(&rhos, &cs);

    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_commitment_inputs(&mut b, &rho_cols, &cs, &combined).expect("alloc rlc inputs");
    enforce_rlc_commitment_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.inputs[0].rho_coeffs[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "circuit accepted a tampered ρ coefficient");
}

#[test]
fn rlc_commitment_combination_rejects_pair_count_mismatch() {
    let rhos = vec![deterministic_rq(81)];
    let cs = vec![deterministic_commitment(91), deterministic_commitment(92)];
    let combined = native_combine(&[deterministic_rq(81), deterministic_rq(82)], &cs);

    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let result = alloc_rlc_commitment_inputs(&mut b, &rho_cols, &cs, &combined);
    assert!(result.is_err(), "alloc must reject pair-count mismatch");
}

// ── X-combination ─────────────────────────────────────────────────────────

const M_IN: usize = D;

fn deterministic_x_matrix(seed: u64) -> Mat<F> {
    let active_cols = M_IN / D;
    let mut m = Mat::zero(D, active_cols, F::ZERO);
    let mut s = seed.wrapping_mul(0xDEADBEEFCAFEBABE);
    for rr in 0..D {
        for col in 0..active_cols {
            s = s
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            m.set(rr, col, F::from_u64(s & 0xFF));
        }
    }
    m
}

fn native_x_combine(rhos: &[Rq], xs: &[Mat<F>]) -> Mat<F> {
    let x_cols = M_IN / D;
    let mut acc = Mat::zero(D, x_cols, F::ZERO);
    for (rho, x_i) in rhos.iter().zip(xs.iter()) {
        for col in 0..x_cols {
            let mut x_col = [F::ZERO; D];
            for (rr, slot) in x_col.iter_mut().enumerate() {
                *slot = x_i[(rr, col)];
            }
            let prod = rot_apply_vec(rho, &x_col);
            for rr in 0..D {
                let cur = acc[(rr, col)];
                acc.set(rr, col, cur + prod[rr]);
            }
        }
    }
    acc
}

#[test]
fn rlc_x_combination_accepts_honest_combination() {
    let rhos = vec![deterministic_rq(101), deterministic_rq(102)];
    let xs = vec![deterministic_x_matrix(201), deterministic_x_matrix(202)];
    let combined = native_x_combine(&rhos, &xs);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_x_inputs(&mut b, &rho_cols, &xs, &combined).expect("alloc rlc x inputs");
    enforce_rlc_x_combination(&mut b, &wires);

    assert!(
        b.is_satisfied(),
        "X-combination circuit must accept native (rhos, Xs, combined) — first bad row: {:?}",
        b.first_unsatisfied_row()
    );
}

#[test]
fn rlc_x_combination_rejects_tampered_combined_x() {
    let rhos = vec![deterministic_rq(111), deterministic_rq(112)];
    let xs = vec![deterministic_x_matrix(211), deterministic_x_matrix(212)];
    let combined = native_x_combine(&rhos, &xs);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_x_inputs(&mut b, &rho_cols, &xs, &combined).expect("alloc");
    enforce_rlc_x_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.combined_x_flat[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "X-circuit accepted tampered combined X[0,0]");
}

#[test]
fn rlc_x_combination_rejects_tampered_input_x() {
    let rhos = vec![deterministic_rq(121), deterministic_rq(122)];
    let xs = vec![deterministic_x_matrix(221), deterministic_x_matrix(222)];
    let combined = native_x_combine(&rhos, &xs);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_x_inputs(&mut b, &rho_cols, &xs, &combined).expect("alloc");
    enforce_rlc_x_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.inputs[0].x_flat[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "X-circuit accepted tampered input X[0,0]");
}

// ── y_ring row combination ────────────────────────────────────────────────

fn deterministic_k(seed: u64) -> K {
    let c0 = F::from_u64(seed.wrapping_mul(123).wrapping_add(7) & 0xFFFF);
    let c1 = F::from_u64(seed.wrapping_mul(456).wrapping_add(11) & 0xFFFF);
    K::from_coeffs([c0, c1])
}

fn deterministic_y_row(seed: u64) -> Vec<K> {
    (0..D)
        .map(|i| deterministic_k(seed.wrapping_mul(31).wrapping_add(i as u64)))
        .collect()
}

fn native_y_row_combine(rhos: &[Rq], ys: &[Vec<K>]) -> Vec<K> {
    let mut out = vec![K::ZERO; D];
    for (rho, y_i) in rhos.iter().zip(ys.iter()) {
        let mut y_c0 = [F::ZERO; D];
        let mut y_c1 = [F::ZERO; D];
        for (kk, val) in y_i.iter().enumerate() {
            let [c0, c1] = val.as_coeffs();
            y_c0[kk] = c0;
            y_c1[kk] = c1;
        }
        let prod_c0 = rot_apply_vec(rho, &y_c0);
        let prod_c1 = rot_apply_vec(rho, &y_c1);
        for rr in 0..D {
            out[rr] += K::from_coeffs([prod_c0[rr], prod_c1[rr]]);
        }
    }
    out
}

#[test]
fn rlc_y_row_combination_accepts_honest_combination() {
    let rhos = vec![deterministic_rq(301), deterministic_rq(302)];
    let ys = vec![deterministic_y_row(401), deterministic_y_row(402)];
    let combined = native_y_row_combine(&rhos, &ys);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_row_inputs(&mut b, &rho_cols, &ys, &combined).expect("alloc rlc y row inputs");
    enforce_rlc_y_row_combination(&mut b, &wires);

    assert!(
        b.is_satisfied(),
        "y_ring-row circuit must accept native (rhos, y_i, combined) — first bad row: {:?}",
        b.first_unsatisfied_row()
    );
}

#[test]
fn rlc_y_row_combination_rejects_tampered_combined_c0() {
    let rhos = vec![deterministic_rq(311), deterministic_rq(312)];
    let ys = vec![deterministic_y_row(411), deterministic_y_row(412)];
    let combined = native_y_row_combine(&rhos, &ys);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_row_inputs(&mut b, &rho_cols, &ys, &combined).expect("alloc");
    enforce_rlc_y_row_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.combined_c0[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "y_ring-row circuit accepted tampered combined.c0[0]");
}

#[test]
fn rlc_y_row_combination_rejects_tampered_combined_c1() {
    let rhos = vec![deterministic_rq(321), deterministic_rq(322)];
    let ys = vec![deterministic_y_row(421), deterministic_y_row(422)];
    let combined = native_y_row_combine(&rhos, &ys);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_row_inputs(&mut b, &rho_cols, &ys, &combined).expect("alloc");
    enforce_rlc_y_row_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.combined_c1[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "y_ring-row circuit accepted tampered combined.c1[0]");
}

#[test]
fn rlc_y_row_combination_rejects_tampered_input_y() {
    let rhos = vec![deterministic_rq(331), deterministic_rq(332)];
    let ys = vec![deterministic_y_row(431), deterministic_y_row(432)];
    let combined = native_y_row_combine(&rhos, &ys);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_row_inputs(&mut b, &rho_cols, &ys, &combined).expect("alloc");
    enforce_rlc_y_row_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "baseline must be satisfied");

    let target_col = wires.inputs[0].y_c0[0].col();
    let tampered = b.witness()[target_col] + F::ONE;
    b.tamper_witness(target_col, tampered);

    assert!(!b.is_satisfied(), "y_ring-row circuit accepted tampered input y_c0[0]");
}

// ── Transcript-driven Π_RLC.V (Phase 6d′-b wire) ─────────────────────────

use neo_fold_clean::engine::r1cs_circuit::alphabet_sampling::enforce_pi_rlc_rhos_from_transcript;
use neo_fold_clean::engine::r1cs_circuit::TranscriptGadget;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::alloc_rlc_commitment_inputs_with_rhos;
use neo_transcript::{Poseidon2Transcript, Transcript as NeoTranscript};

const TR_APP: &[u8] = b"neo.test.pi_rlc/transcript/v1";
const TR_ALPHABET: [i8; 5] = [-2, -1, 0, 1, 2];

/// Native mirror of `enforce_pi_rlc_rhos_from_transcript`, with `[0, i]`
/// outer separator + 4-iteration alphabet sampler.
fn native_pi_rlc_rho_coeffs(tr: &mut Poseidon2Transcript, count: usize) -> Vec<[F; D]> {
    let mut out = Vec::with_capacity(count);
    for i in 0..count {
        tr.append_fields_raw(&[F::ZERO, F::from_u64(i as u64)]);
        let mut symbols = Vec::with_capacity(D);
        let mut ctr = i as u64;
        let bucket = 65535u32;
        while symbols.len() < D {
            tr.append_fields_raw(&[F::from_u64(1), F::from_u64(ctr)]);
            let dig = tr.digest32();
            for w in dig.chunks_exact(2) {
                let x = u16::from_le_bytes([w[0], w[1]]) as u32;
                if x < bucket {
                    let idx = (x % 5) as usize;
                    symbols.push(TR_ALPHABET[idx]);
                    if symbols.len() == D {
                        break;
                    }
                }
            }
            ctr = ctr.wrapping_add(1);
        }
        let mut coeffs = [F::ZERO; D];
        for (slot, &s) in coeffs.iter_mut().zip(symbols.iter()) {
            *slot = if s >= 0 {
                F::from_u64(s as u64)
            } else {
                -F::from_u64((-s) as u64)
            };
        }
        out.push(coeffs);
    }
    out
}

#[test]
fn pi_rlc_commitment_combination_with_transcript_derived_rhos() {
    // Use count = 3 to keep the ring-mul rows manageable while still
    // exercising the full transcript-derivation → commitment-combination chain.
    const COUNT: usize = 3;

    // 1. Native side: derive ρ coefficients, build commitments, combine.
    let mut native = Poseidon2Transcript::new(TR_APP);
    let rho_cols = native_pi_rlc_rho_coeffs(&mut native, COUNT);
    let rhos: Vec<Rq> = rho_cols.iter().map(|c| Rq(*c)).collect();
    let cs: Vec<Commitment> = (0..COUNT)
        .map(|i| deterministic_commitment(31 + i as u64 * 17))
        .collect();
    let combined = native_combine(&rhos, &cs);

    // 2. Circuit: drive transcript, derive ρ, allocate combination inputs.
    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, TR_APP);
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(&mut b, &mut tr, COUNT);

    // Sanity: the derived ρ wires should witness-equal the native coefficients.
    for (i, rho_arr) in rho_wires.iter().enumerate() {
        for (j, var) in rho_arr.iter().enumerate() {
            assert_eq!(
                b.witness()[var.col()],
                rho_cols[i][j],
                "ρ_{i}[{j}] derivation diverges from native"
            );
        }
    }

    let wires = alloc_rlc_commitment_inputs_with_rhos(&mut b, &rho_wires, &cs, &combined)
        .expect("alloc rlc inputs with derived ρ");
    enforce_rlc_commitment_combination(&mut b, &wires);

    assert!(
        b.is_satisfied(),
        "Π_RLC.V with transcript-derived ρ must accept honest native combination — first bad row: {:?}",
        b.first_unsatisfied_row()
    );
}

#[test]
fn pi_rlc_commitment_combination_rejects_tampered_commitment_under_transcript_derived_rhos() {
    const COUNT: usize = 3;

    let mut native = Poseidon2Transcript::new(TR_APP);
    let rho_cols = native_pi_rlc_rho_coeffs(&mut native, COUNT);
    let rhos: Vec<Rq> = rho_cols.iter().map(|c| Rq(*c)).collect();
    let cs: Vec<Commitment> = (0..COUNT)
        .map(|i| deterministic_commitment(101 + i as u64 * 7))
        .collect();
    let combined = native_combine(&rhos, &cs);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, TR_APP);
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(&mut b, &mut tr, COUNT);
    let wires = alloc_rlc_commitment_inputs_with_rhos(&mut b, &rho_wires, &cs, &combined).expect("alloc");
    enforce_rlc_commitment_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "baseline");

    // Tamper child 0's first c lane. Should break the lane-combination equality.
    let target = wires.inputs[0].c_data[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered input commitment must be rejected");
}

#[test]
fn pi_rlc_full_combination_with_transcript_derived_rhos() {
    // End-to-end Π_RLC.V transcript-driven test covering all three combinations:
    //   c   = Σ ρ_i · c_i        (commitment, ring action)
    //   X   = Σ ρ_i · X_i        (per-column ring action)
    //   y_j = Σ ρ_i · y_{i,j}    (K-element row mixing)
    //
    // All three share the SAME transcript-derived ρ wires.
    use neo_fold_clean::paper::reductions::pi_rlc_circuit::{
        alloc_rlc_x_inputs_with_rhos, alloc_rlc_y_row_inputs_with_rhos,
    };
    const COUNT: usize = 3;

    let mut native = Poseidon2Transcript::new(TR_APP);
    let rho_cols = native_pi_rlc_rho_coeffs(&mut native, COUNT);
    let rhos: Vec<Rq> = rho_cols.iter().map(|c| Rq(*c)).collect();

    // Commitment side
    let cs: Vec<Commitment> = (0..COUNT)
        .map(|i| deterministic_commitment(71 + i as u64 * 19))
        .collect();
    let combined_c = native_combine(&rhos, &cs);

    // X side
    let xs: Vec<Mat<F>> = (0..COUNT)
        .map(|i| deterministic_x_matrix(83 + i as u64 * 23))
        .collect();
    let combined_x = native_x_combine(&rhos, &xs);

    // y_ring side (one row)
    let ys: Vec<Vec<K>> = (0..COUNT)
        .map(|i| deterministic_y_row(97 + i as u64 * 29))
        .collect();
    let combined_y = native_y_row_combine(&rhos, &ys);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, TR_APP);
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(&mut b, &mut tr, COUNT);

    // Commitment
    let wires_c = alloc_rlc_commitment_inputs_with_rhos(&mut b, &rho_wires, &cs, &combined_c).expect("alloc c");
    enforce_rlc_commitment_combination(&mut b, &wires_c);

    // X
    let wires_x = alloc_rlc_x_inputs_with_rhos(&mut b, &rho_wires, &xs, &combined_x).expect("alloc x");
    enforce_rlc_x_combination(&mut b, &wires_x);

    // y_ring
    let wires_y = alloc_rlc_y_row_inputs_with_rhos(&mut b, &rho_wires, &ys, &combined_y).expect("alloc y");
    enforce_rlc_y_row_combination(&mut b, &wires_y);

    assert!(
        b.is_satisfied(),
        "full Π_RLC.V (c + X + y_ring) with transcript-derived ρ must accept honest native combination (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    assert_no_unconstrained_columns(&b, "full transcript-derived Π_RLC.V c+X+y");

    // Spot-check: tampering an X entry breaks satisfaction.
    let target = wires_x.inputs[0].x_flat[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "full Π_RLC.V must reject a tampered input X entry");
}

#[test]
fn pi_rlc_commitment_combination_rejects_tampered_rho_under_transcript() {
    const COUNT: usize = 3;

    let mut native = Poseidon2Transcript::new(TR_APP);
    let rho_cols = native_pi_rlc_rho_coeffs(&mut native, COUNT);
    let rhos: Vec<Rq> = rho_cols.iter().map(|c| Rq(*c)).collect();
    let cs: Vec<Commitment> = (0..COUNT)
        .map(|i| deterministic_commitment(53 + i as u64 * 19))
        .collect();
    let combined = native_combine(&rhos, &cs);

    let mut b = R1csBuilder::new();
    let mut tr = TranscriptGadget::new(&mut b, TR_APP);
    let rho_wires = enforce_pi_rlc_rhos_from_transcript(&mut b, &mut tr, COUNT);
    let wires = alloc_rlc_commitment_inputs_with_rhos(&mut b, &rho_wires, &cs, &combined).expect("alloc");
    enforce_rlc_commitment_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "baseline");

    // Tamper a derived ρ coefficient wire. Multiple constraints depend on
    // it: the alphabet-sample's selection equality (output_symbol = symbol_k
    // via one-hot extraction) AND the ring-action gadget downstream. Either
    // must catch the tamper.
    let target = rho_wires[0][0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(
        !b.is_satisfied(),
        "tampered transcript-derived ρ coefficient must be rejected"
    );
}

// ── Native Π_RLC.V sidecar authority ─────────────────────────────────────

#[test]
fn pi_rlc_native_rejects_combined_ct_not_derived_from_y_ring() {
    let (prep, fresh_claims, running, mut proof) = native_pi_rlc_fixture(903);
    assert!(!proof.pi_rlc.combined.ct.is_empty(), "fixture must carry ct");
    proof.pi_rlc.combined.ct[0] += K::ONE;

    let err = verify_pi_rlc_only(&prep, &fresh_claims, &running, &proof)
        .expect_err("native Π_RLC.V accepted combined.ct not derived from combined.y_ring");
    assert!(
        matches!(err, pi_rlc::Error::CtConsistency("combined")),
        "expected combined ct-consistency rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_extra_self_consistent_y_ring_row() {
    let (prep, fresh_claims, running, mut proof) = native_pi_rlc_fixture(904);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");

    let extra_row = vec![K::ZERO; D.next_power_of_two()];
    for output in &mut outputs {
        output.y_ring.push(extra_row.clone());
        output.ct.push(K::ZERO);
    }
    proof.pi_rlc.combined.y_ring.push(extra_row);
    proof.pi_rlc.combined.ct.push(K::ZERO);

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted an extra self-consistent y_ring/ct row");
    assert!(
        matches!(
            err,
            pi_rlc::Error::YRingShape("input") | pi_rlc::Error::YRingShape("combined")
        ),
        "expected y_ring row-count rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_extra_self_consistent_r_limb() {
    let (prep, fresh_claims, running, mut proof) = native_pi_rlc_fixture(906);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");

    for output in &mut outputs {
        output.r.push(K::ZERO);
    }
    proof.pi_rlc.combined.r.push(K::ZERO);

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted an extra self-consistent r limb");
    assert!(
        matches!(err, pi_rlc::Error::RShape("input") | pi_rlc::Error::RShape("combined")),
        "expected r shape rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_input_r_not_shared_by_all_outputs() {
    let (prep, fresh_claims, running, proof) = native_pi_rlc_fixture_many(910, 2);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");
    assert!(outputs.len() > 1, "fixture must expose at least two Π_CCS outputs");
    assert!(!outputs[1].r.is_empty(), "fixture must carry an r point");

    outputs[1].r[0] += K::ONE;

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted inputs with different evaluation points");
    assert!(
        matches!(err, pi_rlc::Error::RConsistency),
        "expected r-consistency rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_input_fold_digest_not_inherited_by_combined_parent() {
    let (prep, fresh_claims, running, proof) = native_pi_rlc_fixture_many(907, 2);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");
    assert!(outputs.len() > 1, "fixture must expose at least two Π_CCS outputs");

    outputs[1].fold_digest[0] ^= 1;

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted an input fold_digest not inherited by the combined parent");
    assert!(
        matches!(err, pi_rlc::Error::FoldDigest),
        "expected fold_digest propagation rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_noncanonical_input_x_width() {
    let (prep, fresh_claims, running, mut proof) = native_pi_rlc_fixture_complete_public_ring(911, 2);

    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("paper Π_CCS.V fixture must accept");

    let widen = |claim: &mut neo_fold_clean::CeClaim| {
        let old_cols = claim.X.cols();
        let mut widened = Mat::zero(D, old_cols + 1, F::ZERO);
        for row in 0..D {
            for column in 0..old_cols {
                widened[(row, column)] = claim.X[(row, column)];
            }
        }
        claim.X = widened;
    };
    for output in &mut outputs {
        widen(output);
    }
    widen(&mut proof.pi_rlc.combined);

    pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted zero-padded X matrices wider than the SuperNeo coefficient embedding");
}

#[test]
fn pi_rlc_native_rejects_nonzero_input_y_ring_padding_lane() {
    let (prep, fresh_claims, running, proof) = native_pi_rlc_fixture(912);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");
    assert!(outputs[0].y_ring[0].len() > D, "fixture must carry padded y_ring lanes");

    outputs[0].y_ring[0][D] += K::ONE;

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted a non-zero input y_ring padding lane");
    assert!(
        matches!(err, pi_rlc::Error::YRingPadding("input")),
        "expected y_ring padding rejection, got {err:?}"
    );
}

#[test]
fn pi_rlc_native_rejects_truncated_input_y_ring_row() {
    let (prep, fresh_claims, running, proof) = native_pi_rlc_fixture(913);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");
    assert!(outputs[0].y_ring[0].len() > D, "fixture must carry padded y_ring lanes");

    outputs[0].y_ring[0].truncate(D);

    let err = pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
    .expect_err("native Π_RLC.V accepted a non-canonical truncated input y_ring row");
    assert!(
        matches!(err, pi_rlc::Error::YRingShape("input")),
        "expected y_ring shape rejection, got {err:?}"
    );
}

fn native_pi_rlc_fixture(
    seed: u64,
) -> (
    neo_fold_clean::Preprocessing,
    Vec<CcsClaim>,
    RunningInstance,
    nifs::NifsProof,
) {
    let prep = support::toy_preprocessing();
    let fresh = vec![support::toy_instance(&prep, seed)];
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let running = RunningInstance::default();
    let mut tr = PaperTranscript::session();
    let (_next, proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P fixture");
    (prep, fresh_claims, running, proof)
}

fn native_pi_rlc_fixture_many(
    seed: u64,
    fresh_count: usize,
) -> (
    neo_fold_clean::Preprocessing,
    Vec<CcsClaim>,
    RunningInstance,
    nifs::NifsProof,
) {
    assert!(fresh_count > 0);
    let prep = support::toy_preprocessing();
    let fresh = (0..fresh_count)
        .map(|idx| support::toy_instance(&prep, seed + idx as u64))
        .collect::<Vec<_>>();
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let running = RunningInstance::default();
    let mut tr = PaperTranscript::session();
    let (_next, proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P fixture");
    (prep, fresh_claims, running, proof)
}

fn native_pi_rlc_fixture_complete_public_ring(
    seed: u64,
    fresh_count: usize,
) -> (Preprocessing, Vec<CcsClaim>, RunningInstance, nifs::NifsProof) {
    assert!(fresh_count > 0);
    let structure =
        CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, vec![])).expect("whole-ring toy CCS structure");
    let params = config::r1cs_params(structure.n, structure.m).expect("production-core toy params");
    support::install_ajtai_module(&params, &structure);
    let prep = preprocess(params, structure, Some(D)).expect("whole-ring toy preprocessing");
    let fresh = (0..fresh_count)
        .map(|idx| {
            let assignment = vec![F::ZERO; D];
            CcsInstance::from_low_norm_assignment(&prep.params, &prep.log, prep.structure(), &assignment, D)
                .unwrap_or_else(|err| panic!("whole-ring toy instance {}: {err}", seed + idx as u64))
        })
        .collect::<Vec<_>>();
    let fresh_claims = fresh
        .iter()
        .map(|instance| instance.claim.clone())
        .collect::<Vec<_>>();
    let running = RunningInstance::default();
    let mut tr = PaperTranscript::session();
    let (_next, proof) = nifs::prove(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &prep.log,
        None,
        prep.mix_rhos_commits(),
        prep.combine_b_pows(),
        fresh,
        &running,
    )
    .expect("NIFS.P whole-ring fixture");
    (prep, fresh_claims, running, proof)
}

fn verify_pi_rlc_only(
    prep: &neo_fold_clean::Preprocessing,
    fresh_claims: &[CcsClaim],
    running: &RunningInstance,
    proof: &nifs::NifsProof,
) -> Result<neo_fold_clean::CeClaim, pi_rlc::Error> {
    let mut tr = PaperTranscript::session();
    let outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        fresh_claims,
        running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");
    pi_rlc::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &proof.pi_rlc,
    )
}

#[test]
fn pi_rlc_native_rejects_noncanonical_fold_digest_limb_alias() {
    let (prep, fresh_claims, running, proof) = native_pi_rlc_fixture(971);
    let mut tr = PaperTranscript::session();
    let mut outputs = pi_ccs::verify(
        &mut tr,
        &prep.params,
        prep.structure(),
        prep.optimized_cache(),
        &fresh_claims,
        &running,
        &proof.pi_ccs,
    )
    .expect("Π_CCS.V fixture must accept");

    let mut noncanonical_zero = [0u8; 32];
    noncanonical_zero[..8].copy_from_slice(&F::ORDER_U64.to_le_bytes());
    for output in &mut outputs {
        output.fold_digest = noncanonical_zero;
    }

    let mut rho_tr = Poseidon2Transcript::new(b"neo.fold.clean/session/v1");
    let input_claims_digest = pi_ccs_outputs_digest(&outputs);
    rho_tr.append_fields(b"pi_rlc/input_claims_digest", &input_claims_digest);
    let rhos = optimized::sample_rho_n(&mut rho_tr, &prep.params, outputs.len()).expect("sample forged rhos");
    let dummy_witnesses = outputs
        .iter()
        .map(|_| Mat::zero(D, prep.structure().m.div_ceil(D), F::ZERO))
        .collect::<Vec<_>>();
    let (combined, _) = optimized::prove_pi_rlc(
        &prep.params,
        prep.structure(),
        &rhos,
        &outputs,
        &dummy_witnesses,
        prep.mix_rhos_commits(),
    )
    .expect("recompute forged public RLC parent");

    let forged = pi_rlc::Proof { combined };
    let mut verify_tr = PaperTranscript::session();
    let err = pi_rlc::verify(
        &mut verify_tr,
        &prep.params,
        prep.structure(),
        prep.mix_rhos_commits(),
        &outputs,
        &forged,
    )
    .expect_err("native Π_RLC.V accepted a noncanonical fold_digest limb aliasing to zero");
    assert!(
        matches!(
            err,
            pi_rlc::Error::FoldDigestCanonicality {
                owner: "input",
                lane: 0
            }
        ),
        "expected input fold-digest canonicality rejection, got {err:?}"
    );
}

fn assert_no_unconstrained_columns(builder: &R1csBuilder, label: &str) {
    let unconstrained = builder.unconstrained_columns();
    assert!(
        unconstrained.is_empty(),
        "{label} left unconstrained columns: {unconstrained:?}"
    );
}
