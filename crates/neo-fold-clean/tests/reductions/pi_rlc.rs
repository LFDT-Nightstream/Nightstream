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

use neo_ajtai::{s_mul_add, Commitment};
use neo_ccs::Mat;
use neo_fold_clean::engine::r1cs_circuit::field_ext::KVar;
use neo_fold_clean::engine::r1cs_circuit::Lc;
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::reductions::pi_rlc_circuit::{
    alloc_rlc_commitment_inputs, alloc_rlc_x_inputs, alloc_rlc_y_row_inputs, alloc_rlc_y_zcol_inputs,
    enforce_rlc_commitment_combination, enforce_rlc_s_col_consistency, enforce_rlc_x_combination,
    enforce_rlc_y_row_combination, enforce_rlc_y_zcol_combination,
};
use neo_math::ring::{cf, rot_apply_vec, Rq, D};
use neo_math::{KExtensions, F, K};
use p3_field::PrimeCharacteristicRing;

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

const M_IN: usize = 2; // small for tests

fn deterministic_x_matrix(seed: u64) -> Mat<F> {
    // Fill only the active ring columns (`ceil(M_IN / D)`); the rest must
    // be structural zeros to match native `project_x_from_witness_mat` and
    // the circuit's `enforce_rlc_x_combination` inactive-zero constraint.
    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(M_IN);
    let mut m = Mat::zero(D, M_IN, F::ZERO);
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
    let mut acc = Mat::zero(D, M_IN, F::ZERO);
    for (rho, x_i) in rhos.iter().zip(xs.iter()) {
        for col in 0..M_IN {
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

#[test]
fn rlc_x_combination_rejects_nonzero_inactive_input_x() {
    // active_cols = ceil(M_IN / D); the X fold enforces input cols
    // >= active_cols are zero so a prover can't smuggle data into them.
    let rhos = vec![deterministic_rq(131), deterministic_rq(132)];
    let xs = vec![deterministic_x_matrix(231), deterministic_x_matrix(232)];
    let combined = native_x_combine(&rhos, &xs);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_x_inputs(&mut b, &rho_cols, &xs, &combined).expect("alloc");
    enforce_rlc_x_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "baseline must be satisfied");

    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(M_IN);
    assert!(active_cols < M_IN, "test setup: expected at least one inactive col");
    // Tamper input[0]'s X[0, active_cols] (an inactive slot).
    let target_col = wires.inputs[0].x_flat[0 * M_IN + active_cols].col();
    b.tamper_witness(target_col, F::ONE);
    assert!(!b.is_satisfied(), "X fold accepted non-zero inactive input X col");
}

#[test]
fn rlc_x_combination_rejects_nonzero_inactive_combined_x() {
    let rhos = vec![deterministic_rq(141), deterministic_rq(142)];
    let xs = vec![deterministic_x_matrix(241), deterministic_x_matrix(242)];
    let combined = native_x_combine(&rhos, &xs);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_x_inputs(&mut b, &rho_cols, &xs, &combined).expect("alloc");
    enforce_rlc_x_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "baseline must be satisfied");

    let active_cols = neo_fold_clean::paper::relations::superneo_public_x_cols(M_IN);
    assert!(active_cols < M_IN, "test setup: expected at least one inactive col");
    let target_col = wires.combined_x_flat[0 * M_IN + active_cols].col();
    b.tamper_witness(target_col, F::ONE);
    assert!(!b.is_satisfied(), "X fold accepted non-zero inactive combined X col");
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

// ── SplitNc NC-channel: y_zcol combination + s_col consistency ────────────

/// Native mirror of the padded-K-vector RLC combination:
/// `acc[0..D] = Σ_i (ρ_i · input_i[0..D])`, `acc[D..d_pad] = 0`. This is
/// the production semantics that SplitNc y_zcol / y_ring rows obey.
fn native_padded_y_combine(rhos: &[Rq], ys: &[Vec<K>], d_pad: usize) -> Vec<K> {
    let mut out = vec![K::ZERO; d_pad];
    for (rho, y_i) in rhos.iter().zip(ys.iter()) {
        let mut y_c0 = [F::ZERO; D];
        let mut y_c1 = [F::ZERO; D];
        for kk in 0..D {
            let [c0, c1] = y_i[kk].as_coeffs();
            y_c0[kk] = c0;
            y_c1[kk] = c1;
        }
        let prod_c0 = rot_apply_vec(rho, &y_c0);
        let prod_c1 = rot_apply_vec(rho, &y_c1);
        for rr in 0..D {
            out[rr] += K::from_coeffs([prod_c0[rr], prod_c1[rr]]);
        }
        // Lanes [D, d_pad) deliberately untouched — native leaves them zero.
    }
    out
}

fn deterministic_padded_y(seed: u64, d_pad: usize) -> Vec<K> {
    // First D lanes carry data; lanes [D, d_pad) are zero on input too
    // (real SplitNc proofs have zero-padded tail on both inputs and
    // outputs after one fold).
    let head = deterministic_y_row(seed);
    let mut out = vec![K::ZERO; d_pad];
    for (i, v) in head.into_iter().enumerate() {
        out[i] = v;
    }
    out
}

#[test]
fn rlc_y_zcol_combination_accepts_honest_combination_at_d_pad_equal_to_d() {
    // Degenerate case: d_pad == D so there's no tail to zero.
    let d_pad = D;
    let rhos = vec![deterministic_rq(701), deterministic_rq(702)];
    let ys = vec![deterministic_padded_y(801, d_pad), deterministic_padded_y(802, d_pad)];
    let combined = native_padded_y_combine(&rhos, &ys, d_pad);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_zcol_inputs(&mut b, &rho_cols, &ys, &combined, d_pad).expect("alloc y_zcol");
    enforce_rlc_y_zcol_combination(&mut b, &wires);

    assert!(b.is_satisfied(), "honest y_zcol combination rejected at d_pad=D");
}

#[test]
fn rlc_y_zcol_combination_accepts_honest_combination_at_production_d_pad() {
    // Production shape: D=54, d_pad=64. The combined output's lanes
    // [D, d_pad) are constrained to zero in-circuit; verifying that
    // native_padded_y_combine produces zero there.
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D, "test only meaningful when d_pad > D");

    let rhos = vec![deterministic_rq(721), deterministic_rq(722)];
    let ys = vec![deterministic_padded_y(821, d_pad), deterministic_padded_y(822, d_pad)];
    let combined = native_padded_y_combine(&rhos, &ys, d_pad);
    // Sanity: native leaves the tail zero.
    for rr in D..d_pad {
        assert_eq!(combined[rr], K::ZERO, "native combine should zero lane {rr}");
    }
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_zcol_inputs(&mut b, &rho_cols, &ys, &combined, d_pad).expect("alloc y_zcol");
    enforce_rlc_y_zcol_combination(&mut b, &wires);

    assert!(
        b.is_satisfied(),
        "honest y_zcol combination rejected at production d_pad (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
}

#[test]
fn rlc_y_zcol_combination_rejects_nonzero_tail() {
    // Soundness witness: if the combined.y_zcol tail [D, d_pad) is non-zero,
    // the gadget must reject. This catches a prover trying to smuggle a
    // non-rotation contribution through the padded slot.
    let d_pad = D.next_power_of_two();
    assert!(d_pad > D);

    let rhos = vec![deterministic_rq(731), deterministic_rq(732)];
    let ys = vec![deterministic_padded_y(831, d_pad), deterministic_padded_y(832, d_pad)];
    let mut combined = native_padded_y_combine(&rhos, &ys, d_pad);
    // Stuff a non-zero value into the first tail lane.
    combined[D] = K::ONE;
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_zcol_inputs(&mut b, &rho_cols, &ys, &combined, d_pad).expect("alloc y_zcol");
    enforce_rlc_y_zcol_combination(&mut b, &wires);

    assert!(
        !b.is_satisfied(),
        "circuit accepted a non-zero combined.y_zcol[D] (tail-must-be-zero violated)"
    );
}

#[test]
fn rlc_y_zcol_combination_rejects_tampered_combined() {
    let d_pad = D;
    let rhos = vec![deterministic_rq(711), deterministic_rq(712)];
    let ys = vec![deterministic_padded_y(811, d_pad), deterministic_padded_y(812, d_pad)];
    let combined = native_padded_y_combine(&rhos, &ys, d_pad);
    let rho_cols: Vec<[F; D]> = rhos.iter().copied().map(cf).collect();

    let mut b = R1csBuilder::new();
    let wires = alloc_rlc_y_zcol_inputs(&mut b, &rho_cols, &ys, &combined, d_pad).expect("alloc y_zcol");
    enforce_rlc_y_zcol_combination(&mut b, &wires);
    assert!(b.is_satisfied(), "baseline");

    let target = wires.combined_c0[0].col();
    b.tamper_witness(target, b.witness()[target] + F::ONE);

    assert!(!b.is_satisfied(), "tampered y_zcol combined was accepted");
}

#[test]
fn rlc_s_col_consistency_accepts_shared_s_col() {
    // Π_RLC propagates s_col by assertion: every input must already share
    // s_col, and the combined parent inherits that same value. The gadget
    // emits lane-wise K equalities between each input.s_col and combined.s_col.
    let mut bd = R1csBuilder::new();
    let s_col: Vec<K> = (0..3).map(|i| K::from_u64((i + 1) as u64)).collect();
    let s_col_vars: Vec<KVar> = s_col
        .iter()
        .copied()
        .map(|v| {
            let [c0, c1] = v.as_coeffs();
            KVar::alloc(&mut bd, c0, c1)
        })
        .collect();
    let input_vars = vec![s_col_vars.clone(), s_col_vars.clone()];

    enforce_rlc_s_col_consistency(&mut bd, &input_vars, &s_col_vars).expect("emit");
    assert!(bd.is_satisfied(), "honest s_col consistency rejected");
}

#[test]
fn rlc_s_col_consistency_rejects_tampered_input_s_col() {
    let mut bd = R1csBuilder::new();
    let s_col: Vec<K> = (0..3).map(|i| K::from_u64((i + 7) as u64)).collect();
    let combined_vars: Vec<KVar> = s_col
        .iter()
        .copied()
        .map(|v| {
            let [c0, c1] = v.as_coeffs();
            KVar::alloc(&mut bd, c0, c1)
        })
        .collect();
    let input_a: Vec<KVar> = s_col
        .iter()
        .copied()
        .map(|v| {
            let [c0, c1] = v.as_coeffs();
            KVar::alloc(&mut bd, c0, c1)
        })
        .collect();
    let input_b: Vec<KVar> = s_col
        .iter()
        .copied()
        .map(|v| {
            let [c0, c1] = v.as_coeffs();
            KVar::alloc(&mut bd, c0, c1)
        })
        .collect();
    let inputs = vec![input_a, input_b.clone()];
    enforce_rlc_s_col_consistency(&mut bd, &inputs, &combined_vars).expect("emit");
    assert!(bd.is_satisfied(), "baseline");

    // Tamper input_b[0].c0 — the equality `inputs[1][0].c0 == combined[0].c0`
    // must break.
    let target = input_b[0].c0.col();
    bd.tamper_witness(target, bd.witness()[target] + F::ONE);
    assert!(!bd.is_satisfied(), "tampered input.s_col was accepted");
}

#[allow(dead_code)]
fn _silence_unused_lc(_x: Lc) {}
