//! In-circuit digest gadgets — parity against native `paper::digest` functions.
//!
//! Native digest functions are called directly here (no inlining): the
//! in-circuit gadgets and the native ones are required to produce
//! byte-identical outputs for the same inputs.

use neo_ajtai::Commitment;
use neo_ccs::{CeClaim as NeoCeClaim, Mat};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::digest::{
    accumulator_digest_from_claims, boundary_update_digest, digest32_as_fields, public_trace_update_digest,
    state_x_out_digest,
};
use neo_fold_clean::paper::f_prime::digest_circuit::{
    enforce_boundary_update_digest_circuit, enforce_public_trace_update_digest_circuit,
    enforce_state_x_out_digest_circuit, StateXOutDigestInputs,
};
use neo_fold_clean::paper::reductions::accumulator_digest_circuit::{
    enforce_accumulator_digest_from_children_circuit, enforce_accumulator_digest_from_parent_circuit,
};
use neo_math::ring::D;
use neo_math::{F, K};
use p3_field::PrimeCharacteristicRing;

type CeClaim = NeoCeClaim<Commitment, F, K>;

// ── Helpers ────────────────────────────────────────────────────────────

fn alloc_4(b: &mut R1csBuilder, vals: [F; 4]) -> [neo_fold_clean::engine::r1cs_circuit::Var; 4] {
    let mut out = [b.alloc(F::ZERO); 4];
    for (slot, v) in out.iter_mut().zip(vals.iter()) {
        *slot = b.alloc(*v);
    }
    out
}

fn extract_4(b: &R1csBuilder, vars: [neo_fold_clean::engine::r1cs_circuit::Var; 4]) -> [F; 4] {
    let mut out = [F::ZERO; 4];
    for (slot, var) in out.iter_mut().zip(vars.iter()) {
        *slot = b.witness()[var.col()];
    }
    out
}

fn seeded_bytes(seed: u64) -> [u8; 32] {
    let mut out = [0u8; 32];
    let mut s = seed;
    for byte in out.iter_mut() {
        s = s
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        *byte = (s >> 8) as u8;
    }
    out
}

fn seeded_digest_fields(seed: u64) -> [F; 4] {
    digest32_as_fields(seeded_bytes(seed))
}

// ── boundary_update parity ───────────────────────────────────────────────

#[test]
fn boundary_update_circuit_matches_native() {
    let prev_bytes = seeded_bytes(0xC0FFEE);
    let chunk = seeded_digest_fields(0xBADBEEF);
    let expected_bytes = boundary_update_digest(prev_bytes, chunk);
    let expected = digest32_as_fields(expected_bytes);

    let mut b = R1csBuilder::new();
    let prev_vars = alloc_4(&mut b, digest32_as_fields(prev_bytes));
    let chunk_vars = alloc_4(&mut b, chunk);
    let out_vars = enforce_boundary_update_digest_circuit(&mut b, prev_vars, chunk_vars);

    assert!(
        b.is_satisfied(),
        "boundary_update circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected, "boundary_update digest diverges from native");
}

#[test]
fn boundary_update_circuit_rejects_tampered_prev() {
    let prev = seeded_digest_fields(0x1234);
    let chunk = seeded_digest_fields(0x5678);

    let mut b = R1csBuilder::new();
    let prev_vars = alloc_4(&mut b, prev);
    let chunk_vars = alloc_4(&mut b, chunk);
    let _ = enforce_boundary_update_digest_circuit(&mut b, prev_vars, chunk_vars);
    assert!(b.is_satisfied(), "baseline");

    let target = prev_vars[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(!b.is_satisfied(), "tampered prev must break boundary_update digest");
}

// ── public_trace_update parity ───────────────────────────────────────────

#[test]
fn public_trace_update_circuit_matches_native() {
    let prev_bytes = seeded_bytes(0xA11CE);
    let chunk = seeded_digest_fields(0xB0B);
    let expected = digest32_as_fields(public_trace_update_digest(prev_bytes, chunk));

    let mut b = R1csBuilder::new();
    let prev_vars = alloc_4(&mut b, digest32_as_fields(prev_bytes));
    let chunk_vars = alloc_4(&mut b, chunk);
    let out_vars = enforce_public_trace_update_digest_circuit(&mut b, prev_vars, chunk_vars);

    assert!(
        b.is_satisfied(),
        "public_trace_update circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected);
}

// ── state_x_out parity ───────────────────────────────────────────────────

#[test]
fn state_x_out_circuit_matches_native_typical_values() {
    let vk_fs = seeded_bytes(0x100);
    let structure = seeded_digest_fields(0x200);
    let z0 = seeded_bytes(0x300);
    let zi = seeded_bytes(0x400);
    let sa = seeded_bytes(0x500);
    let ca = seeded_bytes(0x600);
    let pt = seeded_bytes(0x700);
    let chunk_count = 42u64;
    let step_count = 100u64;
    let pc = 1u64;

    let expected_bytes = state_x_out_digest(vk_fs, &structure, chunk_count, step_count, z0, zi, pc, sa, ca, pt);
    let expected = digest32_as_fields(expected_bytes);

    let mut b = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(chunk_count)),
        step_count: b.alloc(F::from_u64(step_count)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(pc)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let out_vars = enforce_state_x_out_digest_circuit(&mut b, &inputs);

    assert!(
        b.is_satisfied(),
        "state_x_out circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected, "state_x_out digest diverges from native");
}

#[test]
fn state_x_out_circuit_matches_native_at_u32_boundary() {
    // Exercise the u64_halves split: pick counters whose lo half is at the
    // 32-bit boundary so a wrong bit-decomposition would corrupt the
    // resulting digest.
    let vk_fs = seeded_bytes(0x10);
    let structure = seeded_digest_fields(0x20);
    let z0 = seeded_bytes(0x30);
    let zi = seeded_bytes(0x40);
    let sa = seeded_bytes(0x50);
    let ca = seeded_bytes(0x60);
    let pt = seeded_bytes(0x70);
    let chunk_count = 0xFFFF_FFFFu64;
    let step_count = 0x1_0000_0000u64;
    let pc = 0xDEAD_BEEFu64;

    let expected = digest32_as_fields(state_x_out_digest(
        vk_fs,
        &structure,
        chunk_count,
        step_count,
        z0,
        zi,
        pc,
        sa,
        ca,
        pt,
    ));

    let mut b = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(chunk_count)),
        step_count: b.alloc(F::from_u64(step_count)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(pc)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let out_vars = enforce_state_x_out_digest_circuit(&mut b, &inputs);

    assert!(b.is_satisfied(), "u32-boundary circuit unsatisfied");
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected);
}

// ── accumulator_digest parity ────────────────────────────────────────────

const KAPPA: usize = 2; // small for fast tests

fn make_commitment(seed: u64) -> Commitment {
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

fn make_ce_claim(seed: u64) -> CeClaim {
    CeClaim {
        c: make_commitment(seed),
        X: Mat::zero(D, 1, F::ZERO),
        r: Vec::new(),
        s_col: Vec::new(),
        y_ring: Vec::new(),
        ct: Vec::new(),
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in: 1,
        fold_digest: [0u8; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

/// Compute the b-ary weighted sum of `claims[i].c.data` lane-by-lane. This
/// equals `dec_wires.parent.c_data` under `enforce_dec_v`, so the circuit
/// just hashes the parent wires.
fn weighted_sum_c_data(base: u32, claims: &[CeClaim]) -> Vec<F> {
    if claims.is_empty() {
        return Vec::new();
    }
    let parent_len = claims[0].c.data.len();
    let base_f = F::from_u64(base as u64);
    let mut out = vec![F::ZERO; parent_len];
    let mut pow = F::ONE;
    for claim in claims {
        for (slot, &v) in out.iter_mut().zip(claim.c.data.iter()) {
            *slot += v * pow;
        }
        pow *= base_f;
    }
    out
}

#[test]
fn accumulator_digest_circuit_matches_native_empty() {
    let expected = digest32_as_fields(accumulator_digest_from_claims(2, &[]));

    let mut b = R1csBuilder::new();
    let out_vars = enforce_accumulator_digest_from_parent_circuit(&mut b, 0, &[]);

    assert!(b.is_satisfied(), "accumulator (empty) circuit unsatisfied");
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected, "empty-acc digest diverges from native");
}

#[test]
fn accumulator_digest_circuit_matches_native_nonempty() {
    let claims = vec![make_ce_claim(0xC0FFEE), make_ce_claim(0xBADBEEF), make_ce_claim(0xCAFE)];
    let base: u32 = 2;
    let expected = digest32_as_fields(accumulator_digest_from_claims(base, &claims));

    // Build parent.c.data as the b-ary weighted sum of children commitments
    // (which is what `enforce_dec_v` constrains it to equal).
    let parent_c_data = weighted_sum_c_data(base, &claims);
    let mut b = R1csBuilder::new();
    let parent_wires: Vec<_> = parent_c_data.iter().map(|&v| b.alloc(v)).collect();

    let out_vars = enforce_accumulator_digest_from_parent_circuit(&mut b, claims.len(), &parent_wires);

    assert!(
        b.is_satisfied(),
        "accumulator (nonempty) circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected, "acc digest diverges from native");
}

#[test]
fn accumulator_digest_from_children_matches_native() {
    let claims = vec![make_ce_claim(0xC0FFEE), make_ce_claim(0xBADBEEF), make_ce_claim(0xCAFE)];
    let base: u32 = 2;
    let expected = digest32_as_fields(accumulator_digest_from_claims(base, &claims));

    let mut b = R1csBuilder::new();
    let children_c_data: Vec<Vec<_>> = claims
        .iter()
        .map(|c| c.c.data.iter().map(|&v| b.alloc(v)).collect())
        .collect();
    let out_vars = enforce_accumulator_digest_from_children_circuit(&mut b, base, &children_c_data)
        .expect("accumulator-from-children emit");

    assert!(
        b.is_satisfied(),
        "from-children digest circuit unsatisfied (first bad row: {:?})",
        b.first_unsatisfied_row()
    );
    let got = extract_4(&b, out_vars);
    assert_eq!(got, expected, "from-children digest diverges from native");
}

#[test]
fn accumulator_digest_from_children_rejects_mismatched_child_lengths() {
    // Mismatched commitment lengths must be a clean verifier error, NOT
    // a silently-padded plausible digest.
    let mut b = R1csBuilder::new();
    let mut child0: Vec<_> = (0..(D * KAPPA))
        .map(|i| b.alloc(F::from_u64(i as u64)))
        .collect();
    let mut child1: Vec<_> = (0..(D * KAPPA))
        .map(|i| b.alloc(F::from_u64((i + 100) as u64)))
        .collect();
    child1.pop(); // truncate to wrong length

    let children = vec![child0.clone(), child1];
    let result = enforce_accumulator_digest_from_children_circuit(&mut b, 2, &children);
    assert!(result.is_err(), "must reject mismatched child commitment lengths");

    // Sanity: same-length children produce a valid result.
    child0.pop(); // truncate child0 to match length we'll use for child2
    let child2: Vec<_> = (0..(D * KAPPA - 1))
        .map(|i| b.alloc(F::from_u64((i + 200) as u64)))
        .collect();
    let children_ok = vec![child0, child2];
    let _ok = enforce_accumulator_digest_from_children_circuit(&mut b, 2, &children_ok)
        .expect("same-length children must succeed");
}

#[test]
fn accumulator_digest_circuit_rejects_tampered_parent() {
    let claims = vec![make_ce_claim(0x10), make_ce_claim(0x20)];
    let parent_c_data = weighted_sum_c_data(2, &claims);

    let mut b = R1csBuilder::new();
    let parent_wires: Vec<_> = parent_c_data.iter().map(|&v| b.alloc(v)).collect();
    let _ = enforce_accumulator_digest_from_parent_circuit(&mut b, claims.len(), &parent_wires);
    assert!(b.is_satisfied(), "baseline");

    let target = parent_wires[0].col();
    let tampered = b.witness()[target] + F::ONE;
    b.tamper_witness(target, tampered);
    assert!(
        !b.is_satisfied(),
        "tampered parent.c.data must break accumulator digest"
    );
}

#[test]
fn state_x_out_circuit_rejects_tampered_chunk_count() {
    let vk_fs = seeded_bytes(0x10);
    let structure = seeded_digest_fields(0x20);
    let z0 = seeded_bytes(0x30);
    let zi = seeded_bytes(0x40);
    let sa = seeded_bytes(0x50);
    let ca = seeded_bytes(0x60);
    let pt = seeded_bytes(0x70);

    let mut b = R1csBuilder::new();
    let chunk_count_var = b.alloc(F::from_u64(7));
    let inputs = StateXOutDigestInputs {
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: chunk_count_var,
        step_count: b.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(1)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[chunk_count_var.col()] + F::ONE;
    b.tamper_witness(chunk_count_var.col(), tampered);
    assert!(
        !b.is_satisfied(),
        "tampered chunk_count must break the bit-decomposition consistency constraint"
    );
}
