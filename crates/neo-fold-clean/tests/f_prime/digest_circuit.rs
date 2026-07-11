//! In-circuit digest gadgets — parity against native `paper::digest` functions.
//!
//! Native digest functions are called directly here (no inlining): the
//! in-circuit gadgets and the native ones are required to produce
//! byte-identical outputs for the same inputs.

use neo_ajtai::Commitment;
use neo_ccs::{CcsMatrix, CcsStructure, SparsePoly, Term};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::digest::{
    boundary_update_digest, digest32_as_fields, f_prime_chunk_public_digest, public_trace_update_digest,
    state_x_out_digest, state_x_out_digest_with_mode, structure_digest, vk_fs_digest, StateXOutDigestMode,
};
use neo_fold_clean::paper::f_prime::digest_circuit::{
    enforce_boundary_update_digest_circuit, enforce_f_prime_chunk_public_digest_circuit,
    enforce_public_trace_update_digest_circuit, enforce_state_x_out_digest_circuit,
    enforce_state_x_out_digest_with_nebula_circuit, StateXOutDigestInputs,
};
use neo_fold_clean::paper::relations::CcsClaim;
use neo_math::{D, F};
use neo_params::{goldilocks_paper_b2, NeoParams};
use neo_reductions::engines::utils::digest_ccs_matrices_with_sparse_cache;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

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

#[test]
fn f_prime_chunk_shape_digest_circuit_matches_native_and_binds_start_index() {
    let start_index = 9u64;
    let fresh_len = 3usize;
    let kappa = goldilocks_paper_b2::KAPPA as usize;
    let m_in = 257usize;
    let template = CcsClaim {
        c: Commitment::zeros(D, kappa),
        x: vec![F::ZERO; m_in],
        m_in,
        adv: None,
    };
    let fresh = vec![template; fresh_len];
    let expected = f_prime_chunk_public_digest(start_index, &fresh);

    let mut builder = R1csBuilder::new();
    let start = builder.alloc(F::from_u64(start_index));
    let output = enforce_f_prime_chunk_public_digest_circuit(&mut builder, start, fresh_len, D, kappa, m_in);

    assert!(builder.is_satisfied(), "honest chunk-shape digest must satisfy");
    assert_eq!(extract_4(&builder, output), expected);

    builder.tamper_witness(start.col(), F::from_u64(start_index + 1));
    assert!(
        !builder.is_satisfied(),
        "chunk-shape digest must constrain its in-circuit start index"
    );
}

#[test]
fn vk_fs_digest_binds_large_u64_params_without_field_aliasing() {
    let structure = seeded_digest_fields(0xFA11);
    let initial_semantic = seeded_bytes(0x51A7E);
    let base = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        goldilocks_paper_b2::KAPPA,
        1,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        goldilocks_paper_b2::LAMBDA,
    )
    .expect("base params");
    let aliased_m = NeoParams::new(
        goldilocks_paper_b2::Q,
        goldilocks_paper_b2::ETA as u32,
        goldilocks_paper_b2::D as u32,
        goldilocks_paper_b2::KAPPA,
        goldilocks_paper_b2::Q + 1,
        goldilocks_paper_b2::B_BASE,
        goldilocks_paper_b2::K_RHO,
        goldilocks_paper_b2::T,
        goldilocks_paper_b2::EXTENSION_DEGREE,
        goldilocks_paper_b2::LAMBDA,
    )
    .expect("params with m = field modulus + 1");

    let header = [F::from_u64(9); 4];
    let base_digest = vk_fs_digest(&base, &structure, &header, Some(1), initial_semantic);
    let aliased_digest = vk_fs_digest(&aliased_m, &structure, &header, Some(1), initial_semantic);

    assert_ne!(
        base_digest, aliased_digest,
        "vk_fs_digest must encode u64 params injectively; m=1 and m=p+1 \
         are distinct verifier contexts even though F::from_u64 aliases them"
    );
}

#[test]
fn structure_digest_binds_sparse_identity_dimensions_without_field_aliasing() {
    fn identity_structure(n: usize) -> CcsStructure<F> {
        CcsStructure::new_sparse(
            vec![CcsMatrix::Identity { n }],
            SparsePoly::new(
                1,
                vec![Term {
                    coeff: F::ONE,
                    exps: vec![1],
                }],
            ),
        )
        .expect("identity CCS structure")
    }

    let base = identity_structure(1);
    let aliased = identity_structure((F::ORDER_U64 + 1) as usize);

    assert_ne!(
        structure_digest(&base),
        structure_digest(&aliased),
        "structure_digest must encode sparse/identity dimensions injectively; \
         n=m=1 and n=m=p+1 are distinct verifier structures even though \
         F::from_u64 aliases their dimension words"
    );
}

#[test]
fn ccs_matrix_digest_binds_sparse_identity_dimensions_without_field_aliasing() {
    fn identity_structure(n: usize) -> CcsStructure<F> {
        CcsStructure::new_sparse(
            vec![CcsMatrix::Identity { n }],
            SparsePoly::new(
                1,
                vec![Term {
                    coeff: F::ONE,
                    exps: vec![1],
                }],
            ),
        )
        .expect("identity CCS structure")
    }

    let base = identity_structure(1);
    let aliased = identity_structure((F::ORDER_U64 + 1) as usize);

    assert_ne!(
        digest_ccs_matrices_with_sparse_cache(&base, None),
        digest_ccs_matrices_with_sparse_cache(&aliased, None),
        "Π_CCS matrix digest must encode sparse/identity dimensions injectively; \
         the header transcript must not alias n=m=1 with n=m=p+1"
    );
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
    let header = seeded_digest_fields(0x180);
    let structure = seeded_digest_fields(0x200);
    let z0 = seeded_bytes(0x300);
    let zi = seeded_bytes(0x400);
    let sa = seeded_bytes(0x500);
    let ca = seeded_bytes(0x600);
    let pt = seeded_bytes(0x700);
    let chunk_count = 42u64;
    let step_count = 100u64;
    let pc = 1u64;

    let expected_bytes = state_x_out_digest(
        vk_fs,
        header,
        &structure,
        chunk_count,
        step_count,
        z0,
        zi,
        pc,
        sa,
        ca,
        pt,
    );
    let expected = digest32_as_fields(expected_bytes);

    let mut b = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
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
fn state_x_out_circuit_matches_native_with_nebula_lane() {
    let vk_fs = seeded_bytes(0x8100);
    let header = seeded_digest_fields(0x8180);
    let structure = seeded_digest_fields(0x8200);
    let z0 = seeded_bytes(0x8300);
    let zi = seeded_bytes(0x8400);
    let sa = seeded_bytes(0x8500);
    let ca = seeded_bytes(0x8600);
    let pt = seeded_bytes(0x8700);
    let lane = seeded_digest_fields(0x8800);
    let expected = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateful,
        vk_fs,
        header,
        &structure,
        7,
        11,
        z0,
        zi,
        1,
        sa,
        ca,
        pt,
        Some(lane),
    ));

    let mut builder = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut builder, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut builder, header),
        structure_digest: alloc_4(&mut builder, structure),
        chunk_count: builder.alloc(F::from_u64(7)),
        step_count: builder.alloc(F::from_u64(11)),
        initial_boundary: alloc_4(&mut builder, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut builder, digest32_as_fields(zi)),
        pc: builder.alloc(F::ONE),
        semantic_acc: alloc_4(&mut builder, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut builder, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut builder, digest32_as_fields(pt)),
    };
    let lane_wires = alloc_4(&mut builder, lane);
    let out = enforce_state_x_out_digest_with_nebula_circuit(&mut builder, &inputs, lane_wires);
    assert!(builder.is_satisfied(), "Nebula state hash mirror must satisfy");
    assert_eq!(extract_4(&builder, out), expected);
}

#[test]
fn state_x_out_digest_absorbs_pc() {
    let vk_fs = seeded_bytes(0x110);
    let header = seeded_digest_fields(0x180);
    let structure = seeded_digest_fields(0x220);
    let z0 = seeded_bytes(0x330);
    let zi = seeded_bytes(0x440);
    let sa = seeded_bytes(0x550);
    let ca = seeded_bytes(0x660);
    let pt = seeded_bytes(0x770);

    let pc_1 = state_x_out_digest(vk_fs, header, &structure, 3, 5, z0, zi, 1, sa, ca, pt);
    let pc_2 = state_x_out_digest(vk_fs, header, &structure, 3, 5, z0, zi, 2, sa, ca, pt);

    assert_ne!(pc_1, pc_2, "pc must be load-bearing in state_x_out");
}

#[test]
fn stateless_state_x_out_circuit_matches_native_without_semantic_lanes() {
    let vk_fs = seeded_bytes(0x101);
    let header = seeded_digest_fields(0x181);
    let structure = seeded_digest_fields(0x202);
    let z0 = seeded_bytes(0x303);
    let zi = seeded_bytes(0x404);
    let sa = seeded_bytes(0x505);
    let ca = sa;
    let pt = seeded_bytes(0x707);
    let chunk_count = 4u64;
    let step_count = 9u64;
    let pc = 1u64;

    let expected = digest32_as_fields(state_x_out_digest_with_mode(
        StateXOutDigestMode::Stateless,
        vk_fs,
        header,
        &structure,
        chunk_count,
        step_count,
        z0,
        zi,
        pc,
        sa,
        ca,
        pt,
        None,
    ));

    let mut b = R1csBuilder::new();
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateless,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
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

    assert!(b.is_satisfied(), "stateless state_x_out circuit unsatisfied");
    assert_eq!(extract_4(&b, out_vars), expected);
}

#[test]
fn state_x_out_circuit_matches_native_at_u32_boundary() {
    // Exercise the u64_halves split: pick counters whose lo half is at the
    // 32-bit boundary so a wrong bit-decomposition would corrupt the
    // resulting digest.
    let vk_fs = seeded_bytes(0x10);
    let header = seeded_digest_fields(0x18);
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
        header,
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
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
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

#[test]
fn state_x_out_circuit_rejects_tampered_step_count() {
    let vk_fs = seeded_bytes(0x10);
    let header = seeded_digest_fields(0x18);
    let structure = seeded_digest_fields(0x20);
    let z0 = seeded_bytes(0x30);
    let zi = seeded_bytes(0x40);
    let sa = seeded_bytes(0x50);
    let ca = seeded_bytes(0x60);
    let pt = seeded_bytes(0x70);

    let mut b = R1csBuilder::new();
    let step_count_var = b.alloc(F::from_u64(13));
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(7)),
        step_count: step_count_var,
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(1)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[step_count_var.col()] + F::ONE;
    b.tamper_witness(step_count_var.col(), tampered);
    assert!(
        !b.is_satisfied(),
        "tampered step_count must break the bit-decomposition consistency constraint"
    );
}

#[test]
fn state_x_out_circuit_rejects_tampered_pi_ccs_header() {
    let vk_fs = seeded_bytes(0x9010);
    let header = seeded_digest_fields(0x9018);
    let structure = seeded_digest_fields(0x9020);
    let z0 = seeded_bytes(0x9030);
    let zi = seeded_bytes(0x9040);
    let sa = seeded_bytes(0x9050);
    let ca = seeded_bytes(0x9060);
    let pt = seeded_bytes(0x9070);

    let mut builder = R1csBuilder::new();
    let header_wires = alloc_4(&mut builder, header);
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut builder, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: header_wires,
        structure_digest: alloc_4(&mut builder, structure),
        chunk_count: builder.alloc(F::from_u64(7)),
        step_count: builder.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut builder, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut builder, digest32_as_fields(zi)),
        pc: builder.alloc(F::ONE),
        semantic_acc: alloc_4(&mut builder, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut builder, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut builder, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut builder, &inputs);
    assert!(builder.is_satisfied(), "baseline");

    let tampered = builder.witness()[header_wires[0].col()] + F::ONE;
    builder.tamper_witness(header_wires[0].col(), tampered);
    assert!(
        !builder.is_satisfied(),
        "the carried Split-NC header must be state-hash authority"
    );
}

#[test]
fn state_x_out_circuit_rejects_tampered_pc() {
    let vk_fs = seeded_bytes(0x10);
    let header = seeded_digest_fields(0x18);
    let structure = seeded_digest_fields(0x20);
    let z0 = seeded_bytes(0x30);
    let zi = seeded_bytes(0x40);
    let sa = seeded_bytes(0x50);
    let ca = seeded_bytes(0x60);
    let pt = seeded_bytes(0x70);

    let mut b = R1csBuilder::new();
    let pc_var = b.alloc(F::from_u64(1));
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(7)),
        step_count: b.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: pc_var,
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[pc_var.col()] + F::ONE;
    b.tamper_witness(pc_var.col(), tampered);
    assert!(!b.is_satisfied(), "tampered pc must break state_x_out");
}

#[test]
fn state_x_out_circuit_rejects_tampered_current_boundary() {
    let vk_fs = seeded_bytes(0x12);
    let header = seeded_digest_fields(0x1a);
    let structure = seeded_digest_fields(0x22);
    let z0 = seeded_bytes(0x32);
    let zi = seeded_bytes(0x42);
    let sa = seeded_bytes(0x52);
    let ca = seeded_bytes(0x62);
    let pt = seeded_bytes(0x72);

    let mut b = R1csBuilder::new();
    let current_boundary = alloc_4(&mut b, digest32_as_fields(zi));
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(7)),
        step_count: b.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary,
        pc: b.alloc(F::from_u64(1)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[current_boundary[0].col()] + F::ONE;
    b.tamper_witness(current_boundary[0].col(), tampered);
    assert!(
        !b.is_satisfied(),
        "tampered current boundary z_i must break state_x_out"
    );
}

#[test]
fn stateful_state_x_out_circuit_rejects_tampered_semantic_acc() {
    let vk_fs = seeded_bytes(0x11);
    let header = seeded_digest_fields(0x19);
    let structure = seeded_digest_fields(0x21);
    let z0 = seeded_bytes(0x31);
    let zi = seeded_bytes(0x41);
    let sa = seeded_bytes(0x51);
    let ca = seeded_bytes(0x61);
    let pt = seeded_bytes(0x71);

    let mut b = R1csBuilder::new();
    let semantic_acc = alloc_4(&mut b, digest32_as_fields(sa));
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(7)),
        step_count: b.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(1)),
        semantic_acc,
        construction2_acc: alloc_4(&mut b, digest32_as_fields(ca)),
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[semantic_acc[0].col()] + F::ONE;
    b.tamper_witness(semantic_acc[0].col(), tampered);
    assert!(
        !b.is_satisfied(),
        "tampered stateful semantic accumulator must break state_x_out"
    );
}

#[test]
fn state_x_out_circuit_rejects_tampered_construction2_acc() {
    let vk_fs = seeded_bytes(0x13);
    let header = seeded_digest_fields(0x1b);
    let structure = seeded_digest_fields(0x23);
    let z0 = seeded_bytes(0x33);
    let zi = seeded_bytes(0x43);
    let sa = seeded_bytes(0x53);
    let ca = seeded_bytes(0x63);
    let pt = seeded_bytes(0x73);

    let mut b = R1csBuilder::new();
    let construction2_acc = alloc_4(&mut b, digest32_as_fields(ca));
    let inputs = StateXOutDigestInputs {
        mode: StateXOutDigestMode::Stateful,
        vk_fs_digest: alloc_4(&mut b, digest32_as_fields(vk_fs)),
        pi_ccs_header_bundle: alloc_4(&mut b, header),
        structure_digest: alloc_4(&mut b, structure),
        chunk_count: b.alloc(F::from_u64(7)),
        step_count: b.alloc(F::from_u64(13)),
        initial_boundary: alloc_4(&mut b, digest32_as_fields(z0)),
        current_boundary: alloc_4(&mut b, digest32_as_fields(zi)),
        pc: b.alloc(F::from_u64(1)),
        semantic_acc: alloc_4(&mut b, digest32_as_fields(sa)),
        construction2_acc,
        public_trace: alloc_4(&mut b, digest32_as_fields(pt)),
    };
    let _ = enforce_state_x_out_digest_circuit(&mut b, &inputs);
    assert!(b.is_satisfied(), "baseline");

    let tampered = b.witness()[construction2_acc[0].col()] + F::ONE;
    b.tamper_witness(construction2_acc[0].col(), tampered);
    assert!(
        !b.is_satisfied(),
        "tampered Construction-2 accumulator U_i handle must break state_x_out"
    );
}
