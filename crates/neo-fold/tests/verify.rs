use neo_fold::{verify_folding_proof_with_spartan, FoldingProof};
use neo_fold::{pi_ccs::PiCcsProof, pi_rlc::PiRlcProof, pi_dec::PiDecProof};
use neo_ccs::{CcsStructure, Mat, McsInstance, MeInstance, SparsePoly, Term};
use neo_ajtai::{Commitment as Cmt, setup as ajtai_setup, set_global_pp};
use neo_math::{F, K, ring::D};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use rand::{rngs::StdRng, SeedableRng};

/// Create a dummy ProofBundle for testing purposes
/// SECURITY NOTE: This is only for testing - real verification requires actual Spartan2 proofs
fn dummy_proof_bundle() -> neo_spartan_bridge::ProofBundle {
    neo_spartan_bridge::ProofBundle::new_with_vk(
        vec![0u8; 32],  // dummy proof
        vec![0u8; 32],  // dummy verifier key
        vec![0u8; 16],  // dummy public IO
    )
}

/// Set up AjtaiSModule global PP for testing
/// This is required for the fail-closed DEC verification
fn setup_ajtai_for_tests() {
    let mut rng = StdRng::seed_from_u64(1); // Deterministic for tests
    let pp = ajtai_setup(&mut rng, D, 8, 2).expect("Ajtai setup should succeed");
    // Ignore error if already initialized from previous test
    let _ = set_global_pp(pp);
}

fn tiny_ccs() -> CcsStructure<F> {
    let m0 = Mat::from_row_major(1, 1, vec![F::ONE]); // n=1, m=1
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]); // f(y)=y0
    CcsStructure::new(vec![m0], f).unwrap()
}

fn derive_rhos(params: &NeoParams, s: &CcsStructure<F>, input_count: usize) -> (Vec<[F; neo_math::D]>, u64) {
    use neo_transcript::{Poseidon2Transcript, Transcript, labels as tr_labels};
    use neo_challenge::sample_kplus1_invertible;

    let mut tr = Poseidon2Transcript::new(b"neo/fold");

    // Π_CCS header absorb (ell=0 with n=1)
    let ell = s.n.trailing_zeros() as u32;
    let d_sc = s.max_degree() as u32;
    let ext = params.extension_check(ell, d_sc).unwrap();
    tr.append_message(tr_labels::PI_CCS, b"");
    tr.append_message(b"neo/ccs/header/v1", b"");
    tr.append_u64s(b"ccs/header", &[64, ext.s_supported as u64, params.lambda as u64, ell as u64, d_sc as u64, ext.slack_bits.unsigned_abs() as u64]);
    tr.append_message(b"neo/ccs/batch", b"");

    // Π_RLC absorb
    tr.append_message(tr_labels::PI_RLC, b"");
    tr.append_message(b"neo/params/v1", b"");
    tr.append_u64s(b"params", &[params.q, params.lambda as u64, input_count as u64, params.s as u64]);

    let (rhos, t_bound) = sample_kplus1_invertible(&mut tr, &neo_challenge::DEFAULT_STRONGSET, input_count).unwrap();
    let rho_arrs: Vec<[F; neo_math::D]> = rhos.iter().map(|rho| rho.coeffs.as_slice().try_into().unwrap()).collect();
    (rho_arrs, t_bound)
}

#[test]
fn verify_shortcircuit_single_instance() {
    setup_ajtai_for_tests(); // Required for fail-closed DEC verification
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let inst = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 0 };

    // Π_CCS outputs: exactly 1 ME(b,L)
    let me_ccs = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: inst.c.clone(),
        X: Mat::from_row_major(1, 1, vec![F::ZERO]),
        r: vec![],               // ell = 0
        y: vec![vec![K::ZERO]],
        y_scalars: vec![K::ZERO], // Test placeholder
        m_in: 0,
        fold_digest: [0u8; 32],  // Dummy digest for test
    };

    // Output instances (DEC digits): also 1 in short-circuit
    let output_digits = vec![me_ccs.clone()];

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32], sc_initial_sum: None },
        pi_ccs_inputs: vec![McsInstance { c: inst.c.clone(), x: inst.x.clone(), m_in: inst.m_in }],
        pi_ccs_outputs: vec![me_ccs],
        pi_rlc_proof: PiRlcProof {
            rho_elems: vec![],
            guard_params: neo_fold::pi_rlc::GuardParams { k: 0, T: 0, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: None, recomposition_proof: vec![], range_proofs: vec![] },
    };

    let dummy_bundle = dummy_proof_bundle();
    let res = verify_folding_proof_with_spartan(&params, &s, &[inst], &output_digits, &proof, &dummy_bundle);
    assert!(res.is_err(), "single-instance bypass was removed for security - should now error");
}

#[test]
fn verify_multi_instance_zero_commitments_dec_present() {
    setup_ajtai_for_tests(); // Required for fail-closed DEC verification
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    // Two inputs (k+1=2)
    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];

    // Π_CCS outputs (two ME(b,L)), all zeros
    let me0 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    // DEC digits: params.k many zero digits
    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1,1,vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            y_scalars: vec![K::ZERO],
            m_in: 1,
            fold_digest: [0u8; 32],
        });
    }

    // Range "proofs" encoding (deterministic format from pi_dec)
    let mut range = Vec::new();
    for _ in 0..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes());
    }

    // ρ from transcript
    let (rho_elems, t_bound) = derive_rhos(&params, &s, insts.len());

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32], sc_initial_sum: None },
        pi_ccs_inputs: insts.iter().map(|inst| McsInstance { c: inst.c.clone(), x: inst.x.clone(), m_in: inst.m_in }).collect(),
        pi_ccs_outputs: vec![me0, me1],
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof {
            digit_commitments: Some(digits.iter().map(|d| d.c.clone()).collect()),
            recomposition_proof: vec![],
            range_proofs: range,
        },
    };

    let dummy_bundle = dummy_proof_bundle();
    let res = verify_folding_proof_with_spartan(&params, &s, &insts, &digits, &proof, &dummy_bundle);
    assert!(res.is_err(), "multi-instance with dummy proof data should error due to malformed outputs");
}

#[test]
fn verify_rejects_when_rho_mismatch() {
    setup_ajtai_for_tests(); // Required for fail-closed DEC verification
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];
    let me0 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    // DEC digits (zeros)
    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1,1,vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            y_scalars: vec![K::ZERO],
            m_in: 1,
            fold_digest: [0u8; 32],
        });
    }

    // Tamper ρ
    let (mut rho_elems, t_bound) = derive_rhos(&params, &s, insts.len());
    rho_elems[0][0] += F::ONE;

    let mut range = Vec::new();
    for _ in 0..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes());
    }

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32], sc_initial_sum: None },
        pi_ccs_inputs: insts.iter().map(|inst| McsInstance { c: inst.c.clone(), x: inst.x.clone(), m_in: inst.m_in }).collect(),
        pi_ccs_outputs: vec![me0, me1],
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof {
            digit_commitments: Some(digits.iter().map(|d| d.c.clone()).collect()),
            recomposition_proof: vec![],
            range_proofs: range,
        },
    };

    let dummy_bundle = dummy_proof_bundle();
    let res = verify_folding_proof_with_spartan(&params, &s, &insts, &digits, &proof, &dummy_bundle);
    assert!(res.is_err(), "must error when ρ are not transcript-derived or outputs malformed");
}

#[test]
fn verify_rejects_when_range_base_mismatches() {
    setup_ajtai_for_tests(); // Required for fail-closed DEC verification
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];
    let me0 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0, c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
            c: Cmt::zeros(neo_math::D, 1),
            X: Mat::from_row_major(1,1,vec![F::ZERO]),
            r: vec![],
            y: vec![vec![K::ZERO]],
            y_scalars: vec![K::ZERO],
            m_in: 1,
            fold_digest: [0u8; 32],
        });
    }

    let (rho_elems, t_bound) = derive_rhos(&params, &s, insts.len());

    // Wrong base in first digit encoding
    let mut range = Vec::new();
    range.extend_from_slice(&((params.b as u32) + 1).to_le_bytes()); // base mismatch
    range.extend_from_slice(&(1u32).to_le_bytes());
    for _ in 1..k {
        range.extend_from_slice(&(params.b as u32).to_le_bytes());
        range.extend_from_slice(&(1u32).to_le_bytes());
    }

    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32], sc_initial_sum: None },
        pi_ccs_inputs: insts.iter().map(|inst| McsInstance { c: inst.c.clone(), x: inst.x.clone(), m_in: inst.m_in }).collect(),
        pi_ccs_outputs: vec![me0, me1],
        pi_rlc_proof: PiRlcProof {
            rho_elems,
            guard_params: neo_fold::pi_rlc::GuardParams { k: 1, T: t_bound, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof {
            digit_commitments: Some(digits.iter().map(|d| d.c.clone()).collect()),
            recomposition_proof: vec![],
            range_proofs: range,
        },
    };

    let dummy_bundle = dummy_proof_bundle();
    let res = verify_folding_proof_with_spartan(&params, &s, &insts, &digits, &proof, &dummy_bundle);
    assert!(res.is_err(), "DEC must report an error when the encoded base differs or outputs malformed");
}

#[test]
fn verify_rejects_spartan_bundle_mismatch() {
    setup_ajtai_for_tests(); // Required for fail-closed DEC verification
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();
    
    let inst = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 0 };
    
    // Π_CCS outputs: exactly 1 ME(b,L)
    let me_ccs = MeInstance {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c: inst.c.clone(),
        X: Mat::from_row_major(1, 1, vec![F::ZERO]),
        r: vec![],               // ell = 0
        y: vec![vec![K::ZERO]],
        y_scalars: vec![K::ZERO], // Test placeholder
        m_in: 0,
        fold_digest: [0u8; 32],  // Dummy digest for test
    };
    
    let output_digits = vec![me_ccs.clone()];
    
    let proof = FoldingProof {
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32], sc_initial_sum: None },
        pi_ccs_inputs: vec![McsInstance { c: inst.c.clone(), x: inst.x.clone(), m_in: inst.m_in }],
        pi_ccs_outputs: vec![me_ccs],
        pi_rlc_proof: PiRlcProof {
            rho_elems: vec![],
            guard_params: neo_fold::pi_rlc::GuardParams { k: 0, T: 0, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: None, recomposition_proof: vec![], range_proofs: vec![] },
    };
    
    // Create bundle with mismatched public IO bytes (one byte different)
    let mut mismatched_public_io = vec![0u8; 16];
    mismatched_public_io[0] = 1; // Make it different from expected all-zeros
    let mismatched_bundle = neo_spartan_bridge::ProofBundle::new_with_vk(
        vec![0u8; 32],  // dummy proof
        vec![0u8; 32],  // dummy verifier key
        mismatched_public_io,  // mismatched public IO
    );
    
    // Should fail due to public IO mismatch (anti-replay protection)
    let res = verify_folding_proof_with_spartan(&params, &s, &[inst], &output_digits, &proof, &mismatched_bundle);
    assert!(res.is_err(), "verification should error when Spartan bundle public IO doesn't match current instance or outputs malformed");
}
