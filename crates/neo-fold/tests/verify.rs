use neo_fold::{verify_folding_proof, FoldingProof};
use neo_fold::{pi_ccs::PiCcsProof, pi_rlc::PiRlcProof, pi_dec::PiDecProof};
use neo_ccs::{CcsStructure, Mat, McsInstance, MeInstance, SparsePoly, Term};
use neo_ajtai::Commitment as Cmt;
use neo_math::{F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn tiny_ccs() -> CcsStructure<F> {
    let m0 = Mat::from_row_major(1, 1, vec![F::ONE]); // n=1, m=1
    let f = SparsePoly::new(1, vec![Term { coeff: F::ONE, exps: vec![1] }]); // f(y)=y0
    CcsStructure::new(vec![m0], f).unwrap()
}

fn derive_rhos(params: &NeoParams, s: &CcsStructure<F>, input_count: usize) -> (Vec<[F; neo_math::D]>, u64) {
    use neo_fold::transcript::{FoldTranscript, Domain};
    use neo_challenge::sample_kplus1_invertible;

    let mut tr = FoldTranscript::default();

    // Π_CCS header absorb (ell=0 with n=1)
    let ell = s.n.trailing_zeros() as u32;
    let d_sc = s.max_degree() as u32;
    let ext = params.extension_check(ell, d_sc).unwrap();
    tr.domain(Domain::CCS);
    tr.absorb_ccs_header(64, ext.s_supported, params.lambda, ell, d_sc, ext.slack_bits);
    tr.absorb_bytes(b"neo/ccs/batch");

    // Π_RLC absorb
    tr.domain(Domain::Rlc);
    tr.absorb_bytes(b"neo/params/v1");
    tr.absorb_u64(&[params.q, params.lambda as u64, input_count as u64, params.s as u64]);

    let mut ch = tr.challenger();
    let (rhos, t_bound) = sample_kplus1_invertible(&mut ch, &neo_challenge::DEFAULT_STRONGSET, input_count).unwrap();
    let rho_arrs: Vec<[F; neo_math::D]> = rhos.iter().map(|rho| rho.coeffs.as_slice().try_into().unwrap()).collect();
    (rho_arrs, t_bound)
}

#[test]
fn verify_shortcircuit_single_instance() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let inst = McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 0 };

    // Π_CCS outputs: exactly 1 ME(b,L)
    let me_ccs = MeInstance {
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
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
        pi_ccs_outputs: vec![me_ccs],
        pi_rlc_proof: PiRlcProof {
            rho_elems: vec![],
            guard_params: neo_fold::pi_rlc::GuardParams { k: 0, T: 0, b: params.b as u64, B: params.B as u64 },
        },
        pi_dec_proof: PiDecProof { digit_commitments: None, recomposition_proof: vec![], range_proofs: vec![] },
    };

    let ok = verify_folding_proof(&params, &s, &[inst], &output_digits, &proof).unwrap();
    assert!(!ok, "single-instance bypass was removed for security - should now fail");
}

#[test]
fn verify_multi_instance_zero_commitments_dec_present() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    // Two inputs (k+1=2)
    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];

    // Π_CCS outputs (two ME(b,L)), all zeros
    let me0 = MeInstance { c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance { c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    // DEC digits: params.k many zero digits
    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
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
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
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

    let ok = verify_folding_proof(&params, &s, &insts, &digits, &proof).unwrap();
    assert!(!ok, "multi-instance with dummy proof data should fail verification");
}

#[test]
fn verify_rejects_when_rho_mismatch() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];
    let me0 = MeInstance { c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance { c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    // DEC digits (zeros)
    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
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
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
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

    let ok = verify_folding_proof(&params, &s, &insts, &digits, &proof).unwrap();
    assert!(!ok, "must reject when ρ are not transcript-derived");
}

#[test]
fn verify_rejects_when_range_base_mismatches() {
    let params = NeoParams::goldilocks_127();
    let s = tiny_ccs();

    let insts = vec![
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![], m_in: 1 },
        McsInstance { c: Cmt::zeros(neo_math::D, 1), x: vec![F::ZERO], m_in: 1 },
    ];
    let me0 = MeInstance { c: insts[0].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };
    let me1 = MeInstance { c: insts[1].c.clone(), X: Mat::from_row_major(1,1,vec![F::ZERO]), r: vec![], y: vec![vec![K::ZERO]], y_scalars: vec![K::ZERO], m_in: 1, fold_digest: [0u8; 32] };

    let k = params.k as usize;
    let mut digits = Vec::with_capacity(k);
    for _ in 0..k {
        digits.push(MeInstance {
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
        pi_ccs_proof: PiCcsProof { sumcheck_rounds: vec![], header_digest: [0u8; 32] },
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

    let ok = verify_folding_proof(&params, &s, &insts, &digits, &proof).unwrap();
    assert!(!ok, "DEC must reject when the encoded base differs");
}
