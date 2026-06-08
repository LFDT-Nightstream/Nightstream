#![allow(non_snake_case)]

use std::sync::Arc;

use neo_ajtai::{setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsClaim, CcsStructure, CcsWitness, Mat, SparsePoly};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_reductions::api::{prove, FoldingMode};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use rand_chacha::rand_core::SeedableRng;
use rand_chacha::ChaCha8Rng;

fn identity_ccs(n: usize) -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(n)], SparsePoly::new(1, vec![])).expect("valid CCS")
}

fn f_from_i64(v: i64) -> F {
    if v >= 0 {
        F::from_u64(v as u64)
    } else {
        F::ZERO - F::from_u64((-v) as u64)
    }
}

#[test]
fn prove_rejects_out_of_range_packed_witness_early() {
    let n = D; // SuperNeo-packed compatible width
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let mut rng = ChaCha8Rng::seed_from_u64(77);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, ccs.m / D).expect("Ajtai setup");
    let l = AjtaiSModule::new(Arc::new(pp));

    // Packed layout is D x (m/D); fill with values outside the SuperNeo
    // input NC alphabet |x| < b.
    let mut Z = Mat::zero(D, ccs.m / D, F::ZERO);
    for rho in 0..D {
        Z[(rho, 0)] = F::from_u64((1u64 << 60) + rho as u64);
    }
    let w: Vec<F> = (0..ccs.m).map(|c| Z[(c % D, c / D)]).collect();

    let c = l.commit(&Z);
    let mcs_list = vec![CcsClaim { c, x: vec![], m_in: 0 }];
    let mcs_witnesses = vec![CcsWitness { w, Z }];

    let mut tr = Poseidon2Transcript::new(b"neo.reductions/packed_range_guard");
    let err = prove(
        FoldingMode::Optimized,
        &mut tr,
        &params,
        &ccs,
        &mcs_list,
        &mcs_witnesses,
        &[],
        &[],
        &l,
    )
    .expect_err("prove must reject packed witnesses that violate NC range");

    assert!(
        err.to_string().contains("violates NC alphabet"),
        "unexpected error: {err}"
    );
}

#[test]
fn prove_accepts_nc_alphabet_packed_witness_values() {
    let n = D; // SuperNeo-packed compatible width
    let ccs = identity_ccs(n);
    let params = NeoParams::goldilocks_auto_r1cs_ccs(n).expect("params");

    let mut rng = ChaCha8Rng::seed_from_u64(99);
    let pp = ajtai_setup(&mut rng, D, params.kappa as usize, ccs.m / D).expect("Ajtai setup");
    let l = AjtaiSModule::new(Arc::new(pp));

    // Keep witness values inside the NC polynomial support range |x| < b for Π_CCS proving.
    let mut Z = Mat::zero(D, ccs.m / D, F::ZERO);
    for rho in 0..D {
        let centered = (rho % 3) as i64 - 1; // cycles through -1,0,1
        Z[(rho, 0)] = f_from_i64(centered);
    }
    let w: Vec<F> = (0..ccs.m).map(|c| Z[(c % D, c / D)]).collect();

    let c = l.commit(&Z);
    let mcs_list = vec![CcsClaim { c, x: vec![], m_in: 0 }];
    let mcs_witnesses = vec![CcsWitness { w, Z }];

    let mut tr_p = Poseidon2Transcript::new(b"neo.reductions/packed_range_accept");
    let (out, proof) = prove(
        FoldingMode::Optimized,
        &mut tr_p,
        &params,
        &ccs,
        &mcs_list,
        &mcs_witnesses,
        &[],
        &[],
        &l,
    )
    .expect("prove should accept NC-range packed witness values");

    let mut tr_v = Poseidon2Transcript::new(b"neo.reductions/packed_range_accept");
    let ok = neo_reductions::api::verify(
        FoldingMode::Optimized,
        &mut tr_v,
        &params,
        &ccs,
        &mcs_list,
        &[],
        &out,
        &proof,
    )
    .expect("verify should run");
    assert!(ok, "verify should pass for NC-range packed witness values");
}
