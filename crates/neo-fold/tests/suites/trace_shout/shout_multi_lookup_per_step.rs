#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::relations::CeClaim;
use neo_ccs::Mat;
use neo_fold::pi_ccs::FoldingMode;
use neo_fold::shard::{fold_shard_prove, fold_shard_verify};
use neo_math::{F, K};
use neo_memory::plain::{LutTable, PlainLutTrace};
use neo_memory::witness::StepInstanceBundle;
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

use super::multi_table_shout_tests::{
    create_identity_ccs, create_step_with_shout_bus_lanes, default_mixers, setup_ajtai_pp,
};

#[test]
fn shout_multi_lookup_two_lookups_per_step_prove_verify() {
    let ccs = create_identity_ccs(32);
    let mut params = NeoParams::goldilocks_auto_r1cs_ccs(ccs.m).expect("params");
    params.k_rho = 16;

    let l = setup_ajtai_pp(ccs.m, 0x2001);
    let mixers = default_mixers();

    let table = LutTable {
        table_id: 0,
        k: 2,
        d: 1,
        n_side: 2,
        content: vec![F::from_u64(5), F::from_u64(7)],
    };

    let lane0 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![0],
        val: vec![F::from_u64(5)],
    };
    let lane1 = PlainLutTrace {
        has_lookup: vec![F::ONE],
        addr: vec![1],
        val: vec![F::from_u64(7)],
    };
    let step_bundle = create_step_with_shout_bus_lanes(&params, &ccs, &l, 99, &table, &[lane0, lane1]);

    let acc_init: Vec<CeClaim<Cmt, F, K>> = Vec::new();
    let acc_wit_init: Vec<Mat<F>> = Vec::new();

    let mut tr_prove = Poseidon2Transcript::new(b"multi-lookup-same-table");
    let proof = fold_shard_prove(
        FoldingMode::Optimized,
        &mut tr_prove,
        &params,
        &ccs,
        &[step_bundle.clone()],
        &acc_init,
        &acc_wit_init,
        &l,
        mixers,
    )
    .expect("prove should succeed");

    let mut tr_verify = Poseidon2Transcript::new(b"multi-lookup-same-table");
    let steps_public = [StepInstanceBundle::from(&step_bundle)];
    let _ = fold_shard_verify(
        FoldingMode::Optimized,
        &mut tr_verify,
        &params,
        &ccs,
        &steps_public,
        &acc_init,
        &proof,
        mixers,
    )
    .expect("verify should succeed");
}
