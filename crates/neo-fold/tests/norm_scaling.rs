use neo_ccs::{mv_poly, CcsStructure};
use neo_commit::{AjtaiCommitter, TOY_PARAMS};
use neo_fields::ExtF;
use neo_fold::{pi_rlc, EvalInstance, FoldState};
use neo_ring::RingElement;
use p3_field::PrimeCharacteristicRing;

fn dummy_structure() -> CcsStructure {
    let mats = vec![];
    let f = mv_poly(|_| ExtF::ZERO, 0);
    CcsStructure::new(mats, f)
}

#[test]
fn test_norm_scaling() {
    let structure = dummy_structure();
    let mut state = FoldState::new(structure);
    state.max_blind_norm = 0;
    let params = TOY_PARAMS;
    let committer = AjtaiCommitter::setup_unchecked(params);
    let zero_commit = vec![RingElement::zero(params.n); params.k];
    let eval1 = EvalInstance {
        commitment: zero_commit.clone(),
        r: vec![ExtF::ZERO],
        ys: vec![ExtF::ZERO],
        u: ExtF::ZERO,
        e_eval: ExtF::ZERO,
        norm_bound: 2,
        opening_proof: None,
    };
    let eval2 = EvalInstance {
        commitment: zero_commit,
        r: vec![ExtF::ZERO],
        ys: vec![ExtF::ZERO],
        u: ExtF::ZERO,
        e_eval: ExtF::ZERO,
        norm_bound: 3,
        opening_proof: None,
    };
    state.eval_instances = vec![
        eval1.clone(),  // index 0: norm_bound=2
        EvalInstance {
            commitment: vec![],
            r: vec![],
            ys: vec![],
            u: ExtF::ZERO,
            e_eval: ExtF::ZERO,
            norm_bound: 0,
            opening_proof: None,
        },  // index 1: dummy norm=0
        eval2.clone(),  // index 2: norm_bound=3
        EvalInstance {
            commitment: vec![],
            r: vec![],
            ys: vec![],
            u: ExtF::ZERO,
            e_eval: ExtF::ZERO,
            norm_bound: 0,
            opening_proof: None,
        },  // index 3: dummy norm=0
    ];
    let rho_rot = RingElement::zero(params.n);
    let mut transcript = vec![];
    pi_rlc(&mut state, rho_rot.clone(), &committer, &mut transcript);
    let new_eval = state.eval_instances.last().unwrap();
    // Scaling for commitments uses the rotation's ring ∞-norm; the hashed
    // extension-field scalar is only for evaluation-level combination.
    let rho_norm = rho_rot.norm_inf() as f64;
    let expected = ((2f64).powi(2) + (rho_norm.powi(2) * (3f64).powi(2)))
        .sqrt()
        .ceil() as u64;
    assert_eq!(new_eval.norm_bound, expected);
}
