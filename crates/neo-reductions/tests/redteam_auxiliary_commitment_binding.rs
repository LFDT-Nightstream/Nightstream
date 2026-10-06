use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, LaneCommitments, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{rlc_public, rlc_public_matches, verify_dec_public, RotRho};
use p3_field::PrimeCharacteristicRing;

fn commitment(value: u64, kappa: usize) -> Commitment {
    let mut commitment = Commitment::zeros(D, kappa);
    commitment.data[0] = F::from_u64(value);
    commitment
}

fn auxiliary(value: u64, kappa: usize) -> LaneCommitments<Commitment> {
    LaneCommitments {
        ops: commitment(value, kappa),
        is: commitment(value + 1, kappa),
        fs: commitment(value + 2, kappa),
    }
}

fn claim(
    structure: &CcsStructure<F>,
    params: &NeoParams,
    adv: Option<LaneCommitments<Commitment>>,
) -> CeClaim<Commitment, F, K> {
    let point_len = structure
        .domain_rows()
        .max(neo_reductions::common::superneo_carrier_width(structure.m))
        .next_power_of_two()
        .trailing_zeros() as usize;
    CeClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; point_len],
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; structure.t()],
        m_in: D,
        fold_digest: [7; 32],
        adv,
    }
}

#[test]
fn rlc_and_dec_reject_auxiliary_lane_commitments() {
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).unwrap();
    let params = NeoParams::nightstream_goldilocks_k16();
    let kappa = params.kappa as usize;
    let rho = RotRho::new_checked(&params, Mat::identity(D)).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let mix = |_: &[Mat<F>], commitments: &[Commitment]| commitments[0].clone();

    let input = claim(&structure, &params, Some(auxiliary(1, kappa)));
    assert!(
        rlc_public(
            &structure,
            &params,
            std::slice::from_ref(&rho),
            std::slice::from_ref(&input),
            mix,
            ell_d
        )
        .is_err(),
        "PiRLC accepted an input with auxiliary lane commitments"
    );

    let plain = claim(&structure, &params, None);
    let parent = rlc_public(
        &structure,
        &params,
        std::slice::from_ref(&rho),
        std::slice::from_ref(&plain),
        mix,
        ell_d,
    )
    .unwrap();
    let matches = |claimed: &CeClaim<Commitment, F, K>| {
        rlc_public_matches(
            &structure,
            &params,
            std::slice::from_ref(&rho),
            std::slice::from_ref(&plain),
            claimed,
            mix,
            ell_d,
        )
        .unwrap()
    };
    assert!(matches(&parent), "honest PiRLC parent");
    let mut false_parent = parent;
    false_parent.adv = Some(auxiliary(9, kappa));
    assert!(
        !matches(&false_parent),
        "PiRLC accepted a parent with auxiliary lane commitments"
    );

    let children = vec![claim(&structure, &params, None); params.k_rho as usize];
    let zero_mix = |_: &[Commitment], _: u32| Commitment::zeros(D, kappa);
    let dec =
        |parent: &CeClaim<Commitment, F, K>| verify_dec_public(&structure, &params, parent, &children, zero_mix, ell_d);
    assert!(dec(&claim(&structure, &params, None)), "honest PiDEC recomposition");
    assert!(
        !dec(&claim(&structure, &params, Some(auxiliary(17, kappa)))),
        "PiDEC accepted a parent with auxiliary lane commitments"
    );
}
