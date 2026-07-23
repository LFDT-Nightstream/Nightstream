use neo_ajtai::Commitment;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;

fn column_point_len(structure: &CcsStructure<F>) -> usize {
    structure.m.next_power_of_two().max(2).trailing_zeros() as usize
}

fn f_signed(value: i64) -> F {
    if value >= 0 {
        F::from_u64(value as u64)
    } else {
        F::ZERO - F::from_u64(value.unsigned_abs())
    }
}

fn params_with_base_and_digits(base: u32, digits: u32) -> NeoParams {
    let paper = NeoParams::goldilocks_paper_b2();
    NeoParams::new(
        paper.q,
        paper.eta,
        paper.d,
        paper.kappa,
        paper.m,
        base,
        digits,
        1,
        paper.s,
        paper.lambda,
    )
    .expect("valid focused DEC parameters")
}

fn structure() -> CcsStructure<F> {
    CcsStructure::new(vec![Mat::identity(D)], neo_ccs::poly::SparsePoly::new(1, Vec::new()))
        .expect("valid focused CCS structure")
}

fn claim(params: &NeoParams, x: F) -> CeClaim<Commitment, F, K> {
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let mut public_x = Mat::zero(D, 1, F::ZERO);
    public_x[(0, 0)] = x;
    CeClaim {
        adv: None,
        c: Commitment::zeros(params.d as usize, 1),
        X: public_x,
        r: Vec::new(),
        s_col: Vec::new(),
        y_ring: vec![vec![K::ZERO; 1usize << ell_d]],
        ct: vec![K::ZERO],
        aux_openings: Vec::new(),
        y_zcol: Vec::new(),
        m_in: 1,
        fold_digest: [0; 32],
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
    }
}

fn children_for_digits(params: &NeoParams, digits: &[i64]) -> Vec<CeClaim<Commitment, F, K>> {
    digits
        .iter()
        .map(|&digit| claim(params, f_signed(digit)))
        .collect()
}

fn combine_zero_commitments(commits: &[Commitment], _base: u32) -> Commitment {
    Commitment::zeros(commits[0].d, commits[0].kappa)
}

fn verify(params: &NeoParams, parent: &CeClaim<Commitment, F, K>, children: &[CeClaim<Commitment, F, K>]) -> bool {
    let structure = structure();
    neo_reductions::api::verify_dec_public(
        &structure,
        params,
        column_point_len(&structure),
        parent,
        children,
        combine_zero_commitments,
        D.next_power_of_two().trailing_zeros() as usize,
    )
}

#[test]
fn verifier_rejects_recomposing_signed_alias_and_accepts_honest_split() {
    let params = NeoParams::goldilocks_paper_b2();
    let parent = claim(&params, F::ONE);

    let mut honest_digits = vec![0; params.k_rho as usize];
    honest_digits[0] = 1;
    assert!(verify(&params, &parent, &children_for_digits(&params, &honest_digits)));

    // Both [1, 0, ...] and [-1, 1, 0, ...] recompose to one in base two,
    // but only the former is verifier-computed split_b(parent.X).
    let mut alias_digits = vec![0; params.k_rho as usize];
    alias_digits[0] = -1;
    alias_digits[1] = 1;
    assert!(!verify(&params, &parent, &children_for_digits(&params, &alias_digits)));
}

#[test]
fn verifier_rejects_parent_outside_fixed_split_range() {
    let params = NeoParams::goldilocks_paper_b2();
    let parent = claim(&params, F::from_u64(params.B));
    let children = children_for_digits(&params, &vec![0; params.k_rho as usize]);

    assert!(!verify(&params, &parent, &children));
}

#[test]
fn verifier_rejects_wrong_nonzero_child_count() {
    let params = NeoParams::goldilocks_paper_b2();
    let parent = claim(&params, F::ONE);
    let children = children_for_digits(&params, &[1]);

    assert!(!verify(&params, &parent, &children));
}

#[test]
fn verifier_accepts_canonical_splits_for_base_greater_than_two() {
    let params = params_with_base_and_digits(3, 4);

    for (parent_value, digits) in [(17, [2, 2, 1, 0]), (-17, [-2, -2, -1, 0])] {
        let parent = claim(&params, f_signed(parent_value));
        assert!(verify(&params, &parent, &children_for_digits(&params, &digits)));
    }
}
