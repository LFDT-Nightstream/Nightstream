use neo_ajtai::{s_mul_add_from_rot_col, scale_commitment_add_inplace, Commitment};
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, dec_children_with_commit_superneo_cached_from_trusted_split_digits, rlc_public,
    rlc_public_matches, rlc_with_commit, split_b_matrix_k_with_nonzero_flags, verify_dec_public, FoldingMode, RotRho,
};
use p3_field::PrimeCharacteristicRing;

fn zero_claim(params: &NeoParams) -> CeClaim<Commitment, F, K> {
    CeClaim {
        c: Commitment::zeros(D, params.kappa as usize),
        X: Mat::zero(D, 1, F::ZERO),
        r: vec![K::ZERO; D.next_power_of_two().trailing_zeros() as usize],
        eval_k: vec![K::ZERO; D.next_power_of_two()],
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]],
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    }
}

fn rlc_mix(rhos: &[Mat<F>], commitments: &[Commitment]) -> Commitment {
    let mut result = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    for (rho, commitment) in rhos.iter().zip(commitments) {
        let first_column = std::array::from_fn(|row| rho[(row, 0)]);
        s_mul_add_from_rot_col(&mut result, &first_column, commitment);
    }
    result
}

fn dec_mix(commitments: &[Commitment], base: u32) -> Commitment {
    let mut result = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    let mut power = F::ONE;
    for commitment in commitments {
        scale_commitment_add_inplace(&mut result, power, commitment);
        power *= F::from_u64(base as u64);
    }
    result
}

#[test]
fn rlc_rejects_mixed_commitment_widths() {
    let params = NeoParams::nightstream_goldilocks_k16();
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let inputs = vec![zero_claim(&params); 2];
    let witnesses = vec![Mat::zero(D, 1, F::ZERO); inputs.len()];
    let rhos = vec![RotRho::new_checked(&params, Mat::identity(D)).unwrap(); inputs.len()];
    let parent = rlc_public(&structure, &params, &rhos, &inputs, rlc_mix, ell_d).unwrap();
    let (prover_parent, _) = rlc_with_commit(
        FoldingMode::Optimized,
        &structure,
        &params,
        &rhos,
        &inputs,
        &witnesses,
        ell_d,
        rlc_mix,
    )
    .unwrap();
    assert_eq!(prover_parent, parent, "honest RLC prover and public control");
    assert!(rlc_public_matches(&structure, &params, &rhos, &inputs, &parent, rlc_mix, ell_d).unwrap());

    for width in [params.kappa as usize - 1, params.kappa as usize + 1] {
        let mut mixed = inputs.clone();
        mixed[1].c = Commitment::zeros(D, width);
        if width > params.kappa as usize {
            mixed[1].c.data[D * params.kappa as usize] = F::ONE;
        }
        assert!(
            rlc_public(&structure, &params, &rhos, &mixed, rlc_mix, ell_d).is_err(),
            "public RLC accepted mixed commitment widths"
        );
        assert!(
            !rlc_public_matches(&structure, &params, &rhos, &mixed, &parent, rlc_mix, ell_d).unwrap(),
            "RLC match verification accepted mixed commitment widths"
        );
        assert!(
            rlc_with_commit(
                FoldingMode::Optimized,
                &structure,
                &params,
                &rhos,
                &mixed,
                &witnesses,
                ell_d,
                rlc_mix,
            )
            .is_err(),
            "RLC prover accepted mixed commitment widths"
        );
    }
}

#[test]
fn dec_verifier_rejects_mixed_commitment_widths() {
    let params = NeoParams::nightstream_goldilocks_k16();
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let parent = zero_claim(&params);
    let children = vec![parent.clone(); params.k_rho as usize];
    assert!(verify_dec_public(
        &structure, &params, &parent, &children, dec_mix, ell_d
    ));

    for width in [params.kappa as usize - 1, params.kappa as usize + 1] {
        let mut mixed = children.clone();
        let later = mixed.last_mut().unwrap();
        later.c = Commitment::zeros(D, width);
        if width > params.kappa as usize {
            later.c.data[D * params.kappa as usize] = F::ONE;
        }
        assert!(
            !verify_dec_public(&structure, &params, &parent, &mixed, dec_mix, ell_d),
            "DEC verifier accepted a child with a different commitment width"
        );
    }
}

#[test]
fn dec_constructors_reject_malformed_child_commitments() {
    let params = NeoParams::nightstream_goldilocks_k16();
    let structure = CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let parent = zero_claim(&params);
    let (digits, flags) =
        split_b_matrix_k_with_nonzero_flags(&Mat::zero(D, 1, F::ZERO), params.k_rho as usize, params.b).unwrap();
    let commitments = vec![parent.c.clone(); digits.len()];
    let construct = |commitments: &[Commitment]| {
        [
            (
                "checked",
                dec_children_with_commit(
                    FoldingMode::Optimized,
                    &structure,
                    &params,
                    &parent,
                    &digits,
                    ell_d,
                    commitments,
                    dec_mix,
                ),
            ),
            (
                "trusted split",
                dec_children_with_commit_superneo_cached_from_trusted_split_digits(
                    FoldingMode::Optimized,
                    &structure,
                    &params,
                    &parent,
                    &digits,
                    &flags,
                    ell_d,
                    commitments,
                    dec_mix,
                    None,
                    None,
                    None,
                ),
            ),
        ]
    };
    for (path, (children, y, x, c)) in construct(&commitments) {
        assert!(y && x && c, "{path}: honest DEC constructor");
        assert!(verify_dec_public(
            &structure, &params, &parent, &children, dec_mix, ell_d
        ));
    }

    let mut truncated = parent.c.clone();
    truncated.data.clear();
    let mut wider = Commitment::zeros(D, params.kappa as usize + 1);
    wider.data[D * params.kappa as usize] = F::ONE;
    for bad_commitment in [truncated, wider] {
        let mut malformed = commitments.clone();
        *malformed.last_mut().unwrap() = bad_commitment;
        for (path, (children, y, x, c)) in construct(&malformed) {
            assert!(
                children.is_empty() && !y && !x && !c,
                "{path}: DEC constructor accepted a malformed child commitment"
            );
        }
    }
}
