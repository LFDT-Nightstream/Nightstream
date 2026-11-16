#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit_superneo_cached_from_trusted_split_digits,
    dec_children_with_commit_superneo_cached_with_digit_flags, rlc_public, rlc_with_commit, rlc_with_commit_refs,
    split_b_matrix_k_with_nonzero_flags, DecOutcome, FoldingMode, RotRho, RotRing, SplitDigits,
};
use neo_reductions::common::{compute_v1_2_evaluations_from_z_and_r, project_x_from_witness_mat};
use neo_reductions::superneo_eval::{build_superneo_eval_cache, SuperneoEvalCache};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

fn modes() -> [FoldingMode; 3] {
    [
        FoldingMode::Optimized,
        FoldingMode::PaperExact,
        FoldingMode::OptimizedWithCrosscheck,
    ]
}

fn ell_d() -> usize {
    D.next_power_of_two().trailing_zeros() as usize
}

fn structure() -> CcsStructure<F> {
    let m0 = Mat::identity(D);
    let mut m1 = Mat::zero(D, D, F::ZERO);
    for r in 0..D {
        m1[(r, (r + 1) % D)] = F::ONE;
    }
    let f = SparsePoly::new(
        2,
        vec![
            Term {
                coeff: F::ONE,
                exps: vec![1, 0],
            },
            Term {
                coeff: F::ONE,
                exps: vec![0, 1],
            },
        ],
    );
    CcsStructure::new(vec![m0, m1], f).expect("valid CCS structure")
}

fn commitment(seed: u64) -> Commitment {
    let mut c = Commitment::zeros(1);
    c.data_mut()[0] = F::from_u64(seed);
    c
}

fn mix_commitments(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(commits[0].kappa());
    for (rho, c) in rhos.iter().zip(commits) {
        for lane in 0..c.kappa() {
            for row in 0..D {
                for col in 0..D {
                    acc.data_mut()[lane * D + row] += rho[(row, col)] * c.data()[lane * D + col];
                }
            }
        }
    }
    acc
}

fn combine_b_pows(commits: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(commits[0].kappa());
    let mut pow = F::ONE;
    for c in commits {
        let mut term = c.clone();
        for v in term.data_mut().iter_mut() {
            *v *= pow;
        }
        acc.add_inplace(&term);
        pow *= F::from_u64(u64::from(b));
    }
    acc
}

fn claim_from_z(s: &CcsStructure<F>, Z: &Mat<F>, r: &[K], c: Commitment) -> CeClaim<Commitment, F, K> {
    let evaluations = compute_v1_2_evaluations_from_z_and_r(s, Z, r, ell_d());
    CeClaim {
        adv: None,
        c,
        X: project_x_from_witness_mat(Z, s.m(), D).expect("project X"),
        r: r.to_vec(),
        eval_k: evaluations.eval_k,
        eval_a: evaluations.eval_a,
        m_in: D,
        fold_digest: [0u8; 32],
    }
}

fn witness(seed: u64) -> Mat<F> {
    let data = (0..D)
        .map(|row| F::from_u64(seed + row as u64 * 11 + 1))
        .collect();
    Mat::from_row_major(D, 1, data)
}

struct RlcFixture {
    params: NeoParams,
    s: CcsStructure<F>,
    rhos: Vec<RotRho>,
    inputs: Vec<CeClaim<Commitment, F, K>>,
    witnesses: Vec<Mat<F>>,
}

fn rlc_fixture() -> RlcFixture {
    let params = NeoParams::nightstream_p64_k14();
    let s = structure();
    let r = vec![K::from(F::from_u64(5)); 6];
    let witnesses: Vec<Mat<F>> = (0..3).map(|i| witness(400 + i * 90)).collect();
    let inputs = witnesses
        .iter()
        .enumerate()
        .map(|(i, Z)| claim_from_z(&s, Z, &r, commitment(8_000 + i as u64)))
        .collect();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&[F::from_u64(42)]);
    let rhos = neo_reductions::api::sample_rot_rhos_n_typed(&mut transcript, &params, &RotRing::profile(), 3)
        .expect("sample rhos");
    RlcFixture {
        params,
        s,
        rhos,
        inputs,
        witnesses,
    }
}

struct DecFixture {
    params: NeoParams,
    s: CcsStructure<F>,
    parent: CeClaim<Commitment, F, K>,
    split: SplitDigits,
    child_commitments: Vec<Commitment>,
    cache: SuperneoEvalCache,
}

fn dec_fixture() -> DecFixture {
    let params = NeoParams::nightstream_p64_k14();
    let s = structure();
    let r = vec![K::from(F::from_u64(17)); 6];
    let data = (0..D).map(|row| F::from_u64((row % 7) as u64)).collect();
    let z_parent = Mat::from_row_major(D, 1, data);
    let k_rho = params.k_rho() as usize;
    let split = split_b_matrix_k_with_nonzero_flags(&z_parent, k_rho, params.b()).expect("split parent");
    let child_commitments: Vec<Commitment> = (0..k_rho).map(|i| commitment(72_000 + i as u64)).collect();
    let mut parent = claim_from_z(&s, &z_parent, &r, commitment(0));
    parent.c = combine_b_pows(&child_commitments, params.b());
    let cache = build_superneo_eval_cache(&s).expect("SuperNeo cache");
    DecFixture {
        params,
        s,
        parent,
        split,
        child_commitments,
        cache,
    }
}

fn accepted(outcome: &DecOutcome) -> bool {
    outcome.ok_y && outcome.ok_x && outcome.ok_c
}

#[test]
fn rlc_with_commit_refs_returns_the_same_parent_in_every_mode() {
    let fx = rlc_fixture();
    let refs: Vec<&Mat<F>> = fx.witnesses.iter().collect();
    let expected = rlc_with_commit(
        FoldingMode::Optimized,
        &fx.s,
        &fx.params,
        &fx.rhos,
        &fx.inputs,
        &fx.witnesses,
        ell_d(),
        mix_commitments,
    )
    .expect("optimized rlc_with_commit");
    let public = rlc_public(&fx.s, &fx.params, &fx.rhos, &fx.inputs, mix_commitments, ell_d()).expect("rlc_public");
    assert_eq!(expected.0, public, "the folded parent is the public RLC");

    for mode in modes() {
        let label = format!("{mode:?}");
        let got = rlc_with_commit_refs(
            mode,
            &fx.s,
            &fx.params,
            &fx.rhos,
            &fx.inputs,
            &refs,
            ell_d(),
            mix_commitments,
        )
        .unwrap_or_else(|e| panic!("{label} rlc_with_commit_refs: {e}"));
        assert_eq!(got, expected, "{label} rlc_with_commit_refs");
    }
}

#[test]
fn dec_with_digit_flags_returns_the_same_children_in_every_mode() {
    let fx = dec_fixture();
    let run = |mode| {
        dec_children_with_commit_superneo_cached_with_digit_flags(
            mode,
            &fx.s,
            &fx.params,
            &fx.parent,
            fx.split.planes(),
            fx.split.nonzero(),
            ell_d(),
            &fx.child_commitments,
            combine_b_pows,
            &fx.cache,
        )
    };
    let expected = run(FoldingMode::Optimized);
    assert!(accepted(&expected), "canonical split must verify");
    assert_eq!(expected.children.len(), fx.child_commitments.len());
    for mode in modes() {
        let label = format!("{mode:?}");
        assert_eq!(run(mode), expected, "{label} digit-flag DEC");
    }
}

#[test]
fn dec_from_trusted_split_returns_the_same_children_in_every_mode() {
    let fx = dec_fixture();
    let run = |mode| {
        dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            mode,
            &fx.s,
            &fx.params,
            &fx.parent,
            &fx.split,
            ell_d(),
            &fx.child_commitments,
            combine_b_pows,
            Some(&fx.cache),
            None,
        )
    };
    let expected = run(FoldingMode::Optimized);
    assert!(accepted(&expected), "trusted split must verify");
    for mode in modes() {
        let label = format!("{mode:?}");
        assert_eq!(run(mode), expected, "{label} trusted-split DEC");
    }
}

#[test]
#[should_panic(expected = "PiDEC optimized and PaperExact results differ")]
fn crosscheck_panics_when_a_wrong_digit_flag_splits_the_engines() {
    let fx = dec_fixture();
    let mut flags = fx.split.nonzero().to_vec();
    let last = flags.len() - 1;
    assert!(!flags[last], "the top digit plane of a small witness is zero");
    flags[last] = true;
    let run = |mode| {
        dec_children_with_commit_superneo_cached_with_digit_flags(
            mode,
            &fx.s,
            &fx.params,
            &fx.parent,
            fx.split.planes(),
            &flags,
            ell_d(),
            &fx.child_commitments,
            combine_b_pows,
            &fx.cache,
        )
    };
    assert!(
        !run(FoldingMode::Optimized).ok_c,
        "optimized rejects a wrong digit flag"
    );
    assert!(
        accepted(&run(FoldingMode::PaperExact)),
        "PaperExact ignores the digit flags"
    );
    run(FoldingMode::OptimizedWithCrosscheck);
}
