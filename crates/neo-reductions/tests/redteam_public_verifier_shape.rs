#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, rlc_public, rlc_public_matches_with_perf, rlc_with_commit, verify_dec_public, FoldingMode,
};
use neo_reductions::common::{compute_y_from_Z_and_r, sample_rot_rhos_n_typed, RotRing};
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
}

fn build_structure(n: usize, m: usize) -> CcsStructure<F> {
    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for row in 0..n {
        m1[(row, (row + 1) % m)] = F::ONE;
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

fn make_z(seed: u64, m: usize) -> Mat<F> {
    assert!(m.is_multiple_of(D));
    let mut data = Vec::with_capacity(m);
    for row in 0..D {
        for block in 0..(m / D) {
            let column = block * D + row;
            data.push(F::from_u64(seed + (row as u64) * 11 + (column as u64) * 17 + 1));
        }
    }
    Mat::from_row_major(D, m / D, data)
}

fn make_commitment(params: &NeoParams, seed: u64) -> Commitment {
    let mut commitment = Commitment::zeros(params.d as usize, 1);
    commitment.data[0] = F::from_u64(seed);
    commitment
}

fn scale_commitment(commitment: &Commitment, scale: F) -> Commitment {
    let mut out = commitment.clone();
    for value in &mut out.data {
        *value *= scale;
    }
    out
}

fn mix_commitments_from_rhos(rhos: &[Mat<F>], commitments: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    for (rho, commitment) in rhos.iter().zip(commitments) {
        acc.add_inplace(&scale_commitment(commitment, rho[(0, 0)]));
    }
    acc
}

fn combine_commitments_b_pows(commitments: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(commitments[0].d, commitments[0].kappa);
    let mut power = F::ONE;
    let b_f = F::from_u64(b as u64);
    for commitment in commitments {
        acc.add_inplace(&scale_commitment(commitment, power));
        power *= b_f;
    }
    acc
}

fn diag_rho(scale: u64) -> Mat<F> {
    let mut rho = Mat::zero(D, D, F::ZERO);
    for index in 0..D {
        rho[(index, index)] = F::from_u64(scale);
    }
    rho
}

fn typed_rhos(params: &NeoParams, rhos: &[Mat<F>]) -> Vec<neo_reductions::api::RotRho> {
    neo_reductions::api::rot_rhos_from_mats(params, rhos, "redteam public-verifier rhos").expect("typed rhos")
}

fn build_me_from_z(
    params: &NeoParams,
    structure: &CcsStructure<F>,
    z: &Mat<F>,
    r: &[K],
    ell_d: usize,
    m_in: usize,
    commitment: Commitment,
    _aux_seed: u64,
) -> CeClaim<Commitment, F, K> {
    let (y_ring, ct) = compute_y_from_Z_and_r(structure, z, r, ell_d, params.b);
    let X = neo_reductions::common::project_x_from_witness_mat(z, structure.m, m_in).expect("project X");
    CeClaim {
        c: commitment,
        X,
        r: r.to_vec(),
        y_ring,
        ct,
        m_in,
        fold_digest: [0; 32],
        adv: None,
    }
}

#[test]
fn rlc_public_matches_rejects_common_short_row_point() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let m_in = D;
    let r = vec![k(5); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];

    let z0 = make_z(91_000, structure.m);
    let z1 = make_z(92_000, structure.m);
    let mut inputs = vec![
        build_me_from_z(
            &params,
            &structure,
            &z0,
            &r,
            ell_d,
            m_in,
            make_commitment(&params, 93_000),
            94_000,
        ),
        build_me_from_z(
            &params,
            &structure,
            &z1,
            &r,
            ell_d,
            m_in,
            make_commitment(&params, 95_000),
            96_000,
        ),
    ];
    let rhos = typed_rhos(&params, &[diag_rho(1), diag_rho(2)]);
    let (mut expected, _) = rlc_with_commit(
        FoldingMode::Optimized,
        &structure,
        &params,
        &rhos,
        &inputs,
        &[z0, z1],
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("valid RLC fixture");

    for input in &mut inputs {
        input.r.pop();
    }
    expected.r.pop();
    let (accepted, _) = rlc_public_matches_with_perf(
        &structure,
        &params,
        &rhos,
        &inputs,
        &expected,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("malformed inputs should receive a verifier verdict");

    assert!(
        !accepted,
        "public RLC verifier accepted a CE row point shorter than log2(n)"
    );
}

#[test]
fn verify_dec_public_rejects_common_short_row_point() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let m_in = D;
    let r = vec![k(13); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let z_split = vec![Mat::zero(D, structure.m / D, F::ZERO); params.k_rho as usize];

    let mut z_parent = Mat::zero(D, z_split[0].cols(), F::ZERO);
    let mut power = F::ONE;
    let b_f = F::from_u64(params.b as u64);
    for z_i in &z_split {
        for row in 0..D {
            for column in 0..z_parent.cols() {
                z_parent[(row, column)] += power * z_i[(row, column)];
            }
        }
        power *= b_f;
    }
    let child_commitments = (0..params.k_rho)
        .map(|index| make_commitment(&params, 103_000 + index as u64))
        .collect::<Vec<_>>();
    let mut parent = build_me_from_z(
        &params,
        &structure,
        &z_parent,
        &r,
        ell_d,
        m_in,
        combine_commitments_b_pows(&child_commitments, params.b),
        105_000,
    );
    let (mut children, ok_y, ok_x, ok_c) = dec_children_with_commit(
        FoldingMode::Optimized,
        &structure,
        &params,
        &parent,
        &z_split,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
    );
    assert!(ok_y && ok_x && ok_c, "valid DEC fixture");

    parent.r.pop();
    for child in &mut children {
        child.r.pop();
    }

    assert!(
        !verify_dec_public(
            &structure,
            &params,
            &parent,
            &children,
            combine_commitments_b_pows,
            ell_d,
        ),
        "public DEC verifier accepted CE row points shorter than log2(n)"
    );
}

#[test]
fn verify_dec_public_rejects_child_count_different_from_k_rho() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![k(17); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let mut z = Mat::zero(D, structure.m / D, F::ZERO);
    z[(0, 0)] = F::ONE;
    let child = build_me_from_z(
        &params,
        &structure,
        &z,
        &r,
        ell_d,
        D,
        make_commitment(&params, 120_001),
        120_002,
    );
    let parent = child.clone();

    assert_eq!(params.k_rho, 14);
    assert!(
        !verify_dec_public(
            &structure,
            &params,
            &parent,
            core::slice::from_ref(&child),
            combine_commitments_b_pows,
            ell_d,
        ),
        "public DEC verifier accepted one child although params.k_rho is 14"
    );
}

#[test]
fn verify_dec_public_rejects_out_of_range_child_x_at_correct_arity() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![k(19); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let child_count = params.k_rho as usize;

    let mut child_zs = vec![Mat::zero(D, structure.m / D, F::ZERO); child_count];
    child_zs[0][(0, 0)] = F::from_u64(params.b as u64);

    let child_commitments = (0..child_count)
        .map(|index| make_commitment(&params, 121_000 + index as u64))
        .collect::<Vec<_>>();
    let children = child_zs
        .iter()
        .zip(&child_commitments)
        .enumerate()
        .map(|(index, (z, commitment))| {
            let claim = build_me_from_z(
                &params,
                &structure,
                z,
                &r,
                ell_d,
                D,
                commitment.clone(),
                122_000 + 2 * index as u64,
            );
            claim
        })
        .collect::<Vec<_>>();
    assert!(
        !neo_math::balanced::within_nc_bound(children[0].X[(0, 0)], params.b),
        "control: public child coordinate b must be outside CE(b)"
    );

    let mut parent_z = Mat::zero(D, structure.m / D, F::ZERO);
    let mut power = F::ONE;
    let b_f = F::from_u64(params.b as u64);
    for z in &child_zs {
        for row in 0..D {
            for column in 0..parent_z.cols() {
                parent_z[(row, column)] += power * z[(row, column)];
            }
        }
        power *= b_f;
    }
    let parent = build_me_from_z(
        &params,
        &structure,
        &parent_z,
        &r,
        ell_d,
        D,
        combine_commitments_b_pows(&child_commitments, params.b),
        123_000,
    );

    assert_eq!(children.len(), child_count);
    assert!(
        !verify_dec_public(
            &structure,
            &params,
            &parent,
            &children,
            combine_commitments_b_pows,
            ell_d,
        ),
        "public DEC verifier accepted a child X coordinate outside CE(b)"
    );
}

#[cfg(target_pointer_width = "64")]
#[test]
fn rlc_public_rejects_ell_d_that_cannot_describe_a_usize_domain() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let d_pad = 1usize << ell_d;
    let structure =
        CcsStructure::new(vec![Mat::identity(D)], SparsePoly::new(1, Vec::new())).expect("valid CCS structure");
    let input = CeClaim::<Commitment, F, K> {
        c: Commitment::zeros(D, 1),
        X: Mat::zero(D, neo_ccs::superneo_public_x_cols(D), F::ZERO),
        r: vec![K::ZERO; D.next_power_of_two().trailing_zeros() as usize],
        y_ring: vec![vec![K::ZERO; d_pad]; 2],
        ct: vec![K::ZERO; 2],
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    };
    let rhos = neo_reductions::api::rot_rhos_from_mats(&params, &[Mat::identity(D)], "ell_d narrowing regression")
        .expect("typed rho");
    let run = |declared_ell_d| {
        rlc_public(
            &structure,
            &params,
            &rhos,
            core::slice::from_ref(&input),
            |_, _| Commitment::zeros(D, 1),
            declared_ell_d,
        )
    };

    assert!(run(ell_d).is_ok(), "control fixture must be valid");
    let aliased_ell_d = ell_d + (1usize << 32);
    assert!(
        run(aliased_ell_d).is_err(),
        "2^ell_d cannot fit usize; verifier must not reinterpret it as 2^{ell_d}"
    );
}

#[test]
fn rlc_public_rejects_instance_count_above_parameter_guard() {
    const INPUT_COUNT: usize = 76;

    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![k(23); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let z = Mat::zero(D, structure.m / D, F::ZERO);
    let input = build_me_from_z(
        &params,
        &structure,
        &z,
        &r,
        ell_d,
        D,
        make_commitment(&params, 130_000),
        130_001,
    );
    let inputs = vec![input; INPUT_COUNT];

    let mut transcript = Poseidon2Transcript::new(b"redteam/rlc-count-guard");
    assert!(
        sample_rot_rhos_n_typed(&mut transcript, &params, &RotRing::goldilocks(), INPUT_COUNT,).is_err(),
        "control: the parameter guard must reject sampling 76 challenges"
    );
    let rho_mats = vec![Mat::identity(D); INPUT_COUNT];
    let rhos = typed_rhos(&params, &rho_mats);
    let result = rlc_public(&structure, &params, &rhos, &inputs, mix_commitments_from_rhos, ell_d);

    assert!(
        result.is_err(),
        "public RLC verifier accepted {INPUT_COUNT} instances although count*T*(b-1) is not below B"
    );
}

#[test]
fn rlc_public_rejects_inputs_from_different_fold_transcripts() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![k(29); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let z = Mat::zero(D, structure.m / D, F::ZERO);
    let first = build_me_from_z(
        &params,
        &structure,
        &z,
        &r,
        ell_d,
        D,
        make_commitment(&params, 131_000),
        131_001,
    );
    let mut second = first.clone();
    second.c = make_commitment(&params, 131_002);
    second.fold_digest[0] = 1;
    let inputs = vec![first, second];
    let rhos = typed_rhos(&params, &[Mat::identity(D), Mat::identity(D)]);

    let result = rlc_public(&structure, &params, &rhos, &inputs, mix_commitments_from_rhos, ell_d);

    assert!(
        result.is_err(),
        "public RLC combined CE claims authenticated by different Pi_CCS fold transcripts"
    );
}

#[test]
fn verify_dec_public_rejects_children_from_different_fold_transcript() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![k(31); structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let z = Mat::zero(D, structure.m / D, F::ZERO);
    let parent = build_me_from_z(&params, &structure, &z, &r, ell_d, D, Commitment::zeros(D, 1), 132_000);
    let mut children = vec![parent.clone(); params.k_rho as usize];
    children[1].fold_digest[0] = 1;

    assert!(
        !verify_dec_public(
            &structure,
            &params,
            &parent,
            &children,
            combine_commitments_b_pows,
            ell_d,
        ),
        "public DEC accepted a child authenticated by a different fold transcript"
    );
}

#[test]
fn reduction_policy_rejects_parameter_modulus_different_from_field() {
    let mut actual_field_params = NeoParams::goldilocks_paper_b2();
    actual_field_params.lambda = 100;
    let mut claimed_larger_field_params = actual_field_params;
    claimed_larger_field_params.q = u64::MAX;
    let structure =
        CcsStructure::new(vec![Mat::identity(1)], SparsePoly::new(1, Vec::new())).expect("minimal CCS structure");

    let actual = neo_reductions::engines::pi_ccs_joint::build_joint_dims(&actual_field_params, &structure, 1, 0);
    assert!(actual.is_ok(), "control: the real Goldilocks profile must pass");
    let claimed =
        neo_reductions::engines::pi_ccs_joint::build_joint_dims(&claimed_larger_field_params, &structure, 1, 0);

    assert!(
        claimed.is_err(),
        "Goldilocks reduction accepted a policy justified only by params.q=u64::MAX"
    );
}

#[test]
fn rlc_with_commit_rejects_mixed_public_widths_without_panicking() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let structure = build_structure(D, D);
    let r = vec![K::ZERO; structure.n.next_power_of_two().max(2).trailing_zeros() as usize];
    let z = Mat::zero(D, structure.m / D, F::ZERO);
    let inputs = vec![
        build_me_from_z(
            &params,
            &structure,
            &z,
            &r,
            ell_d,
            D,
            make_commitment(&params, 140_000),
            140_001,
        ),
        build_me_from_z(
            &params,
            &structure,
            &z,
            &r,
            ell_d,
            0,
            make_commitment(&params, 140_002),
            140_003,
        ),
    ];
    let rhos = typed_rhos(&params, &[Mat::identity(D), Mat::identity(D)]);
    let witnesses = vec![z.clone(), z];

    let result = std::panic::catch_unwind(|| {
        rlc_with_commit(
            FoldingMode::Optimized,
            &structure,
            &params,
            &rhos,
            &inputs,
            &witnesses,
            ell_d,
            mix_commitments_from_rhos,
        )
    });

    assert!(
        matches!(result, Ok(Err(_))),
        "RLC must reject claims from different public-input relations instead of panicking"
    );
}
