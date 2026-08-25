#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, dec_children_with_commit_superneo_cached_from_trusted_split_digits,
    dec_children_with_commit_superneo_cached_with_digit_flags, rlc_public,
    rlc_public_matches_verified_inputs_with_perf, rlc_public_matches_with_perf, rlc_with_commit,
    rlc_with_commit_refs_and_resident_witness, verify_dec_public, FoldingMode,
};
use neo_reductions::common::{
    compute_v1_1_evaluations_from_z_and_r, left_mul_acc, project_x_from_witness_mat, rot_rhos_to_mats,
    sample_rot_rhos_n, sample_rot_rhos_n_typed, split_b_matrix_k_with_nonzero_flags, validate_packed_witness_nc_range,
    RotRing,
};
use neo_reductions::superneo_eval::build_superneo_eval_cache;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
}

#[test]
fn packed_signed_unit_range_validation_matches_dense_semantics() {
    let params = NeoParams::goldilocks_paper_b2();
    let neg_one = F::ZERO - F::ONE;
    let values = (0..D)
        .map(|index| match index % 3 {
            0 => F::ZERO,
            1 => F::ONE,
            _ => neg_one,
        })
        .collect::<Vec<_>>();
    let dense = Mat::from_row_major(D, 1, values.clone());
    let compact = Mat::compact_signed_unit(D, 1, values);
    let zero = Mat::virtual_constant(D, 1, F::ZERO);

    validate_packed_witness_nc_range(&params, &dense, D, "dense").expect("dense signed units");
    validate_packed_witness_nc_range(&params, &compact, D, "compact").expect("compact signed units");
    validate_packed_witness_nc_range(&params, &zero, D, "constant").expect("constant zero");

    let mut invalid = Mat::zero(D, 1, F::ZERO);
    invalid[(0, 0)] = F::from_u64(1u64 << 60);
    assert!(validate_packed_witness_nc_range(&params, &invalid, D, "invalid").is_err());
}

fn build_structure(n: usize, m: usize) -> CcsStructure<F> {
    let m0 = Mat::identity(n);
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        m1[(r, (r + 1) % m)] = F::ONE;
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

fn build_wide_structure(n: usize, m: usize) -> CcsStructure<F> {
    let mut m0 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        m0[(r, r)] = F::ONE;
    }
    let mut m1 = Mat::zero(n, m, F::ZERO);
    for r in 0..n {
        m1[(r, (r + 1) % m)] = F::ONE;
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
    CcsStructure::new(vec![m0, m1], f).expect("valid wide CCS structure")
}

fn make_z(seed: u64, m: usize) -> Mat<F> {
    assert!(m.is_multiple_of(D), "SuperNeo-only test requires m divisible by D");
    let cols = m / D;
    let mut data = Vec::with_capacity(D * cols);
    for rho in 0..D {
        for blk in 0..cols {
            let c = blk * D + rho;
            data.push(F::from_u64(seed + (rho as u64) * 11 + (c as u64) * 17 + 1));
        }
    }
    Mat::from_row_major(D, cols, data)
}

fn make_sparse_signed_unit_z(seed: usize, m: usize) -> Mat<F> {
    assert!(m.is_multiple_of(D), "SuperNeo-only test requires m divisible by D");
    let cols = m / D;
    let mut out = Mat::zero(D, cols, F::ZERO);
    let neg_one = F::ZERO - F::ONE;
    for col in 0..cols {
        let row = (seed + col * 7) % D;
        out[(row, col)] = if (seed + col) % 2 == 0 { F::ONE } else { neg_one };
    }
    out
}

fn make_commitment(params: &NeoParams, seed: u64) -> Commitment {
    let mut c = Commitment::zeros(params.d as usize, 1);
    c.data[0] = F::from_u64(seed);
    c
}

fn scale_commitment(c: &Commitment, scale: F) -> Commitment {
    let mut out = c.clone();
    for v in out.data.iter_mut() {
        *v *= scale;
    }
    out
}

fn add_commitments(a: &Commitment, b: &Commitment) -> Commitment {
    let mut out = a.clone();
    out.add_inplace(b);
    out
}

fn mix_commitments_from_rhos(rhos: &[Mat<F>], commits: &[Commitment]) -> Commitment {
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    for (rho, c) in rhos.iter().zip(commits.iter()) {
        for lane in 0..c.kappa {
            for row in 0..D {
                for coefficient in 0..D {
                    acc.data[lane * D + row] += rho[(row, coefficient)] * c.data[lane * D + coefficient];
                }
            }
        }
    }
    acc
}

fn combine_commitments_b_pows(commits: &[Commitment], b: u32) -> Commitment {
    let mut acc = Commitment::zeros(commits[0].d, commits[0].kappa);
    let mut pow = F::ONE;
    let bF = F::from_u64(b as u64);
    for c in commits {
        let term = scale_commitment(c, pow);
        acc = add_commitments(&acc, &term);
        pow *= bF;
    }
    acc
}

fn typed_rhos(params: &NeoParams, rhos: &[Mat<F>]) -> Vec<neo_reductions::api::RotRho> {
    neo_reductions::api::rot_rhos_from_mats(params, rhos, "rlc_dec_k_gt1:test rhos").expect("typed rhos")
}

fn v1_1_test_transcript(domain: &[u8]) -> Poseidon2Transcript {
    let mut transcript = Poseidon2Transcript::new_v1_1();
    let domain_fields = domain
        .iter()
        .map(|byte| F::from_u64(u64::from(*byte)))
        .collect::<Vec<_>>();
    transcript.absorb_block_v1_1(&domain_fields);
    transcript
}

fn sampled_rhos(params: &NeoParams, count: usize, alternate: bool) -> (Vec<neo_reductions::api::RotRho>, Vec<Mat<F>>) {
    let domain: &'static [u8] = if alternate {
        b"rlc_dec_k_gt1/alternate"
    } else {
        b"rlc_dec_k_gt1/primary"
    };
    let mut transcript = v1_1_test_transcript(domain);
    let typed = sample_rot_rhos_n_typed(&mut transcript, params, &RotRing::goldilocks(), count)
        .expect("sample selected strong-set rhos");
    let matrices = rot_rhos_to_mats(&typed);
    (typed, matrices)
}

fn combine_z_with_rhos(rhos: &[Mat<F>], Zs: &[Mat<F>]) -> Mat<F> {
    let mut out = Mat::zero(D, Zs[0].cols(), F::ZERO);
    for (rho, Z) in rhos.iter().zip(Zs.iter()) {
        let mut term = Mat::zero(D, Z.cols(), F::ZERO);
        left_mul_acc(&mut term, rho, Z);
        for r in 0..D {
            for c in 0..Z.cols() {
                out[(r, c)] += term[(r, c)];
            }
        }
    }
    out
}

fn build_me_from_z(
    _params: &NeoParams,
    s: &CcsStructure<F>,
    Z: &Mat<F>,
    r: &[K],
    ell_d: usize,
    m_in: usize,
    c: Commitment,
    _aux_seed: u64,
) -> CeClaim<Commitment, F, K> {
    let evaluations = compute_v1_1_evaluations_from_z_and_r(s, Z, r, ell_d);
    let X = neo_reductions::common::project_x_from_witness_mat(Z, s.m, m_in).expect("project X");
    CeClaim {
        adv: None,
        c,
        X,
        r: r.to_vec(),
        eval_k: evaluations.eval_k,
        eval_a: evaluations.eval_a,
        m_in,
        fold_digest: [0u8; 32],
    }
}

#[test]
fn rlc_with_commit_k4_matches_public_recompute_and_detects_rho_tamper() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(3); 6];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..4usize {
        let Z = make_z(1000 + i as u64 * 100, s.m);
        let c = make_commitment(&params, 2000 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            3000 + i as u64 * 10,
        ));
        Zs.push(Z);
    }

    let (rhos_typed, rhos) = sampled_rhos(&params, Zs.len(), false);

    let (parent, Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("rlc_with_commit optimized");

    let parent_public = rlc_public(&s, &params, &rhos_typed, &me_inputs, mix_commitments_from_rhos, ell_d)
        .expect("rlc_public recompute");
    assert_eq!(parent, parent_public, "public RLC recompute must match engine output");

    let witness_refs = Zs.iter().collect::<Vec<_>>();
    let (resident_parent, resident_shape) = rlc_with_commit_refs_and_resident_witness(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &witness_refs,
        ell_d,
        mix_commitments_from_rhos,
        |rho_mats, witnesses| (rho_mats.len(), witnesses.len(), witnesses[0].cols()),
    )
    .expect("resident-witness RLC");
    assert_eq!(resident_parent, parent);
    assert_eq!(resident_shape, (rhos.len(), Zs.len(), Zs[0].cols()));

    let want_Z_mix = combine_z_with_rhos(&rhos, &Zs);
    assert_eq!(Z_mix, want_Z_mix, "Z_mix must equal Σ ρ_i · Z_i");

    let (rhos_tampered_typed, _) = sampled_rhos(&params, Zs.len(), true);
    let parent_tampered = rlc_public(
        &s,
        &params,
        &rhos_tampered_typed,
        &me_inputs,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("rlc_public tampered rho");
    assert_ne!(parent_tampered, parent, "tampered ρ must change the RLC parent");
}

#[test]
fn rlc_with_commit_sampled_rotation_rhos_matches_public_z_mix() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(31); 6];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..5usize {
        let Z = make_z(4_200 + i as u64 * 137, s.m);
        let c = make_commitment(&params, 5_200 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            6_200 + i as u64 * 11,
        ));
        Zs.push(Z);
    }

    let mut transcript = v1_1_test_transcript(b"rlc sampled rotation rho test");
    let rhos_typed =
        sample_rot_rhos_n_typed(&mut transcript, &params, &RotRing::goldilocks(), Zs.len()).expect("sample rhos");
    let rho_mats = rot_rhos_to_mats(&rhos_typed);

    let (parent, Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("optimized rlc_with_commit with sampled rotation rhos");

    let want_Z_mix = combine_z_with_rhos(&rho_mats, &Zs);
    assert_eq!(
        Z_mix, want_Z_mix,
        "Z_mix must equal Σ ρ_i · Z_i for non-diagonal rotation rhos"
    );

    let parent_public =
        rlc_public(&s, &params, &rhos_typed, &me_inputs, mix_commitments_from_rhos, ell_d).expect("rlc_public");
    assert_eq!(parent, parent_public, "public RLC recompute must match engine output");
}

#[test]
fn rlc_with_commit_sparse_rotation_rhs_matches_public_z_mix() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_wide_structure(D, D * 512);
    let m_in = D;
    let r = vec![k(37); s.n.max(s.m).next_power_of_two().trailing_zeros() as usize];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..5usize {
        let Z = make_sparse_signed_unit_z(i, s.m);
        let c = make_commitment(&params, 7_200 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            8_200 + i as u64 * 11,
        ));
        Zs.push(Z);
    }

    let mut transcript = v1_1_test_transcript(b"rlc sparse rotation rhs test");
    let rhos_typed =
        sample_rot_rhos_n_typed(&mut transcript, &params, &RotRing::goldilocks(), Zs.len()).expect("sample rhos");
    let rho_mats = rot_rhos_to_mats(&rhos_typed);

    let (parent, Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("optimized rlc_with_commit with sparse rotation RHS");

    let want_Z_mix = combine_z_with_rhos(&rho_mats, &Zs);
    assert_eq!(
        Z_mix, want_Z_mix,
        "sparse RHS rotation fast path must equal generic Σ ρ_i · Z_i"
    );

    let parent_public =
        rlc_public(&s, &params, &rhos_typed, &me_inputs, mix_commitments_from_rhos, ell_d).expect("rlc_public");
    assert_eq!(parent, parent_public, "public RLC recompute must match engine output");
}

#[test]
fn rlc_x_projection_tracks_mixed_witness_under_rotation_rhos() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(11); 6];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..2usize {
        let Z = make_z(4000 + i as u64 * 100, s.m);
        let c = make_commitment(&params, 5000 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            6000 + i as u64 * 10,
        ));
        Zs.push(Z);
    }

    let mut transcript = v1_1_test_transcript(b"rlc_x_projection_tracks_mixed_witness_under_rotation_rhos");
    let rhos = sample_rot_rhos_n(&mut transcript, &params, &RotRing::goldilocks(), 2).expect("sample rhos");
    assert!(
        rhos.iter()
            .any(|rho| (0..D).any(|row| (0..D).any(|col| row != col && rho[(row, col)] != F::ZERO))),
        "test requires at least one non-diagonal rotation rho"
    );
    let rhos_typed = typed_rhos(&params, &rhos);

    let (parent, Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("optimized rlc_with_commit");
    let projected = project_x_from_witness_mat(&Z_mix, s.m, m_in).expect("project mixed witness X");

    assert_eq!(
        parent.X, projected,
        "Π_RLC public X must stay equal to the CE ring-slot projection L_in(Z_mix)"
    );
}

#[test]
fn rlc_public_verified_inputs_fast_path_matches_full_public_check() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(5); 6];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..4usize {
        let Z = make_z(2200 + i as u64 * 97, s.m);
        let c = make_commitment(&params, 3300 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            4400 + i as u64 * 10,
        ));
        Zs.push(Z);
    }

    let (rhos_typed, _) = sampled_rhos(&params, Zs.len(), false);
    let (parent, _) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("optimized rlc_with_commit");

    let (full_ok, _) = rlc_public_matches_with_perf(
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &parent,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("full rlc_public_matches_with_perf");
    let (verified_ok, _) = rlc_public_matches_verified_inputs_with_perf(
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &parent,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("verified-inputs rlc_public_matches_with_perf");
    assert_eq!(verified_ok, full_ok, "fast path must agree on valid verified inputs");
    assert!(verified_ok, "valid verified inputs must satisfy the public RLC check");

    let (rhos_tampered, _) = sampled_rhos(&params, Zs.len(), true);
    let (full_bad, _) = rlc_public_matches_with_perf(
        &s,
        &params,
        &rhos_tampered,
        &me_inputs,
        &parent,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("full tampered rlc_public_matches_with_perf");
    let (verified_bad, _) = rlc_public_matches_verified_inputs_with_perf(
        &s,
        &params,
        &rhos_tampered,
        &me_inputs,
        &parent,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("verified-inputs tampered rlc_public_matches_with_perf");
    assert_eq!(verified_bad, full_bad, "fast path must agree on tampered rho checks");
    assert!(!verified_bad, "tampered rho must fail the public RLC check");
}

#[cfg(feature = "paper-exact")]
#[test]
fn rlc_with_commit_k4_optimized_matches_paper_exact() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(7); 6];

    let mut Zs = Vec::new();
    let mut me_inputs = Vec::new();
    for i in 0..4usize {
        let Z = make_z(1500 + i as u64 * 101, s.m);
        let c = make_commitment(&params, 6000 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            7000 + i as u64 * 10,
        ));
        Zs.push(Z);
    }

    let (rhos_typed, _) = sampled_rhos(&params, Zs.len(), false);

    let (opt_parent, opt_Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("optimized rlc_with_commit");

    let (paper_parent, paper_Z_mix) = rlc_with_commit(
        FoldingMode::PaperExact,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("paper-exact rlc_with_commit");

    assert_eq!(opt_parent, paper_parent, "RLC parent mismatch between engines");
    assert_eq!(opt_Z_mix, paper_Z_mix, "RLC Z_mix mismatch between engines");
}

#[test]
fn dec_children_with_commit_fixed_arity_public_and_tamper_checks() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(13); 6];

    let k_dec = params.k_rho as usize;
    let Z_parent = make_z(1000, s.m);
    let (Z_split, _) =
        split_b_matrix_k_with_nonzero_flags(&Z_parent, k_dec, params.b).expect("parent must have a canonical split");
    let mut parent = build_me_from_z(
        &params,
        &s,
        &Z_parent,
        &r,
        ell_d,
        m_in,
        make_commitment(&params, 55_000),
        88_000,
    );

    let child_commitments: Vec<Commitment> = (0..k_dec)
        .map(|i| make_commitment(&params, 56_000 + i as u64))
        .collect();
    parent.c = combine_commitments_b_pows(&child_commitments, params.b);

    let (children, ok_y, ok_X, ok_c) = dec_children_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &parent,
        &Z_split,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
    );
    assert!(ok_y && ok_X && ok_c, "optimized DEC must satisfy y/X/c checks");
    assert!(verify_dec_public(
        &s,
        &params,
        &parent,
        &children,
        combine_commitments_b_pows,
        ell_d
    ));

    let mut tampered_child = children.clone();
    tampered_child[2].eval_a[0][0] += K::ONE;
    assert!(!verify_dec_public(
        &s,
        &params,
        &parent,
        &tampered_child,
        combine_commitments_b_pows,
        ell_d
    ));
}

#[test]
fn dec_children_trusted_split_digits_matches_checked_path() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(17); 6];
    let k_dec = params.k_rho as usize;

    let cols = s.m / D;
    let mut z_parent = Mat::zero(D, cols, F::ZERO);
    for rho in 0..D {
        for blk in 0..cols {
            let logical_col = blk * D + rho;
            let value = (logical_col % 7) as u64;
            z_parent[(rho, blk)] = F::from_u64(value);
        }
    }

    let (z_split, digit_nonzero) =
        split_b_matrix_k_with_nonzero_flags(&z_parent, k_dec, params.b).expect("split small witness");
    let child_commitments: Vec<Commitment> = (0..k_dec)
        .map(|i| make_commitment(&params, 72_000 + i as u64))
        .collect();
    let mut parent = build_me_from_z(
        &params,
        &s,
        &z_parent,
        &r,
        ell_d,
        m_in,
        make_commitment(&params, 71_000),
        91_000,
    );
    parent.c = combine_commitments_b_pows(&child_commitments, params.b);
    let superneo_cache = build_superneo_eval_cache(&s).expect("SuperNeo-compatible test structure");

    let checked = dec_children_with_commit_superneo_cached_with_digit_flags(
        FoldingMode::Optimized,
        &s,
        &params,
        &parent,
        &z_split,
        &digit_nonzero,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
        &superneo_cache,
    );
    let trusted = dec_children_with_commit_superneo_cached_from_trusted_split_digits(
        FoldingMode::Optimized,
        &s,
        &params,
        &parent,
        &z_split,
        &digit_nonzero,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
        &superneo_cache,
        None,
        None,
    );

    assert_eq!(trusted.1, checked.1, "ok_y mismatch");
    assert_eq!(trusted.2, checked.2, "ok_X mismatch");
    assert_eq!(trusted.3, checked.3, "ok_c mismatch");
    assert_eq!(trusted.0, checked.0, "trusted split DEC children diverged");
    assert!(trusted.1 && trusted.2 && trusted.3, "trusted split DEC must verify");
}

#[cfg(feature = "paper-exact")]
#[test]
fn dec_children_with_commit_optimized_matches_paper_exact() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(19); 6];

    let k_dec = params.k_rho as usize;
    let Z_parent = make_z(1300, s.m);
    let (Z_split, _) =
        split_b_matrix_k_with_nonzero_flags(&Z_parent, k_dec, params.b).expect("parent must have a canonical split");

    let mut parent = build_me_from_z(
        &params,
        &s,
        &Z_parent,
        &r,
        ell_d,
        m_in,
        make_commitment(&params, 66_000),
        99_000,
    );

    let child_commitments: Vec<Commitment> = (0..k_dec)
        .map(|i| make_commitment(&params, 67_000 + i as u64))
        .collect();
    parent.c = combine_commitments_b_pows(&child_commitments, params.b);

    let out_opt = dec_children_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &parent,
        &Z_split,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
    );
    let out_paper = dec_children_with_commit(
        FoldingMode::PaperExact,
        &s,
        &params,
        &parent,
        &Z_split,
        ell_d,
        &child_commitments,
        combine_commitments_b_pows,
    );

    assert_eq!(out_opt.1, out_paper.1, "ok_y mismatch");
    assert_eq!(out_opt.2, out_paper.2, "ok_X mismatch");
    assert_eq!(out_opt.3, out_paper.3, "ok_c mismatch");
    assert_eq!(out_opt.0, out_paper.0, "DEC children mismatch between engines");
    assert!(out_opt.1 && out_opt.2 && out_opt.3, "canonical DEC split must verify");
}

#[test]
fn rlc_with_commit_k61_boundary_smoke() {
    let params = NeoParams::goldilocks_paper_b2();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = D;
    let r = vec![k(29); 6];

    let k_inputs = 61usize;
    let mut Zs = Vec::with_capacity(k_inputs);
    let mut me_inputs = Vec::with_capacity(k_inputs);

    for i in 0..k_inputs {
        let Z = make_z(1000 + i as u64 * 19, s.m);
        let c = make_commitment(&params, 30_000 + i as u64);
        me_inputs.push(build_me_from_z(
            &params,
            &s,
            &Z,
            &r,
            ell_d,
            m_in,
            c,
            40_000 + i as u64 * 2,
        ));
        Zs.push(Z);
    }
    let (rhos_typed, _) = sampled_rhos(&params, k_inputs, false);

    let (parent, _Z_mix) = rlc_with_commit(
        FoldingMode::Optimized,
        &s,
        &params,
        &rhos_typed,
        &me_inputs,
        &Zs,
        ell_d,
        mix_commitments_from_rhos,
    )
    .expect("k=61 rlc_with_commit");

    let parent_public =
        rlc_public(&s, &params, &rhos_typed, &me_inputs, mix_commitments_from_rhos, ell_d).expect("k=61 rlc_public");
    assert_eq!(parent, parent_public, "k=61: public recompute mismatch");
    assert_eq!(parent.eval_a.len(), s.t());
    assert_eq!(parent.eval_k.len(), D.next_power_of_two());
}
