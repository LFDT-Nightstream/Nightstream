#![allow(non_snake_case)]

use neo_ajtai::Commitment;
use neo_ccs::{poly::SparsePoly, poly::Term, CcsStructure, CeClaim, Mat};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::api::{
    dec_children_with_commit, rlc_public, rlc_public_matches_verified_inputs_with_perf, rlc_public_matches_with_perf,
    rlc_with_commit, verify_dec_public, FoldingMode,
};
use neo_reductions::common::{compute_y_from_Z_and_r, left_mul_acc};
use p3_field::PrimeCharacteristicRing;

fn k(v: u64) -> K {
    K::from(F::from_u64(v))
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
        let term = scale_commitment(c, rho[(0, 0)]);
        acc = add_commitments(&acc, &term);
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

fn diag_rho(scale: u64) -> Mat<F> {
    let mut rho = Mat::zero(D, D, F::ZERO);
    let s = F::from_u64(scale);
    for i in 0..D {
        rho[(i, i)] = s;
    }
    rho
}

fn typed_rhos(params: &NeoParams, rhos: &[Mat<F>]) -> Vec<neo_reductions::api::RotRho> {
    neo_reductions::api::rot_rhos_from_mats(params, rhos, "rlc_dec_k_gt1:test rhos").expect("typed rhos")
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
    params: &NeoParams,
    s: &CcsStructure<F>,
    Z: &Mat<F>,
    r: &[K],
    ell_d: usize,
    m_in: usize,
    c: Commitment,
    aux_seed: u64,
) -> CeClaim<Commitment, F, K> {
    let (y_ring, ct) = compute_y_from_Z_and_r(s, Z, r, ell_d, params.b);
    let X = neo_reductions::common::project_x_from_witness_mat(Z, s.m, m_in).expect("project X");
    CeClaim {
        c_step_coords: vec![],
        u_offset: 0,
        u_len: 0,
        c,
        X,
        r: r.to_vec(),
        s_col: vec![],
        y_ring,
        ct,
        aux_openings: vec![k(aux_seed), k(aux_seed + 1)],
        y_zcol: vec![],
        m_in,
        fold_digest: [0u8; 32],
    }
}

#[test]
fn rlc_with_commit_k4_matches_public_recompute_and_detects_rho_tamper() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 2usize;
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

    let rhos = vec![diag_rho(1), diag_rho(2), diag_rho(3), diag_rho(4)];
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
    .expect("rlc_with_commit optimized");

    let parent_public = rlc_public(&s, &params, &rhos_typed, &me_inputs, mix_commitments_from_rhos, ell_d)
        .expect("rlc_public recompute");
    assert_eq!(parent, parent_public, "public RLC recompute must match engine output");

    let mut me_inputs_stale = me_inputs.clone();
    me_inputs_stale[1].ct[0] += K::ONE;
    let parent_public_stale = rlc_public(
        &s,
        &params,
        &rhos_typed,
        &me_inputs_stale,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("rlc_public stale ct");
    assert_eq!(
        parent_public_stale, parent,
        "public RLC recompute must ignore stale ct shell on inputs"
    );

    let want_Z_mix = combine_z_with_rhos(&rhos, &Zs);
    assert_eq!(Z_mix, want_Z_mix, "Z_mix must equal Σ ρ_i · Z_i");

    let mut rhos_tampered = rhos.clone();
    rhos_tampered[0] = diag_rho(9);
    let rhos_tampered_typed = typed_rhos(&params, &rhos_tampered);
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
fn rlc_public_verified_inputs_fast_path_matches_full_public_check() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 2usize;
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

    let rhos = vec![diag_rho(1), diag_rho(3), diag_rho(4), diag_rho(7)];
    let rhos_typed = typed_rhos(&params, &rhos);
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

    let mut me_inputs_stale = me_inputs.clone();
    me_inputs_stale[2].ct[0] += K::ONE;
    let mut parent_stale = parent.clone();
    parent_stale.ct[0] += K::ONE;
    let (full_ok_stale, _) = rlc_public_matches_with_perf(
        &s,
        &params,
        &rhos_typed,
        &me_inputs_stale,
        &parent_stale,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("full stale-ct rlc_public_matches_with_perf");
    let (verified_ok_stale, _) = rlc_public_matches_verified_inputs_with_perf(
        &s,
        &params,
        &rhos_typed,
        &me_inputs_stale,
        &parent_stale,
        mix_commitments_from_rhos,
        ell_d,
    )
    .expect("verified-inputs stale-ct rlc_public_matches_with_perf");
    assert_eq!(
        verified_ok_stale, full_ok_stale,
        "fast path must agree on stale-ct shell inputs"
    );
    assert!(verified_ok_stale, "stale ct shell must not fail the public RLC check");

    let rhos_tampered = typed_rhos(&params, &[diag_rho(9), diag_rho(3), diag_rho(4), diag_rho(7)]);
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
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 2usize;
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

    let rhos = vec![diag_rho(1), diag_rho(2), diag_rho(3), diag_rho(5)];
    let rhos_typed = typed_rhos(&params, &rhos);

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
fn dec_children_with_commit_k4_public_and_tamper_checks() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 2usize;
    let r = vec![k(13); 6];

    let k_dec = 4usize;
    let mut Z_split = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        Z_split.push(make_z(1000 + i as u64 * 131, s.m));
    }

    let bF = F::from_u64(params.b as u64);
    let z_cols = Z_split[0].cols();
    let mut Z_parent = Mat::zero(D, z_cols, F::ZERO);
    let mut pow = F::ONE;
    for Zi in &Z_split {
        for r_ in 0..D {
            for c_ in 0..z_cols {
                Z_parent[(r_, c_)] += pow * Zi[(r_, c_)];
            }
        }
        pow *= bF;
    }
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
    tampered_child[2].ct[0] += K::ONE;
    assert!(verify_dec_public(
        &s,
        &params,
        &parent,
        &tampered_child,
        combine_commitments_b_pows,
        ell_d
    ));

    let mut tampered_aux = children.clone();
    tampered_aux[0].aux_openings[0] += K::ONE;
    assert!(!verify_dec_public(
        &s,
        &params,
        &parent,
        &tampered_aux,
        combine_commitments_b_pows,
        ell_d
    ));
}

#[cfg(feature = "paper-exact")]
#[test]
fn dec_children_with_commit_k4_optimized_matches_paper_exact() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 2usize;
    let r = vec![k(19); 6];

    let k_dec = 4usize;
    let mut Z_split = Vec::with_capacity(k_dec);
    for i in 0..k_dec {
        Z_split.push(make_z(1300 + i as u64 * 127, s.m));
    }

    let bF = F::from_u64(params.b as u64);
    let z_cols = Z_split[0].cols();
    let mut Z_parent = Mat::zero(D, z_cols, F::ZERO);
    let mut pow = F::ONE;
    for Zi in &Z_split {
        for r_ in 0..D {
            for c_ in 0..z_cols {
                Z_parent[(r_, c_)] += pow * Zi[(r_, c_)];
            }
        }
        pow *= bF;
    }

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
}

#[test]
fn rlc_with_commit_k61_boundary_smoke() {
    let params = NeoParams::goldilocks_127();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;
    let s = build_structure(D, D);
    let m_in = 1usize;
    let r = vec![k(29); 6];

    let k_inputs = 61usize;
    let mut Zs = Vec::with_capacity(k_inputs);
    let mut me_inputs = Vec::with_capacity(k_inputs);
    let mut rhos = Vec::with_capacity(k_inputs);

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
        rhos.push(diag_rho(1 + (i as u64 % 7)));
    }
    let rhos_typed = typed_rhos(&params, &rhos);

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
    assert_eq!(parent.aux_openings.len(), me_inputs[0].aux_openings.len());
}
