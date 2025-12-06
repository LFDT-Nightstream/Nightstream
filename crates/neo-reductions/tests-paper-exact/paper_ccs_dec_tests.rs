#![cfg(feature = "paper-exact")]
#![allow(non_snake_case)]

use neo_reductions::pi_ccs_paper_exact as refimpl;

use neo_ajtai::{set_global_pp, setup as ajtai_setup, AjtaiSModule};
use neo_ccs::traits::SModuleHomomorphism;
use neo_ccs::{CcsStructure, Mat, McsInstance, McsWitness, SparsePoly, Term};

use neo_math::{D, F, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing};
use rand_chacha::rand_core::SeedableRng;

/// --- Shared helpers ----------------------------------------

fn setup_ajtai_for_dims(m: usize) {
    let mut rng = rand_chacha::ChaCha8Rng::seed_from_u64(7);
    let pp = ajtai_setup(&mut rng, D, 4, m).expect("Ajtai setup should succeed");
    let _ = set_global_pp(pp);
}

fn tiny_ccs_id(n: usize, m: usize) -> CcsStructure<F> {
    assert_eq!(n, m, "use square tiny ccs");
    let m0 = Mat::identity(n);
    let f = SparsePoly::new(
        1,
        vec![Term {
            coeff: F::ONE,
            exps: vec![1],
        }],
    ); // f(y0) = y0
    CcsStructure::new(vec![m0], f).unwrap()
}

fn mat_eq<Ff: Field + PrimeCharacteristicRing + Copy>(a: &Mat<Ff>, b: &Mat<Ff>) -> bool {
    if a.rows() != b.rows() || a.cols() != b.cols() {
        return false;
    }
    for r in 0..a.rows() {
        for c in 0..a.cols() {
            if a[(r, c)] != b[(r, c)] {
                return false;
            }
        }
    }
    true
}

/// --- Π_DEC tests -----------------------------------------------------------

#[test]
fn paper_exact_dec_reconstruction_and_checks_hold() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Split components (Z_1, Z_2, Z_3)
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(1); D * m]);
    let z2 = Mat::from_row_major(D, m, vec![F::from_u64(2); D * m]);
    let z3 = Mat::from_row_major(D, m, vec![F::from_u64(3); D * m]);
    let Z_split = vec![z1.clone(), z2.clone(), z3.clone()];

    // Parent Z := Σ b^{i-1} Z_i
    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m, F::ZERO);
    let mut pow = F::ONE;
    for Zi in &Z_split {
        for r in 0..D {
            for c in 0..m {
                Z_parent.set(r, c, Z_parent[(r, c)] + pow * Zi[(r, c)]);
            }
        }
        pow = pow * bF;
    }

    // Build parent ME(B, L)
    let r = vec![K::from(F::from_u64(11)); 1];
    let w_parent = McsWitness {
        w: vec![],
        Z: Z_parent.clone(),
    };
    let inst_parent = McsInstance {
        c: l.commit(&Z_parent),
        x: vec![],
        m_in: 1,
    };
    let parent_out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_parent.clone()],
        &[w_parent.clone()],
        &[],
        &[],
        &r,
        ell_d,
        [0; 32],
        &l,
    );
    let parent = parent_out[0].clone();

    // Run DEC
    let (children, ok_y, ok_X) = refimpl::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_split, ell_d);

    assert!(ok_y && ok_X, "DEC must satisfy both reconstruction checks");

    // Each child must match literal outputs for its Z_i
    for (i, Zi) in Z_split.iter().enumerate() {
        let wi = McsWitness {
            w: vec![],
            Z: Zi.clone(),
        };
        let insti = McsInstance {
            c: l.commit(Zi),
            x: vec![],
            m_in: 1,
        };
        let outi =
            refimpl::build_me_outputs_paper_exact(&s, &params, &[insti], &[wi], &[], &[], &r, ell_d, [0; 32], &l);
        assert!(
            mat_eq(&children[i].X, &outi[0].X),
            "DEC child X_i must match literal projection for i={}",
            i
        );
        for j in 0..s.t() {
            assert_eq!(
                children[i].y[j], outi[0].y[j],
                "DEC child y_(i,j) must match literal for (i={}, j={})",
                i, j
            );
        }
        // y_scalars recomposition (digits → scalar)
        let bK = K::from(F::from_u64(params.b as u64));
        let mut pw = vec![K::ONE; D];
        for u in 1..D {
            pw[u] = pw[u - 1] * bK;
        }
        for j in 0..s.t() {
            let mut rec = K::ZERO;
            for rho in 0..D {
                rec += children[i].y[j][rho] * pw[rho];
            }
            assert_eq!(
                children[i].y_scalars[j], rec,
                "DEC child y_scalars must recompose for i={}, j={}",
                i, j
            );
        }
    }
}

#[test]
fn paper_exact_dec_k1_identity() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    let Z = Mat::from_row_major(D, m, vec![F::from_u64(4); D * m]);
    let r = vec![K::from(F::from_u64(13)); 1];

    let w = McsWitness {
        w: vec![],
        Z: Z.clone(),
    };
    let inst = McsInstance {
        c: l.commit(&Z),
        x: vec![],
        m_in: 1,
    };
    let parent_out = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst.clone()],
        &[w.clone()],
        &[],
        &[],
        &r,
        ell_d,
        [0; 32],
        &l,
    );
    let parent = parent_out[0].clone();

    let (children, ok_y, ok_X) = refimpl::dec_reduction_paper_exact::<F>(&s, &params, &parent, &[Z.clone()], ell_d);
    assert!(ok_y && ok_X, "DEC must pass checks for k=1");

    assert_eq!(children.len(), 1);
    assert!(mat_eq(&children[0].X, &parent.X), "k=1: X must be identical");
    for j in 0..s.t() {
        assert_eq!(children[0].y[j], parent.y[j], "k=1: y must be identical (j={})", j);
    }
}

#[test]
fn paper_exact_dec_wrong_split_detected() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Honest split
    let z1 = Mat::from_row_major(D, m, vec![F::from_u64(1); D * m]);
    let z2 = Mat::from_row_major(D, m, vec![F::from_u64(2); D * m]);

    // Parent Z := z1 + b·z2
    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m, F::ZERO);
    for r_ in 0..D {
        for c_ in 0..m {
            Z_parent.set(r_, c_, z1[(r_, c_)] + bF * z2[(r_, c_)]);
        }
    }

    let r = vec![K::from(F::from_u64(17)); 1];
    let w_parent = McsWitness {
        w: vec![],
        Z: Z_parent.clone(),
    };
    let inst_parent = McsInstance {
        c: l.commit(&Z_parent),
        x: vec![],
        m_in: 1,
    };
    let parent = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_parent],
        &[w_parent],
        &[],
        &[],
        &r,
        ell_d,
        [0; 32],
        &l,
    )[0]
    .clone();

    // Bad split: replace Z_2 with 2·Z_2
    let mut Z2_bad = Mat::zero(D, m, F::ZERO);
    for r_ in 0..D {
        for c_ in 0..m {
            Z2_bad.set(r_, c_, F::from_u64(2) * z2[(r_, c_)]);
        }
    }
    let Z_bad = vec![z1.clone(), Z2_bad];

    let (_children, ok_y, ok_X) = refimpl::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_bad, ell_d);
    assert!(
        !ok_y || !ok_X,
        "DEC must detect a wrong split (at least one check fails)"
    );
}

#[test]
fn paper_exact_dec_rlc_roundtrip() {
    let params = NeoParams::goldilocks_127();
    let (n, m) = (2usize, 2usize);
    setup_ajtai_for_dims(m);

    let s = tiny_ccs_id(n, m);
    let l = AjtaiSModule::from_global_for_dims(D, m).unwrap();
    let ell_d = D.next_power_of_two().trailing_zeros() as usize;

    // Split components Z_i
    let k = 3usize;
    let mut Z_split = Vec::with_capacity(k);
    for i in 0..k {
        let val = F::from_u64((i as u64) + 1);
        Z_split.push(Mat::from_row_major(D, m, vec![val; D * m]));
    }

    // Parent Z := Σ b^{i-1} Z_i
    let bF = F::from_u64(params.b as u64);
    let mut Z_parent = Mat::zero(D, m, F::ZERO);
    let mut pow = F::ONE;
    for Zi in &Z_split {
        for r_ in 0..D {
            for c_ in 0..m {
                Z_parent.set(r_, c_, Z_parent[(r_, c_)] + pow * Zi[(r_, c_)]);
            }
        }
        pow = pow * bF;
    }

    let r = vec![K::from(F::from_u64(19)); 1];

    // Parent ME from witness
    let w_parent = McsWitness {
        w: vec![],
        Z: Z_parent.clone(),
    };
    let inst_parent = McsInstance {
        c: l.commit(&Z_parent),
        x: vec![],
        m_in: 1,
    };
    let parent = refimpl::build_me_outputs_paper_exact(
        &s,
        &params,
        &[inst_parent.clone()],
        &[w_parent.clone()],
        &[],
        &[],
        &r,
        ell_d,
        [0; 32],
        &l,
    )[0]
    .clone();

    // DEC
    let (children, ok_y, ok_X) = refimpl::dec_reduction_paper_exact::<F>(&s, &params, &parent, &Z_split, ell_d);
    assert!(ok_y && ok_X, "DEC must pass checks");

    // RLC with ρ_i := b^{i-1}·I
    let mut rhos = Vec::with_capacity(k);
    let mut rho_pow = F::ONE;
    for _i in 0..k {
        let mut rho = Mat::identity(D);
        for d_ in 0..D {
            rho.set(d_, d_, rho_pow);
        }
        rhos.push(rho);
        rho_pow = rho_pow * bF;
    }

    // Use the DEC children as inputs to RLC
    let inputs_for_rlc: Vec<_> = children.clone();
    let (combined_me, combined_Z) =
        refimpl::rlc_reduction_paper_exact::<F>(&s, &params, &rhos, &inputs_for_rlc, &Z_split, ell_d);

    // Combined must equal the original parent
    assert!(mat_eq(&combined_Z, &Z_parent), "RLC∘DEC Z must reconstruct parent Z");
    assert!(mat_eq(&combined_me.X, &parent.X), "RLC∘DEC X must reconstruct parent X");
    for j in 0..s.t() {
        assert_eq!(
            combined_me.y[j][..D],
            parent.y[j][..D],
            "RLC∘DEC y_j digits must match (j={})",
            j
        );
        assert!(
            combined_me.y[j][D..].iter().all(|&v| v == K::ZERO),
            "padding after roundtrip remains zero"
        );
    }
}
