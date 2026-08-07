//! CE child/running continuity behavior tests.

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_fold_clean::engine::decider::__test_isolation::{enforce_ce_continuity_against_self, CeContinuityProbeWires};
use neo_fold_clean::engine::r1cs_circuit::R1csBuilder;
use neo_fold_clean::paper::params::Params;
use neo_fold_clean::paper::relations::CeClaim;
use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

fn f(value: u64) -> F {
    F::from_u64(value)
}

fn k(c0: u64, c1: u64) -> K {
    K::from_coeffs([f(c0), f(c1)])
}

fn claim_fixture() -> CeClaim {
    let mut x = Mat::zero(D, 1, F::ZERO);
    for row in 0..D {
        x[(row, 0)] = f(100 + row as u64);
    }
    let kappa = Params::production().kappa() as usize;
    let d_pad = D.next_power_of_two();
    let mut y_ring = (0..d_pad)
        .map(|idx| k(11 + idx as u64 * 2, 12 + idx as u64 * 2))
        .collect::<Vec<_>>();
    for lane in D..d_pad {
        y_ring[lane] = K::ZERO;
    }
    CeClaim {
        adv: None,
        c: Commitment {
            d: D,
            kappa,
            data: (0..D * kappa).map(|idx| f(10 + idx as u64)).collect(),
        },
        X: x,
        r: vec![k(1, 2)],
        y_ring: vec![y_ring.clone()],
        ct: vec![y_ring[0]],
        m_in: D,
        fold_digest: [42u8; 32],
    }
}

fn build() -> (R1csBuilder, CeContinuityProbeWires) {
    enforce_ce_continuity_against_self(&claim_fixture()).expect("emit CE continuity")
}

fn equality_pairs(builder: &R1csBuilder) -> Vec<(usize, usize)> {
    let (a, b, c) = builder.sparse_triplets();
    (0..builder.rows())
        .filter_map(|row| {
            let a_terms = a
                .iter()
                .filter(|&&(candidate, _, _)| candidate == row)
                .map(|&(_, column, coefficient)| (column, coefficient))
                .collect::<Vec<_>>();
            let b_terms = b
                .iter()
                .filter(|&&(candidate, _, _)| candidate == row)
                .map(|&(_, column, coefficient)| (column, coefficient))
                .collect::<Vec<_>>();
            assert_eq!(b_terms, vec![(0, F::ONE)]);
            assert!(c.iter().all(|&(candidate, _, _)| candidate != row));
            match a_terms.as_slice() {
                [(_, coefficient)] => {
                    assert_eq!(*coefficient, F::ONE);
                    None
                }
                [(left, left_coefficient), (right, right_coefficient)] if *right == 0 => {
                    assert_eq!(*left_coefficient, F::ONE);
                    assert_ne!(*right_coefficient, F::ZERO);
                    None
                }
                [(left, left_coefficient), (right, right_coefficient)] => {
                    assert_eq!(*left_coefficient, F::ONE);
                    assert_eq!(*right_coefficient, -F::ONE);
                    Some((*left, *right))
                }
                _ => panic!("CE continuity row {row} is neither a constant pin nor direct equality"),
            }
        })
        .collect()
}

#[test]
fn ce_continuity_accepts_honest_and_has_only_direct_equalities() {
    let (builder, _) = build();
    let pairs = equality_pairs(&builder);
    assert!(!pairs.is_empty());
    assert!(pairs.len() < builder.rows(), "shape metadata pins must remain explicit");
    assert!(builder.unconstrained_columns().is_empty());
    assert!(builder.is_satisfied());
}

#[test]
fn ce_continuity_rejects_each_authority_family() {
    let selectors: [fn(&CeContinuityProbeWires) -> usize; 12] = [
        |w| w.c_data0.col(),
        |w| w.x0.col(),
        |w| w.c_d.col(),
        |w| w.c_kappa.col(),
        |w| w.x_rows.col(),
        |w| w.x_cols.col(),
        |w| w.m_in.col(),
        |w| w.r_c0.col(),
        |w| w.r_c1.col(),
        |w| w.ct_c1.col(),
        |w| w.y_ring_c1.col(),
        |w| w.fold_digest0.col(),
    ];
    for select in selectors {
        let (mut builder, probes) = build();
        let column = select(&probes);
        builder.tamper_witness(column, builder.witness()[column] + F::ONE);
        assert!(!builder.is_satisfied(), "CE continuity disconnected column {column}");
    }
}
