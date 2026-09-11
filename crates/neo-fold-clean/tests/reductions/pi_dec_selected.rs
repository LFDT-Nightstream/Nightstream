//! Low-level selected-key D test. Empty matrix slots do not model the selected package.

use neo_ajtai::{
    nightstream_fprime_setup::{
        coefficient, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
    },
    Commitment,
};
use neo_ccs::{CcsStructure, CeClaim, Mat, SparsePoly};
use neo_math::{superneo_bar_block, KExtensions, Rq, D, F, K};
use neo_reductions::superneo_eval::SuperneoEvalCacheBuilder;
use nightstream_fprime::PI_CCS_V1_1_MATRIX_COUNT;
use p3_field::PrimeCharacteristicRing;

use super::{prove_with_production_key, verify};
use crate::paper::{params::Params, relations::ajtai_dec_mixer};

fn weight(point: &[K], row: usize) -> K {
    point
        .iter()
        .enumerate()
        .fold(K::ONE, |product, (bit, &coordinate)| {
            product
                * if row & (1 << bit) == 0 {
                    K::ONE - coordinate
                } else {
                    coordinate
                }
        })
}

#[test]
fn selected_d_wrapper_computes_sparse_parent_children() {
    let started = std::time::Instant::now();
    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let matrix_count = PI_CCS_V1_1_MATRIX_COUNT;
    // This is a shape-only low-level header. The accompanying row visitor
    // supplies every actual matrix row (all zero here), not an artifact digest.
    let structure = CcsStructure::new_verifier_artifact_header(
        1,
        PRODUCTION_CARRIER_WIDTH,
        matrix_count,
        SparsePoly::new(matrix_count, vec![]),
    )
    .unwrap();
    let mut builder = SuperneoEvalCacheBuilder::new(1, structure.m, matrix_count).unwrap();
    for matrix in 0..matrix_count {
        builder.push_row(matrix, 0, []).unwrap();
    }
    let cache = builder.finish().unwrap();
    let params = Params::for_ccs_shape(structure.n, structure.m, structure.t(), structure.max_degree()).unwrap();
    let count = params.k_rho() as usize;
    let positive = params.big_b() - 1;
    let negative = params.big_b() / 2 + 1;
    // The normal R output is dense F storage. Only the first and last blocks
    // contain nonzero coefficients; no dense matrix or key is constructed.
    let mut witness = Mat::zero(D, columns, F::ZERO);
    witness[(0, 0)] = F::from_u64(positive);
    witness[(D - 1, columns - 1)] = -F::from_u64(negative);
    let support = [
        (0, 0, F::from_u64(positive)),
        (columns - 1, D - 1, -F::from_u64(negative)),
    ];
    let point = (0..PRODUCTION_CARRIER_WIDTH
        .next_power_of_two()
        .trailing_zeros())
        .map(|index| K::from_coeffs([F::from_u64(u64::from(index) + 3), F::ONE]))
        .collect::<Vec<_>>();
    let mut commitment = Commitment::zeros(D, PRODUCTION_VERIFIER_ROWS as usize);
    let mut eval_k = vec![K::ZERO; D.next_power_of_two()];
    for (block, lane, coefficient_value) in support {
        let mut value = [F::ZERO; D];
        value[lane] = coefficient_value;
        for row in 0..PRODUCTION_VERIFIER_ROWS as usize {
            let key = Rq(std::array::from_fn(|index| {
                F::from_u64(coefficient(&PRODUCTION_SEED, row as u32, block as u64, index as u32))
            }));
            let product = key.mul(&Rq(value));
            for index in 0..D {
                commitment.data[row * D + index] += product.0[index];
            }
        }
        for local in 0..D {
            let mut basis = [F::ZERO; D];
            basis[local] = F::ONE;
            let product = Rq(superneo_bar_block(basis)).mul(&Rq(value));
            for index in 0..D {
                eval_k[index] += weight(&point, block * D + local) * K::from(product.0[index]);
            }
        }
    }
    let mut public = Mat::zero(D, 1, F::ZERO);
    public[(0, 0)] = F::from_u64(positive);
    let parent = CeClaim {
        c: commitment,
        X: public,
        r: point,
        eval_k,
        eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; matrix_count],
        m_in: D,
        fold_digest: [0; 32],
        adv: None,
    };
    let (children, proof) = prove_with_production_key(&params, &structure, &cache, &parent, &witness).unwrap();
    assert_eq!(children.claims, proof.children);
    assert_eq!(children.witnesses.len(), count);
    for (index, digit) in children.witnesses.iter().enumerate() {
        let negative_bit = (negative >> index) & 1;
        assert_eq!(digit[(0, 0)], F::ONE);
        assert_eq!(digit[(D - 1, columns - 1)], -F::from_u64(negative_bit));
        assert_eq!(
            digit.packed_signed_unit_nonzero_count(),
            Some(1 + negative_bit as usize)
        );
        assert!(digit.packed_signed_unit_column_masks().is_some());
        assert!(children.claims[index]
            .eval_a
            .iter()
            .flatten()
            .all(|value| *value == K::ZERO));
    }
    assert_eq!(
        verify(&params, &structure, ajtai_dec_mixer, &parent, &proof).unwrap(),
        children.claims
    );
    let mut changed = proof.clone();
    changed.children[0].c.data[0] += F::ONE;
    assert!(verify(&params, &structure, ajtai_dec_mixer, &parent, &changed).is_err());
    witness[(0, 0)] = F::from_u64(params.big_b());
    assert!(prove_with_production_key(&params, &structure, &cache, &parent, &witness).is_err());
    println!("selected_sparse_d_wrapper_elapsed={:?}", started.elapsed());
}
