//! Direct paper reference for the RLC and DEC reductions.
//!
//! This file uses explicit matrix and vector loops. It does not use a
//! transformed matrix evaluator or an optimized evaluation cache.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{D, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

fn field_entry<Ff>(matrix: &Mat<Ff>, row: usize, column: usize) -> Ff
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    if row < matrix.rows() && column < matrix.cols() {
        matrix[(row, column)]
    } else {
        Ff::ZERO
    }
}

fn left_multiply_add<Ff>(accumulator: &mut Mat<Ff>, left: &Mat<Ff>, right: &Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + Copy,
{
    for row in 0..accumulator.rows() {
        for column in 0..accumulator.cols() {
            for inner in 0..left.cols() {
                accumulator[(row, column)] += field_entry(left, row, inner) * field_entry(right, inner, column);
            }
        }
    }
}

fn direct_column_opening<Ff>(
    witness: &Mat<Ff>,
    logical_width: usize,
    column_weights: &[K],
    output_width: usize,
) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let mut output = vec![K::ZERO; output_width];
    for column in 0..logical_width {
        output[column % D] += K::from(witness[(column % D, column / D)]) * column_weights[column];
    }
    output
}

/// Apply the paper RLC relation with explicit matrix loops.
pub fn rlc_reduction_paper_exact<Ff>(
    structure: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    inputs: &[CeClaim<Cmt, Ff, K>],
    witnesses: &[Mat<Ff>],
    ell_d: usize,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!inputs.is_empty(), "paper RLC needs at least one input");
    assert_eq!(rhos.len(), inputs.len(), "paper RLC rho count mismatch");
    assert_eq!(witnesses.len(), inputs.len(), "paper RLC witness count mismatch");
    crate::common::validate_rhos_are_rotation_matrices(params, rhos, "paper RLC rhos")
        .unwrap_or_else(|error| panic!("paper RLC invalid rho set: {error}"));

    let d_pad = 1usize << ell_d;
    let witness_columns = witnesses[0].cols();
    let matrix_outputs = inputs[0].y_ring.len();
    let m_in = inputs[0].m_in;
    let wants_column = !inputs[0].s_col.is_empty() || !inputs[0].y_zcol.is_empty();
    for (index, (input, witness)) in inputs.iter().zip(witnesses).enumerate() {
        crate::common::validate_superneo_witness_mat(witness, structure.m)
            .unwrap_or_else(|error| panic!("paper RLC witness {index}: {error}"));
        assert_eq!(witness.cols(), witness_columns, "paper RLC witness width mismatch");
        assert_eq!(input.r, inputs[0].r, "paper RLC row point mismatch");
        assert_eq!(input.m_in, m_in, "paper RLC public-input width mismatch");
        assert_eq!(input.y_ring.len(), matrix_outputs, "paper RLC matrix count mismatch");
        assert_eq!(input.s_col, inputs[0].s_col, "paper RLC column point mismatch");
        assert_eq!(
            !input.s_col.is_empty() || !input.y_zcol.is_empty(),
            wants_column,
            "paper RLC column-channel presence mismatch"
        );
    }

    let mut mixed_witness = Mat::zero(D, witness_columns, Ff::ZERO);
    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for ((rho, input), witness) in rhos.iter().zip(inputs).zip(witnesses) {
        left_multiply_add(&mut mixed_witness, rho, witness);
        left_multiply_add(&mut X, rho, &input.X);
    }

    let mut y_ring = vec![vec![K::ZERO; d_pad]; matrix_outputs];
    for (source, rho) in rhos.iter().enumerate() {
        for (matrix, output) in y_ring.iter_mut().enumerate() {
            for row in 0..D.min(d_pad) {
                for column in 0..D.min(inputs[source].y_ring[matrix].len()) {
                    output[row] += K::from(field_entry(rho, row, column)) * inputs[source].y_ring[matrix][column];
                }
            }
        }
    }

    let mut y_zcol = if wants_column { vec![K::ZERO; d_pad] } else { Vec::new() };
    if wants_column {
        for (source, rho) in rhos.iter().enumerate() {
            assert_eq!(
                inputs[source].y_zcol.len(),
                d_pad,
                "paper RLC column opening width mismatch"
            );
            for row in 0..D {
                for column in 0..D {
                    y_zcol[row] += K::from(field_entry(rho, row, column)) * inputs[source].y_zcol[column];
                }
            }
        }
    }

    let ct = y_ring.iter().map(|output| output[0]).collect();
    let output = CeClaim {
        c: inputs[0].c.clone(),
        X,
        r: inputs[0].r.clone(),
        s_col: inputs[0].s_col.clone(),
        y_ring,
        ct,
        aux_openings: Vec::new(),
        y_zcol,
        m_in,
        fold_digest: inputs[0].fold_digest,
        c_step_coords: Vec::new(),
        u_offset: 0,
        u_len: 0,
        adv: None,
    };
    (output, mixed_witness)
}

fn mix_aux_openings<Ff>(rhos: &[Mat<Ff>], inputs: &[CeClaim<Cmt, Ff, K>]) -> Vec<K>
where
    Ff: Field + PrimeCharacteristicRing + Copy,
    K: From<Ff>,
{
    let width = inputs[0].aux_openings.len();
    let mut output = vec![K::ZERO; width];
    for (source, input) in inputs.iter().enumerate() {
        assert_eq!(input.aux_openings.len(), width, "paper RLC auxiliary width mismatch");
        let weight = K::from(field_entry(&rhos[source], 0, 0));
        for (target, &value) in output.iter_mut().zip(&input.aux_openings) {
            *target += weight * value;
        }
    }
    output
}

/// Apply RLC and replace the placeholder commitment with the supplied linear
/// commitment combination.
pub fn rlc_reduction_paper_exact_with_commit_mix<Ff, Combine>(
    structure: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    inputs: &[CeClaim<Cmt, Ff, K>],
    witnesses: &[Mat<Ff>],
    ell_d: usize,
    combine: Combine,
) -> (CeClaim<Cmt, Ff, K>, Mat<Ff>)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Combine: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    let (mut output, witness) = rlc_reduction_paper_exact(structure, params, rhos, inputs, witnesses, ell_d);
    let commitments: Vec<Cmt> = inputs.iter().map(|input| input.c.clone()).collect();
    output.c = combine(rhos, &commitments);
    output.aux_openings = mix_aux_openings(rhos, inputs);
    (output, witness)
}

/// Apply the paper DEC relation and compute child openings with direct matrix
/// loops.
pub fn dec_reduction_paper_exact<Ff>(
    structure: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    split_witnesses: &[Mat<Ff>],
    ell_d: usize,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
{
    assert!(!split_witnesses.is_empty(), "paper DEC needs child witnesses");
    let d_pad = 1usize << ell_d;
    let row_weights = neo_ccs::utils::tensor_point::<K>(&parent.r);
    let column_weights = neo_ccs::utils::tensor_point::<K>(&parent.s_col);
    let wants_column = !parent.s_col.is_empty() || !parent.y_zcol.is_empty();
    if wants_column {
        assert!(!parent.s_col.is_empty(), "paper DEC parent column point is missing");
        assert_eq!(
            parent.y_zcol.len(),
            d_pad,
            "paper DEC parent column opening width mismatch"
        );
    }

    let mut children = Vec::with_capacity(split_witnesses.len());
    for (index, witness) in split_witnesses.iter().enumerate() {
        let assignment = crate::common::decode_superneo_coeffs_from_witness_mat(witness, structure.m)
            .unwrap_or_else(|error| panic!("paper DEC witness {index}: {error}"));
        let y_ring: Vec<Vec<K>> = structure
            .matrices
            .iter()
            .map(|matrix| {
                let mut coefficients =
                    super::paper_rectangular::direct_ring_mle(matrix, &assignment, &row_weights).to_vec();
                coefficients.resize(d_pad, K::ZERO);
                coefficients
            })
            .collect();
        let ct = y_ring.iter().map(|output| output[0]).collect();
        let y_zcol = if wants_column {
            direct_column_opening(witness, structure.m, &column_weights, d_pad)
        } else {
            Vec::new()
        };
        children.push(CeClaim {
            c: parent.c.clone(),
            X: crate::common::project_x_from_witness_mat(witness, structure.m, parent.m_in)
                .unwrap_or_else(|error| panic!("paper DEC child projection: {error}")),
            r: parent.r.clone(),
            s_col: parent.s_col.clone(),
            y_ring,
            ct,
            aux_openings: Vec::new(),
            y_zcol,
            m_in: parent.m_in,
            fold_digest: parent.fold_digest,
            c_step_coords: Vec::new(),
            u_offset: 0,
            u_len: 0,
            adv: None,
        });
    }

    let base_f = Ff::from_u64(params.b as u64);
    let base_k = K::from(base_f);
    let mut y_valid = parent.y_ring.len() == structure.t();
    for matrix in 0..structure.t() {
        let mut reconstructed = vec![K::ZERO; d_pad];
        let mut power = K::ONE;
        for child in &children {
            for (target, &value) in reconstructed.iter_mut().zip(&child.y_ring[matrix]) {
                *target += power * value;
            }
            power *= base_k;
        }
        y_valid &= parent.y_ring.get(matrix) == Some(&reconstructed);
    }

    let mut x_valid = parent.X.rows() == D && parent.X.cols() == parent.m_in;
    for row in 0..D {
        for column in 0..parent.m_in {
            let mut reconstructed = Ff::ZERO;
            let mut power = Ff::ONE;
            for child in &children {
                reconstructed += power * child.X[(row, column)];
                power *= base_f;
            }
            x_valid &= reconstructed == parent.X[(row, column)];
        }
    }
    (children, y_valid, x_valid)
}

/// Apply DEC and check the supplied child commitments against the parent.
pub fn dec_reduction_paper_exact_with_commit_check<Ff, Combine>(
    structure: &CcsStructure<Ff>,
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    split_witnesses: &[Mat<Ff>],
    ell_d: usize,
    child_commitments: &[Cmt],
    combine: Combine,
) -> (Vec<CeClaim<Cmt, Ff, K>>, bool, bool, bool)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    K: From<Ff>,
    Combine: Fn(&[Cmt], u32) -> Cmt,
{
    let (mut children, y_valid, x_valid) = dec_reduction_paper_exact(structure, params, parent, split_witnesses, ell_d);
    assert_eq!(
        children.len(),
        child_commitments.len(),
        "paper DEC commitment count mismatch"
    );
    for (child, commitment) in children.iter_mut().zip(child_commitments) {
        child.c = commitment.clone();
    }
    for (index, child) in children.iter_mut().enumerate() {
        child.aux_openings = if index == 0 {
            parent.aux_openings.clone()
        } else {
            vec![K::ZERO; parent.aux_openings.len()]
        };
    }
    let commitment_valid = combine(child_commitments, params.b) == parent.c;
    (children, y_valid, x_valid, commitment_valid)
}
