//! Direct paper reference for PiRLC and PiDEC.
//!
//! PiRLC reads the raw quotient-ring coefficient of each challenge and uses
//! schoolbook ring multiplication. PiDEC independently checks the canonical
//! signed `split_(b,k)` map before it constructs child evaluation claims.

#![allow(non_snake_case)]

use neo_ajtai::Commitment as Cmt;
use neo_ccs::{CcsStructure, CeClaim, Mat};
use neo_math::{Fq, D, ETA, K};
use neo_params::NeoParams;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use super::paper_ring::PaperRing;

fn read_rho<Ff>(params: &NeoParams, matrix: &Mat<Ff>) -> [Fq; D]
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    assert_eq!(params.d as usize, D, "PaperExact PiRLC ring degree mismatch");
    assert_eq!(params.eta as usize, ETA, "PaperExact PiRLC cyclotomic mismatch");
    assert_eq!(matrix.rows(), D, "PaperExact PiRLC rho row count");
    assert_eq!(matrix.cols(), D, "PaperExact PiRLC rho column count");

    let coefficients = core::array::from_fn(|row| Fq::from_u64(matrix[(row, 0)].as_canonical_u64()));
    let mut column = coefficients;
    for column_index in 0..D {
        for row in 0..D {
            assert_eq!(
                matrix[(row, column_index)].as_canonical_u64(),
                column[row].as_canonical_u64(),
                "PaperExact PiRLC rho is not quotient-ring multiplication"
            );
        }
        let last = column[D - 1];
        let mut next = [Fq::ZERO; D];
        next[0] = -last;
        next[1..].copy_from_slice(&column[..D - 1]);
        next[27] -= last;
        column = next;
    }
    coefficients
}

fn base_block<Ff>(matrix: &Mat<Ff>, column: usize) -> [Fq; D]
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    core::array::from_fn(|row| Fq::from_u64(matrix[(row, column)].as_canonical_u64()))
}

fn extension_block(values: &[K]) -> [K; D] {
    core::array::from_fn(|coefficient| values[coefficient])
}

fn validate_claim<Ff>(structure: &CcsStructure<Ff>, claim: &CeClaim<Cmt, Ff, K>, ell_d: usize)
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    assert_eq!(ell_d, D.next_power_of_two().trailing_zeros() as usize);
    assert_eq!(claim.m_in % D, 0, "PaperExact requires whole-ring public inputs");
    assert_eq!(claim.X.rows(), D);
    assert_eq!(claim.X.cols(), claim.m_in);
    assert_eq!(claim.y_ring.len(), structure.t() + 1);
    assert_eq!(claim.ct.len(), structure.t() + 1);
    for (matrix, image) in claim.y_ring.iter().enumerate() {
        assert_eq!(image.len(), D.next_power_of_two());
        assert_eq!(image[0], claim.ct[matrix]);
        assert!(image.iter().skip(D).all(|&value| value == K::ZERO));
    }
}

/// Compute the public PaperExact PiRLC claim without a prover witness.
#[doc(hidden)]
pub fn rlc_claim_paper_exact_with_commit_mix<Ff, Combine>(
    structure: &CcsStructure<Ff>,
    params: &NeoParams,
    rhos: &[Mat<Ff>],
    inputs: &[CeClaim<Cmt, Ff, K>],
    ell_d: usize,
    combine: Combine,
) -> CeClaim<Cmt, Ff, K>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy + Send + Sync,
    Combine: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    assert!(!inputs.is_empty(), "PaperExact PiRLC needs at least one source");
    assert_eq!(rhos.len(), inputs.len());
    let ring = PaperRing::new();
    let coefficients: Vec<[Fq; D]> = rhos.iter().map(|rho| read_rho(params, rho)).collect();
    let m_in = inputs[0].m_in;
    let point = inputs[0].r.clone();
    for claim in inputs {
        validate_claim(structure, claim, ell_d);
        assert_eq!(claim.m_in, m_in);
        assert_eq!(claim.r, point);
    }

    let mut X = Mat::zero(D, m_in, Ff::ZERO);
    for (rho, input) in coefficients.iter().zip(inputs) {
        for column in 0..m_in {
            let product = ring.multiply_base(*rho, base_block(&input.X, column));
            for row in 0..D {
                X[(row, column)] += Ff::from_u64(product[row].as_canonical_u64());
            }
        }
    }

    let matrix_count = structure.t() + 1;
    let mut y_ring = vec![vec![K::ZERO; D.next_power_of_two()]; matrix_count];
    for (rho, input) in coefficients.iter().zip(inputs) {
        let rho_extension = rho.map(K::from);
        for (matrix, output) in y_ring.iter_mut().enumerate() {
            let product = ring.multiply_extension(rho_extension, extension_block(&input.y_ring[matrix]));
            for coefficient in 0..D {
                output[coefficient] += product[coefficient];
            }
        }
    }
    let ct = y_ring.iter().map(|image| image[0]).collect();
    let commitments: Vec<Cmt> = inputs.iter().map(|input| input.c.clone()).collect();
    CeClaim {
        c: combine(rhos, &commitments),
        X,
        r: point,
        y_ring,
        ct,
        m_in,
        fold_digest: inputs[0].fold_digest,
        adv: None,
    }
}

/// Apply the complete paper PiRLC relation and its commitment action.
#[doc(hidden)]
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
    Combine: Fn(&[Mat<Ff>], &[Cmt]) -> Cmt,
{
    assert_eq!(witnesses.len(), inputs.len());
    let output = rlc_claim_paper_exact_with_commit_mix(structure, params, rhos, inputs, ell_d, combine);
    let ring = PaperRing::new();
    let coefficients: Vec<[Fq; D]> = rhos.iter().map(|rho| read_rho(params, rho)).collect();
    let witness_columns = structure.m.div_ceil(D);
    let mut mixed_witness = Mat::zero(D, witness_columns, Ff::ZERO);
    for (rho, witness) in coefficients.iter().zip(witnesses) {
        assert_eq!(witness.rows(), D);
        assert_eq!(witness.cols(), witness_columns);
        for column in 0..witness_columns {
            let product = ring.multiply_base(*rho, base_block(witness, column));
            for row in 0..D {
                mixed_witness[(row, column)] += Ff::from_u64(product[row].as_canonical_u64());
            }
        }
    }
    (output, mixed_witness)
}

fn centered<Ff>(value: Ff) -> i128
where
    Ff: PrimeField64,
{
    let canonical = value.as_canonical_u64() as i128;
    let modulus = Ff::ORDER_U64 as i128;
    if canonical <= (modulus - 1) / 2 {
        canonical
    } else {
        canonical - modulus
    }
}

fn signed_field<Ff>(value: i128) -> Ff
where
    Ff: Field + PrimeCharacteristicRing,
{
    if value >= 0 {
        Ff::from_u64(value as u64)
    } else {
        -Ff::from_u64((-value) as u64)
    }
}

fn canonical_digits<Ff>(value: Ff, count: usize, base: u32) -> Option<Vec<Ff>>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    let mut remaining = centered(value);
    let base = base as i128;
    let mut digits = Vec::with_capacity(count);
    for _ in 0..count {
        let digit = remaining % base;
        digits.push(signed_field(digit));
        remaining = (remaining - digit) / base;
    }
    (remaining == 0).then_some(digits)
}

fn canonical_split_matches<Ff>(split: &[Mat<Ff>], base: u32) -> bool
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
{
    if split.is_empty() || base < 2 {
        return false;
    }
    let rows = split[0].rows();
    let columns = split[0].cols();
    if split
        .iter()
        .any(|matrix| matrix.rows() != rows || matrix.cols() != columns)
    {
        return false;
    }
    let base_field = Ff::from_u64(base as u64);
    for row in 0..rows {
        for column in 0..columns {
            let mut value = Ff::ZERO;
            let mut power = Ff::ONE;
            for digit in split {
                value += power * digit[(row, column)];
                power *= base_field;
            }
            let Some(expected) = canonical_digits(value, split.len(), base) else {
                return false;
            };
            if split
                .iter()
                .zip(expected)
                .any(|(matrix, digit)| matrix[(row, column)] != digit)
            {
                return false;
            }
        }
    }
    true
}

/// Apply the complete paper PiDEC relation and check child commitments.
#[doc(hidden)]
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
    validate_claim(structure, parent, ell_d);
    assert_eq!(split_witnesses.len(), params.k_rho as usize);
    assert_eq!(split_witnesses.len(), child_commitments.len());
    assert!(params.b >= 2);
    let split_valid = canonical_split_matches(split_witnesses, params.b);
    let ring = PaperRing::new();
    let matrix_count = structure.t() + 1;
    let mut children = Vec::with_capacity(split_witnesses.len());

    for (witness, commitment) in split_witnesses.iter().zip(child_commitments) {
        assert_eq!(witness.rows(), D);
        assert_eq!(witness.cols(), structure.m.div_ceil(D));
        let assignment: Vec<K> = (0..witness.cols() * D)
            .map(|column| K::from(witness[(column % D, column / D)]))
            .collect();
        let mut y_ring = Vec::with_capacity(matrix_count);
        let mut identity = super::paper_joint::direct_identity_ring_mle(&ring, &assignment, &parent.r).to_vec();
        identity.resize(D.next_power_of_two(), K::ZERO);
        y_ring.push(identity);
        for matrix in &structure.matrices {
            let mut image = super::paper_joint::direct_ring_mle(&ring, matrix, &assignment, &parent.r).to_vec();
            image.resize(D.next_power_of_two(), K::ZERO);
            y_ring.push(image);
        }
        let ct = y_ring.iter().map(|image| image[0]).collect();
        children.push(CeClaim {
            c: commitment.clone(),
            X: super::paper_joint::direct_public_input(witness, structure.m, parent.m_in)
                .expect("validated PaperExact PiDEC public projection"),
            r: parent.r.clone(),
            y_ring,
            ct,
            m_in: parent.m_in,
            fold_digest: parent.fold_digest,
            adv: None,
        });
    }

    let base_f = Ff::from_u64(params.b as u64);
    let base_k = K::from(base_f);
    let mut y_valid = split_valid;
    for matrix in 0..matrix_count {
        for coefficient in 0..D.next_power_of_two() {
            let mut reconstructed = K::ZERO;
            let mut power = K::ONE;
            for child in &children {
                reconstructed += power * child.y_ring[matrix][coefficient];
                power *= base_k;
            }
            y_valid &= reconstructed == parent.y_ring[matrix][coefficient];
        }
    }

    let mut x_valid = split_valid;
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
    let commitment_valid = split_valid && combine(child_commitments, params.b) == parent.c;
    (children, y_valid, x_valid, commitment_valid)
}

/// Verify the public PaperExact PiDEC recomposition with direct loops.
#[doc(hidden)]
pub fn verify_dec_public_paper_exact<Ff, Combine>(
    params: &NeoParams,
    parent: &CeClaim<Cmt, Ff, K>,
    children: &[CeClaim<Cmt, Ff, K>],
    combine: Combine,
) -> bool
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
    K: From<Ff>,
    Combine: Fn(&[Cmt], u32) -> Cmt,
{
    if children.len() != params.k_rho as usize || children.is_empty() {
        return false;
    }
    if children.iter().any(|child| {
        child.X.rows() != parent.X.rows()
            || child.X.cols() != parent.X.cols()
            || child.r != parent.r
            || child.y_ring.len() != parent.y_ring.len()
            || child.ct.len() != parent.ct.len()
            || child.m_in != parent.m_in
            || child.fold_digest != parent.fold_digest
            || child
                .y_ring
                .iter()
                .zip(&parent.y_ring)
                .any(|(child_row, parent_row)| child_row.len() != parent_row.len())
    }) {
        return false;
    }

    let commitments = children
        .iter()
        .map(|child| child.c.clone())
        .collect::<Vec<_>>();
    if combine(&commitments, params.b) != parent.c {
        return false;
    }
    let base_f = Ff::from_u64(params.b as u64);
    let base_k = K::from(base_f);

    for row in 0..parent.X.rows() {
        for column in 0..parent.X.cols() {
            let Some(expected_digits) = canonical_digits(parent.X[(row, column)], children.len(), params.b) else {
                return false;
            };
            if children
                .iter()
                .zip(expected_digits)
                .any(|(child, expected)| child.X[(row, column)] != expected)
            {
                return false;
            }
            let mut value = Ff::ZERO;
            let mut power = Ff::ONE;
            for child in children {
                value += power * child.X[(row, column)];
                power *= base_f;
            }
            if value != parent.X[(row, column)] {
                return false;
            }
        }
    }
    for matrix in 0..parent.y_ring.len() {
        for coefficient in 0..parent.y_ring[matrix].len() {
            let mut value = K::ZERO;
            let mut power = K::ONE;
            for child in children {
                value += power * child.y_ring[matrix][coefficient];
                power *= base_k;
            }
            if value != parent.y_ring[matrix][coefficient] {
                return false;
            }
        }
    }
    for coordinate in 0..parent.ct.len() {
        let mut value = K::ZERO;
        let mut power = K::ONE;
        for child in children {
            value += power * child.ct[coordinate];
            power *= base_k;
        }
        if value != parent.ct[coordinate] {
            return false;
        }
    }
    true
}
