//! Original matrix coefficients for the direct paper evaluator.
//! Export readers supply rows; ring products and multilinear evaluation stay here.

use neo_ccs::CcsStructure;
use neo_math::{Fq, D, K};
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};

use crate::PiCcsError;

use super::{paper_joint, paper_ring::PaperRing};

/// A prepared source of original, untransformed CCS coefficients.
/// The circuit owner must validate the rows before providing this view.
/// This is prover input, not an authority for the verifier's circuit.
pub trait PaperMatrixRows<Ff>: Sync {
    /// Active row count, logical column count, and matrix count.
    fn shape(&self) -> (usize, usize, usize);

    /// Return every nonzero term in one original matrix row.
    fn row(&self, matrix: usize, row: usize) -> Vec<(usize, Ff)>;
}

pub(super) struct Matrices<'a, Ff> {
    structure: &'a CcsStructure<Ff>,
    rows: Option<&'a dyn PaperMatrixRows<Ff>>,
}

impl<'a, Ff> Matrices<'a, Ff>
where
    Ff: Field + PrimeCharacteristicRing + PrimeField64 + Copy,
    K: From<Ff>,
{
    pub(super) fn new(
        structure: &'a CcsStructure<Ff>,
        rows: Option<&'a dyn PaperMatrixRows<Ff>>,
    ) -> Result<Self, PiCcsError> {
        if rows.is_some_and(|rows| rows.shape() != (structure.n, structure.m, structure.t())) {
            return Err(PiCcsError::InvalidInput(
                "PaperExact row source differs from the circuit dimensions".into(),
            ));
        }
        if rows.is_none()
            && structure
                .matrices
                .iter()
                .any(|matrix| matches!(matrix, neo_ccs::CcsMatrix::VerifierArtifact { .. }))
        {
            return Err(PiCcsError::InvalidInput(
                "PaperExact needs original rows for a verifier-artifact header".into(),
            ));
        }
        Ok(Self { structure, rows })
    }

    pub(super) fn evaluate(&self, ring: &PaperRing, matrix: usize, assignment: &[K], point: &[K]) -> [K; D] {
        let Some(rows) = self.rows else {
            return paper_joint::direct_ring_mle(ring, &self.structure.matrices[matrix], assignment, point);
        };
        let mut output = [K::ZERO; D];
        for row in 0..self.structure.n {
            let entries = rows.row(matrix, row);
            let weight = paper_joint::boolean_weight(point, row);
            for block in 0..assignment.len().div_ceil(D) {
                let coefficients = core::array::from_fn(|lane| {
                    let column = block * D + lane;
                    entries
                        .iter()
                        .filter(|(index, _)| *index == column)
                        .fold(Fq::ZERO, |sum, (_, value)| sum + Fq::from_u64(value.as_canonical_u64()))
                });
                let product = paper_joint::ring_product(ring, coefficients, assignment, block);
                for coefficient in 0..D {
                    output[coefficient] += weight * product[coefficient];
                }
            }
        }
        output
    }
}
