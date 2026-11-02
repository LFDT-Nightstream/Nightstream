//! One option-B step circuit's relation as a fold reads it (Lean
//! `LogicalPlan.plan`): the plan rows read through the slot layout, the gate
//! `A · B − C + S^7` (`GatePolynomial.terms`) over the 28-variable cube, and
//! the parameters that its shape allows.
use std::ops::Range;

use crate::folding::{CcsClaim, CcsInstance, CcsWitness, FoldContext, Params, Structure};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{GeometricRowRun, Mat};
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow};
use neo_reductions::PiCcsError;
use nightstream_fprime::{LogicalLayout, PackageError, PlanRows, StepPackage, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS};
use p3_field::PrimeCharacteristicRing;

const MATRICES: usize = 4;

#[derive(Debug, thiserror::Error)]
pub(crate) enum RelationError {
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
    #[error("option-B relation rows: {0}")]
    Rows(#[from] PiCcsError),
    #[error(transparent)]
    Commitment(#[from] neo_ajtai::AjtaiError),
    #[error("option-B relation header: {0}")]
    Header(#[from] neo_ccs::RelationError),
    #[error("option-B logical assignment has {got} columns, expected {expected}")]
    Width { expected: usize, got: usize },
}

pub(crate) struct Relation {
    rows: PlanRows,
    layout: LogicalLayout,
    structure: Structure,
    params: Params,
    workspace_bytes: usize,
}

impl Relation {
    pub(crate) fn new(package: &StepPackage, minimum_security_bits: u32) -> Result<Self, RelationError> {
        let rows = package.plan_rows()?;
        let layout = package.layout()?;
        let gate = SparsePoly::new(
            MATRICES,
            vec![
                Term {
                    coeff: F::ONE,
                    exps: vec![1, 1, 0, 0],
                },
                Term {
                    coeff: -F::ONE,
                    exps: vec![0, 0, 1, 0],
                },
                Term {
                    coeff: F::ONE,
                    exps: vec![0, 0, 0, 7],
                },
            ],
        );
        let structure = Structure::new_verifier_artifact_header(rows.row_count(), layout.width(), MATRICES, gate)?
            .with_domain_variables(PI_CCS_V1_2_ROUND_COUNT)?;
        let params = Params::for_ccs_shape(
            structure.domain_rows(),
            structure.m(),
            structure.t(),
            structure.max_degree(),
            minimum_security_bits,
        )?;
        let mut relation = Self {
            rows,
            layout,
            structure,
            params,
            workspace_bytes: 0,
        };
        let payload = 2 * MATRICES * size_of::<K>() + size_of::<F>() + size_of::<K>();
        relation.workspace_bytes =
            MatrixWindow::required_workspace(&relation, 0..relation.rows.row_count(), payload)? * 2;
        Ok(relation)
    }

    pub(crate) fn context(&self) -> FoldContext<'_> {
        FoldContext {
            params: &self.params,
            structure: &self.structure,
            rows: self,
            workspace_bytes: self.workspace_bytes,
        }
    }

    /// The committed instance of a logical assignment: the public prefix and
    /// the witness matrix, one D-lane column per block, under the production
    /// key.
    pub(crate) fn instance(&self, logical: &[F]) -> Result<CcsInstance, RelationError> {
        if logical.len() != self.layout.width() {
            return Err(RelationError::Width {
                expected: self.layout.width(),
                got: logical.len(),
            });
        }
        let blocks = self.blocks();
        let witness = Mat::from_row_major(
            D,
            blocks,
            (0..D)
                .flat_map(|lane| (0..blocks).map(move |block| block * D + lane))
                .map(|column| logical.get(column).copied().unwrap_or(F::ZERO))
                .collect(),
        );
        let commitment = commit_production_signed_unit_prefix_matrix(&witness)?;
        Ok(CcsInstance {
            claim: CcsClaim {
                c: commitment,
                x: logical[..PUBLIC_WORDS].to_vec(),
                m_in: PUBLIC_WORDS,
                adv: None,
            },
            witness: CcsWitness { Z: witness },
        })
    }

    pub(crate) fn blocks(&self) -> usize {
        self.layout.width().div_ceil(D)
    }

    pub(crate) fn params(&self) -> &Params {
        &self.params
    }

    pub(crate) fn structure(&self) -> &Structure {
        &self.structure
    }
}

impl MatrixRows for Relation {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.rows.row_count(),
            columns: self.blocks() * D,
            matrices: MATRICES,
        }
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        for row in rows {
            for port in 0..MATRICES {
                let mut failure = None;
                self.layout
                    .visit_runs(&self.rows, row, port, |run| {
                        if failure.is_none() {
                            let run = GeometricRowRun::new(row, run.column, run.count, run.coefficient, run.ratio);
                            failure = sink.push_run(row, port, run).err();
                        }
                    })
                    .map_err(|error| PiCcsError::InvalidInput(error.to_string()))?;
                if let Some(error) = failure {
                    return Err(error);
                }
                if sink.finish_matrix_row(row, port)?.is_break() {
                    return Ok(());
                }
            }
        }
        Ok(())
    }
}
