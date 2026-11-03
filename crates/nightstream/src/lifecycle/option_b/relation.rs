//! One option-B step circuit's relation as a fold reads it (Lean
//! `LogicalPlan.plan`): the plan rows read through the slot layout, the gate
//! `A · B − C + S^7` (`GatePolynomial.terms`) over the 28-variable cube, and
//! the parameters that its shape allows.
use std::ops::Range;

use crate::folding::{
    has_canonical_evaluations, superneo_has_canonical_x_shape, CcsClaim, CcsInstance, CcsWitness, CeClaim, FoldContext,
    Params, Structure,
};
use neo_ajtai::nightstream_fprime_setup::{commit_production_signed_unit_prefix_matrix, PRODUCTION_VERIFIER_ROWS};
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::{GeometricRowRun, Mat};
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow};
use neo_reductions::PiCcsError;
use nightstream_fprime::{
    LogicalLayout, PackageError, PlanRows, StepPackage, StepWords, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS,
};
use p3_field::PrimeCharacteristicRing;

const MATRICES: usize = 4;

#[derive(Debug, thiserror::Error)]
pub enum RelationError {
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

    /// The logical assignment of a step: the package's witness program fills
    /// the circuit's columns from the step words, the relation's fresh
    /// columns follow, and the slot layout lowers them to digits.
    pub(crate) fn assignment(&self, package: &StepPackage, words: &StepWords) -> Result<Vec<F>, RelationError> {
        let columns = package.circuit_columns(&words.columns()?)?;
        let columns = package.relation_columns(columns, &self.rows)?;
        Ok(self.layout.assignment(&columns)?)
    }

    /// The first row of the relation that a step's columns fail; the
    /// application's rows come first.
    #[cfg(test)]
    pub(crate) fn first_failing_row(
        &self,
        package: &StepPackage,
        words: &StepWords,
    ) -> Result<Option<usize>, RelationError> {
        let columns = package.circuit_columns(&words.columns()?)?;
        let columns = package.relation_columns(columns, &self.rows)?;
        Ok((0..self.rows.row_count()).find(|&row| self.rows.residual(row, &columns) != F::ZERO))
    }

    /// The shape that a slot's claim and the state words need: the selected
    /// commitment, the 270 public words, the 28-coordinate point and canonical
    /// evaluations.
    pub(crate) fn shape(&self, claim: &CeClaim) -> Result<(), &'static str> {
        if claim.adv.is_some()
            || claim.c.d() != D
            || claim.c.kappa() != PRODUCTION_VERIFIER_ROWS as usize
            || claim.m_in != PUBLIC_WORDS
            || !superneo_has_canonical_x_shape(&claim.X, claim.m_in)
            || claim.r.len() != self.structure.cube_variables()
            || !has_canonical_evaluations(claim, MATRICES)
        {
            return Err("claim shape");
        }
        Ok(())
    }

    /// `CCS.Holds` for a fresh instance: its witness is one logical
    /// assignment with a zero tail, its public prefix is `x`, the production
    /// key commits it to `c` (so every entry is in {-1, 0, 1}), and every row
    /// of the relation holds on it.
    pub(crate) fn holds(&self, instance: &CcsInstance) -> Result<(), &'static str> {
        let CcsInstance { claim, witness } = instance;
        let z = &witness.Z;
        if claim.adv.is_some() || claim.m_in != PUBLIC_WORDS || claim.x.len() != PUBLIC_WORDS {
            return Err("claim shape");
        }
        if z.rows() != D || z.cols() != self.blocks() {
            return Err("witness shape");
        }
        let logical: Vec<F> = (0..self.blocks() * D)
            .map(|column| z[(column % D, column / D)])
            .collect();
        let width = self.layout.width();
        if logical[width..].iter().any(|value| *value != F::ZERO) {
            return Err("nonzero witness tail");
        }
        if logical[..PUBLIC_WORDS] != claim.x[..] {
            return Err("witness public projection differs from x");
        }
        if commit_production_signed_unit_prefix_matrix(z).map_err(|_| "witness outside {-1, 0, 1}")? != claim.c {
            return Err("fixed-key commitment differs from the witness");
        }
        let port = |row, port| {
            self.layout
                .evaluate(&self.rows, row, port, &logical[..width])
        };
        for row in 0..self.rows.row_count() {
            let [a, b, c, s] = [0, 1, 2, 3].map(|index| port(row, index));
            let (Ok(a), Ok(b), Ok(c), Ok(s)) = (a, b, c, s) else {
                return Err("row evaluation");
            };
            if a * b - c + s.exp_u64(7) != F::ZERO {
                return Err("a relation row fails");
            }
        }
        Ok(())
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
