//! One bounded-depth step circuit's relation as a fold reads it (Lean
//! `LogicalPlan.plan`): the plan rows read through the slot layout, in
//! factored form (each entry reads one slot, `SlotLayout.layout`), the gate
//! `A · B − C + S^7` (`GatePolynomial.terms`) over the 28-variable cube, and
//! the parameters that its shape allows.
use rayon::prelude::*;
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};

use crate::engine::{Backend, CommitError, DeviceError, EngineError};
use crate::folding::{
    has_canonical_evaluations, superneo_has_canonical_x_shape, CcsClaim, CcsInstance, CcsWitness, CeClaim, FoldContext,
    Params, Structure,
};
use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{
    FactoredEntries, FactoredRows, FactoredSlots, MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow,
    RetainedMatrixWindow,
};
use neo_reductions::PiCcsError;
use nightstream_fprime::{
    LogicalLayout, PackageError, PlanRows, SlotSpans, StepPackage, StepWords, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS,
};
use p3_field::PrimeCharacteristicRing;

const MATRICES: usize = 4;
const ROWS_PER_RANGE: usize = 1024;

#[derive(Debug, thiserror::Error)]
pub enum RelationError {
    #[error(transparent)]
    Package(#[from] PackageError),
    #[error(transparent)]
    Parameters(#[from] neo_params::ParamsError),
    #[error("bounded-depth relation rows: {0}")]
    Rows(#[from] PiCcsError),
    #[error(transparent)]
    Commitment(#[from] CommitError),
    #[error("bounded-depth relation header: {0}")]
    Header(#[from] neo_ccs::RelationError),
    #[error("bounded-depth logical assignment has {got} columns, expected {expected}")]
    Width { expected: usize, got: usize },
}

/// Why `Relation::holds` did not accept an instance: a check failed, or the
/// engine failed and judged nothing.
#[derive(Debug)]
pub(crate) enum Holds {
    Fails(&'static str),
    Engine(EngineError),
}

pub(crate) struct Relation {
    rows: PlanRows,
    layout: LogicalLayout,
    spans: SlotSpans,
    structure: Structure,
    params: Params,
    workspace_bytes: usize,
    /// The compact matrices of every row, kept after their first complete
    /// load: each fold, opening and row check reads them again.
    window: RetainedMatrixWindow,
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
            spans: layout.spans(),
            rows,
            layout,
            structure,
            params,
            workspace_bytes: 0,
            window: RetainedMatrixWindow::default(),
        };
        relation.factored_rows().checked()?;
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
    /// key. The key commits entries in {-1, 0, 1} only, so the matrix holds
    /// one mask of the 1 lanes and one of the -1 lanes for each block (Lean
    /// `FreshInstance.Masks`, `FreshInstance.built`).
    pub(crate) fn instance(&self, backend: &Backend, logical: &[F]) -> Result<CcsInstance, RelationError> {
        if logical.len() != self.layout.width() {
            return Err(RelationError::Width {
                expected: self.layout.width(),
                got: logical.len(),
            });
        }
        let masks = |block: &[F]| {
            let (mut positive, mut negative) = (0u64, 0u64);
            for (lane, &value) in block.iter().enumerate() {
                if value == F::ONE {
                    positive |= 1 << lane;
                } else if value == -F::ONE {
                    negative |= 1 << lane;
                } else if value != F::ZERO {
                    let error = neo_ajtai::AjtaiError::RangeViolation {
                        value: neo_math::balanced::to_balanced_i128(value),
                        bound: 2,
                    };
                    return Err(CommitError::Input(error.to_string()));
                }
            }
            Ok((positive, negative))
        };
        let (positive, negative): (Vec<u64>, Vec<u64>) = logical
            .par_chunks(D)
            .map(masks)
            .collect::<Result<Vec<_>, _>>()?
            .into_iter()
            .unzip();
        let witness = Mat::compact_signed_unit_from_column_masks(D, self.blocks(), &positive, &negative)
            .expect("one disjoint mask pair of D lanes for each block");
        let commitment = backend.commit(&[&witness])?.remove(0);
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
    pub(crate) fn assignment(
        &self,
        package: &StepPackage,
        words: &StepWords,
        locals: Option<&[F]>,
    ) -> Result<Vec<F>, RelationError> {
        let columns = package.circuit_columns_with(&words.columns()?, locals)?;
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
    /// commitment, the public words, the 28-coordinate point and canonical
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

    /// Lean `CCS.Holds` for a fresh instance, and a zero tail that Lean does
    /// not require (SPEC.md Algorithm 6): its witness is one logical
    /// assignment with a zero tail, its public prefix is `x`, the production
    /// key commits it to `c` (so every entry is in {-1, 0, 1}), and every row
    /// of the relation holds on it.
    pub(crate) fn holds(&self, backend: &Backend, instance: &CcsInstance) -> Result<(), Holds> {
        let CcsInstance { claim, witness } = instance;
        let z = &witness.Z;
        if claim.adv.is_some() || claim.m_in != PUBLIC_WORDS || claim.x.len() != PUBLIC_WORDS {
            return Err(Holds::Fails("claim shape"));
        }
        if z.rows() != D || z.cols() != self.blocks() {
            return Err(Holds::Fails("witness shape"));
        }
        let logical: Vec<F> = match z.packed_signed_unit_column_masks() {
            Some((positive, negative)) => positive
                .iter()
                .zip(negative)
                .flat_map(|(&positive, &negative)| {
                    (0..D).map(move |lane| F::from_u64((positive >> lane) & 1) - F::from_u64((negative >> lane) & 1))
                })
                .collect(),
            None => (0..self.blocks() * D)
                .map(|column| z[(column % D, column / D)])
                .collect(),
        };
        let width = self.layout.width();
        if logical[width..].iter().any(|value| *value != F::ZERO) {
            return Err(Holds::Fails("nonzero witness tail"));
        }
        if logical[..PUBLIC_WORDS] != claim.x[..] {
            return Err(Holds::Fails("witness public projection differs from x"));
        }
        let commitment = match backend.commit(&[z]) {
            Ok(mut commitments) => commitments.remove(0),
            Err(CommitError::Input(_)) => return Err(Holds::Fails("witness outside {-1, 0, 1}")),
            Err(CommitError::Engine(error)) => return Err(Holds::Engine(error)),
        };
        if commitment != claim.c {
            return Err(Holds::Fails("fixed-key commitment differs from the witness"));
        }
        if let Some(first) = backend.first_failing_row(self, self.workspace_bytes, &self.structure, z) {
            return match first {
                Ok(None) => Ok(()),
                Ok(Some(_)) => Err(Holds::Fails("a relation row fails")),
                Err(DeviceError::Engine(error)) => Err(Holds::Engine(error)),
                Err(DeviceError::Rejected(_)) => Err(Holds::Fails("row evaluation")),
            };
        }
        self.rows_hold(&logical[..width]).map_err(Holds::Fails)
    }

    /// Every row of the relation holds on the logical assignment `logical`.
    pub(crate) fn rows_hold(&self, logical: &[F]) -> Result<(), &'static str> {
        let holds = |row: usize| {
            let [a, b, c, s] = [0, 1, 2, 3].map(|port| self.layout.evaluate(&self.rows, row, port, logical));
            let (Ok(a), Ok(b), Ok(c), Ok(s)) = (a, b, c, s) else {
                return Err("row evaluation");
            };
            if a * b - c + s.exp_u64(7) != F::ZERO {
                return Err("a relation row fails");
            }
            Ok(())
        };
        let count = self.rows.row_count();
        let next = AtomicUsize::new(0);
        // The cost of a row varies a lot along the rows, so each worker takes
        // the next small range until none is left.
        let first_failure = || loop {
            let start = next.fetch_add(ROWS_PER_RANGE, Ordering::Relaxed);
            if start >= count {
                return None;
            }
            for row in start..(start + ROWS_PER_RANGE).min(count) {
                if let Err(error) = holds(row) {
                    return Some((row, error));
                }
            }
        };
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads).map(|_| scope.spawn(first_failure)).collect();
            let failures = workers
                .into_iter()
                .map(|worker| worker.join().map_err(|_| "row worker panicked"))
                .collect::<Result<Vec<_>, _>>()?;
            failures
                .into_iter()
                .flatten()
                .min_by_key(|&(row, _)| row)
                .map_or(Ok(()), |(_, error)| Err(error))
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
        self.factored_rows().visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        Ok(Some(self.factored_rows()))
    }

    fn retained_window(&self) -> Option<&RetainedMatrixWindow> {
        Some(&self.window)
    }
}

impl Relation {
    /// The factored rows. `new` checks them once, and they do not change.
    fn factored_rows(&self) -> FactoredRows<'_> {
        FactoredRows::new_unchecked(
            MatrixRows::shape(self),
            FactoredEntries {
                offsets: self.rows.offsets(),
                slots: self.rows.columns(),
                coefficients: self.rows.coefficients(),
            },
            FactoredSlots {
                starts: &self.spans.starts,
                widths: &self.spans.widths,
                ratio: self.spans.ratio,
            },
        )
    }
}
