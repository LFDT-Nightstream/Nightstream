//! Lean `LogicalPlan.plan`, `SlotLayout.layout`, `GatePolynomial.terms`.
use std::ops::Range;
use std::sync::atomic::{AtomicUsize, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, TryLockError, Weak};

use crate::engine::{Backend, CommitError, DeviceError, EngineError};
use crate::folding::{
    has_canonical_evaluations, superneo_has_canonical_x_shape, CcsClaim, CcsInstance, CcsWitness, CeClaim, FoldContext,
    Params, Structure,
};
use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use neo_ccs::crypto::poseidon2::SBOX_DEGREE;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::Mat;
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::{
    FactoredEntries, FactoredRows, FactoredSlots, MatrixRowSink, MatrixRows, MatrixShape, MatrixWindow,
    RetainedMatrixWindow,
};
use neo_reductions::PiCcsError;
use nightstream_fprime::{ExpandedPackage, PackageError, SlotSpans, StepWords, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS};
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
}

#[derive(Debug)]
pub(crate) enum Holds {
    Fails(&'static str),
    Engine(EngineError),
}

pub(crate) struct CompiledStep {
    opening_columns: neo_reductions::superneo_eval::OpeningColumns,
    expanded: ExpandedPackage,
    spans: SlotSpans,
    workspace_bytes: usize,
}

impl CompiledStep {
    pub(crate) fn new(expanded: ExpandedPackage) -> Result<Self, RelationError> {
        let mut step = Self {
            opening_columns: Default::default(),
            spans: expanded.layout().spans(),
            expanded,
            workspace_bytes: 0,
        };
        step.factored_rows().checked()?;
        let payload = 2 * MATRICES * size_of::<K>() + size_of::<F>() + size_of::<K>();
        step.workspace_bytes =
            MatrixWindow::required_workspace(&step, 0..step.expanded.rows().row_count(), payload)? * 2;
        Ok(step)
    }

    #[cfg(test)]
    pub(crate) fn expanded(&self) -> &ExpandedPackage {
        &self.expanded
    }

    fn blocks(&self) -> usize {
        self.expanded.layout().width().div_ceil(D)
    }

    fn factored_rows(&self) -> FactoredRows<'_> {
        let rows = self.expanded.rows();
        FactoredRows::new_unchecked(
            MatrixRows::shape(self),
            FactoredEntries {
                offsets: rows.offsets(),
                slots: rows.columns(),
                coefficients: rows.coefficients(),
            },
            FactoredSlots {
                starts: &self.spans.starts,
                widths: &self.spans.widths,
                ratio: self.spans.ratio,
            },
        )
        .with_opening_columns(&self.opening_columns)
    }
}

impl MatrixRows for CompiledStep {
    fn shape(&self) -> MatrixShape {
        MatrixShape {
            rows: self.expanded.rows().row_count(),
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
}

pub(super) struct CheckedFreshRows {
    witness: Mat<F>,
    values: Mutex<Option<Vec<[F; MATRICES]>>>,
}

impl CheckedFreshRows {
    pub(super) fn new(witness: &Mat<F>, values: Vec<[F; MATRICES]>) -> Self {
        Self {
            witness: witness.clone(),
            values: Mutex::new(Some(values)),
        }
    }

    fn matches(&self, witness: &Mat<F>) -> bool {
        self.witness
            .packed_signed_unit_column_masks()
            .zip(witness.packed_signed_unit_column_masks())
            .is_some_and(|((stored_p, stored_n), (p, n))| std::ptr::eq(stored_p, p) && std::ptr::eq(stored_n, n))
    }

    fn take(&self) -> Option<Vec<[F; MATRICES]>> {
        self.values
            .lock()
            .unwrap_or_else(std::sync::PoisonError::into_inner)
            .take()
    }

    #[cfg(test)]
    pub(super) fn retained(&self) -> bool {
        self.values.lock().unwrap().is_some()
    }
}

pub(super) struct FreshMatrixRows<'a> {
    pub(super) relation: &'a Relation,
    pub(super) prepared: Option<&'a CheckedFreshRows>,
    pub(super) witness: &'a Mat<F>,
}

impl MatrixRows for FreshMatrixRows<'_> {
    fn shape(&self) -> MatrixShape {
        MatrixRows::shape(self.relation)
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.relation.visit_rows(rows, sink)
    }

    fn retained_window(&self) -> Option<&RetainedMatrixWindow> {
        self.relation.retained_window()
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        self.relation.factored()
    }

    fn take_checked_fresh_rows(&self) -> Option<Vec<[F; MATRICES]>> {
        let prepared = self.prepared?;
        let mut slot = self.relation.checked_slot()?;
        let ready = slot.as_ref()?.upgrade()?;
        if !std::ptr::eq(prepared, &*ready) || !ready.matches(self.witness) {
            return None;
        }
        *slot = None;
        ready.take()
    }
}

pub(crate) struct Relation {
    step: Arc<CompiledStep>,
    structure: Structure,
    params: Params,
    window: RetainedMatrixWindow,
    checked: Mutex<Option<Weak<CheckedFreshRows>>>,
}

impl Relation {
    pub(crate) fn new(step: &Arc<CompiledStep>, minimum_security_bits: u32) -> Result<Self, RelationError> {
        let (rows, layout) = (step.expanded.rows(), step.expanded.layout());
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
                    exps: vec![0, 0, 0, SBOX_DEGREE as u32],
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
        Ok(Self {
            step: Arc::clone(step),
            structure,
            params,
            window: RetainedMatrixWindow::default(),
            checked: Mutex::new(None),
        })
    }

    #[cfg(test)]
    pub(crate) fn step(&self) -> &Arc<CompiledStep> {
        &self.step
    }

    pub(crate) fn context(&self) -> FoldContext<'_> {
        FoldContext {
            params: &self.params,
            structure: &self.structure,
            rows: self,
            workspace_bytes: self.step.workspace_bytes,
        }
    }

    fn checked_slot(&self) -> Option<MutexGuard<'_, Option<Weak<CheckedFreshRows>>>> {
        match self.checked.try_lock() {
            Ok(slot) => Some(slot),
            Err(TryLockError::Poisoned(error)) => Some(error.into_inner()),
            Err(TryLockError::WouldBlock) => None,
        }
    }

    pub(super) fn reserve_checked_rows(&self) -> Option<MutexGuard<'_, Option<Weak<CheckedFreshRows>>>> {
        let mut slot = self.checked_slot()?;
        if let Some(previous) = slot.take().and_then(|entry| entry.upgrade()) {
            drop(previous.take());
        }
        Some(slot)
    }

    /// Lean `FreshInstance.Masks`, `FreshInstance.built`.
    pub(crate) fn instance(&self, backend: &Backend, columns: &[F]) -> Result<CcsInstance, RelationError> {
        let witness = self.step.expanded.layout().signed_assignment(columns)?;
        let commitment = backend.commit(&[&witness])?.remove(0);
        Ok(CcsInstance {
            claim: CcsClaim {
                c: commitment,
                x: columns[..PUBLIC_WORDS].to_vec(),
                m_in: PUBLIC_WORDS,
                adv: None,
            },
            witness: CcsWitness { Z: witness },
        })
    }

    pub(crate) fn assignment_columns(&self, words: &StepWords, locals: Option<&[F]>) -> Result<Vec<F>, RelationError> {
        let expanded = &self.step.expanded;
        Ok(expanded.assignment_columns(&words.columns()?, locals)?)
    }

    #[cfg(test)]
    pub(crate) fn first_failing_row(&self, words: &StepWords) -> Result<Option<usize>, RelationError> {
        let expanded = &self.step.expanded;
        let columns = expanded.circuit_columns_with(&words.columns()?, None)?;
        let columns = expanded.relation_columns(columns)?;
        let rows = expanded.rows();
        Ok((0..rows.row_count()).find(|&row| rows.residual(row, &columns) != F::ZERO))
    }

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

    /// Lean `CCS.Holds`; SPEC.md Algorithm 6.
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
        let width = self.step.expanded.layout().width();
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
        if let Some(first) = backend.first_failing_row(self, self.step.workspace_bytes, &self.structure, z) {
            return match first {
                Ok(None) => Ok(()),
                Ok(Some(_)) => Err(Holds::Fails("a relation row fails")),
                Err(DeviceError::Engine(error)) => Err(Holds::Engine(error)),
                Err(DeviceError::Rejected(_)) => Err(Holds::Fails("row evaluation")),
            };
        }
        self.rows_hold(&logical[..width]).map_err(Holds::Fails)
    }

    pub(super) fn checked_source_rows(&self, columns: &[F]) -> Result<Vec<[F; MATRICES]>, &'static str> {
        let rows = self.step.expanded.rows();
        let mut values = vec![[F::ZERO; MATRICES]; rows.row_count()];
        let pending = Mutex::new(values.chunks_mut(ROWS_PER_RANGE).enumerate());
        let produce = || loop {
            let (index, group) = pending
                .lock()
                .unwrap_or_else(std::sync::PoisonError::into_inner)
                .next()?;
            for (local, output) in group.iter_mut().enumerate() {
                let row = index * ROWS_PER_RANGE + local;
                let [a, b, c, s] = std::array::from_fn(|matrix| rows.evaluate(row, matrix, columns));
                if a * b - c + s.exp_u64(SBOX_DEGREE) != F::ZERO {
                    return Some(row);
                }
                *output = [a, b, c, s];
            }
        };
        let threads = std::thread::available_parallelism().map_or(1, usize::from);
        std::thread::scope(|scope| {
            let workers: Vec<_> = (0..threads).map(|_| scope.spawn(produce)).collect();
            let failures = workers
                .into_iter()
                .map(|worker| worker.join().map_err(|_| "row worker panicked"))
                .collect::<Result<Vec<_>, _>>()?;
            failures
                .into_iter()
                .flatten()
                .min()
                .map_or(Ok(()), |_| Err("a relation row fails"))
        })?;
        Ok(values)
    }

    pub(crate) fn source_rows_hold(&self, columns: &[F]) -> Result<(), &'static str> {
        let rows = self.step.expanded.rows();
        self.check_rows(|row| {
            if rows.residual(row, columns) == F::ZERO {
                Ok(())
            } else {
                Err("a relation row fails")
            }
        })
    }

    fn rows_hold(&self, logical: &[F]) -> Result<(), &'static str> {
        let (rows, layout) = (self.step.expanded.rows(), self.step.expanded.layout());
        self.check_rows(|row| {
            let [a, b, c, s] = [0, 1, 2, 3].map(|port| layout.evaluate(rows, row, port, logical));
            let (Ok(a), Ok(b), Ok(c), Ok(s)) = (a, b, c, s) else {
                return Err("row evaluation");
            };
            if a * b - c + s.exp_u64(SBOX_DEGREE) != F::ZERO {
                return Err("a relation row fails");
            }
            Ok(())
        })
    }

    fn check_rows(&self, holds: impl Fn(usize) -> Result<(), &'static str> + Sync) -> Result<(), &'static str> {
        let count = self.step.expanded.rows().row_count();
        let next = AtomicUsize::new(0);
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
        self.step.blocks()
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
        self.step.shape()
    }

    fn visit_rows(&self, rows: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.step.visit_rows(rows, sink)
    }

    fn factored(&self) -> Result<Option<FactoredRows<'_>>, PiCcsError> {
        self.step.factored()
    }

    fn retained_window(&self) -> Option<&RetainedMatrixWindow> {
        Some(&self.window)
    }
}
