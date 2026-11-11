//! A memory application from the R1CS relation of one row (SPEC.md section
//! 7.6). The WASM relation of `neo-wasm` and the Starstream interleaving
//! relation both use it. One Nebula step proves `rows` consecutive rows, so
//! the rows share the fixed F′ cost of a fold. Its private words are the
//! witnesses `z` of its rows, one block of the one-row width each, and every
//! block uses the constant-one column of block 0. Its relation is the
//! rank-one rows of each row, then one equality row for each continuity link
//! between neighbor rows. Its state is the continuity words, one per link:
//! the `next_step_column`s of its first row in, the `previous_step_column`s
//! of its last row out. Each group of `Shared::rows` consecutive rows shares
//! the last `Shared::columns` columns of the one-row relation: they follow the
//! blocks of the rows, one block for each group, and a rank-one row that reads
//! shared columns and no other column but the constant-one column is kept
//! once for the group. `CatalogBinding` checks every memory of the batched
//! catalog (Lean `Spec.Nebula.MemoryTrace.batchCatalog`) in the regions of a
//! `Size`, so one circuit serves every run whose memories fit those regions.

use std::num::NonZeroUsize;
use std::sync::Arc;

use neo_application::{ContinuityCatalog, MemoryCatalog, MemoryPortActivation, MemoryPortKind, R1csRelation};
use neo_ccs::CcsMatrix;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

use super::app::{Application, Memory, Transition};
use super::catalog::{CatalogBinding, CatalogBindingError, CatalogRowError, Region};
use super::expr::Expr;
use super::geometry::Geometry;
use super::rows::View;
use super::{MemoryPreload, Program};

/// The terms `(column, coefficient)` of one side of a rank-one row.
pub type Terms = Vec<(usize, F)>;

/// The R1CS relation of one row: `[A_i, B_i, C_i]` of each rank-one row
/// `(A_i·z)(B_i·z) = C_i·z` over the row's `width` columns. Column `one`
/// holds the constant 1. The last `shared.columns` columns are the columns
/// that a group of `shared.rows` rows shares.
#[derive(Clone, Debug)]
pub struct RowRelation {
    pub rows: Vec<[Terms; 3]>,
    pub width: usize,
    pub one: usize,
    pub shared: Shared,
}

/// The columns that each group of `rows` consecutive rows of a step shares
/// (SPEC.md section 7.6, "Shared columns"): the last `columns` columns of
/// the one-row relation. Each row of a group reads the group's one block of
/// them, so the group holds exactly when each of its rows holds the one-row
/// relation on its own columns and the block (Lean
/// `Lifecycle.Nebula.SharedAbsorb.groupRows_iff_forall`).
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Shared {
    pub columns: usize,
    pub rows: NonZeroUsize,
}

impl Shared {
    /// No shared column: each row is a group of one.
    pub const NONE: Self = Self {
        columns: 0,
        rows: NonZeroUsize::MIN,
    };

    /// The step witness of the one-row witnesses `rows` of one step, each
    /// of `width` words: the first `width − columns` words of each row in
    /// row order, then, for each group of `self.rows` rows, the last
    /// `columns` words of the row of the group that `source` names. `source`
    /// returns an index into the group it gets.
    pub fn step_witness(self, width: usize, rows: &[&[F]], source: impl Fn(&[&[F]]) -> usize) -> Vec<F> {
        let block = width - self.columns;
        let mut z = Vec::with_capacity(rows.len() * block + rows.len().div_ceil(self.rows.get()) * self.columns);
        for row in rows {
            z.extend_from_slice(&row[..block]);
        }
        for group in rows.chunks(self.rows.get()) {
            z.extend_from_slice(&group[source(group)][block..width]);
        }
        z
    }
}

impl RowRelation {
    /// The rank-one rows of a relation that the relation builders made
    /// (`crate::relation::R1csBuilder`), with its constant-one column. A
    /// memory verifier takes no column values, so the relation may have no
    /// verifier-supplied column other than the constant-one column.
    pub fn of<Owner>(relation: &R1csRelation<Owner>) -> Result<Self, R1csApplicationError> {
        if (0..relation.public_input_count()).any(|column| column != relation.const_one_column()) {
            return Err(R1csApplicationError::PublicInput);
        }
        let structure = relation.structure();
        let mut rows = vec![[Terms::new(), Terms::new(), Terms::new()]; structure.n()];
        for (side, matrix) in structure.matrices().iter().take(3).enumerate() {
            let CcsMatrix::Csc(csc) = matrix else {
                return Err(R1csApplicationError::Matrix);
            };
            for column in 0..csc.ncols() {
                for entry in csc.column_range(column) {
                    rows[csc.row_index(entry)][side].push((column, csc.vals()[entry]));
                }
            }
        }
        Ok(Self {
            rows,
            width: relation.column_count(),
            one: relation.const_one_column(),
            shared: Shared::NONE,
        })
    }
}

/// One fixed circuit of an `R1csApplication`: its geometry, the rows of each
/// step, the region of each checked memory, and the slot plan over the ranks
/// of the batched catalog (memories, then rows, then ports).
#[derive(Clone, Debug)]
pub struct Size<Id> {
    pub geometry: Geometry,
    pub rows: NonZeroUsize,
    pub regions: Vec<(Id, Region)>,
    pub slots: Vec<Vec<usize>>,
}

/// The slot plan of `rows` rows in each step from the one-row slot plan
/// `plan` of `catalog`: the plan for row 0, then for row 1, and so on. Each
/// one-row rank `s` of memory `m` becomes the batched rank
/// `rows · offset_m + row · P_m + (s − offset_m)`, where `offset_m` is the
/// first one-row rank of `m` and `P_m` its port count. Inside one memory the
/// batched ranks are in row order. A plan that keeps each dependent pair of
/// one row in order keeps it in order across the rows too (SPEC.md section
/// 7.6, rule 2), because two dependent ports share a memory. The batched plan
/// has `rows` times the one-row slots.
/// A rank outside the catalog or an unrepresentable batched plan returns `Plan`.
pub fn batched_slots<Id>(
    catalog: &MemoryCatalog<Id>,
    plan: &[Vec<usize>],
    rows: NonZeroUsize,
) -> Result<Vec<Vec<usize>>, CatalogBindingError> {
    let memories = catalog.entries();
    let mut offsets = Vec::with_capacity(memories.len());
    let mut ranks = 0usize;
    for memory in memories {
        offsets.push(ranks);
        ranks = ranks
            .checked_add(memory.ports.len())
            .ok_or(CatalogBindingError::Plan)?;
    }
    if plan.iter().flatten().any(|rank| *rank >= ranks) {
        return Err(CatalogBindingError::Plan);
    }
    if plan.is_empty() {
        return Ok(Vec::new());
    }
    ranks
        .checked_mul(rows.get())
        .ok_or(CatalogBindingError::Plan)?;
    let slots = plan
        .len()
        .checked_mul(rows.get())
        .ok_or(CatalogBindingError::Plan)?;
    let mut batched = Vec::new();
    batched
        .try_reserve_exact(slots)
        .map_err(|_| CatalogBindingError::Plan)?;
    let rank = |row: usize, s: usize| {
        let m = offsets.partition_point(|offset| *offset <= s) - 1;
        rows.get() * offsets[m] + row * memories[m].ports.len() + (s - offsets[m])
    };
    batched.extend((0..rows.get()).flat_map(|row| {
        plan.iter()
            .map(move |slot| slot.iter().map(|s| rank(row, *s)).collect())
    }));
    Ok(batched)
}

/// A relation of one row as a Nebula application, `rows` rows in each step.
#[derive(Clone, Debug)]
pub struct R1csApplication<Id> {
    /// `[A_i, B_i, C_i]` of each rank-one row of one step.
    relation: Arc<Vec<[Terms; 3]>>,
    /// The one-row `(previous_step_column, next_step_column)` of each
    /// continuity link, in link order.
    links: Arc<Vec<(usize, usize)>>,
    layout: StepLayout,
    binding: CatalogBinding<Id>,
}

/// Where a step witness of `rows` rows holds each one-row column: the block
/// of `block` words of each row, then the shared block of each group.
#[derive(Clone, Copy, Debug)]
struct StepLayout {
    rows: NonZeroUsize,
    /// The one-row columns that are not shared.
    block: usize,
    one: usize,
    shared: Shared,
}

impl StepLayout {
    /// The column of the one-row column `column` in row `row` of a step.
    /// Every row uses the constant-one column of row 0, and the rows of a
    /// group use the group's shared block.
    fn column(&self, row: usize, column: usize) -> usize {
        if column == self.one {
            column
        } else if column < self.block {
            column + row * self.block
        } else {
            self.rows.get() * self.block + row / self.shared.rows.get() * self.shared.columns + (column - self.block)
        }
    }

    /// The words of a step witness.
    fn words(&self) -> usize {
        self.rows.get() * self.block + self.rows.get() / self.shared.rows.get() * self.shared.columns
    }
}

#[derive(Debug, thiserror::Error)]
pub enum R1csApplicationError {
    #[error("a matrix of the relation is not in CSC form")]
    Matrix,
    #[error(transparent)]
    Binding(#[from] CatalogBindingError),
    #[error("a memory of the catalog has no region, so the circuit would not check it")]
    Unchecked,
    #[error("the relation has a verifier-supplied column other than the constant-one column, which no memory verifier supplies")]
    PublicInput,
    #[error("a relation, continuity, or memory column is outside the row witness, or the constant-one, a continuity, or a memory column is shared")]
    Column,
    #[error("the rows of a step are not whole groups of the shared columns' rows")]
    Group,
    #[error("the batched witness or relation dimensions overflow")]
    Dimension,
}

/// Why a step input has no step.
#[derive(Debug, thiserror::Error)]
pub enum R1csStepError {
    #[error("the step input has {words} words, not the {expected} words of its rows")]
    Width { words: usize, expected: usize },
    #[error("the first row does not start from the state")]
    State,
    #[error(transparent)]
    Row(#[from] CatalogRowError),
    #[error("rank-one row {row} of the step relation does not hold")]
    Relation { row: usize },
}

fn previous(link: &(usize, usize)) -> usize {
    link.0
}

fn next(link: &(usize, usize)) -> usize {
    link.1
}

impl<Id: Clone + Eq> R1csApplication<Id> {
    /// The application of `relation`, its continuity links, and its memory
    /// catalog for `size`. The size must give each memory of the catalog a
    /// region, so that the circuit checks every memory.
    /// Every relation, continuity, and memory column must be inside the row,
    /// and the constant-one, continuity, and memory columns must not be
    /// shared. The rows of a step must be whole groups.
    /// Batched witness and relation dimensions must fit the address space.
    pub fn new(
        relation: RowRelation,
        continuity: &ContinuityCatalog,
        catalog: &MemoryCatalog<Id>,
        size: Size<Id>,
    ) -> Result<Self, R1csApplicationError> {
        let links: Vec<(usize, usize)> = continuity
            .links()
            .map(|link| (link.previous_step_column, link.next_step_column))
            .collect();
        let shared = relation.shared;
        let block = relation
            .width
            .checked_sub(shared.columns)
            .ok_or(R1csApplicationError::Column)?;
        if !size.rows.get().is_multiple_of(shared.rows.get()) {
            return Err(R1csApplicationError::Group);
        }
        let check_column = |column| {
            if column < block {
                Ok(())
            } else {
                Err(R1csApplicationError::Column)
            }
        };
        check_column(relation.one)?;
        if relation
            .rows
            .iter()
            .flatten()
            .flatten()
            .any(|(column, _)| *column >= relation.width)
        {
            return Err(R1csApplicationError::Column);
        }
        for (previous, next) in &links {
            check_column(*previous)?;
            check_column(*next)?;
        }
        for port in catalog.entries().iter().flat_map(|memory| &memory.ports) {
            for column in &port.address_columns {
                check_column(*column)?;
            }
            check_column(port.value_column)?;
            if let MemoryPortKind::Write {
                value_before_column: Some(column),
            } = port.kind
            {
                check_column(column)?;
            }
            match port.activation {
                MemoryPortActivation::When(column) | MemoryPortActivation::Unless(column) => check_column(column)?,
                MemoryPortActivation::Always => {}
            }
        }
        block
            .checked_mul(size.rows.get())
            .and_then(|rows| {
                shared
                    .columns
                    .checked_mul(size.rows.get() / shared.rows.get())
                    .and_then(|groups| rows.checked_add(groups))
            })
            .ok_or(R1csApplicationError::Dimension)?;
        relation
            .rows
            .len()
            .checked_mul(size.rows.get())
            .and_then(|rows| {
                links
                    .len()
                    .checked_mul(size.rows.get() - 1)
                    .and_then(|continuity| rows.checked_add(continuity))
            })
            .ok_or(R1csApplicationError::Dimension)?;
        let layout = StepLayout {
            rows: size.rows,
            block,
            one: relation.one,
            shared,
        };
        let batched = catalog.batched(size.rows, |row, column| layout.column(row, column));
        let binding = CatalogBinding::new(&batched, &size.geometry, size.regions, &size.slots)?;
        if !binding.host_checked().is_empty() {
            return Err(R1csApplicationError::Unchecked);
        }
        Ok(Self {
            relation: Arc::new(step_relation(&relation.rows, &links, &layout)),
            links: Arc::new(links),
            layout,
            binding,
        })
    }

    /// The program images of a preload, in the application's regions.
    pub fn program(&self, preload: &MemoryPreload<Id>) -> Result<Program, CatalogBindingError>
    where
        Id: Copy + Ord,
    {
        self.binding.program(preload)
    }

    /// The rank-one rows of one step's relation: the rows of each row, the
    /// rows that read only shared columns and the constant-one column once for
    /// each group, then the continuity rows between neighbor rows.
    pub fn relation_rows(&self) -> usize {
        self.relation.len()
    }

    /// Whether the step input `z` holds each rank-one row of one step's
    /// relation. The memory rows are not checked here.
    pub fn check(&self, z: &[F]) -> Result<(), R1csStepError> {
        if z.len() != self.aux() {
            return Err(R1csStepError::Width {
                words: z.len(),
                expected: self.aux(),
            });
        }
        let affine = |terms: &Terms| {
            terms
                .iter()
                .fold(F::ZERO, |sum, (column, coefficient)| sum + *coefficient * z[*column])
        };
        match self
            .relation
            .iter()
            .position(|[a, b, c]| affine(a) * affine(b) != affine(c))
        {
            Some(row) => Err(R1csStepError::Relation { row }),
            None => Ok(()),
        }
    }

    /// The state words that step input `z` ends in: the
    /// `previous_step_column`s of its last row.
    pub fn output_words(&self, z: &[F]) -> Vec<u64> {
        self.link_words(z, self.layout.rows.get() - 1, previous)
    }

    /// The position in the state words of the link whose one-row
    /// `previous_step_column` is `column`.
    pub fn state_position(&self, column: usize) -> Option<usize> {
        self.links.iter().position(|link| previous(link) == column)
    }

    fn link_words(&self, z: &[F], row: usize, column: fn(&(usize, usize)) -> usize) -> Vec<u64> {
        self.links
            .iter()
            .map(|link| z[self.layout.column(row, column(link))].as_canonical_u64())
            .collect()
    }
}

impl<Id: Clone + Eq> Application for R1csApplication<Id> {
    type Input = Vec<F>;
    type State = Vec<u64>;
    type Error = R1csStepError;

    fn width(&self) -> usize {
        self.links.len()
    }

    fn aux(&self) -> usize {
        self.layout.words()
    }

    fn port_count(&self) -> usize {
        self.binding.port_count()
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        state.clone()
    }

    /// The binding rows read a slot only through `a_p · (…)` for a port `p`
    /// of the slot, and `1 − pad_j = Σ a_p` (SPEC.md section 7.6, rules 1 to
    /// 4), so an inactive slot's fields reach no row of the relation.
    fn ignores_idle_slots(&self) -> bool {
        true
    }

    /// Each rank-one row as `C·z − (A·z)(B·z)`; `z[one] = 1`; the
    /// `next_step_column`s of the first row equal the input state and the
    /// `previous_step_column`s of the last row the output state; then the
    /// binding rows. Every row but `z[one] = 1` reads the constant 1 in place
    /// of `z[one]` (Lean `PortApp.fixColumn`), so a row `x · (x − one)` reads
    /// only `x` and gives `x` a bit slot (SPEC.md section 7.6).
    fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
        let one = view.aux(self.layout.one);
        let mut z: Vec<Expr> = (0..self.aux()).map(|column| view.aux(column)).collect();
        z[self.layout.one] = Expr::constant(1);
        let affine = |terms: &Terms| {
            terms
                .iter()
                .fold(Expr::constant(0), |sum, (column, coefficient)| {
                    sum + Expr::Const(*coefficient) * z[*column].clone()
                })
        };
        let mut rows: Vec<Expr> = self
            .relation
            .iter()
            .map(|[a, b, c]| affine(c) - affine(a) * affine(b))
            .collect();
        rows.push(one - Expr::constant(1));
        let last = self.layout.rows.get() - 1;
        for (k, link) in self.links.iter().enumerate() {
            rows.push(z[self.layout.column(0, next(link))].clone() - view.input(k));
            rows.push(z[self.layout.column(last, previous(link))].clone() - view.output(k));
        }
        rows.extend(self.binding.rows(view, &z)?);
        Ok(rows)
    }

    fn step(&self, state: &Vec<u64>, z: &Vec<F>, memory: &Memory) -> Result<Transition<Vec<u64>>, R1csStepError> {
        if z.len() != self.aux() {
            return Err(R1csStepError::Width {
                words: z.len(),
                expected: self.aux(),
            });
        }
        if self.link_words(z, 0, next) != *state {
            return Err(R1csStepError::State);
        }
        Ok(Transition {
            next: self.output_words(z),
            ports: self.binding.ports(z, memory)?,
            aux: z.clone(),
        })
    }
}

/// The rank-one rows of one step: the rows of `relation` for row 0, then for
/// row 1, and so on, except the rows that read shared columns and no other
/// column but the constant-one column; then those rows once for each group,
/// since they are the same row for each row of the group; then for each pair
/// of neighbor rows one row `z[previous] − z[next] = 0` for each link, as
/// `(A·z)(z[one]) = 0`.
fn step_relation(relation: &[[Terms; 3]], links: &[(usize, usize)], layout: &StepLayout) -> Vec<[Terms; 3]> {
    let block_only = |sides: &[Terms; 3]| {
        let mut columns = sides.iter().flatten().map(|(column, _)| *column);
        columns.clone().any(|column| column >= layout.block)
            && columns.all(|column| column == layout.one || column >= layout.block)
    };
    let (shared, own): (Vec<_>, Vec<_>) = relation.iter().partition(|sides| block_only(sides));
    let rows = layout.rows.get();
    let group = layout.shared.rows.get();
    let mut step = Vec::with_capacity(rows * own.len() + rows / group * shared.len() + (rows - 1) * links.len());
    let place = |row: usize, sides: &[Terms; 3]| {
        sides.clone().map(|terms| {
            terms
                .into_iter()
                .map(|(c, coefficient)| (layout.column(row, c), coefficient))
                .collect()
        })
    };
    for row in 0..rows {
        step.extend(own.iter().map(|sides| place(row, sides)));
    }
    for row in (0..rows).step_by(group) {
        step.extend(shared.iter().map(|sides| place(row, sides)));
    }
    for row in 1..rows {
        for link in links {
            step.push([
                vec![
                    (layout.column(row - 1, previous(link)), F::ONE),
                    (layout.column(row, next(link)), -F::ONE),
                ],
                vec![(layout.one, F::ONE)],
                Vec::new(),
            ]);
        }
    }
    step
}
