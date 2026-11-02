//! A memory application from the R1CS relation of one row (SPEC.md section
//! 7.6). The WASM relation of `neo-wasm` and the Starstream interleaving
//! relation both use it. One Nebula step proves `rows` consecutive rows, so
//! the rows share the fixed F′ cost of a fold. Its private words are the
//! witnesses `z` of its rows, one block of the one-row width each, and every
//! block uses the constant-one column of block 0. Its relation is the
//! rank-one rows of each row, then one equality row for each continuity link
//! between neighbor rows. Its state is the continuity words, one per link:
//! the `next_step_column`s of its first row in, the `previous_step_column`s
//! of its last row out. `CatalogBinding` checks every memory of the batched
//! catalog (Lean `Spec.Nebula.MemoryTrace.batchCatalog`) in the regions of a
//! `Size`, so one circuit serves every run whose memories fit those regions.

use std::num::NonZeroUsize;
use std::sync::Arc;

use neo_application::{ContinuityCatalog, MemoryCatalog};
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
/// holds the constant 1.
#[derive(Clone, Debug)]
pub struct RowRelation {
    pub rows: Vec<[Terms; 3]>,
    pub width: usize,
    pub one: usize,
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
pub fn batched_slots<Id>(catalog: &MemoryCatalog<Id>, plan: &[Vec<usize>], rows: NonZeroUsize) -> Vec<Vec<usize>> {
    let memories = catalog.entries();
    let mut offsets = Vec::with_capacity(memories.len());
    let mut ranks = 0;
    for memory in memories {
        offsets.push(ranks);
        ranks += memory.ports.len();
    }
    let rank = |row: usize, s: usize| {
        let m = offsets.partition_point(|offset| *offset <= s) - 1;
        rows.get() * offsets[m] + row * memories[m].ports.len() + (s - offsets[m])
    };
    (0..rows.get())
        .flat_map(|row| {
            plan.iter()
                .map(move |slot| slot.iter().map(|s| rank(row, *s)).collect())
        })
        .collect()
}

/// A relation of one row as a Nebula application, `rows` rows in each step.
#[derive(Clone, Debug)]
pub struct R1csApplication<Id> {
    /// `[A_i, B_i, C_i]` of each rank-one row of one step.
    relation: Arc<Vec<[Terms; 3]>>,
    /// The one-row `(previous_step_column, next_step_column)` of each
    /// continuity link, in link order.
    links: Arc<Vec<(usize, usize)>>,
    rows: NonZeroUsize,
    /// The witness width of one row.
    row_width: usize,
    one: usize,
    binding: CatalogBinding<Id>,
}

#[derive(Debug, thiserror::Error)]
pub enum R1csApplicationError {
    #[error(transparent)]
    Binding(#[from] CatalogBindingError),
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
}

fn previous(link: &(usize, usize)) -> usize {
    link.0
}

fn next(link: &(usize, usize)) -> usize {
    link.1
}

/// The column of the one-row column `column` in row `row` of a step. Every
/// row uses the constant-one column of row 0.
fn step_column(row_width: usize, one: usize, row: usize, column: usize) -> usize {
    if column == one {
        column
    } else {
        column + row * row_width
    }
}

impl<Id: Clone + Eq> R1csApplication<Id> {
    /// The application of `relation`, its continuity links, and its memory
    /// catalog for `size`.
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
        let (row_width, one) = (relation.width, relation.one);
        let column = |row: usize, column: usize| step_column(row_width, one, row, column);
        let batched = catalog.batched(size.rows, column);
        Ok(Self {
            relation: Arc::new(step_relation(&relation, &links, size.rows)),
            links: Arc::new(links),
            rows: size.rows,
            row_width,
            one,
            binding: CatalogBinding::new(&batched, &size.geometry, size.regions, &size.slots)?,
        })
    }

    pub fn binding(&self) -> &CatalogBinding<Id> {
        &self.binding
    }

    /// The program images of a preload, in the application's regions.
    pub fn program(&self, preload: &MemoryPreload<Id>) -> Result<Program, CatalogBindingError>
    where
        Id: Copy + Ord,
    {
        self.binding.program(preload)
    }

    /// The rank-one rows of one step's relation: the rows of each row, then
    /// the continuity rows between neighbor rows.
    pub fn relation_rows(&self) -> usize {
        self.relation.len()
    }

    /// The state words that step input `z` ends in: the
    /// `previous_step_column`s of its last row.
    pub fn output_words(&self, z: &[F]) -> Vec<u64> {
        self.link_words(z, self.rows.get() - 1, previous)
    }

    /// The position in the state words of the link whose one-row
    /// `previous_step_column` is `column`.
    pub fn state_position(&self, column: usize) -> Option<usize> {
        self.links.iter().position(|link| previous(link) == column)
    }

    fn column(&self, row: usize, column: usize) -> usize {
        step_column(self.row_width, self.one, row, column)
    }

    fn link_words(&self, z: &[F], row: usize, column: fn(&(usize, usize)) -> usize) -> Vec<u64> {
        self.links
            .iter()
            .map(|link| z[self.column(row, column(link))].as_canonical_u64())
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
        self.rows.get() * self.row_width
    }

    fn port_count(&self) -> usize {
        self.binding.port_count()
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        state.clone()
    }

    /// Each rank-one row as `C·z − (A·z)(B·z)`; `z[one] = 1`; the
    /// `next_step_column`s of the first row equal the input state and the
    /// `previous_step_column`s of the last row the output state; then the
    /// binding rows.
    fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
        let z: Vec<Expr> = (0..self.aux()).map(|column| view.aux(column)).collect();
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
        rows.push(z[self.one].clone() - Expr::constant(1));
        let last = self.rows.get() - 1;
        for (k, link) in self.links.iter().enumerate() {
            rows.push(z[self.column(0, next(link))].clone() - view.input(k));
            rows.push(z[self.column(last, previous(link))].clone() - view.output(k));
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

/// The rank-one rows of one step of `rows` rows: the rows of `relation` for
/// row 0, then for row 1, and so on, then for each pair of neighbor rows one
/// row `z[previous] − z[next] = 0` for each link, as `(A·z)(z[one]) = 0`.
fn step_relation(relation: &RowRelation, links: &[(usize, usize)], rows: NonZeroUsize) -> Vec<[Terms; 3]> {
    let column = |row: usize, column: usize| step_column(relation.width, relation.one, row, column);
    let rows = rows.get();
    let mut step = Vec::with_capacity(rows * relation.rows.len() + (rows - 1) * links.len());
    for row in 0..rows {
        for sides in &relation.rows {
            step.push(sides.clone().map(|terms| {
                terms
                    .into_iter()
                    .map(|(c, coefficient)| (column(row, c), coefficient))
                    .collect()
            }));
        }
    }
    for row in 1..rows {
        for link in links {
            step.push([
                vec![
                    (column(row - 1, previous(link)), F::ONE),
                    (column(row, next(link)), -F::ONE),
                ],
                vec![(relation.one, F::ONE)],
                Vec::new(),
            ]);
        }
    }
    step
}
