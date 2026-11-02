//! Enzo's WASM relation as a Nebula memory application (`SPEC.md` section
//! 7.6): `nightstream::nebula::R1csApplication` of the one-row WASM relation
//! with its operation-table rows, its continuity links, and its memory
//! catalog. One Nebula step proves `rows` consecutive trace rows. Its private
//! words are the witnesses `z` of its rows (`build_batched_witness`), and its
//! state is the continuity words. One circuit serves every program whose
//! memories fit the regions that the caller gives.

use std::num::NonZeroUsize;

use neo_ccs::CcsMatrix;
use neo_math::F;
pub use nightstream::nebula::R1csStepError as WasmStepError;
use nightstream::nebula::{
    batched_slots, Application, CatalogBinding, CatalogBindingError, Expr, Geometry, Memory, MemoryKind,
    MemoryPortActivation, Program, R1csApplication, R1csApplicationError, Region, RowRelation, Size, Terms, Transition,
    View,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::batch::{build_batched_witness, single_wasm_relation, BatchError};
use crate::ir::{WasmStepState, WasmVmStep};
use crate::layout::COL_ONE;
use crate::{
    build_wasm_relation_layout, continuity_words, plan_ordered_wasm_memory_slots, WasmMemoryId, WasmMemoryPreload,
};

/// The WASM relation as a Nebula application.
#[derive(Clone)]
pub struct WasmApplication(R1csApplication<WasmMemoryId>);

#[derive(Debug, thiserror::Error)]
pub enum WasmApplicationError {
    #[error(transparent)]
    Relation(#[from] BatchError),
    #[error("a matrix of the WASM relation is not in CSC form")]
    Matrix,
    #[error(transparent)]
    Binding(#[from] CatalogBindingError),
    #[error("the trace of {trace} rows does not fit {steps} steps of {rows} rows")]
    Steps {
        trace: usize,
        steps: usize,
        rows: NonZeroUsize,
    },
}

impl From<R1csApplicationError> for WasmApplicationError {
    fn from(error: R1csApplicationError) -> Self {
        match error {
            R1csApplicationError::Binding(error) => Self::Binding(error),
        }
    }
}

/// The slot plan of `WasmApplication::new`: `batched_slots` of the one-row
/// plan of `plan_ordered_wasm_memory_slots`, so each dependent pair stays in
/// order (`SPEC.md` section 7.6 rule 2). It has `rows` times the one-row
/// slots.
pub fn plan_ordered_slots(rows: NonZeroUsize) -> Vec<Vec<usize>> {
    let catalog = &build_wasm_relation_layout().auxiliary.memory;
    batched_slots(catalog, &plan_ordered_wasm_memory_slots(catalog.entries()).slots, rows)
}

impl WasmApplication {
    /// The application of `rows` trace rows per step for `geometry`, with one
    /// region for each memory of the catalog and the slot plan of
    /// `plan_ordered_slots`.
    pub fn new(
        geometry: &Geometry,
        regions: Vec<(WasmMemoryId, Region)>,
        rows: NonZeroUsize,
    ) -> Result<Self, WasmApplicationError> {
        Self::with_slots(geometry, regions, rows, &plan_ordered_slots(rows))
    }

    /// The application of `rows` trace rows per step for `geometry`,
    /// `regions`, and the slot plan `slots` over the ranks of the batched
    /// catalog (memories, then rows, then ports).
    pub fn with_slots(
        geometry: &Geometry,
        regions: Vec<(WasmMemoryId, Region)>,
        rows: NonZeroUsize,
        slots: &[Vec<usize>],
    ) -> Result<Self, WasmApplicationError> {
        let layout = build_wasm_relation_layout();
        let size = Size {
            geometry: *geometry,
            rows,
            regions,
            slots: slots.to_vec(),
        };
        Ok(Self(R1csApplication::new(
            relation()?,
            &layout.auxiliary.continuity,
            &layout.auxiliary.memory,
            size,
        )?))
    }

    pub fn binding(&self) -> &CatalogBinding<WasmMemoryId> {
        self.0.binding()
    }

    /// The program images of a preload, in the application's regions.
    pub fn program(&self, preload: &WasmMemoryPreload) -> Result<Program, CatalogBindingError> {
        self.0.program(preload)
    }

    /// The rank-one rows of one step's relation: the rows of each trace row,
    /// then the continuity rows between neighbor rows.
    pub fn relation_rows(&self) -> usize {
        self.0.relation_rows()
    }

    /// The state words that step input `z` ends in: the `_AFTER` columns of
    /// its last row.
    pub fn output_words(&self, z: &[F]) -> Vec<u64> {
        self.0.output_words(z)
    }

    /// The position in the state words of the link whose one-row `_AFTER`
    /// column is `column`.
    pub fn state_position(&self, column: usize) -> Option<usize> {
        self.0.state_position(column)
    }
}

impl Application for WasmApplication {
    type Input = Vec<F>;
    type State = Vec<u64>;
    type Error = WasmStepError;

    fn width(&self) -> usize {
        self.0.width()
    }

    fn aux(&self) -> usize {
        self.0.aux()
    }

    fn port_count(&self) -> usize {
        self.0.port_count()
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        self.0.words(state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Expr>, nightstream::Error> {
        self.0.rows(view)
    }

    fn step(&self, state: &Vec<u64>, z: &Vec<F>, memory: &Memory) -> Result<Transition<Vec<u64>>, WasmStepError> {
        self.0.step(state, z, memory)
    }
}

/// The one-row WASM relation with its operation-table rows
/// (`build_batched_wasm_ccs(1)` without the batch).
fn relation() -> Result<RowRelation, WasmApplicationError> {
    let r1cs = single_wasm_relation()?.relation;
    let mut terms = vec![[Terms::new(), Terms::new(), Terms::new()]; r1cs.n];
    for (k, matrix) in [&r1cs.a, &r1cs.b, &r1cs.c].into_iter().enumerate() {
        let CcsMatrix::Csc(csc) = matrix else {
            return Err(WasmApplicationError::Matrix);
        };
        for column in 0..csc.ncols() {
            for entry in csc.column_range(column) {
                terms[csc.row_index(entry)][k].push((column, csc.vals()[entry]));
            }
        }
    }
    Ok(RowRelation {
        rows: terms,
        width: r1cs.m,
        one: COL_ONE,
    })
}

/// The state words of `state`: its continuity words.
pub fn state_words(state: WasmStepState) -> Vec<u64> {
    continuity_words(state)
        .iter()
        .map(F::as_canonical_u64)
        .collect()
}

/// The step inputs of a run of `steps` steps of `rows` trace rows each: the
/// witnesses of the rows of `trace`, then of padding rows, which activate no
/// port and keep the state, cut into `steps` inputs of `rows` rows each.
/// `rows = 1` gives the one-row witnesses that `smallest_regions` takes.
pub fn step_inputs(
    trace: &[WasmVmStep],
    steps: usize,
    rows: NonZeroUsize,
) -> Result<Vec<Vec<F>>, WasmApplicationError> {
    let capacity = steps
        .checked_mul(rows.get())
        .filter(|capacity| trace.len() <= *capacity)
        .ok_or(WasmApplicationError::Steps {
            trace: trace.len(),
            steps,
            rows,
        })?;
    let batch = build_batched_witness(trace, capacity, 0);
    Ok(batch
        .chunks(batch.len() / steps)
        .map(<[F]>::to_vec)
        .collect())
}

/// Lean `active`: whether a port with `activation` is active on row `z`.
pub fn is_active(z: &[F], activation: MemoryPortActivation) -> bool {
    match activation {
        MemoryPortActivation::Always => true,
        MemoryPortActivation::When(column) => z[column] == F::ONE,
        MemoryPortActivation::Unless(column) => z[column] == F::ZERO,
    }
}

fn bits(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

/// The smallest region of each memory of the catalog that holds the cells of
/// `preloads` and the addresses of the active ports of `rows`, the one-row
/// witnesses of `step_inputs(trace, steps * rows, 1)`. Each part's
/// width holds the largest part of that memory. In each namespace the regions
/// go from the largest to the smallest, so each base is a multiple of its
/// size. Returns the regions and the `log2` size of the ROM and of the RAM
/// that hold them. A deployment that proves programs it does not know yet
/// needs fixed regions instead.
pub fn smallest_regions(preloads: &[&WasmMemoryPreload], rows: &[Vec<F>]) -> (Vec<(WasmMemoryId, Region)>, [usize; 2]) {
    let catalog = &build_wasm_relation_layout().auxiliary.memory;
    let mut needs: Vec<(WasmMemoryId, bool, Vec<usize>)> = catalog
        .entries()
        .iter()
        .map(|memory| {
            let mut largest = vec![0; memory.ports[0].address_columns.len()];
            let mut see = |parts: Vec<u64>| {
                for (largest, part) in largest.iter_mut().zip(parts) {
                    *largest = part.max(*largest);
                }
            };
            for (id, address, _) in preloads.iter().flat_map(|preload| preload.entries()) {
                if id == memory.id {
                    see(address.into_iter().map(u64::from).collect());
                }
            }
            for port in &memory.ports {
                for z in rows.iter().filter(|z| is_active(z, port.activation)) {
                    see(port
                        .address_columns
                        .iter()
                        .map(|column| z[*column].as_canonical_u64())
                        .collect());
                }
            }
            let widths = largest.into_iter().map(bits).collect();
            (memory.id, memory.kind == MemoryKind::Ram, widths)
        })
        .collect();
    needs.sort_by_key(|(_, is_ram, widths)| (*is_ram, std::cmp::Reverse(widths.iter().sum::<usize>())));
    let mut used = [0u64; 2];
    let regions = needs
        .into_iter()
        .map(|(id, is_ram, widths)| {
            let next = &mut used[usize::from(is_ram)];
            let base = *next;
            *next += 1 << widths.iter().sum::<usize>();
            (id, Region { base, widths })
        })
        .collect();
    (regions, used.map(|cells| bits(cells - 1)))
}

/// The smallest geometry with ROM and RAM of at least `2^log2` cells, `b_ops`
/// slots, segments of `n` steps, and at most `s_max` segments. `n` must be a
/// power of two; each namespace has at least `n` cells, so that the scan
/// covers both in `n` steps (`SPEC.md` section 7.1, rule 1).
pub fn smallest_geometry(log2: [usize; 2], b_ops: usize, n: usize, s_max: usize) -> Geometry {
    let low = n.trailing_zeros() as usize;
    // Rule 4: the RAM is at least as large as the ROM.
    let r = log2[0].max(low);
    let mu = log2[1].max(r);
    Geometry {
        r,
        mu,
        w_ts: bits((s_max * n * b_ops) as u64),
        b_ops,
        b_scan: ((1 << r) + (1 << mu)) / n,
        n,
        s_max,
    }
}
