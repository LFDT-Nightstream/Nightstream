//! Enzo's WASM relation as a Nebula memory application (`SPEC.md` section
//! 7.6). One trace row is one Nebula step: its private words are the row's
//! witness `z`, and its state is the continuity words, one per link (`_AFTER`
//! of a row equal to `_BEFORE` of the next). `CatalogBinding` checks every
//! memory of the catalog in the regions that the caller gives, so one circuit
//! serves every program whose memories fit those regions.

use std::sync::Arc;

use neo_ccs::CcsMatrix;
use neo_math::F;
use nightstream::nebula::{
    Application, CatalogBinding, CatalogBindingError, CatalogRowError, Expr, Geometry, Memory, MemoryKind,
    MemoryPortActivation, Program, Region, Transition, View,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::batch::{build_batched_wasm_ccs, build_batched_witness, BatchError};
use crate::ir::{WasmStepState, WasmVmStep};
use crate::{
    build_wasm_relation_layout, continuity_words, plan_ordered_wasm_memory_slots, WasmMemoryId, WasmMemoryPreload,
};

/// The terms `(column, coefficient)` of one row of a sparse matrix.
type Terms = Vec<(usize, F)>;

/// The WASM relation as a Nebula application.
#[derive(Clone)]
pub struct WasmApplication {
    /// `[A_i, B_i, C_i]` of each rank-one row `(A_i·z)(B_i·z) = C_i·z`.
    relation: Arc<Vec<[Terms; 3]>>,
    /// The `(_AFTER, _BEFORE)` columns of each continuity link, in link order.
    links: Arc<Vec<(usize, usize)>>,
    columns: usize,
    binding: CatalogBinding<WasmMemoryId>,
}

#[derive(Debug, thiserror::Error)]
pub enum WasmApplicationError {
    #[error(transparent)]
    Relation(#[from] BatchError),
    #[error("a matrix of the WASM relation is not in CSC form")]
    Matrix,
    #[error(transparent)]
    Binding(#[from] CatalogBindingError),
    #[error("the trace has {rows} rows, more than the {steps} steps")]
    Steps { rows: usize, steps: usize },
}

#[derive(Debug, thiserror::Error)]
pub enum WasmStepError {
    #[error("the row does not start from the state")]
    State,
    #[error(transparent)]
    Row(#[from] CatalogRowError),
}

fn after(link: &(usize, usize)) -> usize {
    link.0
}

fn before(link: &(usize, usize)) -> usize {
    link.1
}

impl WasmApplication {
    /// The application for `geometry`, with one region for each memory of the
    /// catalog and the slot plan of `plan_ordered_wasm_memory_slots`, which
    /// keeps each pair of one memory with a write in catalog order.
    pub fn new(geometry: &Geometry, regions: Vec<(WasmMemoryId, Region)>) -> Result<Self, WasmApplicationError> {
        let catalog = &build_wasm_relation_layout().auxiliary.memory;
        Self::with_slots(
            geometry,
            regions,
            &plan_ordered_wasm_memory_slots(catalog.entries()).slots,
        )
    }

    /// The application for `geometry`, `regions`, and the slot plan `slots`.
    pub fn with_slots(
        geometry: &Geometry,
        regions: Vec<(WasmMemoryId, Region)>,
        slots: &[Vec<usize>],
    ) -> Result<Self, WasmApplicationError> {
        let layout = build_wasm_relation_layout();
        let (relation, columns) = relation()?;
        Ok(Self {
            relation: Arc::new(relation),
            links: Arc::new(
                layout
                    .auxiliary
                    .continuity
                    .links()
                    .map(|link| (link.previous_step_column, link.next_step_column))
                    .collect(),
            ),
            columns,
            binding: CatalogBinding::new(&layout.auxiliary.memory, geometry, regions, slots)?,
        })
    }

    pub fn binding(&self) -> &CatalogBinding<WasmMemoryId> {
        &self.binding
    }

    /// The program images of a preload, in the application's regions.
    pub fn program(&self, preload: &WasmMemoryPreload) -> Result<Program, CatalogBindingError> {
        self.binding.program(preload)
    }

    /// The rank-one rows of one trace row's relation.
    pub fn relation_rows(&self) -> usize {
        self.relation.len()
    }

    /// The state words that row `z` ends in: its `_AFTER` columns.
    pub fn output_words(&self, z: &[F]) -> Vec<u64> {
        self.words(z, after)
    }

    /// The position in the state words of the link whose `_AFTER` column is
    /// `column`.
    pub fn state_position(&self, column: usize) -> Option<usize> {
        self.links.iter().position(|link| after(link) == column)
    }

    fn words(&self, z: &[F], column: fn(&(usize, usize)) -> usize) -> Vec<u64> {
        self.links
            .iter()
            .map(|link| z[column(link)].as_canonical_u64())
            .collect()
    }
}

impl Application for WasmApplication {
    type Input = Vec<F>;
    type State = Vec<u64>;
    type Error = WasmStepError;

    fn width(&self) -> usize {
        self.links.len()
    }

    fn aux(&self) -> usize {
        self.columns
    }

    fn port_count(&self) -> usize {
        self.binding.port_count()
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        state.clone()
    }

    /// Each R1CS row as `C·z − (A·z)(B·z)`, one rank-one row each; `z[0] = 1`;
    /// the `_BEFORE` columns equal the input state and the `_AFTER` columns
    /// the output state; then the binding rows.
    fn rows(&self, view: &View) -> Result<Vec<Expr>, nightstream::Error> {
        let z: Vec<Expr> = (0..self.columns).map(|column| view.aux(column)).collect();
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
        rows.push(z[0].clone() - Expr::constant(1));
        for (k, link) in self.links.iter().enumerate() {
            rows.push(z[before(link)].clone() - view.input(k));
            rows.push(z[after(link)].clone() - view.output(k));
        }
        rows.extend(self.binding.rows(view, &z)?);
        Ok(rows)
    }

    fn step(&self, state: &Vec<u64>, z: &Vec<F>, memory: &Memory) -> Result<Transition<Vec<u64>>, WasmStepError> {
        if self.words(z, before) != *state {
            return Err(WasmStepError::State);
        }
        Ok(Transition {
            next: self.words(z, after),
            ports: self.binding.ports(z, memory)?,
            aux: z.clone(),
        })
    }
}

/// The rows of `A`, `B`, and `C` of one trace row's relation, lookup rows
/// included (`build_batched_wasm_ccs(1)` has no link rows).
fn relation() -> Result<(Vec<[Terms; 3]>, usize), WasmApplicationError> {
    let r1cs = build_batched_wasm_ccs(1)?.sparse_r1cs;
    let mut rows = vec![[Terms::new(), Terms::new(), Terms::new()]; r1cs.n];
    for (k, matrix) in [&r1cs.a, &r1cs.b, &r1cs.c].into_iter().enumerate() {
        let CcsMatrix::Csc(csc) = matrix else {
            return Err(WasmApplicationError::Matrix);
        };
        for column in 0..csc.ncols() {
            for entry in csc.column_range(column) {
                rows[csc.row_index(entry)][k].push((column, csc.vals()[entry]));
            }
        }
    }
    Ok((rows, r1cs.m))
}

/// The state words of `state`: its continuity words.
pub fn state_words(state: WasmStepState) -> Vec<u64> {
    continuity_words(state)
        .iter()
        .map(F::as_canonical_u64)
        .collect()
}

/// The step inputs of a run of `steps` steps: the witness of each row of
/// `trace`, then of padding rows, which activate no port and keep the state.
pub fn step_inputs(trace: &[WasmVmStep], steps: usize) -> Result<Vec<Vec<F>>, WasmApplicationError> {
    if trace.len() > steps {
        return Err(WasmApplicationError::Steps {
            rows: trace.len(),
            steps,
        });
    }
    let batch = build_batched_witness(trace, steps, 0);
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
/// `preloads` and the addresses of the active ports of `rows`. Each part's
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
