//! Enzo's WASM relation as a Nebula memory application (`SPEC.md` section
//! 7.6): `nightstream::nebula::R1csApplication` of the one-row WASM relation
//! with its operation-table rows, its continuity links, and its memory
//! catalog. One Nebula step proves `rows` consecutive trace rows. Its private
//! words are the witnesses `z` of its rows (`build_batched_witness`). Its
//! state is a `WasmStepState`, whose words are the continuity words
//! (`state_words`), so a verifier reads the last state of a proof as a typed
//! value. Every field has words (test `every_state_field_changes_the_state_words`),
//! and a field of `q` or more has no canonical word. Absence of a default
//! memory uses a reserved word above the wasm32 page limit, so it differs
//! from a present zero-page memory. One circuit serves every program whose memories fit its
//! regions. A deployment proves in the fixed sizes of `WasmSize`.

use std::num::NonZeroUsize;

use neo_ccs::CcsMatrix;
use neo_math::F;
use nightstream::nebula::{
    batched_slots, pack_regions, Application, CatalogBindingError, Expr, Geometry, Memory, Program, R1csApplication,
    R1csApplicationError, R1csStepError, Region, RowRelation, Size, Terms, Transition, View,
};
use nightstream::relation::MemoryPortActivation;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use crate::batch::{build_batched_witness, single_wasm_relation, BatchError};
use crate::ir::{WasmStepState, WasmVmStep};
use crate::layout::COL_ONE;
use crate::{build_wasm_relation_layout, plan_ordered_wasm_memory_slots, WasmMemoryId, WasmMemoryPreload};

/// The WASM relation as a Nebula application.
#[derive(Clone)]
pub struct WasmApplication(R1csApplication<WasmMemoryId>);

/// One step input: the witnesses of its rows and the state after its last
/// trace row.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct WasmStep {
    pub z: Vec<F>,
    /// Its words must be the `_AFTER` columns of the last row of `z`.
    pub next: WasmStepState,
}

#[derive(Debug, thiserror::Error)]
pub enum WasmStepError {
    #[error(transparent)]
    Step(#[from] R1csStepError),
    #[error("the step's next state has other words than its last row")]
    Next,
}

#[derive(Debug, thiserror::Error)]
pub enum WasmApplicationError {
    #[error(transparent)]
    Relation(#[from] BatchError),
    #[error("a matrix of the WASM relation is not in CSC form")]
    Matrix,
    #[error(transparent)]
    Application(#[from] R1csApplicationError),
    #[error(transparent)]
    Catalog(#[from] CatalogBindingError),
    #[error("witness row has {actual} words, needs at least {minimum}")]
    RowLength { actual: usize, minimum: usize },
    #[error("a run needs at least one trace row")]
    EmptyTrace,
    #[error("no WASM size holds the program and its run, which need {0:?}")]
    TooLarge(WasmLimits),
    #[error("the trace of {trace} rows does not fit {steps} steps of {rows} rows")]
    Steps {
        trace: usize,
        steps: usize,
        rows: NonZeroUsize,
    },
}

/// The slot plan of `WasmApplication::new`: `batched_slots` of the one-row
/// plan of `plan_ordered_wasm_memory_slots`, so each dependent pair stays in
/// order (`SPEC.md` section 7.6 rule 2). It has `rows` times the one-row
/// slots.
pub fn plan_ordered_slots(rows: NonZeroUsize) -> Result<Vec<Vec<usize>>, CatalogBindingError> {
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
        Self::with_slots(geometry, regions, rows, &plan_ordered_slots(rows)?)
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
    type Input = WasmStep;
    type State = WasmStepState;
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

    fn words(&self, state: &WasmStepState) -> Vec<u64> {
        state_words(*state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Expr>, nightstream::Error> {
        self.0.rows(view)
    }

    fn step(
        &self,
        state: &WasmStepState,
        input: &WasmStep,
        memory: &Memory,
    ) -> Result<Transition<WasmStepState>, WasmStepError> {
        let transition = self.0.step(&state_words(*state), &input.z, memory)?;
        if transition.next != state_words(input.next) {
            return Err(WasmStepError::Next);
        }
        Ok(Transition {
            next: input.next,
            ports: transition.ports,
            aux: transition.aux,
        })
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

/// The state words of `state`: the integers of its continuity words, before
/// their reduction to field words. A field of `q` or more keeps a word of `q`
/// or more, which the terminal check of SPEC.md Algorithm 8 refuses, so a
/// claim cannot name a field that differs from the proved word by a multiple
/// of `q` (test `a_state_field_of_q_or_more_has_no_canonical_word`).
pub fn state_words(state: WasmStepState) -> Vec<u64> {
    crate::preprocess::continuity_values(state)
}

/// The step inputs of a run of `steps` steps of `rows` trace rows each: the
/// witnesses of the rows of `trace`, then of padding rows, which activate no
/// port and keep the state, cut into `steps` inputs of `rows` rows each. The
/// next state of a step is the state after its last trace row. `rows = 1`
/// gives the one-row witnesses that `WasmLimits::of` takes.
pub fn step_inputs(
    trace: &[WasmVmStep],
    steps: usize,
    rows: NonZeroUsize,
) -> Result<Vec<WasmStep>, WasmApplicationError> {
    if trace.is_empty() {
        return Err(WasmApplicationError::EmptyTrace);
    }
    let capacity = steps
        .checked_mul(rows.get())
        .filter(|capacity| trace.len() <= *capacity)
        .ok_or(WasmApplicationError::Steps {
            trace: trace.len(),
            steps,
            rows,
        })?;
    let batch = build_batched_witness(trace, capacity, 0)?;
    Ok(batch
        .chunks(batch.len() / steps)
        .enumerate()
        .map(|(step, z)| WasmStep {
            z: z.to_vec(),
            next: trace[((step + 1) * rows.get()).min(trace.len()) - 1].state_after,
        })
        .collect())
}

/// Lean `active`: whether a port with `activation` is active on row `z`.
pub fn is_active(z: &[F], activation: MemoryPortActivation) -> Result<bool, WasmApplicationError> {
    let (column, expected) = match activation {
        MemoryPortActivation::Always => return Ok(true),
        MemoryPortActivation::When(column) => (column, F::ONE),
        MemoryPortActivation::Unless(column) => (column, F::ZERO),
    };
    Ok(*z.get(column).ok_or(WasmApplicationError::RowLength {
        actual: z.len(),
        minimum: column.saturating_add(1),
    })? == expected)
}

fn bits(value: u64) -> usize {
    (u64::BITS - value.leading_zeros()) as usize
}

/// What one address part of a WASM memory counts.
#[derive(Clone, Copy)]
enum Dimension {
    Pc,
    Choice,
    Function,
    Event,
    Slot,
    Stack,
    LinearMemory,
    Global,
    CallDepth,
    HostDepth,
    Table,
    TableIndex,
}

impl Dimension {
    const ALL: [Self; 12] = [
        Self::Pc,
        Self::Choice,
        Self::Function,
        Self::Event,
        Self::Slot,
        Self::Stack,
        Self::LinearMemory,
        Self::Global,
        Self::CallDepth,
        Self::HostDepth,
        Self::Table,
        Self::TableIndex,
    ];
}

/// The dimension of each address part of memory `id`, least significant part
/// first, as `preload_from_program_artifacts`, `preload_host_event_tables`
/// and the catalog's address columns key it.
fn dimensions(id: WasmMemoryId) -> &'static [Dimension] {
    use Dimension as D;
    use WasmMemoryId as M;
    match id {
        M::ProgramOpcode
        | M::ProgramBrTableLen
        | M::ProgramLocalIndex
        | M::ProgramGlobalIndex
        | M::ProgramTableId
        | M::ProgramMemoryOffset
        | M::ProgramCallIndirectExpectedTypeId
        | M::ProgramI32ConstValue
        | M::ProgramI64ConstValueLo
        | M::ProgramI64ConstValueHi
        | M::ProgramRefFuncRef
        | M::PcEdgeKind
        | M::PcFunctionRef
        | M::CallTarget => &[D::Pc],
        M::PcRom | M::FrameSlideRemaining | M::FrameSlideDistance => &[D::Pc, D::Choice],
        M::FunctionEntry
        | M::FunctionType
        | M::FunctionCallMetadata
        | M::FunctionLocalCount
        | M::HostEventExportEntryScheduleCount
        | M::HostEventExportExitScheduleCount
        | M::HostEventImportScheduleCount => &[D::Function],
        M::HostEventSlotKind
        | M::HostEventSlotArg
        | M::HostEventSlotVariant
        | M::HostEventSlotImmediate0
        | M::HostEventSlotImmediate1 => &[D::Function, D::Event, D::Slot],
        M::Stack => &[D::Stack],
        M::LinearMemory => &[D::LinearMemory],
        M::GlobalLo | M::GlobalHi => &[D::Global],
        M::CallStackReturnPc | M::CallStackCallerFbp | M::CallStackCallerSpBase => &[D::CallDepth],
        M::HostContextExport
        | M::HostContextImport
        | M::HostContextArgsBase
        | M::HostContextParentDepth
        | M::HostContextSp => &[D::HostDepth],
        M::TableSize => &[D::Table],
        M::TableElement => &[D::Table, D::TableIndex],
    }
}

/// The bit width of each address dimension of a WASM program and its run.
/// The region of a memory has, for each address part, the width of its
/// dimension (`WasmLimits::regions`). So a program and its run fit limits
/// when each address part of the preload and of the active ports fits.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct WasmLimits {
    /// A program counter.
    pub pc: usize,
    /// A control choice at one program counter (`PcRom`, frame slides).
    pub choice: usize,
    /// A function reference, imports included.
    pub function: usize,
    /// A host event of one function.
    pub event: usize,
    /// A slot of one host event.
    pub slot: usize,
    /// A stack cell.
    pub stack: usize,
    /// A word of linear memory.
    pub linear_memory: usize,
    /// A global.
    pub global: usize,
    /// A call-stack frame.
    pub call_depth: usize,
    /// A host-context frame.
    pub host_depth: usize,
    /// A table.
    pub table: usize,
    /// An element of one table.
    pub table_index: usize,
}

impl WasmLimits {
    /// The smallest limits that hold the cells of `preloads` and the address
    /// parts of the active ports of `rows`, the one-row witnesses of
    /// `step_inputs(trace, trace.len(), 1)`.
    pub fn of(preloads: &[&WasmMemoryPreload], rows: &[Vec<F>]) -> Result<Self, WasmApplicationError> {
        let mut limits = Self::default();
        for (id, address, _) in preloads.iter().flat_map(|preload| preload.entries()) {
            limits.hold(id, address.into_iter().map(u64::from));
        }
        for z in rows {
            limits.hold_row(z)?;
        }
        Ok(limits)
    }

    /// Widen the limits so that they hold the address parts of the active
    /// ports of the one-row witness `z`.
    fn hold_row(&mut self, z: &[F]) -> Result<(), WasmApplicationError> {
        if z.len() < crate::RANGE_CHECKED_WITNESS_WIDTH {
            return Err(WasmApplicationError::RowLength {
                actual: z.len(),
                minimum: crate::RANGE_CHECKED_WITNESS_WIDTH,
            });
        }
        for memory in build_wasm_relation_layout().auxiliary.memory.entries() {
            for port in &memory.ports {
                if is_active(z, port.activation)? {
                    self.hold(
                        memory.id,
                        port.address_columns
                            .iter()
                            .map(|column| z[*column].as_canonical_u64()),
                    );
                }
            }
        }
        Ok(())
    }

    /// Whether each width is at least the width of `other`.
    pub fn holds(&self, other: &Self) -> bool {
        let (mut this, mut other) = (*self, *other);
        Dimension::ALL
            .into_iter()
            .all(|dimension| *this.width(dimension) >= *other.width(dimension))
    }

    /// One region for each memory of the catalog, with the widths of its
    /// dimensions (`pack_regions`), and the cells that they take in the ROM
    /// and in the RAM.
    pub fn regions(&self) -> Result<(Vec<(WasmMemoryId, Region)>, [u64; 2]), CatalogBindingError> {
        let mut limits = *self;
        pack_regions(
            build_wasm_relation_layout()
                .auxiliary
                .memory
                .entries()
                .iter()
                .map(|memory| {
                    let widths = dimensions(memory.id)
                        .iter()
                        .map(|dimension| *limits.width(*dimension))
                        .collect();
                    (memory.id, memory.kind, widths)
                }),
        )
    }

    /// Widen the limits so that they hold the address `parts` of memory `id`.
    fn hold(&mut self, id: WasmMemoryId, parts: impl IntoIterator<Item = u64>) {
        for (dimension, part) in dimensions(id).iter().zip(parts) {
            let width = self.width(*dimension);
            *width = (*width).max(bits(part));
        }
    }

    fn width(&mut self, dimension: Dimension) -> &mut usize {
        match dimension {
            Dimension::Pc => &mut self.pc,
            Dimension::Choice => &mut self.choice,
            Dimension::Function => &mut self.function,
            Dimension::Event => &mut self.event,
            Dimension::Slot => &mut self.slot,
            Dimension::Stack => &mut self.stack,
            Dimension::LinearMemory => &mut self.linear_memory,
            Dimension::Global => &mut self.global,
            Dimension::CallDepth => &mut self.call_depth,
            Dimension::HostDepth => &mut self.host_depth,
            Dimension::Table => &mut self.table,
            Dimension::TableIndex => &mut self.table_index,
        }
    }
}

/// A fixed size of WASM proofs (`decisions/wasm-fixed-sizes.md`): limits, a
/// geometry, and the rows of one step. Each size is one circuit, so a
/// verifier compiles at most three. It takes the size of a proof from the
/// proof's statement, never from the prover's run. A prover takes the
/// smallest size that holds the run (`WasmSize::for_run`). Each size has the
/// most segments for which the memory fingerprint term of all its segments,
/// `S_max · 2·m_mem/q²`, is at most `2^−109` (SPEC.md section 7.1), so the
/// runtime gate accepts at most 109 bits. This is not an end-to-end security
/// level: the deployment query count, retries, and recursive depth remain
/// separate.
#[derive(Clone, Copy, Debug, PartialEq, Eq, PartialOrd, Ord, Hash)]
pub enum WasmSize {
    /// ROM of `2^14` cells and RAM of `2^10`: 256 instructions, 32 function
    /// references, steps of 64 rows, and at most 10 segments of 8 steps:
    /// runs of up to 5,120 rows.
    Small,
    /// ROM of `2^15` cells and RAM of `2^11`: 512 instructions, 64 function
    /// references, steps of 64 rows, and at most 5 segments of 16 steps: runs
    /// of up to 5,120 rows.
    Medium,
    /// ROM of `2^16` cells and RAM of `2^12`: 1,024
    /// instructions, 128 function references, steps of 32 rows, and at most 3
    /// segments of 16 steps: runs of up to 1,536 rows.
    Large,
}

impl WasmSize {
    pub const ALL: [Self; 3] = [Self::Small, Self::Medium, Self::Large];

    /// The `log2` cells of the ROM, the rows of one step, the steps of one
    /// segment, and the most segments.
    fn shape(self) -> (usize, usize, usize, usize) {
        match self {
            Self::Small => (14, 64, 8, 10),
            Self::Medium => (15, 64, 16, 5),
            Self::Large => (16, 32, 16, 3),
        }
    }

    /// The `log2` cells of the RAM: a sixteenth of the ROM. A segment scans
    /// each cell of both, and the program tables in the ROM need far more
    /// cells than a run's stack and linear memory.
    fn ram_log2(self) -> usize {
        self.shape().0 - 4
    }

    /// The limits of the programs and runs that the size holds. Each memory
    /// family gets a fixed part of its space: in the RAM, half for linear
    /// memory and a quarter for the stack.
    pub fn limits(self) -> WasmLimits {
        let log2 = self.ram_log2();
        let (pc, function) = match self {
            Self::Small => (8, 5),
            Self::Medium => (9, 6),
            Self::Large => (10, 7),
        };
        WasmLimits {
            pc,
            choice: 1,
            function,
            event: 3,
            slot: 3,
            stack: log2 - 2,
            linear_memory: log2 - 1,
            global: log2 - 5,
            call_depth: log2 - 6,
            host_depth: 4,
            table: 1,
            table_index: log2 - 6,
        }
    }

    /// The trace rows of one step.
    pub fn rows(self) -> NonZeroUsize {
        NonZeroUsize::new(self.shape().1).expect("a step has rows")
    }

    pub fn geometry(self) -> Geometry {
        let (log2, _, n, s_max) = self.shape();
        Geometry::new(
            log2,
            self.ram_log2(),
            plan_ordered_slots(self.rows())
                .expect("fixed WASM slot plan")
                .len(),
            n,
            s_max,
        )
        .expect("each WASM size keeps SPEC.md section 7.1")
    }

    /// The application of the size: its regions (`WasmLimits::regions`) and
    /// the slot plan of `plan_ordered_slots`.
    pub fn application(self) -> Result<WasmApplication, WasmApplicationError> {
        WasmApplication::new(&self.geometry(), self.limits().regions()?.0, self.rows())
    }

    /// The trace rows of one segment.
    pub fn segment_rows(self) -> usize {
        let (_, rows, n, _) = self.shape();
        rows * n
    }

    /// The most trace rows of one run.
    pub fn max_rows(self) -> usize {
        self.segment_rows() * self.shape().3
    }

    /// The smallest size whose limits hold `limits` and whose runs hold
    /// `rows` trace rows.
    pub fn smallest(limits: &WasmLimits, rows: usize) -> Option<Self> {
        Self::ALL
            .into_iter()
            .find(|size| size.limits().holds(limits) && rows <= size.max_rows())
    }

    /// The smallest size that holds the program of `preload` and its run
    /// `trace`.
    pub fn for_run(preload: &WasmMemoryPreload, trace: &[WasmVmStep]) -> Result<Self, WasmApplicationError> {
        if trace.is_empty() {
            return Err(WasmApplicationError::EmptyTrace);
        }
        let mut limits = WasmLimits::of(&[preload], &[])?;
        for row in trace {
            limits.hold_row(&build_batched_witness(std::slice::from_ref(row), 1, 0)?)?;
        }
        Self::smallest(&limits, trace.len()).ok_or(WasmApplicationError::TooLarge(limits))
    }

    /// The step inputs of each segment of the run `trace` in this size: the
    /// fewest segments, padded as `step_inputs` pads. The iterator builds the
    /// witnesses of a segment when it yields that segment.
    pub fn segments(
        self,
        trace: &[WasmVmStep],
    ) -> Result<impl Iterator<Item = Result<Vec<WasmStep>, WasmApplicationError>> + '_, WasmApplicationError> {
        let (_, _, n, s_max) = self.shape();
        if trace.is_empty() {
            return Err(WasmApplicationError::EmptyTrace);
        }
        let segments = trace.len().div_ceil(self.segment_rows());
        if segments > s_max {
            return Err(WasmApplicationError::Steps {
                trace: trace.len(),
                steps: s_max * n,
                rows: self.rows(),
            });
        }
        Ok(trace
            .chunks(self.segment_rows())
            .map(move |segment| step_inputs(segment, n, self.rows())))
    }
}
