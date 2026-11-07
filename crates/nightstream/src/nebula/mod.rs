//! Native side of Nebula memory checking (SPEC.md section 7) for any application of
//! the `Application` contract: the geometry and the program, the record
//! chains and challenge transcript, the carry, the segment runner that writes
//! each invocation's witness words, the SPEC.md Algorithm 10 terminal checks,
//! the memory circuit, the memory prover and verifier, the memory backend of
//! catalog applications (`CatalogBinding`), and the application of an R1CS
//! relation of one row (`R1csApplication`). A public memory circuit comes
//! from the Rust-built relation (`Circuit::compile`). The tests also load the
//! pinned Lean-emitted packages of the toy `Machine` (`Package::load`), and
//! `tests/nebula/parity.rs` requires the machine's Rust relation to equal
//! each Lean package's, row for row.

mod app;
mod carry;
mod catalog;
mod circuit;
mod expr;
mod geometry;
#[cfg(test)]
mod machine;
mod proving;
mod r1cs;
mod records;
mod rows;
mod segment;
mod transcript;
mod words;

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

pub use app::{Application, Memory, Transition};
pub(crate) use carry::Carry;
#[cfg(test)]
pub(crate) use catalog::CatalogBinding;
pub use catalog::{pack_regions, CatalogBindingError, CatalogRowError, Region};
pub use expr::Expr;
pub use geometry::Geometry;
#[cfg(test)]
pub(crate) use machine::{Machine, MachineError, MachineState, Step};
use neo_application::MemoryPreload;
pub use proving::{FinalProof, Proof, ProveError, Prover, Proving, Verified, Verifier, VerifyError};
pub use r1cs::{batched_slots, R1csApplication, R1csApplicationError, R1csStepError, RowRelation, Size, Terms};
pub use records::PortAccess;
pub use rows::View;
use segment::Cell;
pub(crate) use segment::Context;
pub use segment::{Invocation, SegmentError};
pub use transcript::Digest;

/// A Lean-emitted memory package of the machine: the memory geometry that it
/// proves and its pinned structural identifier. A program of the geometry is
/// a `Program`.
#[cfg(test)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum Package {
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-v1.json` (`FirstPlan.lean`).
    V1,
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-n1.json` (`SecondPlan.lean`).
    N1,
}

#[cfg(test)]
impl Package {
    /// Whether `identifier` pins a memory package, whose verifier is
    /// `nebula::Circuit::verifier`.
    pub(crate) fn pinned(identifier: [u64; 4]) -> bool {
        [Package::V1, Package::N1]
            .into_iter()
            .any(|package| package.structural_identifier() == identifier)
    }

    /// The memory geometry that the package proves.
    pub fn geometry(self) -> Geometry {
        match self {
            Package::V1 => Geometry::first(),
            Package::N1 => Geometry::second(),
        }
    }

    /// The default program of the Lean plan that the package was emitted
    /// from (`FirstPlan.plan`, `SecondPlan.plan`).
    pub fn program(self) -> Program {
        let (rom, ram) = match self {
            Package::V1 => (vec![23, 2, 1, 0], vec![0; 4]),
            Package::N1 => (vec![1, 6, 5, 2], vec![7, 0, 0, 0, 0, 0, 0, 0]),
        };
        Program::new(self.geometry(), rom, ram).expect("a default program fits its geometry")
    }

    /// Load the package's circuit from its bytes. `Circuit::load_package`
    /// recomputes the structural identifier and requires the package's pin.
    pub fn load(self, bytes: &[u8]) -> Result<Circuit<Machine>, crate::Error> {
        Ok(Circuit {
            geometry: self.geometry(),
            app: Machine,
            circuit: crate::Circuit::load_package(bytes, self.structural_identifier())?,
        })
    }

    /// The package's relation built in Rust, row for row as Lean builds it
    /// (`tests/nebula/parity.rs`).
    pub(crate) fn application(self) -> Result<crate::application::ApplicationCircuit, crate::Error> {
        circuit::memory_circuit(&self.geometry(), &Machine)
    }

    fn structural_identifier(self) -> [u64; 4] {
        match self {
            Package::V1 => [
                16124593816834730366,
                3526578406157962078,
                13420800549100596370,
                2876334031716046261,
            ],
            Package::N1 => [
                8020921799501217406,
                18145530998824220440,
                8946753667183584570,
                9385901100025494867,
            ],
        }
    }
}

/// A program of a memory geometry: its ROM and RAM images of 32-bit words
/// (SPEC.md section 7.1). The program enters a proof only through `D_init`, its key,
/// which `Program::new` computes once from the images (SPEC.md section 7.4). One
/// circuit serves every program of its geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    geometry: Geometry,
    rom: Vec<u32>,
    ram: Vec<u32>,
    d_init: Digest,
}

/// Why images are not a program of a geometry.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    #[error("the geometry breaks SPEC.md section 7.1")]
    Geometry,
    #[error("the ROM image needs 2^r words and the RAM image 2^μ words")]
    Size,
}

impl Program {
    /// Check the geometry (Lean `Geometry.Valid`) and the image sizes, as
    /// Lean `Plan.Valid` does (`2^r` ROM words and `2^μ` RAM words), then
    /// compute `D_init`.
    pub fn new(geometry: Geometry, rom: Vec<u32>, ram: Vec<u32>) -> Result<Self, ProgramError> {
        if !geometry.is_valid() {
            return Err(ProgramError::Geometry);
        }
        if rom.len() != geometry.rom_size() || ram.len() != geometry.ram_size() {
            return Err(ProgramError::Size);
        }
        let d_init = segment::initial_root(&geometry, &rom, &ram);
        Ok(Self {
            geometry,
            rom,
            ram,
            d_init,
        })
    }

    /// `D_init`: the IS root of the images (SPEC.md section 7.4), the program's key.
    pub fn d_init(&self) -> Digest {
        self.d_init
    }
}

/// The SPEC.md Algorithm 10 public statement of a memory proof over application states `S`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub(crate) struct Statement<S> {
    pub steps: u64,
    pub initial: S,
    pub final_: S,
    pub segments: u64,
    pub final_ts: u64,
    pub final_root: Digest,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TerminalError {
    /// Canonical decoding (SPEC.md Algorithm 10, step 2): an integer of the
    /// statement that becomes a field word is not below `q`, so it has the
    /// word of a smaller integer.
    #[error("a statement integer is not a canonical field element")]
    NonCanonical,
    /// A state of the statement has another number of words than the
    /// application's width (Lean `Fin width → F`).
    #[error("a statement state has another word count than the application's width")]
    Width,
    #[error("the final carry is inside a segment")]
    OpenCarry,
    #[error("the segment count is outside 1..=S_max")]
    SegmentRange,
    #[error("the step count differs from S · N")]
    StepCount,
    #[error("a statement field differs from the final carry")]
    Field,
}

fn check_port_count<A: Application>(geometry: &Geometry, app: &A) -> Result<(), crate::Error> {
    if app.port_count() != geometry.b_ops {
        return Err(crate::Error::Circuit(
            "the application's port count is not the geometry's b_ops",
        ));
    }
    Ok(())
}

/// One named part of the application size of a memory circuit.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct SizePart {
    pub name: &'static str,
    pub size: crate::ApplicationSize,
}

/// The parts of the memory circuit of `geometry` and `app`: the phases of its
/// rows (the sponges, the hash bindings, the memory rows, and the
/// application's rows), then the regions of its witness words. They sum to
/// the application size of `Circuit::compile` for `geometry` and `app`, and
/// `scripts/fprime_breakdown.py` charts them. It builds no Stage 1 package.
pub fn size_parts<A: Application>(geometry: &Geometry, app: &A) -> Result<Vec<SizePart>, crate::Error> {
    if !geometry.is_valid() {
        return Err(crate::Error::Circuit("the memory geometry breaks SPEC.md section 7.1"));
    }
    check_port_count(geometry, app)?;
    let (_, phases) = circuit::memory_circuit_phases(geometry, app)?;
    let size = |witness_words, local_words, rows| crate::ApplicationSize {
        witness_words,
        local_words,
        rows,
    };
    let regions = words::Layout::new(geometry, app.width(), app.aux())
        .memory_regions()
        .into_iter()
        .chain([
            ("Application state words", 2 * app.width()),
            ("Application private words", app.aux()),
        ]);
    Ok(phases
        .into_iter()
        .map(|phase| SizePart {
            name: phase.name,
            size: size(0, phase.local_words, phase.rows),
        })
        .chain(regions.map(|(name, words)| SizePart {
            name,
            size: size(words, 0, 0),
        }))
        .collect())
}

/// The circuit of a memory relation: its geometry, its application, and the
/// Stage 1 circuit. Only `Circuit::compile` (the Rust-built relation) makes a
/// public one, so a memory verifier never takes its circuit from a prover.
/// The memory prover and the memory verifier can share it.
pub struct Circuit<A> {
    geometry: Geometry,
    app: A,
    circuit: crate::Circuit,
}

impl<A: Application> Circuit<A> {
    /// Build the memory relation of a valid, secure geometry (Lean
    /// `Geometry.Valid` and `Geometry.Secure`) and `app` in Rust, and compile
    /// it with the embedded Stage 1 reference package. `app` must have the
    /// geometry's `b_ops` ports (Lean `PortApp.portsLength`).
    pub fn compile(geometry: Geometry, app: A) -> Result<Self, crate::Error> {
        if !geometry.is_valid() || !geometry.is_secure() {
            return Err(crate::Error::Circuit("the memory geometry breaks SPEC.md section 7.1"));
        }
        check_port_count(&geometry, &app)?;
        let application = circuit::memory_circuit(&geometry, &app)?;
        Ok(Self {
            circuit: crate::Circuit::compile_memory(application)?,
            geometry,
            app,
        })
    }

    /// Save a `Circuit::compile` circuit for a prover in another process
    /// (`crate::Prover::load`). The destination must not exist. A Lean
    /// package has no Rust application to save. A verifier compiles its own
    /// circuit and never loads this file.
    pub fn write(&self, path: impl AsRef<std::path::Path>) -> Result<(), crate::Error> {
        self.circuit.write(path)
    }

    /// The size of the memory relation and the application in the F′ step,
    /// or `None` for a Lean package.
    pub fn application_size(&self) -> Option<crate::ApplicationSize> {
        self.circuit.application_size()
    }

    /// The prover of the circuit. `minimum_security_bits` bounds the folding
    /// profile and the memory fingerprint term (`Geometry::memory_security_bits`).
    pub fn prover(&self, engine: crate::Engine, minimum_security_bits: u32) -> Result<Prover<A>, crate::Error>
    where
        A: Clone,
    {
        memory_security(&self.geometry, minimum_security_bits)?;
        Ok(Prover::new(
            self.geometry,
            self.app.clone(),
            self.circuit.prover(engine, minimum_security_bits)?,
        ))
    }

    /// The verifier of the circuit. It may share the prover's circuit.
    /// `minimum_security_bits` bounds the folding profile and the memory
    /// fingerprint term (`Geometry::memory_security_bits`).
    pub fn verifier(&self, engine: crate::Engine, minimum_security_bits: u32) -> Result<Verifier<A>, crate::Error>
    where
        A: Clone,
    {
        memory_security(&self.geometry, minimum_security_bits)?;
        Ok(Verifier::new(
            self.geometry,
            self.app.clone(),
            crate::Verifier::of_memory(&self.circuit, engine, minimum_security_bits)?,
        ))
    }
}

/// Charge all allowed segment challenges and the final memory comparison
/// against the caller's minimum (SECURITY_MODEL.md, trust item 7).
fn memory_security(geometry: &Geometry, minimum_security_bits: u32) -> Result<(), crate::Error> {
    let q = u128::from(F::ORDER_U64);
    let available = (geometry.max_tuples() as u128)
        .checked_mul(2)
        .and_then(|count| count.checked_mul((geometry.s_max as u128) + 1))
        .and_then(|error| (q * q).checked_div(error))
        .and_then(u128::checked_ilog2)
        .unwrap_or(0);
    if available < minimum_security_bits {
        return Err(neo_params::ParamsError::InsufficientStatisticalSecurity {
            required: minimum_security_bits,
            available,
        }
        .into());
    }
    Ok(())
}

#[cfg(test)]
impl<A: Application> Context<A> {
    /// SPEC.md Algorithm 10: check the final carry against the statement and return the
    /// initial and final Stage 1 states for the Stage 1 terminal check.
    pub fn terminal_states(
        &self,
        statement: &Statement<A::State>,
        final_carry: &Carry,
    ) -> Result<(Digest, Digest), TerminalError> {
        terminal_states(&self.app, self.program(), statement, final_carry)
    }
}

fn terminal_states<A: Application>(
    app: &A,
    program: &Program,
    statement: &Statement<A::State>,
    final_carry: &Carry,
) -> Result<(Digest, Digest), TerminalError> {
    // Canonical decoding: an integer that becomes a field word must be below `q`, as Lean
    // reads each word with `.val`. The pattern names every field, so a new
    // field does not compile until it is classified here. `steps`,
    // `segments`, and the carry counters must equal values below `q`
    // (checked below), and `final_root` is field-typed.
    let Statement {
        steps: _,
        initial,
        final_,
        segments: _,
        final_ts,
        final_root: _,
    } = statement;
    let (initial, final_) = (app.words(initial), app.words(final_));
    if initial.len() != app.width() || final_.len() != app.width() {
        return Err(TerminalError::Width);
    }
    if initial
        .iter()
        .chain(&final_)
        .chain([final_ts])
        .any(|word| *word >= F::ORDER_U64)
    {
        return Err(TerminalError::NonCanonical);
    }
    let geometry = program.geometry;
    if final_carry.idx != geometry.n as u64 {
        return Err(TerminalError::OpenCarry);
    }
    if statement.segments == 0 || statement.segments > geometry.s_max as u64 {
        return Err(TerminalError::SegmentRange);
    }
    if statement.steps != statement.segments * geometry.n as u64 {
        return Err(TerminalError::StepCount);
    }
    if final_carry.seg_idx != statement.segments
        || final_carry.ts != statement.final_ts
        || final_carry.mem_root != statement.final_root
    {
        return Err(TerminalError::Field);
    }
    let start = Carry::start(geometry.n, program.d_init);
    Ok((
        segment::state_digest(&initial, &start),
        segment::state_digest(&final_, final_carry),
    ))
}

/// A native memory run from the start carry, one segment at a time, with no
/// proof. A caller can check a run with it before proving; `Proving` runs one
/// inside.
#[derive(Clone, Debug)]
pub struct Run<A: Application> {
    context: Context<A>,
    initial: A::State,
    carry: Carry,
    state: A::State,
    memory: Vec<Cell>,
    steps: u64,
}

impl<A: Application> Run<A> {
    pub fn new(app: A, program: &Program, initial: A::State) -> Self {
        let context = Context::new(app, program);
        let memory = segment::initial_memory(&context.program().rom, &context.program().ram);
        Self {
            carry: context.start_carry(),
            context,
            state: initial.clone(),
            initial,
            memory,
            steps: 0,
        }
    }

    #[cfg(test)]
    pub(crate) fn context(&self) -> &Context<A> {
        &self.context
    }

    /// The Stage 1 state before the first invocation.
    pub(crate) fn initial_state(&self) -> Digest {
        self.context
            .state(&self.initial, &self.context.start_carry())
    }

    pub(crate) fn carry(&self) -> &Carry {
        &self.carry
    }

    /// The application state after the last segment.
    pub fn state(&self) -> &A::State {
        &self.state
    }

    /// The number of memory accesses so far (the carry's `ts`).
    pub fn timestamp(&self) -> u64 {
        self.carry.ts
    }

    /// Run the next segment and return its invocations in order.
    pub fn segment(&mut self, inputs: &[A::Input]) -> Result<Vec<Invocation>, SegmentError> {
        let segment = self
            .context
            .run_segment(self.carry, &self.state, &self.memory, inputs)?;
        self.carry = segment.carry;
        self.state = segment.state;
        self.memory = segment.memory;
        self.steps += inputs.len() as u64;
        Ok(segment.invocations)
    }

    /// The SPEC.md Algorithm 10 statement of the run so far.
    #[cfg(test)]
    pub(crate) fn statement(&self) -> Statement<A::State> {
        Statement {
            steps: self.steps,
            initial: self.initial.clone(),
            final_: self.state.clone(),
            segments: self.carry.seg_idx,
            final_ts: self.carry.ts,
            final_root: self.carry.mem_root,
        }
    }
}

#[cfg(test)]
#[path = "../../tests/nebula/memory.rs"]
mod memory;
