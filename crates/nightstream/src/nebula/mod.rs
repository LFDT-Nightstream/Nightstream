//! Native side of Nebula memory checking (SPEC.md section 7) for any application of
//! the `Application` contract: the geometry and the program, the record
//! chains and challenge transcript, the carry, the segment runner that writes
//! each invocation's witness words, the SPEC.md Algorithm 10 terminal checks, the memory
//! circuit, the memory verifier, and the memory backend of catalog
//! applications (`CatalogBinding`). `Machine` is the first application.
//! A memory circuit comes from a pinned Lean-emitted package (`Package::load`)
//! or from the Rust-built relation (`Circuit::compile`), and
//! `tests/nebula/parity.rs` requires the machine's Rust relation to equal each
//! Lean package's, row for row.

mod app;
mod carry;
mod catalog;
mod circuit;
mod expr;
mod geometry;
mod machine;
mod records;
mod rows;
mod segment;
#[cfg(test)]
#[path = "../../tests/nebula/packages.rs"]
pub(crate) mod test_packages;
mod transcript;
mod words;

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

pub use app::{Application, Memory, Transition};
pub use carry::{Carry, Products};
pub use catalog::{CatalogBinding, CatalogBindingError, CatalogRowError, Region};
pub use expr::Expr;
pub use geometry::Geometry;
pub use machine::{Machine, MachineError, MachineState, Step};
pub use neo_application::{
    ColumnFamilySpec, ColumnRegistry, ColumnRegistryError, ColumnWidth, MemoryCatalog, MemoryCatalogError, MemoryKind,
    MemoryPortActivation, MemoryPortKind, MemoryPortSpec, MemoryPreload, MemorySpec,
};
pub use records::PortAccess;
pub use rows::View;
use segment::Cell;
pub use segment::{Context, Invocation, SegmentError};
pub use transcript::{state_words, Digest};

/// A Lean-emitted memory package of the machine: the memory geometry that it
/// proves and its pinned structural identifier. A program of the geometry is
/// a `Program`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Package {
    /// `FirstPlan.lean`: `artifacts/nightstream-fprime-option-b-nebula-v1-{light,refresh}-v1.json`.
    V1,
    /// `SecondPlan.lean`: `artifacts/nightstream-fprime-option-b-nebula-n1-{light,refresh}-v1.json`.
    N1,
}

impl Package {
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

    /// Load the package's option-B circuit: Lean's light and refresh
    /// packages of the memory program (`EmitMain --program`) and the finish
    /// package. `Circuit::load_packages` recomputes their identity
    /// and requires the package's pin.
    pub fn load(self, references: crate::References<'_>) -> Result<Circuit<Machine>, crate::Error> {
        Ok(Circuit {
            geometry: self.geometry(),
            app: Machine,
            circuit: crate::Circuit::load_packages(references, self.option_b_identity())?.into_memory(),
        })
    }

    /// The identity of the package's option-B light and refresh packages.
    fn option_b_identity(self) -> [u64; 4] {
        match self {
            Package::V1 => [
                14_739_391_730_004_183_567,
                4_652_337_019_402_923_359,
                12_693_096_347_412_764_448,
                16_208_379_304_942_263_586,
            ],
            Package::N1 => [
                12_101_932_483_783_393_099,
                1_747_503_734_875_263_425,
                17_844_470_795_173_320_248,
                10_869_441_032_500_514_034,
            ],
        }
    }

    /// The package's relation built in Rust, row for row as Lean builds it
    /// (`tests/nebula/parity.rs`).
    #[cfg(test)]
    pub(crate) fn application(self) -> Result<crate::application::ApplicationCircuit, crate::Error> {
        circuit::memory_circuit(&self.geometry(), &Machine)
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
pub struct Statement<S> {
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
/// the application size of `Circuit::compile` for `geometry` and `app`. It
/// builds no step package.
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
/// Stage 1 circuit. Only `Package::load` (a pinned Lean package) and
/// `Circuit::compile` (the Rust-built relation) make one, so a memory verifier
/// never takes its circuit from a prover. The memory prover and the memory
/// verifier can share it.
pub struct Circuit<A> {
    geometry: Geometry,
    app: A,
    circuit: crate::Circuit,
}

impl<A: Application> Circuit<A> {
    /// Build the memory relation of a valid, secure geometry (Lean
    /// `Geometry.Valid` and `Geometry.Secure`) and `app` in Rust, and compile
    /// it into the step packages of `references` (`Circuit::compile`). `app` must have the geometry's
    /// `b_ops` ports (Lean `PortApp.portsLength`).
    pub fn compile(references: crate::References<'_>, geometry: Geometry, app: A) -> Result<Self, crate::Error> {
        if !geometry.is_valid() || !geometry.is_secure() {
            return Err(crate::Error::Circuit("the memory geometry breaks SPEC.md section 7.1"));
        }
        check_port_count(&geometry, &app)?;
        let application = circuit::memory_circuit(&geometry, &app)?;
        Ok(Self {
            circuit: crate::Circuit::compile(references, application)?.into_memory(),
            geometry,
            app,
        })
    }

    /// Save a `Circuit::compile` circuit for a prover in another process
    /// (`Circuit::load`). The destination must not exist. A Lean
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
    pub fn prover(&self, engine: crate::Engine, minimum_security_bits: u32) -> Result<crate::Prover, crate::Error> {
        self.check_memory_security(minimum_security_bits)?;
        self.circuit.prover(engine, minimum_security_bits)
    }

    fn check_memory_security(&self, minimum_security_bits: u32) -> Result<(), crate::Error> {
        let available = self.geometry.memory_security_bits();
        if available < f64::from(minimum_security_bits) {
            return Err(neo_params::ParamsError::InsufficientStatisticalSecurity {
                required: minimum_security_bits,
                available: available.floor() as u32,
            }
            .into());
        }
        Ok(())
    }
}

/// Why a memory proof does not verify.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error("the program is for another geometry")]
    Geometry,
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
}

/// The memory verifier of one circuit: the SPEC.md Algorithm 10 terminal checks, then
/// the Stage 1 verifier of the circuit's relation. It verifies proofs of
/// every program of the circuit's geometry.
pub struct Verifier<A> {
    geometry: Geometry,
    app: A,
    stage1: crate::Verifier,
}

impl<A: Application + Clone> Verifier<A> {
    /// The verifier of a memory circuit. It may share the prover's circuit.
    /// `minimum_security_bits` bounds the folding profile and the memory
    /// fingerprint term (`Geometry::memory_security_bits`).
    pub fn new(circuit: &Circuit<A>, engine: crate::Engine, minimum_security_bits: u32) -> Result<Self, crate::Error> {
        circuit.check_memory_security(minimum_security_bits)?;
        Ok(Self {
            geometry: circuit.geometry,
            app: circuit.app.clone(),
            stage1: circuit
                .circuit
                .stage1_verifier(engine, minimum_security_bits)?,
        })
    }

    /// Decode untrusted proof bytes (`crate::Verifier::decode_proof`).
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<crate::Proof, crate::Error> {
        self.stage1.decode_proof(bytes)
    }

    /// Verify a memory proof of `program` (SPEC.md Algorithm 10): the terminal checks on
    /// the final carry, then the Stage 1 verification of `proof` with the
    /// initial and final states that the statement and the carry open. The
    /// initial state opens the start carry, which holds the program's
    /// `D_init` (`Program::d_init`).
    pub fn verify(
        &self,
        program: &Program,
        statement: &Statement<A::State>,
        final_carry: &Carry,
        proof: &crate::Proof,
    ) -> Result<(), VerifyError> {
        if program.geometry != self.geometry {
            return Err(VerifyError::Geometry);
        }
        let (initial, last) = terminal_states(&self.app, program, statement, final_carry)?;
        self.stage1
            .verify(&crate::State::new(statement.steps, initial, last), proof)?;
        Ok(())
    }

    /// The trusted compression key of the circuit (`crate::Verifier::compression_key`).
    pub fn compression_key(&self) -> Result<crate::CompressionKey, crate::Error> {
        self.stage1.compression_key()
    }

    /// Verify a finished memory proof (`crate::Prover::finish`)
    /// as `verify` does: the terminal checks, then
    /// `crate::Verifier::verify_final` with `key`.
    pub fn verify_final(
        &self,
        program: &Program,
        statement: &Statement<A::State>,
        final_carry: &Carry,
        key: &crate::CompressionKey,
        proof: &crate::FinalProof,
    ) -> Result<(), VerifyError> {
        if program.geometry != self.geometry {
            return Err(VerifyError::Geometry);
        }
        let (initial, last) = terminal_states(&self.app, program, statement, final_carry)?;
        self.stage1
            .verify_final(&crate::State::new(statement.steps, initial, last), key, proof)?;
        Ok(())
    }
}

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

/// A native memory run from the start carry, one segment at a time.
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
    pub fn new(context: Context<A>, initial: A::State) -> Self {
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

    pub fn context(&self) -> &Context<A> {
        &self.context
    }

    /// The Stage 1 state before the first invocation.
    pub fn initial_state(&self) -> Digest {
        self.context
            .state(&self.initial, &self.context.start_carry())
    }

    pub fn carry(&self) -> &Carry {
        &self.carry
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
    pub fn statement(&self) -> Statement<A::State> {
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
