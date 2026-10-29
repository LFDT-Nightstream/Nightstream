//! Native side of the first Nebula memory application (spec §4–§13): the
//! geometry and the program, the machine, the record chains and challenge
//! transcript, the carry, the segment runner that writes each invocation's
//! witness words, the spec §13 terminal checks, and the memory verifier of the
//! two Lean-emitted packages.
//! The prover and the verifier load the Lean-owned circuit (`Lifecycle/Nebula/`).
//! The test-only `circuit` module builds the same relation in Rust, and
//! `tests/nebula/parity.rs` requires its rows to equal the Lean package's.

mod carry;
#[cfg(test)]
mod circuit;
#[cfg(test)]
mod expr;
mod geometry;
mod machine;
mod records;
#[cfg(test)]
mod rows;
mod segment;
mod transcript;
mod words;

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

pub use carry::{Carry, Products};
pub use geometry::Geometry;
pub use machine::MachineState;
use segment::Cell;
pub use segment::{Context, Invocation, SegmentError, Step};
pub use transcript::{state_words, Digest};

/// A Lean-emitted memory package: the memory geometry that it proves and its
/// pinned structural identifier. The verifier key binds the geometry
/// (spec §4.3), so a memory run and a memory verifier exist only for these
/// packages. A program of the geometry is a `Program`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Package {
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-v1.json` (`FirstPlan.lean`).
    V1,
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-n1.json` (`SecondPlan.lean`).
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
        Program::new(self, rom, ram).expect("a default program fits its geometry")
    }

    /// Load the package's circuit from its bytes. `Circuit::load_package`
    /// recomputes the structural identifier and requires the package's pin.
    pub fn load(self, bytes: &[u8]) -> Result<PackageCircuit, crate::Error> {
        Ok(PackageCircuit {
            package: self,
            circuit: crate::Circuit::load_package(bytes, self.structural_identifier())?,
        })
    }

    /// The package's relation built in Rust, row for row as Lean builds it
    /// (`tests/nebula/parity.rs`).
    #[cfg(test)]
    pub(crate) fn application(
        self,
    ) -> Result<crate::application::ApplicationCircuit, crate::application::ApplicationError> {
        circuit::memory_circuit(&self.geometry())
    }

    fn structural_identifier(self) -> [u64; 4] {
        match self {
            Package::V1 => [
                3172882483849057180,
                8567705847331220519,
                8030146092339936618,
                10432117065660733435,
            ],
            Package::N1 => [
                5065087233990754590,
                15874026430280222715,
                856469097843029471,
                17098359820501084920,
            ],
        }
    }
}

/// A program of a package's geometry: its ROM and RAM images of 32-bit words
/// (spec §4.1). The program enters a proof only through `D_init`, its key,
/// which `Program::new` computes once from the images (spec §9.2). One package
/// serves every program of its geometry.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Program {
    package: Package,
    rom: Vec<u32>,
    ram: Vec<u32>,
    d_init: Digest,
}

/// Why images are not a program of a package's geometry.
#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum ProgramError {
    #[error("the ROM image needs 2^r words and the RAM image 2^μ words")]
    Size,
}

impl Program {
    /// Check the image sizes against the geometry of `package`, as Lean
    /// `Plan.Valid` does (`2^r` ROM words and `2^μ` RAM words), then compute
    /// `D_init`.
    pub fn new(package: Package, rom: Vec<u32>, ram: Vec<u32>) -> Result<Self, ProgramError> {
        let geometry = package.geometry();
        if rom.len() != geometry.rom_size() || ram.len() != geometry.ram_size() {
            return Err(ProgramError::Size);
        }
        let d_init = segment::initial_root(&geometry, &rom, &ram);
        Ok(Self {
            package,
            rom,
            ram,
            d_init,
        })
    }

    /// `D_init`: the IS root of the images (spec §9.2), the program's key.
    pub fn d_init(&self) -> Digest {
        self.d_init
    }
}

/// The spec §13 public statement of a memory proof.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Statement {
    pub steps: u64,
    pub initial: MachineState,
    pub final_: MachineState,
    pub segments: u64,
    pub final_ts: u64,
    pub final_root: Digest,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum TerminalError {
    /// Ob1: an integer of the statement that becomes a field word is not
    /// below `q`, so it has the word of a smaller integer.
    #[error("a statement integer is not a canonical field element")]
    NonCanonical,
    #[error("the final carry is inside a segment")]
    OpenCarry,
    #[error("the segment count is outside 1..=S_max")]
    SegmentRange,
    #[error("the step count differs from S · N")]
    StepCount,
    #[error("a statement field differs from the final carry")]
    Field,
}

/// The circuit of a memory package, which only `Package::load` builds. The
/// memory prover and the memory verifier share it.
pub struct PackageCircuit {
    package: Package,
    circuit: crate::Circuit,
}

impl PackageCircuit {
    /// The prover of the package. `minimum_security_bits` bounds the folding
    /// profile and the memory fingerprint term (`Geometry::memory_security_bits`).
    pub fn prover(&self, engine: crate::Engine, minimum_security_bits: u32) -> Result<crate::Prover, crate::Error> {
        self.check_memory_security(minimum_security_bits)?;
        self.circuit.prover(engine, minimum_security_bits)
    }

    fn check_memory_security(&self, minimum_security_bits: u32) -> Result<(), crate::Error> {
        let available = self.package.geometry().memory_security_bits();
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
    #[error("the program is for another package")]
    Package,
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
}

/// The memory verifier of one package: the spec §13 terminal checks, then
/// the Stage 1 verifier of the package's relation. It verifies proofs of
/// every program of the package's geometry.
pub struct Verifier {
    package: Package,
    stage1: crate::Verifier,
}

impl Verifier {
    /// The verifier of a loaded package. It may share the prover's circuit.
    /// `minimum_security_bits` bounds the folding profile and the memory
    /// fingerprint term (`Geometry::memory_security_bits`).
    pub fn new(
        circuit: &PackageCircuit,
        engine: crate::Engine,
        minimum_security_bits: u32,
    ) -> Result<Self, crate::Error> {
        circuit.check_memory_security(minimum_security_bits)?;
        Ok(Self {
            package: circuit.package,
            stage1: crate::Verifier::from_package(&circuit.circuit, engine, minimum_security_bits)?,
        })
    }

    /// Decode untrusted proof bytes (`crate::Verifier::decode_proof`).
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<crate::Proof, crate::Error> {
        self.stage1.decode_proof(bytes)
    }

    /// Verify a memory proof of `program` (spec §13): the terminal checks on
    /// the final carry, then the Stage 1 verification of `proof` with the
    /// initial and final states that the statement and the carry open. The
    /// initial state opens the start carry, which holds the program's
    /// `D_init` (`Program::d_init`).
    pub fn verify(
        &self,
        program: &Program,
        statement: &Statement,
        final_carry: &Carry,
        proof: &crate::Proof,
    ) -> Result<(), VerifyError> {
        if program.package != self.package {
            return Err(VerifyError::Package);
        }
        let (initial, last) = Context::new(program).terminal_states(statement, final_carry)?;
        self.stage1
            .verify(&crate::State::new(statement.steps, initial, last), proof)?;
        Ok(())
    }
}

impl Context {
    /// Spec §13: check the final carry against the statement and return the
    /// initial and final Stage 1 states for the Stage 1 terminal check.
    pub fn terminal_states(
        &self,
        statement: &Statement,
        final_carry: &Carry,
    ) -> Result<(Digest, Digest), TerminalError> {
        // Ob1: an integer that becomes a field word must be below `q`, as Lean
        // reads each word with `.val`. The patterns name every field, so a new
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
        let MachineState {
            pc: initial_pc,
            acc: initial_acc,
        } = *initial;
        let MachineState {
            pc: final_pc,
            acc: final_acc,
        } = *final_;
        if [initial_pc, initial_acc, final_pc, final_acc, *final_ts]
            .iter()
            .any(|word| *word >= F::ORDER_U64)
        {
            return Err(TerminalError::NonCanonical);
        }
        let geometry = self.geometry();
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
        Ok((
            self.state(statement.initial, &self.start_carry()),
            self.state(statement.final_, final_carry),
        ))
    }
}

/// A native memory run from the start carry, one segment at a time.
#[derive(Clone, Debug)]
pub struct Run {
    context: Context,
    initial: MachineState,
    carry: Carry,
    state: MachineState,
    memory: Vec<Cell>,
    steps: u64,
}

impl Run {
    pub fn new(context: Context, initial: MachineState) -> Self {
        let memory = segment::initial_memory(&context.program().rom, &context.program().ram);
        Self {
            carry: context.start_carry(),
            context,
            initial,
            state: initial,
            memory,
            steps: 0,
        }
    }

    pub fn context(&self) -> &Context {
        &self.context
    }

    /// The Stage 1 state before the first invocation.
    pub fn initial_state(&self) -> Digest {
        self.context
            .state(self.initial, &self.context.start_carry())
    }

    pub fn carry(&self) -> &Carry {
        &self.carry
    }

    /// Run the next segment and return its invocations in order.
    pub fn segment(&mut self, steps: &[Step]) -> Result<Vec<Invocation>, SegmentError> {
        let segment = self
            .context
            .run_segment(self.carry, self.state, &self.memory, steps)?;
        self.carry = segment.carry;
        self.state = segment.state;
        self.memory = segment.memory;
        self.steps += steps.len() as u64;
        Ok(segment.invocations)
    }

    /// The spec §13 statement of the run so far.
    pub fn statement(&self) -> Statement {
        Statement {
            steps: self.steps,
            initial: self.initial,
            final_: self.state,
            segments: self.carry.seg_idx,
            final_ts: self.carry.ts,
            final_root: self.carry.mem_root,
        }
    }
}
