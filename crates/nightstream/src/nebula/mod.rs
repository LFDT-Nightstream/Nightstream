//! Native side of the first Nebula memory application (spec §4–§13): the plan,
//! the machine, the record chains and challenge transcript, the carry, the
//! segment runner that writes each invocation's witness words, the spec §13
//! terminal checks, and the memory verifier of the two Lean-emitted packages.
//! The circuit and its proofs are Lean-owned (`Lifecycle/Nebula/`); this module
//! only computes values for them.

mod carry;
mod machine;
mod plan;
mod records;
mod segment;
mod transcript;
mod words;

use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

pub use carry::{Carry, Products};
pub use machine::MachineState;
pub use plan::Plan;
use segment::Cell;
pub use segment::{Context, Invocation, SegmentError, Step};
pub use transcript::{state_words, Digest};

/// A Lean-emitted memory package: the plan that it proves and its pinned
/// structural identifier. The verifier key binds the plan (spec §4.1), so a
/// memory run and a memory verifier exist only for these packages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Package {
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-v1.json` (`FirstPlan.lean`).
    V1,
    /// `artifacts/nightstream-fprime-stage2-nebula-memory-n1.json` (`SecondPlan.lean`).
    N1,
}

impl Package {
    pub fn plan(self) -> Plan {
        match self {
            Package::V1 => Plan::first(),
            Package::N1 => Plan::second(),
        }
    }

    /// Load the package's circuit from its bytes. `Circuit::load_package`
    /// recomputes the structural identifier and requires the package's pin.
    pub fn load(self, bytes: &[u8]) -> Result<PackageCircuit, crate::Error> {
        Ok(PackageCircuit {
            package: self,
            circuit: crate::Circuit::load_package(bytes, self.structural_identifier())?,
        })
    }

    fn structural_identifier(self) -> [u64; 4] {
        match self {
            Package::V1 => [
                7606418978967494273,
                7012554739737736582,
                16451068804121006463,
                2787927620498818402,
            ],
            Package::N1 => [
                2732929784355259483,
                2352438795120520828,
                6712609518222072634,
                2244383376532064420,
            ],
        }
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
    pub fn prover(&self, engine: crate::Engine, minimum_security_bits: u32) -> Result<crate::Prover, crate::Error> {
        self.circuit.prover(engine, minimum_security_bits)
    }
}

/// Why a memory proof does not verify.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
}

/// The memory verifier of one package: the spec §13 terminal checks, then
/// the Stage 1 verifier of the package's relation.
pub struct Verifier {
    context: Context,
    stage1: crate::Verifier,
}

impl Verifier {
    /// The verifier of a loaded package. It may share the prover's circuit.
    pub fn new(
        circuit: &PackageCircuit,
        engine: crate::Engine,
        minimum_security_bits: u32,
    ) -> Result<Self, crate::Error> {
        Ok(Self {
            context: Context::new(circuit.package),
            stage1: crate::Verifier::from_package(&circuit.circuit, engine, minimum_security_bits)?,
        })
    }

    /// Decode untrusted proof bytes (`crate::Verifier::decode_proof`).
    pub fn decode_proof(&self, bytes: &[u8]) -> Result<crate::Proof, crate::Error> {
        self.stage1.decode_proof(bytes)
    }

    /// Verify a memory proof (spec §13): the terminal checks on the final
    /// carry, then the Stage 1 verification of `proof` with the initial and
    /// final states that the statement and the carry open.
    pub fn verify(&self, statement: &Statement, final_carry: &Carry, proof: &crate::Proof) -> Result<(), VerifyError> {
        let (initial, last) = self.context.terminal_states(statement, final_carry)?;
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
        let plan = self.plan();
        if final_carry.idx != plan.n as u64 {
            return Err(TerminalError::OpenCarry);
        }
        if statement.segments == 0 || statement.segments > plan.s_max as u64 {
            return Err(TerminalError::SegmentRange);
        }
        if statement.steps != statement.segments * plan.n as u64 {
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
        let memory = segment::initial_memory(context.plan());
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
