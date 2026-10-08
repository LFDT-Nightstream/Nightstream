//! Native side of the first Nebula memory application (spec §4–§13): the plan,
//! the machine, the record chains and challenge transcript, the carry, the
//! segment runner that writes each invocation's witness words, and the spec
//! §13 terminal checks. The circuit and its proofs are Lean-owned
//! (`Lifecycle/Nebula/`); this module only computes values for them.

mod carry;
mod machine;
mod plan;
mod records;
mod segment;
mod transcript;
mod words;

pub use carry::{Carry, Products};
pub use machine::MachineState;
pub use plan::Plan;
pub use segment::{Cell, Context, Invocation, Segment, SegmentError, Step};
pub use transcript::{state_words, Digest};

/// The structural identifier of the Lean-emitted first memory-application
/// package (`artifacts/nightstream-fprime-stage2-nebula-memory-v1.json`).
/// A verifier loads the package only with this identifier.
pub const PACKAGE_STRUCTURAL_IDENTIFIER: [u64; 4] = [
    14802976229900588367,
    17868216444329475912,
    15715525866008245071,
    6369654844452736898,
];

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
    #[error("the final carry is inside a segment")]
    OpenCarry,
    #[error("the segment count is outside 1..=S_max")]
    SegmentRange,
    #[error("the step count differs from S · N")]
    StepCount,
    #[error("a statement field differs from the final carry")]
    Field,
}

/// Why a memory proof does not verify.
#[derive(Debug, thiserror::Error)]
pub enum VerifyError {
    #[error(transparent)]
    Terminal(#[from] TerminalError),
    #[error(transparent)]
    Proof(#[from] crate::Error),
}

impl Context {
    /// Verify a memory proof (spec §13): the terminal checks on the final
    /// carry, then the Stage 1 verification of `proof` with the initial and
    /// final states that the statement and the carry open. `verifier` must
    /// come from the package with `PACKAGE_STRUCTURAL_IDENTIFIER`.
    pub fn verify(
        &self,
        verifier: &crate::Verifier,
        statement: &Statement,
        final_carry: &Carry,
        proof: &crate::Proof,
    ) -> Result<(), VerifyError> {
        let (initial, last) = self.terminal_states(statement, final_carry)?;
        verifier.verify(&crate::State::new(statement.steps, initial, last), proof)?;
        Ok(())
    }

    /// Spec §13: check the final carry against the statement and return the
    /// initial and final Stage 1 states for the Stage 1 terminal check.
    pub fn terminal_states(
        &self,
        statement: &Statement,
        final_carry: &Carry,
    ) -> Result<(Digest, Digest), TerminalError> {
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
