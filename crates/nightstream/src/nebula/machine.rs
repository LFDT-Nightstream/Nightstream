//! The first memory application: the two-word machine `(pc, acc)` with its
//! program in ROM, over two ports. Its one private word is the idle flag.
//! Mirrors `Lifecycle/Nebula/Machine.lean` (the step) and
//! `Lifecycle/Nebula/MachineApp.lean` (`Machine.portApp`, the rows).
//!
//! Instruction word `v`: `op = v mod 4`, `arg = v / 4`. `0` halt, `1` load
//! (`acc ← RAM[arg]`), `2` store (`RAM[arg] ← acc`), `3` loadi (`acc ← arg`).
//! Port 0 fetches `ROM[pc]`; port 1 is the data port.

use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

use super::app::{Application, Memory, Transition};
use super::expr::Expr;
use super::records::PortAccess;
use super::rows::{bit_row, chunk, View};

/// The machine as an application.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct Machine;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MachineState {
    pub pc: u64,
    pub acc: u64,
}

/// One step of the machine: no port, or the instruction at `pc`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Step {
    Idle,
    Execute,
}

#[derive(Debug, thiserror::Error, PartialEq, Eq)]
pub enum MachineError {
    #[error("the program counter leaves the ROM")]
    ProgramCounter,
    #[error("a data address leaves the RAM")]
    DataAddress,
}

/// The fetch access of a non-idle step.
fn fetch(state: MachineState, word: u64) -> PortAccess {
    PortAccess {
        is_write: false,
        is_ram: false,
        addr: state.pc,
        vr: word,
        vw: word,
    }
}

/// The data access of instruction `word`; `data` is the value that port 1 reads.
fn data_port(state: MachineState, word: u64, data: u64) -> Option<PortAccess> {
    match word % 4 {
        1 => Some(PortAccess {
            is_write: false,
            is_ram: true,
            addr: word / 4,
            vr: data,
            vw: data,
        }),
        2 => Some(PortAccess {
            is_write: true,
            is_ram: true,
            addr: word / 4,
            vr: data,
            vw: state.acc,
        }),
        _ => None,
    }
}

/// The state after instruction `word`, with `data` read by port 1.
fn exec(state: MachineState, word: u64, data: u64) -> MachineState {
    match word % 4 {
        0 => state,
        1 => MachineState {
            pc: state.pc + 1,
            acc: data,
        },
        2 => MachineState {
            pc: state.pc + 1,
            acc: state.acc,
        },
        _ => MachineState {
            pc: state.pc + 1,
            acc: word / 4,
        },
    }
}

fn c(value: u64) -> Expr {
    Expr::constant(value)
}

impl Application for Machine {
    type Input = Step;
    type State = MachineState;
    type Error = MachineError;

    fn width(&self) -> usize {
        2
    }

    fn aux(&self) -> usize {
        1
    }

    /// Lean `Machine.portApp` is for `bOps = 2`.
    fn port_count(&self) -> usize {
        2
    }

    fn words(&self, state: &MachineState) -> Vec<u64> {
        vec![state.pc, state.acc]
    }

    /// `Machine.rows`, node for node.
    fn rows(&self, view: &View) -> Result<Vec<Expr>, crate::Error> {
        let idle = view.aux(0);
        let instruction = view.vr_bits(0);
        let is_load = instruction[0].clone() * (c(1) - instruction[1].clone());
        let is_store = (c(1) - instruction[0].clone()) * instruction[1].clone();
        let is_loadi = instruction[0].clone() * instruction[1].clone();
        let argument = chunk(instruction[2..].to_vec());
        let touches = || is_load.clone() + is_store.clone();
        Ok(vec![
            bit_row(idle.clone()),
            view.pad(0) - idle.clone(),
            view.is_write(0),
            view.is_ram(0),
            view.addr(0) - (c(1) - idle) * view.input(0),
            view.pad(1) - (c(1) - touches()),
            view.is_write(1) - is_store.clone(),
            view.is_ram(1) - touches(),
            view.addr(1) - touches() * argument.clone(),
            view.vw(1) - (is_store.clone() * view.input(1) + is_load.clone() * view.vr(1)),
            view.output(0) - (view.input(0) + is_load.clone() + is_store + is_loadi.clone()),
            view.output(1)
                - (is_load.clone() * view.vr(1)
                    + is_loadi.clone() * argument
                    + (c(1) - is_load - is_loadi) * view.input(1)),
        ])
    }

    fn step(
        &self,
        state: &MachineState,
        input: &Step,
        memory: &Memory,
    ) -> Result<Transition<MachineState>, MachineError> {
        let state = *state;
        match input {
            Step::Idle => Ok(Transition {
                next: state,
                ports: vec![None, None],
                aux: vec![F::ONE],
            }),
            Step::Execute => {
                let word = memory.rom(state.pc).ok_or(MachineError::ProgramCounter)?;
                let data = match word % 4 {
                    1 | 2 => memory.ram(word / 4).ok_or(MachineError::DataAddress)?,
                    _ => 0,
                };
                Ok(Transition {
                    next: exec(state, word, data),
                    ports: vec![Some(fetch(state, word)), data_port(state, word, data)],
                    aux: vec![F::ZERO],
                })
            }
        }
    }
}
