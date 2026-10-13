//! The first memory application's two-word machine `(pc, acc)` with its
//! program in ROM. Mirrors `Lifecycle/Nebula/Machine.lean`.
//!
//! Instruction word `v`: `op = v mod 4`, `arg = v / 4`. `0` halt, `1` load
//! (`acc ← RAM[arg]`), `2` store (`RAM[arg] ← acc`), `3` loadi (`acc ← arg`).
//! Port 0 fetches `ROM[pc]`; port 1 is the data port.

use super::records::PortAccess;

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct MachineState {
    pub pc: u64,
    pub acc: u64,
}

/// The fetch access of a non-idle step.
pub fn fetch(state: MachineState, word: u64) -> PortAccess {
    PortAccess {
        is_write: false,
        is_ram: false,
        addr: state.pc,
        vr: word,
        vw: word,
    }
}

/// The data access of instruction `word`; `data` is the value that port 1 reads.
pub fn data_port(state: MachineState, word: u64, data: u64) -> Option<PortAccess> {
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
pub fn exec(state: MachineState, word: u64, data: u64) -> MachineState {
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
