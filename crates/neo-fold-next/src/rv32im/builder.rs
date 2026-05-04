//! Owns executing the RV32IM parity slice into concrete steps and lowered rows.

use super::execute::{execute_step, ExecutedStep};
use super::isa::{Rv32BuildError, Rv32Program, Rv32State};
use super::lower::{lower_step, Rv32ExpandedRow};

#[derive(Clone, Debug, Eq, PartialEq)]
pub struct Rv32ProgramBuild {
    pub executed_steps: Vec<ExecutedStep>,
    pub rows: Vec<Rv32ExpandedRow>,
    pub final_state: Rv32State,
}

pub fn build_program(
    program: &Rv32Program,
    initial_state: &Rv32State,
    max_steps: usize,
) -> Result<Rv32ProgramBuild, Rv32BuildError> {
    if max_steps == 0 {
        return Err(Rv32BuildError::Program(
            "RV32 parity slice requires max_steps > 0".into(),
        ));
    }

    let mut state = initial_state.clone();
    let mut executed_steps = Vec::new();
    for step_index in 0..max_steps {
        let step = execute_step(program, &state, step_index)?;
        state = step.next.clone();
        executed_steps.push(step);
        if state.halted {
            break;
        }
    }

    if !state.halted {
        return Err(Rv32BuildError::Program(format!(
            "RV32 parity slice did not halt within {max_steps} steps"
        )));
    }

    let mut trace_index = 0usize;
    let mut rows = Vec::new();
    for step in &executed_steps {
        let lowered = lower_step(step, trace_index);
        trace_index += lowered.len();
        rows.extend(lowered);
    }
    Ok(Rv32ProgramBuild {
        executed_steps,
        rows,
        final_state: state,
    })
}
