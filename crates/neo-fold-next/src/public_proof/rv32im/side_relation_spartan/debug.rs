//! Owns debug and measurement helpers for the side-binding Spartan shell.

use bellpepper_core::test_cs::TestConstraintSystem;
use spartan2::traits::circuit::SpartanCircuit;

use super::*;

pub fn debug_check_rv32im_side_binding_circuit(
    statement: &Rv32imSideBindingStatement,
    public: &Rv32imSideOpeningPublic,
    claim_witnesses: &[FamilyEvalClaimWitness],
) -> Result<(), SimpleKernelError> {
    let circuit = Rv32imSideBindingCircuit::from_claim_witnesses(statement, public, claim_witnesses)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM side binding debug synthesis failed: {err}")))?;
    if !cs.is_satisfied() {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM side binding circuit unsatisfied: {}",
            cs.which_is_unsatisfied()
                .unwrap_or_else(|| "unknown".into())
        )));
    }
    Ok(())
}

pub fn measure_rv32im_side_binding_circuit_constraints(
    statement: &Rv32imSideBindingStatement,
    public: &Rv32imSideOpeningPublic,
) -> Result<usize, SimpleKernelError> {
    let circuit = Rv32imSideBindingCircuit::dummy(statement, public)?;
    let mut cs = TestConstraintSystem::<SpartanF>::new();
    circuit
        .synthesize(&mut cs, &[], &[], None)
        .map_err(|err| SimpleKernelError::Bridge(format!("RV32IM side binding counting synthesis failed: {err}")))?;
    Ok(cs.num_constraints())
}
