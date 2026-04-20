use std::io::{self, Write};
use std::time::Instant;

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, SynthesisError};

use crate::rv64im::ivc_snark::SpartanF;

pub(super) fn mark_unsatisfied<CS: ConstraintSystem<SpartanF>>(cs: &mut CS, label: &str) -> Result<(), SynthesisError> {
    cs.enforce(|| label, |lc| lc + CS::one(), |lc| lc + CS::one(), |lc| lc);
    Ok(())
}

pub(super) fn emit_synthesize_trace(trace_prefix: Option<&str>, label: &str, started: Instant) {
    if let Some(prefix) = trace_prefix {
        eprintln!("{prefix}.{label}={:.2}ms", started.elapsed().as_secs_f64() * 1_000.0);
        let _ = io::stderr().flush();
    }
}

pub(super) fn enforce_pc_range<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    value: &AllocatedNum<SpartanF>,
    ell: u64,
) -> Result<(), SynthesisError> {
    if ell == 0 || ell != 1 {
        return Err(SynthesisError::Unsatisfiable);
    }
    cs.enforce(
        || format!("{label}_eq_one"),
        |lc| lc + value.get_variable() - (SpartanF::from_canonical_u64(1), CS::one()),
        |lc| lc + CS::one(),
        |lc| lc,
    );
    Ok(())
}
