use bellpepper_core::{ConstraintSystem, SynthesisError};
use neo_math::{KExtensions, K};
use p3_field::PrimeField64;

use crate::rv64im::ivc_snark::SpartanF;
use crate::rv64im::main_relation_circuit::k_field::KNumVar;
use crate::rv64im::main_relation_circuit::transcript::Poseidon2TranscriptCircuit;

pub(super) fn append_k_to_transcript<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    transcript: &mut Poseidon2TranscriptCircuit,
    raw_tag: u64,
    value: &KNumVar,
    value_hint: K,
    name: &str,
) -> Result<(), SynthesisError> {
    let coeffs = value_hint.as_coeffs();
    let coeff_fields = [value.c0, value.c1];
    let coeff_values = [
        SpartanF::from_canonical_u64(coeffs[0].as_canonical_u64()),
        SpartanF::from_canonical_u64(coeffs[1].as_canonical_u64()),
    ];
    transcript.append_const_fields_raw(
        cs.namespace(|| format!("{name}_tag")),
        &[SpartanF::from_canonical_u64(raw_tag)],
    )?;
    transcript.append_field_vars_raw(cs.namespace(|| format!("{name}_append")), &coeff_fields, &coeff_values)
}
