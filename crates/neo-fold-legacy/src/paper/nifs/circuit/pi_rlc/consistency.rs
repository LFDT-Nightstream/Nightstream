//! Π_RLC transcript-digest consistency phase.
//!
//! **Owns:** equality between every Π_CCS output and the checked Π_DEC parent
//! for `fold_digest`.
//! **Does not own:** fold-view construction, projection identities, or transcript binding.
//! **Emits constraints:** `fold_digest` equality rows.
//! **Authority boundary:** the parent cannot substitute new shared values
//! after Π_CCS has fixed the input outputs. Whether these equalities are
//! independently necessary belongs to the transcript authority proof.
//!
//! | Stage child | Mathematical obligation | Arithmetic owner |
//! | --- | --- | --- |
//! | `consistency.fold_digest` | `input_i.fold_digest = parent.fold_digest` for every input | `pi_rlc_circuit::consistency` |

use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::reductions::pi_ccs_circuit::PiCcsOutputWires;
use crate::paper::reductions::pi_dec_circuit::DecInputWires;
use crate::paper::reductions::pi_rlc_circuit::{enforce_rlc_fold_digest_consistency, stage};

use super::super::Error;

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    outputs: &[PiCcsOutputWires],
    dec_wires: &DecInputWires,
) -> Result<(), Error> {
    builder.begin_encoding_stage(stage::CONSISTENCY);
    builder.begin_encoding_stage(stage::CONSISTENCY_FOLD_DIGEST);
    let input_fold_digests = outputs
        .iter()
        .map(|output| output.fold_digest_fields.as_slice())
        .collect::<Vec<_>>();
    enforce_rlc_fold_digest_consistency(builder, &input_fold_digests, &dec_wires.parent.fold_digest_fields)?;
    Ok(())
}
