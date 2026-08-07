//! Π_RLC ring-evaluation padding phase.
//!
//! **Owns:** zero constraints for padded ring-evaluation tails.
//! **Does not own:** compact X shape, active-coordinate identities, fold construction, or
//! quotient binding or paper CE arithmetic. **Emits constraints:** direct
//! zero-equality rows.
//! **Authority boundary:** padded ring lanes cannot carry values outside the
//! projection identities.
//!
//! | Stage child | Mathematical obligation | Arithmetic owner |
//! | --- | --- | --- |
//! | `padding.y_ring` | Every padded y_ring tail is zero | `pi_rlc_circuit::padded_k` |

use crate::engine::r1cs_circuit::builder::ProjectionGlueRole;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::reductions::pi_rlc_circuit::{enforce_rlc_padded_k_padding_glue, stage, RlcPaddedKVectorWires};

use super::super::Error;

pub(super) fn enforce_y_ring(
    builder: &mut R1csBuilder,
    wires: &RlcPaddedKVectorWires,
    row: usize,
) -> Result<(), Error> {
    builder.begin_encoding_stage(stage::PADDING_Y_RING);
    let start = builder.rows();
    enforce_rlc_padded_k_padding_glue(builder, wires)?;
    builder.record_projection_glue(ProjectionGlueRole::YRingPaddingZero { row }, start);
    Ok(())
}
