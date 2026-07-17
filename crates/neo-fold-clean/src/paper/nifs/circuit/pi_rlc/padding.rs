//! Π_RLC-adjacent canonical encoding/padding phase.
//!
//! **Owns:** zero constraints for inactive X columns and padded y tails.
//! **Does not own:** active-coordinate identities, fold construction, or
//! quotient binding or paper CE arithmetic. **Emits constraints:** direct
//! zero-equality rows.
//! **Authority boundary:** inactive coordinates cannot carry values outside the
//! projection identities or cancel across folded inputs.
//!
//! | Stage child | Mathematical obligation | Arithmetic owner |
//! | --- | --- | --- |
//! | `padding.x` | Every inactive input/parent X coordinate is zero | `pi_rlc_circuit::x` |
//! | `padding.y_ring` | Every padded y_ring tail is zero | `pi_rlc_circuit::padded_k` |
//! | `padding.y_zcol` | Every padded y_zcol tail is zero | `pi_rlc_circuit::padded_k` |

use crate::engine::r1cs_circuit::builder::ProjectionGlueRole;
use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::reductions::pi_rlc_circuit::{
    enforce_rlc_padded_k_padding_glue, enforce_rlc_x_padding_glue, stage, RlcPaddedKVectorWires, RlcXWires,
};

use super::super::Error;

pub(super) fn enforce_x(builder: &mut R1csBuilder, wires: &RlcXWires) -> Result<(), Error> {
    builder.begin_encoding_stage(stage::PADDING);
    builder.begin_encoding_stage(stage::PADDING_X);
    let start = builder.rows();
    enforce_rlc_x_padding_glue(builder, wires)?;
    builder.record_projection_glue(ProjectionGlueRole::InactiveXZero, start);
    Ok(())
}

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

pub(super) fn enforce_y_zcol(builder: &mut R1csBuilder, wires: &RlcPaddedKVectorWires) -> Result<(), Error> {
    builder.begin_encoding_stage(stage::PADDING_Y_ZCOL);
    let start = builder.rows();
    enforce_rlc_padded_k_padding_glue(builder, wires)?;
    builder.record_projection_glue(ProjectionGlueRole::YZColPaddingZero, start);
    Ok(())
}
