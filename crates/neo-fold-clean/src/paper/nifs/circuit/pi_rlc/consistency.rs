//! Π_RLC-adjacent transcript/NC sidecar consistency phase.
//!
//! **Owns:** equality between every Π_CCS output and the checked Π_DEC parent
//! for `s_col` and `fold_digest`. Neither field belongs to the paper CE point.
//! **Does not own:** fold-view construction, projection identities, or transcript binding.
//! **Emits constraints:** `s_col` and `fold_digest` equality rows.
//! **Authority boundary:** the parent cannot substitute new shared values
//! after Π_CCS has fixed the input outputs. Whether these equalities are
//! independently necessary belongs to the transcript/NC authority proof.
//!
//! | Stage child | Mathematical obligation | Arithmetic owner |
//! | --- | --- | --- |
//! | `consistency.s_col` | `input_i.s_col = parent.s_col` for every input | `pi_rlc_circuit::consistency` |
//! | `consistency.fold_digest` | `input_i.fold_digest = parent.fold_digest` for every input | `pi_rlc_circuit::consistency` |

use crate::engine::r1cs_circuit::R1csBuilder;
use crate::paper::reductions::pi_ccs_split_nc_circuit::SplitNcPiCcsOutputWires;
use crate::paper::reductions::pi_dec_circuit::DecInputWires;
use crate::paper::reductions::pi_rlc_circuit::{
    enforce_rlc_fold_digest_consistency, enforce_rlc_s_col_consistency, stage,
};

use super::super::Error;

pub(super) fn enforce(
    builder: &mut R1csBuilder,
    outputs: &[SplitNcPiCcsOutputWires],
    dec_wires: &DecInputWires,
) -> Result<(), Error> {
    builder.begin_encoding_stage(stage::CONSISTENCY);
    builder.begin_encoding_stage(stage::CONSISTENCY_S_COL);
    let input_s_cols = outputs
        .iter()
        .map(|output| output.s_col.clone())
        .collect::<Vec<_>>();
    enforce_rlc_s_col_consistency(builder, &input_s_cols, &dec_wires.parent.s_col)?;

    builder.begin_encoding_stage(stage::CONSISTENCY_FOLD_DIGEST);
    let input_fold_digests = outputs
        .iter()
        .map(|output| output.fold_digest_fields.as_slice())
        .collect::<Vec<_>>();
    enforce_rlc_fold_digest_consistency(builder, &input_fold_digests, &dec_wires.parent.fold_digest_fields)?;
    Ok(())
}
