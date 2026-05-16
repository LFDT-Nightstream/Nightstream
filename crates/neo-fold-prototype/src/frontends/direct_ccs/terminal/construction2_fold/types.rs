//! Types and logging for the direct F' Construction-2 fold.

use super::*;

#[derive(Clone)]
pub(crate) struct DirectCcsConstruction2FoldContext {
    pub(crate) params: NeoParams,
    pub(crate) structure: CcsStructure<F>,
    pub(crate) dims: Dims,
    pub(crate) mat_digest: [F; 4],
    pub(crate) initial_claims: Vec<CeClaim<Commitment, F, K>>,
    pub(crate) initial_transcript: Option<SuperNeoIvcTranscriptSnapshot>,
    pub(crate) surface: DirectCcsChunkCircuitSurface,
    pub(crate) accumulator_in_digest: [u8; 32],
    pub(crate) accumulator_out_digest: [u8; 32],
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirectCcsConstruction2FoldShapeDelta {
    pub(crate) rows: usize,
    pub(crate) public_cols: usize,
    pub(crate) aux_cols: usize,
}

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub(crate) struct DirectCcsConstruction2FoldBreakdown {
    pub(crate) initial_transcript: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) initial_claims: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_in_digest: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_in_digest_check: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) nifs_v: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_digest: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_digest_check: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) accumulator_out_public_link: DirectCcsConstruction2FoldShapeDelta,
    pub(crate) total: DirectCcsConstruction2FoldShapeDelta,
}

impl DirectCcsConstruction2FoldContext {
    pub(crate) fn validate_digest_linkage(
        &self,
        in_digest: [u8; 32],
        out_digest: [u8; 32],
    ) -> Result<(), DirectCcsFPrimeSnarkError> {
        if self.accumulator_in_digest != in_digest || self.accumulator_out_digest != out_digest {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct F' Construction-2 fold context does not match terminal accumulator digest boundary".into(),
            ));
        }
        Ok(())
    }
}

impl DirectCcsConstruction2FoldBreakdown {
    pub(crate) fn log_lines(&self) -> Vec<String> {
        let mut lines =
            vec!["direct_ccs_ivc.construction2_fold_breakdown stage|rows|public_cols|aux_cols|primitive".to_owned()];
        push_log(
            &mut lines,
            "initial_transcript",
            self.initial_transcript,
            "allocate prior F' transcript state",
        );
        push_log(
            &mut lines,
            "initial_claims",
            self.initial_claims,
            "allocate prior F' carried CE claims",
        );
        push_log(
            &mut lines,
            "accumulator_in_digest",
            self.accumulator_in_digest,
            "Poseidon2 digest of incoming prior F' CE accumulator",
        );
        push_log(
            &mut lines,
            "accumulator_in_digest_check",
            self.accumulator_in_digest_check,
            "check incoming prior F' accumulator digest boundary",
        );
        push_log(
            &mut lines,
            "nested_nifs_v",
            self.nifs_v,
            "SuperNeo NIFS.V replay for the prior F' step",
        );
        push_log(
            &mut lines,
            "accumulator_out_digest",
            self.accumulator_out_digest,
            "Poseidon2 digest of outgoing prior F' CE accumulator",
        );
        push_log(
            &mut lines,
            "accumulator_out_digest_check",
            self.accumulator_out_digest_check,
            "check outgoing prior F' accumulator digest boundary",
        );
        push_log(
            &mut lines,
            "accumulator_out_public_link",
            self.accumulator_out_public_link,
            "link outgoing prior F' accumulator digest to terminal public image",
        );
        push_log(
            &mut lines,
            "total",
            self.total,
            "Construction-2 folded prior F' accumulator update",
        );
        lines
    }
}

fn push_log(lines: &mut Vec<String>, stage: &str, shape: DirectCcsConstruction2FoldShapeDelta, primitive: &str) {
    lines.push(format!(
        "direct_ccs_ivc.construction2_fold_breakdown {stage}|{}|{}|{}|{primitive}",
        shape.rows, shape.public_cols, shape.aux_cols
    ));
}
