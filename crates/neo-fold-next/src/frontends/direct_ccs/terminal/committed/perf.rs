use super::types::{DirectCcsR1csShapeDelta, DirectCcsTerminalCommittedPerf};
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS};

impl DirectCcsTerminalCommittedPerf {
    pub(crate) fn breakdown_log_lines(&self) -> Vec<String> {
        let b = self.breakdown;
        let stage_sum = b.public_input_alloc
            + b.boundary_input_alloc
            + b.packed_witness_alloc
            + b.public_boundary.total
            + b.public_commitment_shape
            + b.committed_image.total
            + b.terminal_body_with_sources
            + b.terminal_ajtai_commitment;
        let mut lines = vec![
            "direct_ccs_ivc.terminal_committed_breakdown stage|constraints".to_owned(),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_input_alloc|{}",
                b.public_input_alloc
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown boundary_input_alloc|{}",
                b.boundary_input_alloc
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown packed_witness_alloc|{}",
                b.packed_witness_alloc
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_boundary.total|{}",
                b.public_boundary.total
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_boundary.digest_checks|{}",
                b.public_boundary.digest_checks
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_boundary.x_i_bit_checks|{}",
                b.public_boundary.x_i_bit_checks
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_boundary.x_i_limb_links|{}",
                b.public_boundary.x_i_limb_links
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown public_commitment_shape|{}",
                b.public_commitment_shape
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown committed_image.total|{}",
                b.committed_image.total
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown committed_image.public_z_links|{}",
                b.committed_image.public_z_links
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown committed_image.constant_one_link|{}",
                b.committed_image.constant_one_link
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown committed_image.low_norm_bit_checks|{}",
                b.committed_image.low_norm_bit_checks
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown committed_image.padding_zero_checks|{}",
                b.committed_image.padding_zero_checks
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown terminal_body.with_sources|{}",
                b.terminal_body_with_sources
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown terminal_body.source_links|{}",
                b.terminal_body_source_links
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown terminal_body.without_source_links|{}",
                b.terminal_body_without_source_links
            ),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown terminal_ajtai_commitment|{}",
                b.terminal_ajtai_commitment
            ),
            format!("direct_ccs_ivc.terminal_committed_breakdown stage_sum|{stage_sum}"),
            format!("direct_ccs_ivc.terminal_committed_breakdown measured_total|{}", b.total),
            format!(
                "direct_ccs_ivc.terminal_committed_breakdown unattributed|{}",
                b.total.saturating_sub(stage_sum)
            ),
            "direct_ccs_ivc.terminal_committed_shape_breakdown stage|rows|public_cols|aux_cols|primitive".to_owned(),
        ];
        push_shape_log(
            &mut lines,
            "public_input_alloc",
            b.public_input_alloc_shape,
            "alloc terminal public statement fields",
        );
        push_shape_log(
            &mut lines,
            "boundary_input_alloc",
            b.boundary_input_alloc_shape,
            "alloc public Construction-2 boundary u_i=(C_i,x_i)",
        );
        push_shape_log(
            &mut lines,
            "packed_witness_alloc",
            b.packed_witness_alloc_shape,
            "alloc private packed low-norm R2 source image",
        );
        push_shape_log(
            &mut lines,
            "public_boundary.digest_checks",
            b.public_boundary.digest_checks_shape,
            "Poseidon2 digests for commitment and public boundary",
        );
        push_shape_log(
            &mut lines,
            "public_boundary.x_i_bit_checks",
            b.public_boundary.x_i_bit_checks_shape,
            "booleanize 256 public x_i bits",
        );
        push_shape_log(
            &mut lines,
            "public_boundary.x_i_limb_links",
            b.public_boundary.x_i_limb_links_shape,
            "pack x_i bits into 4 field limbs",
        );
        push_shape_log(
            &mut lines,
            "public_commitment_shape",
            b.public_commitment_shape_shape,
            "check public commitment dimensions",
        );
        push_shape_log(
            &mut lines,
            "committed_image.public_z_links",
            b.committed_image.public_z_links_shape,
            "link public x_i bits into committed source image",
        );
        push_shape_log(
            &mut lines,
            "committed_image.constant_one_link",
            b.committed_image.constant_one_link_shape,
            "force committed constant-one column",
        );
        push_shape_log(
            &mut lines,
            "committed_image.low_norm_bit_checks",
            b.committed_image.low_norm_bit_checks_shape,
            "boolean low-norm check for every committed source column",
        );
        push_shape_log(
            &mut lines,
            "committed_image.padding_zero_checks",
            b.committed_image.padding_zero_checks_shape,
            "force packed source padding to zero",
        );
        push_shape_log(
            &mut lines,
            "terminal_body.with_sources",
            b.terminal_body_shape,
            "latest F' body plus Construction-2 fold using committed sources",
        );
        push_shape_log(
            &mut lines,
            "terminal_ajtai_commitment",
            b.terminal_ajtai_commitment_shape,
            "linear Ajtai opening check for committed source image",
        );
        push_shape_log(
            &mut lines,
            "total",
            b.total_shape,
            "full terminal committed-step R1CS shape",
        );
        lines
    }
}

fn push_shape_log(lines: &mut Vec<String>, stage: &str, shape: DirectCcsR1csShapeDelta, primitive: &str) {
    lines.push(format!(
        "direct_ccs_ivc.terminal_committed_shape_breakdown {stage}|{}|{}|{}|{primitive}",
        shape.rows, shape.public_cols, shape.aux_cols
    ));
}

pub(super) fn shape_point(cs: &ShapeCS<NeoFoldDeciderEngine>) -> DirectCcsR1csShapeDelta {
    DirectCcsR1csShapeDelta {
        rows: cs.num_constraints(),
        public_cols: cs.num_inputs(),
        aux_cols: cs.num_aux(),
    }
}

pub(super) fn shape_delta(
    start: DirectCcsR1csShapeDelta,
    cs: &ShapeCS<NeoFoldDeciderEngine>,
) -> DirectCcsR1csShapeDelta {
    let end = shape_point(cs);
    DirectCcsR1csShapeDelta {
        rows: end.rows.saturating_sub(start.rows),
        public_cols: end.public_cols.saturating_sub(start.public_cols),
        aux_cols: end.aux_cols.saturating_sub(start.aux_cols),
    }
}
