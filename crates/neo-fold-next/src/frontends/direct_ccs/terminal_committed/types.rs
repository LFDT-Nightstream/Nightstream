use std::collections::BTreeMap;
use std::sync::Arc;

use neo_math::F;
use serde::{Deserialize, Serialize};
use thiserror::Error;

use super::DirectCcsTerminalFPrimeCircuit;
use crate::construction2::Construction2PublicBoundary;
use crate::construction2_terminal::TerminalPrivateColumnEncoding;
use crate::spartan_backend::{
    NeoFoldDeciderEngine, NeoFoldDeciderProverKey, NeoFoldDeciderVerifierKey, SpartanF, SplitR1CSShape,
};

#[derive(Debug, Error)]
pub(crate) enum DirectCcsTerminalError {
    #[error("{0}")]
    Bridge(String),
}

pub(super) type SimpleKernelError = DirectCcsTerminalError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct DirectCcsTerminalCommittedProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalCommittedRelation {
    pub(super) public_boundary: Construction2PublicBoundary,
    pub(super) assignment: DirectCcsTerminalR2Assignment,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalCommittedCircuit {
    pub(super) public_boundary: Construction2PublicBoundary,
    pub(super) assignment: DirectCcsTerminalR2Assignment,
}

#[derive(Clone, Debug)]
pub(crate) struct DirectCcsTerminalCommittedPerf {
    pub constraints: usize,
    pub public_inputs: usize,
    pub committed_width: usize,
    pub commitment_words: usize,
    pub source_values: usize,
    pub source_bit_values: usize,
    pub source_u32_values: usize,
    pub source_u64_values: usize,
    pub unclassified_private_values: usize,
    pub breakdown: DirectCcsTerminalCommittedConstraintBreakdown,
    pub sizes: [usize; 10],
    pub nnz: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsR1csShapeDelta {
    pub rows: usize,
    pub public_cols: usize,
    pub aux_cols: usize,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsTerminalCommittedConstraintBreakdown {
    pub public_input_alloc: usize,
    pub public_input_alloc_shape: DirectCcsR1csShapeDelta,
    pub boundary_input_alloc: usize,
    pub boundary_input_alloc_shape: DirectCcsR1csShapeDelta,
    pub packed_witness_alloc: usize,
    pub packed_witness_alloc_shape: DirectCcsR1csShapeDelta,
    pub public_boundary: DirectCcsPublicBoundaryConstraintBreakdown,
    pub public_commitment_shape: usize,
    pub public_commitment_shape_shape: DirectCcsR1csShapeDelta,
    pub committed_image: DirectCcsCommittedImageConstraintBreakdown,
    pub terminal_body_with_sources: usize,
    pub terminal_body_source_links: usize,
    pub terminal_body_without_source_links: usize,
    pub terminal_body_shape: DirectCcsR1csShapeDelta,
    pub terminal_ajtai_commitment: usize,
    pub terminal_ajtai_commitment_shape: DirectCcsR1csShapeDelta,
    pub total: usize,
    pub total_shape: DirectCcsR1csShapeDelta,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsPublicBoundaryConstraintBreakdown {
    pub digest_checks: usize,
    pub digest_checks_shape: DirectCcsR1csShapeDelta,
    pub x_i_bit_checks: usize,
    pub x_i_bit_checks_shape: DirectCcsR1csShapeDelta,
    pub x_i_limb_links: usize,
    pub x_i_limb_links_shape: DirectCcsR1csShapeDelta,
    pub total: usize,
    pub total_shape: DirectCcsR1csShapeDelta,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DirectCcsCommittedImageConstraintBreakdown {
    pub public_z_links: usize,
    pub public_z_links_shape: DirectCcsR1csShapeDelta,
    pub constant_one_link: usize,
    pub constant_one_link_shape: DirectCcsR1csShapeDelta,
    pub low_norm_bit_checks: usize,
    pub low_norm_bit_checks_shape: DirectCcsR1csShapeDelta,
    pub padding_zero_checks: usize,
    pub padding_zero_checks_shape: DirectCcsR1csShapeDelta,
    pub total: usize,
    pub total_shape: DirectCcsR1csShapeDelta,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalCommittedKeyPair {
    pub(crate) prover: Arc<NeoFoldDeciderProverKey>,
    pub(crate) verifier: Arc<NeoFoldDeciderVerifierKey>,
    pub(crate) perf: DirectCcsTerminalCommittedPerf,
}

#[derive(Clone)]
pub(super) struct DirectCcsTerminalR2Assignment {
    pub(super) layout: DirectCcsTerminalR2Layout,
    pub(super) terminal_public_values: Vec<F>,
    pub(super) r2_public_values: Vec<F>,
    pub(super) witness_values: Vec<F>,
    pub(super) terminal_circuit: DirectCcsTerminalFPrimeCircuit,
}

#[derive(Clone)]
pub(super) struct DirectCcsTerminalR2Layout {
    pub(super) source_labels: Vec<String>,
    pub(super) source_encodings: Vec<TerminalPrivateColumnEncoding>,
    pub(super) source_offsets: Vec<usize>,
    pub(super) source_by_label: BTreeMap<String, usize>,
    pub(super) source_limb_width: usize,
}

#[derive(Clone, Debug)]
pub(super) struct DirectCcsTerminalShapeExport {
    pub(super) split_shape: SplitR1CSShape<NeoFoldDeciderEngine>,
    pub(super) expected_public_values: Vec<SpartanF>,
    pub(super) private_witness_labels: Vec<String>,
}
