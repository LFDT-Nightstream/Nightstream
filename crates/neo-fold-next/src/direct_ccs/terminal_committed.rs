//! Owns the direct-CCS terminal Construction-2 committed-step proof.
//!
//! This is the non-VM analogue of the RV32IM terminal `F'` committed-step
//! boundary: the public output is `u_i = (C_i, x_i)`, where `C_i` opens to a
//! low-norm SuperNeo-packed source image linked to the latest direct-CCS F'
//! terminal circuit.

use std::collections::{BTreeMap, HashMap};
use std::sync::{Arc, Mutex, OnceLock};

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError, Variable};
use neo_ajtai::{
    get_global_pp_seeded_params_for_dims, has_global_pp_for_dims, set_global_pp_seeded, AjtaiSModule, Commitment,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::{D, F};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use spartan2::{
    bellpepper::{r1cs::SpartanWitness, solver::SatisfyingAssignment},
    traits::{transcript::TranscriptEngineTrait, Engine},
};
use thiserror::Error;

use super::ivc::DirectCcsTerminalFPrimeCircuit;
use crate::construction2::{
    Construction2Commitment, Construction2FreshInstance, Construction2PublicBoundary, CONSTRUCTION2_COMMITMENT_RAW_TAG,
    CONSTRUCTION2_ENC_INST_BITS, CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
};
use crate::construction2_terminal::{
    alloc_terminal_boundary_public_inputs, collect_private_witness_labels, committed_nc_range_error,
    enforce_boolean_allocated, enforce_packed_padding_zero, enforce_public_commitment_shape,
    enforce_terminal_ajtai_commitment, enforce_terminal_boundary_digests, low_norm_encoded_values, native_to_spartan,
    padded_private_witness_labels, terminal_boundary_public_values, Construction2TerminalBoundaryInputs,
    Construction2TerminalBoundaryView, TerminalPrivateColumnEncoding,
};
use crate::spartan_backend::{
    NeoFoldDeciderEngine, NeoFoldDeciderProverKey, NeoFoldDeciderSnark, NeoFoldDeciderVerifierKey, R1CSSNARKTrait,
    ShapeCS, SpartanCircuit, SpartanF, SpartanShape, SplitR1CSShape,
};
use crate::superneo_circuit::witness::{alloc_packed_mat_witness, PackedWitnessVar};
use crate::witness_layout::{commit_cols_for_full_width, encode_vector_for_full_width};

#[derive(Debug, Error)]
pub(crate) enum DirectCcsTerminalError {
    #[error("{0}")]
    Bridge(String),
}

type SimpleKernelError = DirectCcsTerminalError;

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub(crate) struct DirectCcsTerminalCommittedProof {
    pub snark_data: Vec<u8>,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalCommittedRelation {
    public_boundary: Construction2PublicBoundary,
    assignment: DirectCcsTerminalR2Assignment,
}

#[derive(Clone)]
pub(crate) struct DirectCcsTerminalCommittedCircuit {
    public_boundary: Construction2PublicBoundary,
    assignment: DirectCcsTerminalR2Assignment,
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

static DIRECT_CCS_TERMINAL_COMMITTED_SETUP_CACHE: OnceLock<
    Mutex<HashMap<[u8; 32], DirectCcsTerminalCommittedKeyPair>>,
> = OnceLock::new();

fn shape_point(cs: &ShapeCS<NeoFoldDeciderEngine>) -> DirectCcsR1csShapeDelta {
    DirectCcsR1csShapeDelta {
        rows: cs.num_constraints(),
        public_cols: cs.num_inputs(),
        aux_cols: cs.num_aux(),
    }
}

fn shape_delta(start: DirectCcsR1csShapeDelta, cs: &ShapeCS<NeoFoldDeciderEngine>) -> DirectCcsR1csShapeDelta {
    let end = shape_point(cs);
    DirectCcsR1csShapeDelta {
        rows: end.rows.saturating_sub(start.rows),
        public_cols: end.public_cols.saturating_sub(start.public_cols),
        aux_cols: end.aux_cols.saturating_sub(start.aux_cols),
    }
}

#[derive(Clone)]
struct DirectCcsTerminalR2Assignment {
    layout: DirectCcsTerminalR2Layout,
    terminal_public_values: Vec<F>,
    r2_public_values: Vec<F>,
    witness_values: Vec<F>,
    terminal_circuit: DirectCcsTerminalFPrimeCircuit,
}

#[derive(Clone)]
struct DirectCcsTerminalR2Layout {
    source_labels: Vec<String>,
    source_encodings: Vec<TerminalPrivateColumnEncoding>,
    source_offsets: Vec<usize>,
    source_by_label: BTreeMap<String, usize>,
    source_limb_width: usize,
}

#[derive(Clone, Debug)]
struct DirectCcsTerminalShapeExport {
    split_shape: SplitR1CSShape<NeoFoldDeciderEngine>,
    expected_public_values: Vec<SpartanF>,
    private_witness_labels: Vec<String>,
}

impl DirectCcsTerminalCommittedRelation {
    pub(crate) fn from_terminal_circuit(circuit: DirectCcsTerminalFPrimeCircuit) -> Result<Self, SimpleKernelError> {
        let assignment = DirectCcsTerminalR2Assignment::from_terminal_circuit(circuit)?;
        let commitment = assignment.commitment()?;
        let fresh_instance = Construction2FreshInstance::from_parts(
            Construction2Commitment::from_commitment(commitment),
            assignment.terminal_circuit.construction2_x_i()?,
        );
        let public_boundary = Construction2PublicBoundary::from_fresh_instance(&fresh_instance);
        Ok(Self {
            public_boundary,
            assignment,
        })
    }

    pub(crate) fn public_boundary(&self) -> &Construction2PublicBoundary {
        &self.public_boundary
    }

    pub(crate) fn committed_circuit(&self) -> DirectCcsTerminalCommittedCircuit {
        DirectCcsTerminalCommittedCircuit {
            public_boundary: self.public_boundary.clone(),
            assignment: self.assignment.clone(),
        }
    }

    pub(crate) fn measure(&self) -> Result<DirectCcsTerminalCommittedPerf, SimpleKernelError> {
        let circuit = self.committed_circuit();
        let public_inputs = circuit
            .public_values()
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed public IO failed: {err}")))?
            .len();
        let mut cs = ShapeCS::<NeoFoldDeciderEngine>::new();
        let breakdown = circuit
            .measure_with_breakdown(&mut cs)
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed shape failed: {err}")))?;
        Ok(DirectCcsTerminalCommittedPerf {
            constraints: cs.num_constraints(),
            public_inputs,
            committed_width: self.assignment.committed_width()?,
            commitment_words: self.public_boundary.commitment_data.len(),
            source_values: self.assignment.layout.source_labels.len(),
            source_bit_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::Bit),
            source_u32_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::U32),
            source_u64_values: self
                .assignment
                .layout
                .source_encoding_count(TerminalPrivateColumnEncoding::U64),
            unclassified_private_values: 0,
            breakdown,
            sizes: [0; 10],
            nnz: 0,
        })
    }
}

impl SpartanCircuit<NeoFoldDeciderEngine> for DirectCcsTerminalCommittedCircuit {
    fn public_values(&self) -> Result<Vec<SpartanF>, SynthesisError> {
        let mut values = self
            .assignment
            .terminal_public_values
            .iter()
            .map(native_to_spartan)
            .collect::<Vec<_>>();
        values.extend(terminal_committed_boundary_public_values(&self.public_boundary));
        Ok(values)
    }

    fn shared<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn precommitted<CS: ConstraintSystem<SpartanF>>(
        &self,
        _: &mut CS,
        _: &[AllocatedNum<SpartanF>],
    ) -> Result<Vec<AllocatedNum<SpartanF>>, SynthesisError> {
        Ok(Vec::new())
    }

    fn num_challenges(&self) -> usize {
        0
    }

    fn synthesize<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        _: &[AllocatedNum<SpartanF>],
        _: &[AllocatedNum<SpartanF>],
        _: Option<&[SpartanF]>,
    ) -> Result<(), SynthesisError> {
        let terminal_public_inputs = self
            .assignment
            .terminal_public_values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("terminal_public_{idx}")), || {
                    Ok(native_to_spartan(value))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        let boundary = self.alloc_public_boundary_inputs(cs)?;
        let (committed_width, packed_z) = self.allocate_committed_packed_z(cs)?;
        self.enforce_public_boundary(cs, &terminal_public_inputs, &boundary)?;
        self.enforce_public_commitment_shape(cs, &packed_z, &boundary)?;
        self.enforce_committed_image(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        self.synthesize_terminal_with_committed_sources(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        self.enforce_terminal_commitment(cs, &packed_z, &boundary.commitment_data)?;
        Ok(())
    }
}

impl DirectCcsTerminalCommittedCircuit {
    fn measure_with_breakdown(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
    ) -> Result<DirectCcsTerminalCommittedConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsTerminalCommittedConstraintBreakdown::default();

        let before = shape_point(cs);
        let terminal_public_inputs = self
            .assignment
            .terminal_public_values
            .iter()
            .enumerate()
            .map(|(idx, value)| {
                AllocatedNum::alloc_input(cs.namespace(|| format!("terminal_public_{idx}")), || {
                    Ok(native_to_spartan(value))
                })
            })
            .collect::<Result<Vec<_>, _>>()?;
        out.public_input_alloc_shape = shape_delta(before, cs);
        out.public_input_alloc = out.public_input_alloc_shape.rows;

        let before = shape_point(cs);
        let boundary = self.alloc_public_boundary_inputs(cs)?;
        out.boundary_input_alloc_shape = shape_delta(before, cs);
        out.boundary_input_alloc = out.boundary_input_alloc_shape.rows;

        let before = shape_point(cs);
        let (committed_width, packed_z) = self.allocate_committed_packed_z(cs)?;
        out.packed_witness_alloc_shape = shape_delta(before, cs);
        out.packed_witness_alloc = out.packed_witness_alloc_shape.rows;

        out.public_boundary = self.measure_public_boundary(cs, &terminal_public_inputs, &boundary)?;

        let before = shape_point(cs);
        self.enforce_public_commitment_shape(cs, &packed_z, &boundary)?;
        out.public_commitment_shape_shape = shape_delta(before, cs);
        out.public_commitment_shape = out.public_commitment_shape_shape.rows;

        out.committed_image = self.measure_committed_image(cs, &terminal_public_inputs, &packed_z, committed_width)?;

        let before = shape_point(cs);
        out.terminal_body_source_links =
            self.synthesize_terminal_with_committed_sources(cs, &terminal_public_inputs, &packed_z, committed_width)?;
        out.terminal_body_shape = shape_delta(before, cs);
        out.terminal_body_with_sources = out.terminal_body_shape.rows;
        out.terminal_body_without_source_links = out
            .terminal_body_with_sources
            .saturating_sub(out.terminal_body_source_links);

        let before = shape_point(cs);
        self.enforce_terminal_commitment(cs, &packed_z, &boundary.commitment_data)?;
        out.terminal_ajtai_commitment_shape = shape_delta(before, cs);
        out.terminal_ajtai_commitment = out.terminal_ajtai_commitment_shape.rows;
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }

    fn alloc_public_boundary_inputs<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<Construction2TerminalBoundaryInputs, SynthesisError> {
        alloc_terminal_boundary_public_inputs(
            cs,
            "direct_terminal_boundary",
            &direct_terminal_boundary_view(&self.public_boundary),
        )
    }

    fn enforce_public_boundary<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_terminal_boundary_digests(
            cs,
            boundary,
            CONSTRUCTION2_COMMITMENT_RAW_TAG,
            CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
            "direct_terminal_boundary",
        )?;

        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != CONSTRUCTION2_ENC_INST_BITS || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for limb_idx in 0usize..4 {
            let mut packed = LinearCombination::<SpartanF>::zero();
            for bit_idx in 0usize..64 {
                let bit_offset = limb_idx
                    .checked_mul(64)
                    .and_then(|value| value.checked_add(bit_idx))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let public_idx = x_range
                    .start
                    .checked_add(bit_offset)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                enforce_boolean_allocated(
                    &mut cs.namespace(|| format!("direct_terminal_x_i_public_bit_{public_idx}")),
                    &terminal_public_inputs[public_idx],
                    &format!("direct_terminal_x_i_public_bit_{public_idx}"),
                );
                packed = packed
                    + (
                        SpartanF::from_canonical_u64(1u64 << bit_idx),
                        terminal_public_inputs[public_idx].get_variable(),
                    );
            }
            cs.enforce(
                || format!("direct_terminal_boundary_x_i_limb_{limb_idx}_eq"),
                |_| packed,
                |lc| lc + CS::one(),
                |lc| lc + boundary.x_i[limb_idx].get_variable(),
            );
        }
        Ok(())
    }

    fn measure_public_boundary(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<DirectCcsPublicBoundaryConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsPublicBoundaryConstraintBreakdown::default();

        let before = shape_point(cs);
        enforce_terminal_boundary_digests(
            cs,
            boundary,
            CONSTRUCTION2_COMMITMENT_RAW_TAG,
            CONSTRUCTION2_PUBLIC_BOUNDARY_RAW_TAG,
            "direct_terminal_boundary",
        )?;
        out.digest_checks_shape = shape_delta(before, cs);
        out.digest_checks = out.digest_checks_shape.rows;

        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != CONSTRUCTION2_ENC_INST_BITS || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for limb_idx in 0usize..4 {
            let mut packed = LinearCombination::<SpartanF>::zero();
            for bit_idx in 0usize..64 {
                let bit_offset = limb_idx
                    .checked_mul(64)
                    .and_then(|value| value.checked_add(bit_idx))
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let public_idx = x_range
                    .start
                    .checked_add(bit_offset)
                    .ok_or(SynthesisError::Unsatisfiable)?;
                let before = shape_point(cs);
                enforce_boolean_allocated(
                    &mut cs.namespace(|| format!("direct_terminal_x_i_public_bit_{public_idx}")),
                    &terminal_public_inputs[public_idx],
                    &format!("direct_terminal_x_i_public_bit_{public_idx}"),
                );
                let delta = shape_delta(before, cs);
                out.x_i_bit_checks += delta.rows;
                out.x_i_bit_checks_shape.rows += delta.rows;
                out.x_i_bit_checks_shape.public_cols += delta.public_cols;
                out.x_i_bit_checks_shape.aux_cols += delta.aux_cols;
                packed = packed
                    + (
                        SpartanF::from_canonical_u64(1u64 << bit_idx),
                        terminal_public_inputs[public_idx].get_variable(),
                    );
            }
            let before = shape_point(cs);
            cs.enforce(
                || format!("direct_terminal_boundary_x_i_limb_{limb_idx}_eq"),
                |_| packed,
                |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
                |lc| lc + boundary.x_i[limb_idx].get_variable(),
            );
            let delta = shape_delta(before, cs);
            out.x_i_limb_links += delta.rows;
            out.x_i_limb_links_shape.rows += delta.rows;
            out.x_i_limb_links_shape.public_cols += delta.public_cols;
            out.x_i_limb_links_shape.aux_cols += delta.aux_cols;
        }
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }

    fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        boundary: &Construction2TerminalBoundaryInputs,
    ) -> Result<(), SynthesisError> {
        enforce_public_commitment_shape(cs, packed_z, boundary, "direct_terminal_boundary")
    }

    fn allocate_committed_packed_z<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
    ) -> Result<(usize, PackedWitnessVar), SynthesisError> {
        let full_width = self
            .assignment
            .committed_width()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let packed_native = self
            .assignment
            .committed_packed_witness()
            .map_err(|_| SynthesisError::Unsatisfiable)?;
        let packed_cols = commit_cols_for_full_width(full_width);
        if packed_native.rows() != D || packed_native.cols() != packed_cols {
            return Err(SynthesisError::Unsatisfiable);
        }
        let packed_z = alloc_packed_mat_witness(
            &mut cs.namespace(|| "direct_terminal_r2_packed_z"),
            &packed_native,
            "direct_terminal_r2_packed_z",
        )?;
        Ok((full_width, packed_z))
    }

    fn enforce_committed_image<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<(), SynthesisError> {
        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != self.assignment.r2_public_values.len() || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..self.assignment.r2_public_values.len() {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            cs.enforce(
                || format!("direct_terminal_r2_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc + terminal_public_inputs[x_range.start + public_idx].get_variable(),
            );
        }
        let constant_one_col = committed_width
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let constant_one = packed_z.logical_entry(committed_width, constant_one_col)?;
        cs.enforce(
            || "direct_terminal_r2_constant_one_link",
            |lc| lc + constant_one.get_variable(),
            |lc| lc + CS::one(),
            |lc| lc + CS::one(),
        );

        for logical_col in 0..committed_width {
            let value = packed_z.logical_entry(committed_width, logical_col)?;
            enforce_boolean_allocated(
                &mut cs.namespace(|| format!("direct_terminal_r2_low_norm_bit_{logical_col}")),
                &value,
                &format!("direct_terminal_r2_low_norm_bit_{logical_col}"),
            );
        }
        enforce_packed_padding_zero(cs, packed_z, committed_width, "direct_terminal_r2_padding_zero")
    }

    fn measure_committed_image(
        &self,
        cs: &mut ShapeCS<NeoFoldDeciderEngine>,
        terminal_public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<DirectCcsCommittedImageConstraintBreakdown, SynthesisError> {
        let start = shape_point(cs);
        let mut out = DirectCcsCommittedImageConstraintBreakdown::default();

        let x_range = self.assignment.terminal_circuit.construction2_x_bit_range();
        if x_range.len() != self.assignment.r2_public_values.len() || x_range.end > terminal_public_inputs.len() {
            return Err(SynthesisError::Unsatisfiable);
        }
        for public_idx in 0..self.assignment.r2_public_values.len() {
            let packed_entry = packed_z.logical_entry(committed_width, public_idx)?;
            let before = shape_point(cs);
            cs.enforce(
                || format!("direct_terminal_r2_public_z_link_{public_idx}"),
                |lc| lc + packed_entry.get_variable(),
                |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
                |lc| lc + terminal_public_inputs[x_range.start + public_idx].get_variable(),
            );
            let delta = shape_delta(before, cs);
            out.public_z_links += delta.rows;
            out.public_z_links_shape.rows += delta.rows;
            out.public_z_links_shape.public_cols += delta.public_cols;
            out.public_z_links_shape.aux_cols += delta.aux_cols;
        }
        let constant_one_col = committed_width
            .checked_sub(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        let constant_one = packed_z.logical_entry(committed_width, constant_one_col)?;
        let before = shape_point(cs);
        cs.enforce(
            || "direct_terminal_r2_constant_one_link",
            |lc| lc + constant_one.get_variable(),
            |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
            |lc| lc + <ShapeCS<NeoFoldDeciderEngine> as ConstraintSystem<SpartanF>>::one(),
        );
        out.constant_one_link_shape = shape_delta(before, cs);
        out.constant_one_link = out.constant_one_link_shape.rows;

        for logical_col in 0..committed_width {
            let value = packed_z.logical_entry(committed_width, logical_col)?;
            let before = shape_point(cs);
            enforce_boolean_allocated(
                &mut cs.namespace(|| format!("direct_terminal_r2_low_norm_bit_{logical_col}")),
                &value,
                &format!("direct_terminal_r2_low_norm_bit_{logical_col}"),
            );
            let delta = shape_delta(before, cs);
            out.low_norm_bit_checks += delta.rows;
            out.low_norm_bit_checks_shape.rows += delta.rows;
            out.low_norm_bit_checks_shape.public_cols += delta.public_cols;
            out.low_norm_bit_checks_shape.aux_cols += delta.aux_cols;
        }
        let before = shape_point(cs);
        enforce_packed_padding_zero(cs, packed_z, committed_width, "direct_terminal_r2_padding_zero")?;
        out.padding_zero_checks_shape = shape_delta(before, cs);
        out.padding_zero_checks = out.padding_zero_checks_shape.rows;
        out.total_shape = shape_delta(start, cs);
        out.total = out.total_shape.rows;
        Ok(out)
    }

    fn synthesize_terminal_with_committed_sources<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        public_inputs: &[AllocatedNum<SpartanF>],
        packed_z: &PackedWitnessVar,
        committed_width: usize,
    ) -> Result<usize, SynthesisError> {
        let mut linking_cs = DirectSourceWitnessLinkingCs::new(
            cs,
            &self.assignment.layout,
            packed_z,
            committed_width,
            self.assignment.r2_public_values.len(),
        );
        self.assignment
            .terminal_circuit
            .synthesize_body_with_public_inputs(&mut linking_cs, public_inputs)?;
        Ok(linking_cs.source_link_constraints)
    }

    fn enforce_terminal_commitment<CS: ConstraintSystem<SpartanF>>(
        &self,
        cs: &mut CS,
        packed_z: &PackedWitnessVar,
        commitment_inputs: &[AllocatedNum<SpartanF>],
    ) -> Result<(), SynthesisError> {
        enforce_terminal_ajtai_commitment(
            &mut cs.namespace(|| "direct_terminal_r2_ajtai_commitment"),
            packed_z,
            commitment_inputs,
            "direct_terminal_r2_ajtai_commitment",
        )
    }
}

impl DirectCcsTerminalR2Assignment {
    fn from_terminal_circuit(circuit: DirectCcsTerminalFPrimeCircuit) -> Result<Self, SimpleKernelError> {
        let export = direct_terminal_shape_export(&circuit)?;
        let private_witness_labels = padded_private_witness_labels(
            &export.split_shape,
            &export.private_witness_labels,
            "direct terminal F'",
        )
        .map_err(SimpleKernelError::Bridge)?;
        let layout = DirectCcsTerminalR2Layout::new(
            export.expected_public_values.len(),
            &private_witness_labels,
            (circuit.params.k_rho as usize).saturating_mul(2),
        )?;
        let terminal_public_values = export
            .expected_public_values
            .iter()
            .map(|value| F::from_u64(value.to_canonical_u64()))
            .collect::<Vec<_>>();
        let x_range = circuit.construction2_x_bit_range();
        if x_range.len() != CONSTRUCTION2_ENC_INST_BITS || x_range.end > terminal_public_values.len() {
            return Err(SimpleKernelError::Bridge(
                "direct terminal F' public image must contain the 256-bit Construction-2 enc_inst image".into(),
            ));
        }
        let r2_public_values = terminal_public_values[x_range].to_vec();

        let (ck, _) = SplitR1CSShape::commitment_key(&[&export.split_shape]).map_err(|err| {
            SimpleKernelError::Bridge(format!("direct terminal F' R1CS commitment key failed: {err}"))
        })?;
        let mut state =
            SatisfyingAssignment::<NeoFoldDeciderEngine>::shared_witness(&export.split_shape, &ck, &circuit, false)
                .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' shared witness failed: {err}")))?;
        SatisfyingAssignment::<NeoFoldDeciderEngine>::precommitted_witness(
            &mut state,
            &export.split_shape,
            &ck,
            &circuit,
            false,
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' precommitted witness failed: {err}")))?;
        let mut transcript = <NeoFoldDeciderEngine as Engine>::TE::new(b"direct_ccs_terminal_f_prime_r2_assignment");
        let (instance, witness) = SatisfyingAssignment::<NeoFoldDeciderEngine>::r1cs_instance_and_witness(
            &mut state,
            &export.split_shape,
            &ck,
            &circuit,
            false,
            &mut transcript,
        )
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' witness export failed: {err}")))?;
        let regular_shape = export.split_shape.to_regular_shape();
        let regular_instance = instance.to_regular_instance().map_err(|err| {
            SimpleKernelError::Bridge(format!("direct terminal F' R1CS instance flatten failed: {err}"))
        })?;
        regular_shape
            .is_sat(&ck, &regular_instance, &witness)
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' exported witness is unsat: {err}")))?;
        if regular_instance.public_values() != export.expected_public_values.as_slice() {
            return Err(SimpleKernelError::Bridge(
                "direct terminal F' exported public IO does not match expected terminal image".into(),
            ));
        }

        let mut witness_values = Vec::with_capacity(layout.source_limb_width + 1);
        for (witness_idx, value) in witness.values().iter().enumerate() {
            let Some(Some(label)) = private_witness_labels.get(witness_idx) else {
                continue;
            };
            let Some((offset, encoding)) = layout.source_binding(label) else {
                continue;
            };
            while witness_values.len() < offset {
                witness_values.push(F::ZERO);
            }
            let native = F::from_u64(value.to_canonical_u64());
            let encoded = low_norm_encoded_values(native, encoding, &format!("direct terminal F' source {label}"))
                .map_err(SimpleKernelError::Bridge)?;
            witness_values.extend(encoded);
        }
        if witness_values.len() != layout.source_limb_width {
            while witness_values.len() < layout.source_limb_width {
                witness_values.push(F::ZERO);
            }
            if witness_values.len() != layout.source_limb_width {
                return Err(SimpleKernelError::Bridge(format!(
                    "direct terminal F' source witness length mismatch: expected {}, got {}",
                    layout.source_limb_width,
                    witness_values.len()
                )));
            }
        }
        witness_values.push(F::ONE);

        let assignment = Self {
            layout,
            terminal_public_values,
            r2_public_values,
            witness_values,
            terminal_circuit: circuit,
        };
        assignment.validate()?;
        Ok(assignment)
    }

    fn committed_width(&self) -> Result<usize, SimpleKernelError> {
        self.r2_public_values
            .len()
            .checked_add(self.witness_values.len())
            .ok_or_else(|| SimpleKernelError::Bridge("direct terminal F' committed width overflow".into()))
    }

    fn committed_full_vector(&self) -> Result<Vec<F>, SimpleKernelError> {
        let mut out = Vec::with_capacity(self.committed_width()?);
        out.extend_from_slice(&self.r2_public_values);
        out.extend_from_slice(&self.witness_values);
        Ok(out)
    }

    fn committed_packed_witness(&self) -> Result<Mat<F>, SimpleKernelError> {
        let full_width = self.committed_width()?;
        let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "direct terminal F' R2 params failed for width {full_width}: {err}"
            ))
        })?;
        let full_vector = self.committed_full_vector()?;
        if let Some(error) = committed_nc_range_error(
            &params,
            &full_vector,
            |idx| format!("committed index {idx}"),
            "direct terminal F'",
        ) {
            return Err(SimpleKernelError::Bridge(error));
        }
        encode_vector_for_full_width(&params, full_width, &full_vector)
            .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' R2 packing failed: {err}")))
    }

    fn commitment(&self) -> Result<Commitment, SimpleKernelError> {
        let full_width = self.committed_width()?;
        let packed = self.committed_packed_witness()?;
        direct_terminal_commit_packed_z(full_width, &packed)
    }

    fn validate(&self) -> Result<(), SimpleKernelError> {
        if self.r2_public_values.len() != CONSTRUCTION2_ENC_INST_BITS {
            return Err(SimpleKernelError::Bridge(
                "direct terminal F' R2 public input must contain the 256-bit Construction-2 image".into(),
            ));
        }
        if self.witness_values.last().copied() != Some(F::ONE) {
            return Err(SimpleKernelError::Bridge(
                "direct terminal F' R2 witness must end with constant-one slot".into(),
            ));
        }
        self.committed_full_vector()?;
        Ok(())
    }
}

impl DirectCcsTerminalR2Layout {
    fn new(
        _r2_public_len: usize,
        private_witness_labels: &[Option<String>],
        reserved_fold_digest_count: usize,
    ) -> Result<Self, SimpleKernelError> {
        let mut out = Self {
            source_labels: Vec::new(),
            source_encodings: Vec::new(),
            source_offsets: Vec::new(),
            source_by_label: BTreeMap::new(),
            source_limb_width: 0,
        };
        let mut actual_fold_digest_count = 0usize;
        for label in private_witness_labels.iter().flatten() {
            let Some(encoding) = direct_terminal_private_encoding_from_label(label)? else {
                continue;
            };
            if label.contains("_fold_digest_len") {
                actual_fold_digest_count = actual_fold_digest_count.checked_add(1).ok_or_else(|| {
                    SimpleKernelError::Bridge("direct terminal F' fold-digest source count overflow".into())
                })?;
            }
            out.push_source_label(label.clone(), encoding)?;
        }
        for missing_idx in actual_fold_digest_count..reserved_fold_digest_count {
            out.push_source_label(
                format!("direct_terminal_reserved_fold_digest_{missing_idx}_len"),
                TerminalPrivateColumnEncoding::U32,
            )?;
            for limb_idx in 0..5 {
                out.push_source_label(
                    format!("direct_terminal_reserved_fold_digest_{missing_idx}_limb_{limb_idx}"),
                    TerminalPrivateColumnEncoding::U64,
                )?;
            }
        }
        if out.source_labels.is_empty() {
            return Err(SimpleKernelError::Bridge(
                "direct terminal F' committed source set is empty".into(),
            ));
        }
        Ok(out)
    }

    fn push_source_label(
        &mut self,
        label: String,
        encoding: TerminalPrivateColumnEncoding,
    ) -> Result<(), SimpleKernelError> {
        self.source_by_label
            .insert(label.clone(), self.source_labels.len());
        self.source_labels.push(label);
        self.source_encodings.push(encoding);
        self.source_offsets.push(self.source_limb_width);
        self.source_limb_width = self
            .source_limb_width
            .checked_add(encoding.limb_count())
            .ok_or_else(|| SimpleKernelError::Bridge("direct terminal F' source limb width overflow".into()))?;
        Ok(())
    }

    fn source_binding(&self, label: &str) -> Option<(usize, TerminalPrivateColumnEncoding)> {
        let source_idx = *self.source_by_label.get(label)?;
        Some((self.source_offsets[source_idx], self.source_encodings[source_idx]))
    }

    fn source_encoding_count(&self, encoding: TerminalPrivateColumnEncoding) -> usize {
        self.source_encodings
            .iter()
            .filter(|candidate| **candidate == encoding)
            .count()
    }
}

pub(crate) fn setup_direct_ccs_terminal_committed_relation(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: DirectCcsTerminalCommittedPerf,
) -> Result<
    (
        NeoFoldDeciderProverKey,
        NeoFoldDeciderVerifierKey,
        DirectCcsTerminalCommittedPerf,
    ),
    SimpleKernelError,
> {
    let circuit = relation.committed_circuit();
    let (pk, vk) = NeoFoldDeciderSnark::setup(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step setup failed: {err}")))?;
    let mut perf = perf;
    perf.sizes = pk.sizes();
    perf.nnz = pk.shape_debug_stats().total_nnz;
    Ok((pk, vk, perf))
}

pub(crate) fn setup_direct_ccs_terminal_committed_relation_cached(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: DirectCcsTerminalCommittedPerf,
) -> Result<DirectCcsTerminalCommittedKeyPair, SimpleKernelError> {
    let cache_key = direct_terminal_committed_setup_cache_key(relation, &perf);
    let cache = DIRECT_CCS_TERMINAL_COMMITTED_SETUP_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
    if let Some(keys) = cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("direct terminal committed-step setup cache poisoned".into()))?
        .get(&cache_key)
        .cloned()
    {
        return Ok(keys);
    }

    let (pk, vk, perf) = setup_direct_ccs_terminal_committed_relation(relation, perf)?;
    let keys = DirectCcsTerminalCommittedKeyPair {
        prover: Arc::new(pk),
        verifier: Arc::new(vk),
        perf,
    };
    cache
        .lock()
        .map_err(|_| SimpleKernelError::Bridge("direct terminal committed-step setup cache poisoned".into()))?
        .insert(cache_key, keys.clone());
    Ok(keys)
}

fn direct_terminal_committed_setup_cache_key(
    relation: &DirectCcsTerminalCommittedRelation,
    perf: &DirectCcsTerminalCommittedPerf,
) -> [u8; 32] {
    let assignment = &relation.assignment;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/terminal_committed_setup_cache");
    tr.append_message(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/version",
        b"v1",
    );
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/shape",
        &[
            perf.constraints as u64,
            perf.public_inputs as u64,
            perf.committed_width as u64,
            perf.commitment_words as u64,
            perf.source_values as u64,
            perf.source_bit_values as u64,
            perf.source_u32_values as u64,
            perf.source_u64_values as u64,
            assignment.terminal_public_values.len() as u64,
            assignment.r2_public_values.len() as u64,
            assignment.witness_values.len() as u64,
            relation.public_boundary.commitment_d as u64,
            relation.public_boundary.commitment_kappa as u64,
        ],
    );
    let encodings = assignment
        .layout
        .source_encodings
        .iter()
        .map(|encoding| match encoding {
            TerminalPrivateColumnEncoding::UnusedPadding => 0,
            TerminalPrivateColumnEncoding::Bit => 1,
            TerminalPrivateColumnEncoding::U32 => 32,
            TerminalPrivateColumnEncoding::U64 => 64,
        })
        .collect::<Vec<_>>();
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/terminal_committed_setup_cache/source_encodings",
        &encodings,
    );
    tr.digest32()
}

pub(crate) fn prove_direct_ccs_terminal_committed_relation(
    pk: &NeoFoldDeciderProverKey,
    relation: &DirectCcsTerminalCommittedRelation,
) -> Result<(DirectCcsTerminalCommittedProof, f64), SimpleKernelError> {
    let circuit = relation.committed_circuit();
    let prep = NeoFoldDeciderSnark::prep_prove(pk, circuit.clone(), false)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step prepare failed: {err}")))?;
    let (proof, perf) = NeoFoldDeciderSnark::prove_with_perf(pk, circuit, &prep, false)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step prove failed: {err}")))?;
    let snark_data = bincode::serialize(&proof).map_err(|err| {
        SimpleKernelError::Bridge(format!("direct terminal committed-step proof encoding failed: {err}"))
    })?;
    Ok((DirectCcsTerminalCommittedProof { snark_data }, perf.pcs_prove_ms))
}

pub(crate) fn verify_direct_ccs_terminal_committed_relation(
    vk: &NeoFoldDeciderVerifierKey,
    expected_terminal_public_values: &[SpartanF],
    expected_public_boundary: &Construction2PublicBoundary,
    proof: &DirectCcsTerminalCommittedProof,
) -> Result<(), SimpleKernelError> {
    let snark: NeoFoldDeciderSnark = bincode::deserialize(&proof.snark_data).map_err(|err| {
        SimpleKernelError::Bridge(format!("direct terminal committed-step proof decoding failed: {err}"))
    })?;
    let public_values = snark
        .verify(vk)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal committed-step verify failed: {err}")))?;
    if public_values.len() < expected_terminal_public_values.len()
        || &public_values[..expected_terminal_public_values.len()] != expected_terminal_public_values
    {
        return Err(SimpleKernelError::Bridge(
            "direct terminal committed-step public IO does not match expected folded F' terminal image".into(),
        ));
    }
    let boundary_values = terminal_committed_boundary_public_values(expected_public_boundary);
    if public_values.len() < boundary_values.len()
        || public_values[public_values.len() - boundary_values.len()..] != boundary_values
    {
        return Err(SimpleKernelError::Bridge(
            "direct terminal committed-step public IO does not bind expected Construction-2 boundary".into(),
        ));
    }
    Ok(())
}

fn direct_terminal_shape_export(
    circuit: &DirectCcsTerminalFPrimeCircuit,
) -> Result<DirectCcsTerminalShapeExport, SimpleKernelError> {
    let expected_public_values = circuit
        .public_values()
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' public IO failed: {err}")))?;
    let split_shape = ShapeCS::<NeoFoldDeciderEngine>::r1cs_shape(circuit)
        .map_err(|err| SimpleKernelError::Bridge(format!("direct terminal F' R1CS export failed: {err}")))?;
    let private_witness_labels =
        collect_private_witness_labels(circuit, "direct terminal F'").map_err(SimpleKernelError::Bridge)?;
    if private_witness_labels.len() != split_shape.num_variables_unpadded() {
        return Err(SimpleKernelError::Bridge(format!(
            "direct terminal F' label count mismatch: expected {}, got {}",
            split_shape.num_variables_unpadded(),
            private_witness_labels.len()
        )));
    }
    Ok(DirectCcsTerminalShapeExport {
        split_shape,
        expected_public_values,
        private_witness_labels,
    })
}

struct DirectSourceWitnessLinkingCs<'a, 'b, CS: ConstraintSystem<SpartanF>> {
    inner: &'a mut CS,
    layout: &'b DirectCcsTerminalR2Layout,
    packed_z: &'b PackedWitnessVar,
    committed_width: usize,
    public_len: usize,
    current_namespace: Vec<String>,
    source_link_constraints: usize,
}

impl<'a, 'b, CS: ConstraintSystem<SpartanF>> DirectSourceWitnessLinkingCs<'a, 'b, CS> {
    fn new(
        inner: &'a mut CS,
        layout: &'b DirectCcsTerminalR2Layout,
        packed_z: &'b PackedWitnessVar,
        committed_width: usize,
        public_len: usize,
    ) -> Self {
        Self {
            inner,
            layout,
            packed_z,
            committed_width,
            public_len,
            current_namespace: Vec::new(),
            source_link_constraints: 0,
        }
    }

    fn alloc_path(&self, annotation: &str) -> String {
        if self.current_namespace.is_empty() {
            return annotation.to_owned();
        }
        let mut path = self.current_namespace.join("/");
        path.push('/');
        path.push_str(annotation);
        path
    }

    fn source_lc_with_encoding(
        &self,
        offset: usize,
        encoding: TerminalPrivateColumnEncoding,
    ) -> Result<LinearCombination<SpartanF>, SynthesisError> {
        let mut lc = LinearCombination::<SpartanF>::zero();
        for limb_idx in 0..encoding.limb_count() {
            let logical_col = self
                .public_len
                .checked_add(offset)
                .and_then(|value| value.checked_add(limb_idx))
                .ok_or(SynthesisError::Unsatisfiable)?;
            let limb = self
                .packed_z
                .logical_entry(self.committed_width, logical_col)?;
            lc = lc + (SpartanF::from_canonical_u64(1u64 << limb_idx), limb.get_variable());
        }
        Ok(lc)
    }
}

impl<CS: ConstraintSystem<SpartanF>> ConstraintSystem<SpartanF> for DirectSourceWitnessLinkingCs<'_, '_, CS> {
    type Root = Self;

    fn alloc<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let annotation = annotation().into();
        let label = self.alloc_path(&annotation);
        let var = self.inner.alloc(|| annotation.clone(), f)?;
        if let Some((offset, encoding)) = self.layout.source_binding(&label) {
            let source_lc = self.source_lc_with_encoding(offset, encoding)?;
            self.inner.enforce(
                || format!("direct_terminal_r2_source_link_{label}"),
                |lc| lc + var,
                |lc| lc + CS::one(),
                |_| source_lc,
            );
            self.source_link_constraints += 1;
        }
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, annotation: A, f: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        self.inner.alloc_input(annotation, f)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, annotation: A, a: LA, b: LB, c: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
        self.inner.enforce(annotation, a, b, c);
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        let name = name_fn().into();
        self.current_namespace.push(name.clone());
        self.inner.push_namespace(|| name);
    }

    fn pop_namespace(&mut self) {
        assert!(self.current_namespace.pop().is_some());
        self.inner.pop_namespace();
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}

fn direct_terminal_private_encoding_from_label(
    label: &str,
) -> Result<Option<TerminalPrivateColumnEncoding>, SimpleKernelError> {
    let root = label.split('/').next().unwrap_or(label);
    if root == "direct_terminal_construction2_input_u_i" {
        if label.contains("construction2_input_u_i_x_") || label.contains("construction2_input_u_i_commitment_data_") {
            return Ok(Some(TerminalPrivateColumnEncoding::U64));
        }
        if label.contains("construction2_input_u_i_commitment_d")
            || label.contains("construction2_input_u_i_commitment_kappa")
        {
            return Ok(Some(TerminalPrivateColumnEncoding::U32));
        }
        return Err(SimpleKernelError::Bridge(format!(
            "direct terminal F' selected Construction-2 source label is unclassified: {label}"
        )));
    }
    if label.contains("_fold_digest_len") {
        return Ok(Some(TerminalPrivateColumnEncoding::U32));
    }
    if label.contains("_fold_digest_limb_") {
        return Ok(Some(TerminalPrivateColumnEncoding::U64));
    }
    Ok(None)
}

fn terminal_committed_boundary_public_values(boundary: &Construction2PublicBoundary) -> Vec<SpartanF> {
    terminal_boundary_public_values(&direct_terminal_boundary_view(boundary))
}

fn direct_terminal_boundary_view(boundary: &Construction2PublicBoundary) -> Construction2TerminalBoundaryView<'_> {
    Construction2TerminalBoundaryView {
        fresh_instance_digest: boundary.fresh_instance_digest,
        commitment_digest: boundary.commitment_digest,
        commitment_d: boundary.commitment_d,
        commitment_kappa: boundary.commitment_kappa,
        commitment_data: &boundary.commitment_data,
        x_i_bytes: boundary.x_i.bytes(),
    }
}

fn direct_terminal_commit_packed_z(full_width: usize, packed_z: &Mat<F>) -> Result<Commitment, SimpleKernelError> {
    let params = NeoParams::goldilocks_auto_r1cs_ccs(full_width).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "direct terminal commitment params failed for width {full_width}: {err}"
        ))
    })?;
    let m = commit_cols_for_full_width(full_width);
    let want_kappa = params.kappa as usize;
    if has_global_pp_for_dims(D, m) {
        let (kappa, _) = get_global_pp_seeded_params_for_dims(D, m).map_err(|err| {
            SimpleKernelError::Bridge(format!("direct terminal commitment PP registry read failed: {err}"))
        })?;
        if kappa != want_kappa {
            return Err(SimpleKernelError::Bridge(format!(
                "direct terminal commitment PP mismatch for (d,m)=({D},{m}): registered kappa={kappa}, want {want_kappa}"
            )));
        }
    } else {
        set_global_pp_seeded(D, want_kappa, m, direct_terminal_commitment_seed(full_width)).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "direct terminal commitment PP setup failed for (d,m)=({D},{m}): {err}"
            ))
        })?;
    }
    if packed_z.rows() != D || packed_z.cols() != m {
        return Err(SimpleKernelError::Bridge(format!(
            "direct terminal packed Z shape mismatch: got {}x{}, expected {D}x{m}",
            packed_z.rows(),
            packed_z.cols()
        )));
    }
    let log = AjtaiSModule::from_global_for_dims(D, m).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "direct terminal commitment module failed for (d,m)=({D},{m}): {err}"
        ))
    })?;
    Ok(log.commit(packed_z))
}

fn direct_terminal_commitment_seed(full_width: usize) -> [u8; 32] {
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/direct_ccs/construction2_commitment_seed");
    tr.append_message(b"neo.fold.next/direct_ccs/construction2_commitment_seed/version", b"v1");
    tr.append_u64s(
        b"neo.fold.next/direct_ccs/construction2_commitment_seed/full_width",
        &[full_width as u64],
    );
    tr.digest32()
}
