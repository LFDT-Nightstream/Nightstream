use std::collections::BTreeMap;

use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::F;
use neo_params::NeoParams;
use p3_field::PrimeCharacteristicRing;
use spartan2::{
    bellpepper::{r1cs::SpartanWitness, solver::SatisfyingAssignment},
    traits::{transcript::TranscriptEngineTrait, Engine},
};

use super::commitment::direct_terminal_commit_packed_z;
use super::types::{
    DirectCcsTerminalR2Assignment, DirectCcsTerminalR2Layout, DirectCcsTerminalShapeExport, SimpleKernelError,
};
use super::DirectCcsTerminalFPrimeCircuit;
use crate::construction2::terminal::{
    collect_private_witness_labels, committed_nc_range_error, low_norm_encoded_values, padded_private_witness_labels,
    TerminalPrivateColumnEncoding,
};
use crate::construction2::CONSTRUCTION2_ENC_INST_BITS;
use crate::spartan_backend::{NeoFoldDeciderEngine, ShapeCS, SpartanCircuit, SpartanShape, SplitR1CSShape};
use crate::witness_layout::encode_vector_for_full_width;

impl DirectCcsTerminalR2Assignment {
    pub(super) fn from_terminal_circuit(circuit: DirectCcsTerminalFPrimeCircuit) -> Result<Self, SimpleKernelError> {
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

    pub(super) fn committed_width(&self) -> Result<usize, SimpleKernelError> {
        self.r2_public_values
            .len()
            .checked_add(self.witness_values.len())
            .ok_or_else(|| SimpleKernelError::Bridge("direct terminal F' committed width overflow".into()))
    }

    pub(super) fn committed_full_vector(&self) -> Result<Vec<F>, SimpleKernelError> {
        let mut out = Vec::with_capacity(self.committed_width()?);
        out.extend_from_slice(&self.r2_public_values);
        out.extend_from_slice(&self.witness_values);
        Ok(out)
    }

    pub(super) fn committed_packed_witness(&self) -> Result<Mat<F>, SimpleKernelError> {
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

    pub(super) fn commitment(&self) -> Result<Commitment, SimpleKernelError> {
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
    pub(super) fn new(
        _r2_public_len: usize,
        private_witness_labels: &[Option<String>],
        reserved_fold_digest_count: usize,
    ) -> Result<Self, SimpleKernelError> {
        let mut out = Self {
            source_labels: Vec::new(),
            sources: Vec::new(),
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
        self.sources.push(encoding);
        self.source_offsets.push(self.source_limb_width);
        self.source_limb_width = self
            .source_limb_width
            .checked_add(encoding.limb_count())
            .ok_or_else(|| SimpleKernelError::Bridge("direct terminal F' source limb width overflow".into()))?;
        Ok(())
    }

    pub(super) fn source_binding(&self, label: &str) -> Option<(usize, TerminalPrivateColumnEncoding)> {
        let source_idx = *self.source_by_label.get(label)?;
        Some((self.source_offsets[source_idx], self.sources[source_idx]))
    }

    pub(super) fn source_count(&self, encoding: TerminalPrivateColumnEncoding) -> usize {
        self.sources
            .iter()
            .filter(|candidate| **candidate == encoding)
            .count()
    }
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
