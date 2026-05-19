//! Owns Ajtai commitments and selected time-point openings for the RV32IM root lane family.

use neo_ajtai::{
    decomp_b_row_major_into, get_global_pp_for_dims, get_global_pp_seeded_params_for_dims, has_global_pp_for_dims,
    set_global_pp_seeded, AjtaiSModule, Commitment, DecompStyle,
};
use neo_ccs::{traits::SModuleHomomorphism, Mat};
use neo_math::{KExtensions, F, K};
use neo_params::NeoParams;
use neo_transcript::{Poseidon2Transcript, Transcript, TranscriptProtocol};
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};

use super::canonical_openings::{AjtaiFamilyKind, AjtaiObjectId, AjtaiOpeningId, SelectedOpeningRef};
use super::root_lane_witness::{build_root_lane_witness, root_lane_row_digest, RootLanePublicWitness, RootLaneWitness};
use super::simple::{SimpleKernelError, SIMPLE_KERNEL_PP_SEED};
use crate::rv32im::ccs::RV32IM_ROOT_ROW_WIDTH;
use crate::rv32im::lower::Rv32ExpandedRow;

const ROOT_LANE_OPENING_DECOMP_BASE: u32 = 2;
const ROOT_LANE_COMMITMENT_BATCH: usize = 256;
const RV32IM_ROOT_LANE_COMMITTED_ROWS_LAYOUT_V1: u64 = 3;

fn append_k_point(tr: &mut Poseidon2Transcript, label: &'static [u8], point: &[K]) {
    tr.append_u64s(
        b"neo.fold.next/rv32im/root_lane_commitment/point_len",
        &[point.len() as u64],
    );
    let coeffs_per_elem = point
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    tr.append_fields_iter(
        label,
        point.len().saturating_mul(coeffs_per_elem),
        point.iter().flat_map(|value| value.as_coeffs()),
    );
}

fn append_k_values(tr: &mut Poseidon2Transcript, label: &'static [u8], values: &[K]) {
    tr.append_u64s(
        b"neo.fold.next/rv32im/root_lane_commitment/value_len",
        &[values.len() as u64],
    );
    let coeffs_per_elem = values
        .first()
        .map(|value| value.as_coeffs().len())
        .unwrap_or(0);
    tr.append_fields_iter(
        label,
        values.len().saturating_mul(coeffs_per_elem),
        values.iter().flat_map(|value| value.as_coeffs()),
    );
}

fn logical_index_point_le(logical_index: usize, padded_time_len: usize) -> Vec<K> {
    debug_assert!(padded_time_len.is_power_of_two());
    let ell = padded_time_len.trailing_zeros() as usize;
    (0..ell)
        .map(|bit| {
            if ((logical_index >> bit) & 1) == 1 {
                K::ONE
            } else {
                K::ZERO
            }
        })
        .collect()
}

fn encoded_time_width(time_len: usize, padded_time_len: usize) -> Result<usize, SimpleKernelError> {
    if time_len > padded_time_len {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM root lane column length {time_len} exceeds padded time length {padded_time_len}"
        )));
    }
    Ok(padded_time_len)
}

fn root_lane_committer(params: &NeoParams, encoded_t: usize) -> Result<AjtaiSModule, SimpleKernelError> {
    let d = params.d as usize;
    let want_kappa = params.kappa as usize;
    let expected_seed = SIMPLE_KERNEL_PP_SEED;

    if has_global_pp_for_dims(d, encoded_t) {
        if let Ok((kappa, seed)) = get_global_pp_seeded_params_for_dims(d, encoded_t) {
            if kappa != want_kappa || seed != expected_seed {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM root lane commitment PP mismatch for (d,m)=({d},{encoded_t})"
                )));
            }
        } else {
            let pp = get_global_pp_for_dims(d, encoded_t).map_err(|err| {
                SimpleKernelError::Bridge(format!(
                    "failed to load RV32IM root lane commitment PP for (d,m)=({d},{encoded_t}): {err}"
                ))
            })?;
            if pp.kappa != want_kappa {
                return Err(SimpleKernelError::Bridge(format!(
                    "RV32IM root lane commitment PP kappa mismatch for (d,m)=({d},{encoded_t})"
                )));
            }
        }
    } else {
        set_global_pp_seeded(d, want_kappa, encoded_t, expected_seed).map_err(|err| {
            SimpleKernelError::Bridge(format!(
                "failed to register RV32IM root lane commitment PP for (d,m)=({d},{encoded_t}): {err}"
            ))
        })?;
    }

    AjtaiSModule::from_global_for_dims(d, encoded_t).map_err(|err| {
        SimpleKernelError::Bridge(format!(
            "failed to initialize RV32IM root lane committer for (d,m)=({d},{encoded_t}): {err}"
        ))
    })
}

struct RootLaneEncodingScratch {
    values: Vec<F>,
    row_major_digits: Vec<F>,
}

impl RootLaneEncodingScratch {
    fn new() -> Self {
        Self {
            values: Vec::new(),
            row_major_digits: Vec::new(),
        }
    }
}

fn encode_time_opening_vector_to_mat_with_scratch(
    params: &NeoParams,
    values: &[F],
    padded_time_len: usize,
    scratch: &mut RootLaneEncodingScratch,
) -> Result<Mat<F>, SimpleKernelError> {
    let time_len = values.len();
    let encoded_t = encoded_time_width(time_len, padded_time_len)?;
    scratch.values.clear();
    scratch.values.extend_from_slice(values);

    let d = params.d as usize;
    let row_major_len = d.saturating_mul(time_len);
    if scratch.values.iter().any(|&value| value != F::ZERO) {
        decomp_b_row_major_into(
            scratch.values.as_slice(),
            ROOT_LANE_OPENING_DECOMP_BASE,
            d,
            DecompStyle::Balanced,
            &mut scratch.row_major_digits,
        );
    } else {
        scratch.row_major_digits.clear();
        scratch.row_major_digits.resize(row_major_len, F::ZERO);
    }

    let zero_pad_len = padded_time_len.saturating_sub(time_len);
    let mut row_major = Vec::with_capacity(d * encoded_t);
    for rho in 0..d {
        let row_start = rho * time_len;
        let row_end = row_start + time_len;
        row_major.extend_from_slice(&scratch.row_major_digits[row_start..row_end]);
        row_major.extend(std::iter::repeat_n(F::ZERO, zero_pad_len));
    }
    Ok(Mat::from_row_major(d, encoded_t, row_major))
}

fn recompose_time_vector_digits_to_scalar(digits: &[K]) -> K {
    let base = K::from(F::from_u64(ROOT_LANE_OPENING_DECOMP_BASE as u64));
    let mut power = K::ONE;
    let mut acc = K::ZERO;
    for &digit in digits {
        acc += power * digit;
        power *= base;
    }
    acc
}

fn extract_time_mat_digits_at_logical_index(
    logical_index: usize,
    padded_time_len: usize,
    encoded: &Mat<F>,
) -> Result<Vec<K>, SimpleKernelError> {
    if encoded.cols() < padded_time_len {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM root lane encoded matrix column count {} is smaller than padded time length {}",
            encoded.cols(),
            padded_time_len
        )));
    }
    let time_len = padded_time_len;
    if logical_index >= encoded.cols() || logical_index >= time_len {
        return Err(SimpleKernelError::Bridge(format!(
            "RV32IM root lane logical index {} is out of range for time_len {}",
            logical_index, time_len
        )));
    }
    let mut digits = vec![K::ZERO; encoded.rows()];
    let cols = encoded.cols();
    let data = encoded.as_slice();
    for rho in 0..encoded.rows() {
        let row = &data[rho * cols..(rho + 1) * cols];
        digits[rho] = K::from(row[logical_index]);
    }
    Ok(digits)
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneCommitmentSet {
    pub commitments: Vec<Commitment>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneCommitmentSetSummary {
    pub commitment_count: u64,
    pub digest: [u8; 32],
}

impl RootLaneCommitmentSet {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_commitments");
        tr.append_u64s(b"rv32im/root_lane_commitments/len", &[self.commitments.len() as u64]);
        for commitment in &self.commitments {
            tr.append_u64s(
                b"rv32im/root_lane_commitments/shape",
                &[commitment.d as u64, commitment.kappa as u64],
            );
            tr.absorb_commit_coords(&commitment.data);
        }
        tr.digest32()
    }
}

impl From<&RootLaneCommitmentSet> for RootLaneCommitmentSetSummary {
    fn from(value: &RootLaneCommitmentSet) -> Self {
        Self {
            commitment_count: value.commitments.len() as u64,
            digest: value.digest,
        }
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneOpeningProof {
    pub logical_index: u64,
    pub point: Vec<K>,
    pub claimed_values: Vec<K>,
    pub value_digest: [u8; 32],
    pub digit_evals: Vec<Vec<K>>,
    pub digest: [u8; 32],
}

impl RootLaneOpeningProof {
    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_opening");
        tr.append_u64s(
            b"rv32im/root_lane_opening/meta",
            &[
                self.logical_index,
                self.point.len() as u64,
                self.claimed_values.len() as u64,
                self.digit_evals.len() as u64,
            ],
        );
        append_k_point(&mut tr, b"rv32im/root_lane_opening/point", &self.point);
        append_k_values(
            &mut tr,
            b"rv32im/root_lane_opening/claimed_values",
            &self.claimed_values,
        );
        tr.append_message(b"rv32im/root_lane_opening/value_digest", &self.value_digest);
        for digits in &self.digit_evals {
            append_k_values(&mut tr, b"rv32im/root_lane_opening/digit_eval", digits);
        }
        tr.digest32()
    }
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneCommitmentArtifact {
    pub row_width: u64,
    pub time_len: u64,
    pub padded_time_len: u64,
    pub commitments: RootLaneCommitmentSet,
    pub first_opening: Option<RootLaneOpeningProof>,
    pub last_opening: Option<RootLaneOpeningProof>,
    pub digest: [u8; 32],
}

#[derive(Clone, Debug, Eq, PartialEq, Serialize, Deserialize)]
pub struct RootLaneCommitmentSummaryArtifact {
    pub time_len: u64,
    pub commitments: RootLaneCommitmentSetSummary,
    pub first_selected_row: Option<SelectedOpeningRef>,
    pub last_selected_row: Option<SelectedOpeningRef>,
    pub digest: [u8; 32],
}

impl RootLaneCommitmentArtifact {
    fn opening_object(&self) -> AjtaiObjectId {
        AjtaiObjectId::new(
            AjtaiFamilyKind::RootMainLaneCommittedRows,
            self.commitments.digest,
            RV32IM_ROOT_LANE_COMMITTED_ROWS_LAYOUT_V1,
        )
    }

    fn selected_ref_from_opening(&self, opening: &RootLaneOpeningProof) -> SelectedOpeningRef {
        SelectedOpeningRef::new(
            AjtaiOpeningId::new(self.opening_object(), opening.logical_index),
            opening.value_digest,
        )
    }

    pub fn first_selected_row(&self) -> Option<SelectedOpeningRef> {
        self.first_opening
            .as_ref()
            .map(|opening| self.selected_ref_from_opening(opening))
    }

    pub fn last_selected_row(&self) -> Option<SelectedOpeningRef> {
        self.last_opening
            .as_ref()
            .map(|opening| self.selected_ref_from_opening(opening))
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_commitment_artifact");
        tr.append_u64s(
            b"rv32im/root_lane_commitment_artifact/meta",
            &[self.row_width, self.time_len, self.padded_time_len],
        );
        tr.append_message(
            b"rv32im/root_lane_commitment_artifact/commitments_digest",
            &self.commitments.digest,
        );
        tr.append_u64s(
            b"rv32im/root_lane_commitment_artifact/first_present",
            &[self.first_opening.is_some() as u64],
        );
        if let Some(opening) = &self.first_opening {
            tr.append_message(b"rv32im/root_lane_commitment_artifact/first_digest", &opening.digest);
        }
        tr.append_u64s(
            b"rv32im/root_lane_commitment_artifact/last_present",
            &[self.last_opening.is_some() as u64],
        );
        if let Some(opening) = &self.last_opening {
            tr.append_message(b"rv32im/root_lane_commitment_artifact/last_digest", &opening.digest);
        }
        tr.digest32()
    }
}

impl RootLaneCommitmentSummaryArtifact {
    pub fn first_selected_row(&self) -> Option<SelectedOpeningRef> {
        self.first_selected_row.clone()
    }

    pub fn last_selected_row(&self) -> Option<SelectedOpeningRef> {
        self.last_selected_row.clone()
    }

    pub fn expected_digest(&self) -> [u8; 32] {
        let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_commitment_artifact");
        tr.append_u64s(b"rv32im/root_lane_commitment_artifact/meta", &[self.time_len]);
        tr.append_message(
            b"rv32im/root_lane_commitment_artifact/commitments_digest",
            &self.commitments.digest,
        );
        tr.append_u64s(
            b"rv32im/root_lane_commitment_artifact/first_present",
            &[self.first_selected_row.is_some() as u64],
        );
        if let Some(reference) = &self.first_selected_row {
            tr.append_message(b"rv32im/root_lane_commitment_artifact/first_digest", &reference.digest);
        }
        tr.append_u64s(
            b"rv32im/root_lane_commitment_artifact/last_present",
            &[self.last_selected_row.is_some() as u64],
        );
        if let Some(reference) = &self.last_selected_row {
            tr.append_message(b"rv32im/root_lane_commitment_artifact/last_digest", &reference.digest);
        }
        tr.digest32()
    }
}

impl From<&RootLaneCommitmentArtifact> for RootLaneCommitmentSummaryArtifact {
    fn from(value: &RootLaneCommitmentArtifact) -> Self {
        let summary = Self {
            time_len: value.time_len,
            commitments: RootLaneCommitmentSetSummary::from(&value.commitments),
            first_selected_row: value.first_selected_row(),
            last_selected_row: value.last_selected_row(),
            digest: [0; 32],
        };
        Self {
            digest: summary.expected_digest(),
            ..summary
        }
    }
}

fn encode_columns_to_mats(
    params: &NeoParams,
    columns: &[Vec<F>],
    padded_time_len: usize,
) -> Result<Vec<Mat<F>>, SimpleKernelError> {
    let mut scratch = RootLaneEncodingScratch::new();
    let mut encoded = Vec::with_capacity(columns.len());
    for column in columns {
        encoded.push(encode_time_opening_vector_to_mat_with_scratch(
            params,
            column,
            padded_time_len,
            &mut scratch,
        )?);
    }
    Ok(encoded)
}

fn build_commitments_from_encoded(
    params: &NeoParams,
    encoded_columns: &[Mat<F>],
) -> Result<RootLaneCommitmentSet, SimpleKernelError> {
    let encoded_t = encoded_columns
        .first()
        .map(|encoded| encoded.cols())
        .unwrap_or(0);
    let committer = root_lane_committer(params, encoded_t)?;
    let mut commitments = Vec::with_capacity(encoded_columns.len());
    for chunk in encoded_columns.chunks(ROOT_LANE_COMMITMENT_BATCH) {
        let refs: Vec<&Mat<F>> = chunk.iter().collect();
        commitments.extend(committer.commit_many(&refs));
    }
    let set = RootLaneCommitmentSet {
        commitments,
        digest: [0; 32],
    };
    Ok(RootLaneCommitmentSet {
        digest: set.expected_digest(),
        ..set
    })
}

fn build_commitment_summary_from_columns(
    params: &NeoParams,
    columns: &[Vec<F>],
    padded_time_len: usize,
) -> Result<RootLaneCommitmentSetSummary, SimpleKernelError> {
    let time_len = columns.first().map(Vec::len).unwrap_or(0);
    let committer = root_lane_committer(params, encoded_time_width(time_len, padded_time_len)?)?;
    let mut tr = Poseidon2Transcript::new(b"neo.fold.next/rv32im/root_lane_commitments");
    tr.append_u64s(b"rv32im/root_lane_commitments/len", &[columns.len() as u64]);

    let mut scratch = RootLaneEncodingScratch::new();
    let mut batch = Vec::with_capacity(ROOT_LANE_COMMITMENT_BATCH);
    for column in columns {
        batch.push(encode_time_opening_vector_to_mat_with_scratch(
            params,
            column,
            padded_time_len,
            &mut scratch,
        )?);
        if batch.len() == ROOT_LANE_COMMITMENT_BATCH {
            let refs: Vec<&Mat<F>> = batch.iter().collect();
            for commitment in committer.commit_many(&refs) {
                tr.append_u64s(
                    b"rv32im/root_lane_commitments/shape",
                    &[commitment.d as u64, commitment.kappa as u64],
                );
                tr.absorb_commit_coords(&commitment.data);
            }
            batch.clear();
        }
    }
    if !batch.is_empty() {
        let refs: Vec<&Mat<F>> = batch.iter().collect();
        for commitment in committer.commit_many(&refs) {
            tr.append_u64s(
                b"rv32im/root_lane_commitments/shape",
                &[commitment.d as u64, commitment.kappa as u64],
            );
            tr.absorb_commit_coords(&commitment.data);
        }
    }

    Ok(RootLaneCommitmentSetSummary {
        commitment_count: columns.len() as u64,
        digest: tr.digest32(),
    })
}

fn build_opening_proof_from_encoded(
    encoded_columns: &[Mat<F>],
    semantic_row: &[F; RV32IM_ROOT_ROW_WIDTH],
    logical_index: usize,
    padded_time_len: usize,
) -> Result<RootLaneOpeningProof, SimpleKernelError> {
    let point = logical_index_point_le(logical_index, padded_time_len);
    let claimed_values = semantic_row
        .iter()
        .map(|&value| K::from(value))
        .collect::<Vec<_>>();
    let mut digit_evals = Vec::with_capacity(encoded_columns.len());
    for (encoded_column, &claimed_value) in encoded_columns.iter().zip(claimed_values.iter()) {
        let digits = extract_time_mat_digits_at_logical_index(logical_index, padded_time_len, encoded_column)?;
        if recompose_time_vector_digits_to_scalar(&digits) != claimed_value {
            return Err(SimpleKernelError::Bridge(format!(
                "RV32IM root lane opening claim mismatch at logical index {logical_index}"
            )));
        }
        digit_evals.push(digits);
    }
    let value_digest = root_lane_row_digest(logical_index as u64, semantic_row);
    let proof = RootLaneOpeningProof {
        logical_index: logical_index as u64,
        point,
        claimed_values,
        value_digest,
        digit_evals,
        digest: [0; 32],
    };
    Ok(RootLaneOpeningProof {
        digest: proof.expected_digest(),
        ..proof
    })
}

fn build_selected_row_reference(
    commitment_digest: [u8; 32],
    logical_index: usize,
    row_digest: [u8; 32],
) -> SelectedOpeningRef {
    SelectedOpeningRef::from_parts(
        AjtaiFamilyKind::RootMainLaneCommittedRows,
        commitment_digest,
        RV32IM_ROOT_LANE_COMMITTED_ROWS_LAYOUT_V1,
        logical_index as u64,
        row_digest,
    )
}

pub(crate) fn build_root_lane_commitment_artifact_from_witness(
    params: &NeoParams,
    witness: &RootLaneWitness,
) -> Result<RootLaneCommitmentArtifact, SimpleKernelError> {
    let encoded_columns = encode_columns_to_mats(params, &witness.columns, witness.padded_time_len())?;
    let commitments = build_commitments_from_encoded(params, &encoded_columns)?;
    let first_opening = witness
        .semantic_rows
        .first()
        .map(|row| build_opening_proof_from_encoded(&encoded_columns, row, 0, witness.padded_time_len()))
        .transpose()?;
    let last_opening = witness
        .semantic_rows
        .last()
        .map(|row| {
            build_opening_proof_from_encoded(
                &encoded_columns,
                row,
                witness.semantic_rows.len().saturating_sub(1),
                witness.padded_time_len(),
            )
        })
        .transpose()?;
    let artifact = RootLaneCommitmentArtifact {
        row_width: RV32IM_ROOT_ROW_WIDTH as u64,
        time_len: witness.time_len() as u64,
        padded_time_len: witness.padded_time_len() as u64,
        commitments,
        first_opening,
        last_opening,
        digest: [0; 32],
    };
    Ok(RootLaneCommitmentArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    })
}

pub(crate) fn build_root_lane_commitment_summary_artifact_from_public_witness(
    params: &NeoParams,
    witness: &RootLanePublicWitness,
) -> Result<RootLaneCommitmentSummaryArtifact, SimpleKernelError> {
    let commitments = build_commitment_summary_from_columns(params, &witness.columns, witness.padded_time_len())?;
    let first_opening = witness
        .first_row_digest
        .map(|digest| build_selected_row_reference(commitments.digest, 0, digest));
    let last_opening = witness
        .last_row_digest
        .map(|digest| build_selected_row_reference(commitments.digest, witness.time_len.saturating_sub(1), digest));
    let artifact = RootLaneCommitmentSummaryArtifact {
        time_len: witness.time_len as u64,
        commitments,
        first_selected_row: first_opening,
        last_selected_row: last_opening,
        digest: [0; 32],
    };
    Ok(RootLaneCommitmentSummaryArtifact {
        digest: artifact.expected_digest(),
        ..artifact
    })
}

pub fn build_root_lane_commitment_artifact(
    params: &NeoParams,
    rows: &[Rv32ExpandedRow],
) -> Result<RootLaneCommitmentArtifact, SimpleKernelError> {
    build_root_lane_commitment_artifact_from_witness(params, &build_root_lane_witness(rows))
}

pub fn verify_root_lane_commitment_artifact(
    params: &NeoParams,
    rows: &[Rv32ExpandedRow],
    artifact: &RootLaneCommitmentArtifact,
) -> Result<(), SimpleKernelError> {
    let expected = build_root_lane_commitment_artifact(params, rows)?;
    if artifact != &expected {
        return Err(SimpleKernelError::Bridge(
            "RV32IM root lane commitment artifact mismatch".into(),
        ));
    }
    Ok(())
}
