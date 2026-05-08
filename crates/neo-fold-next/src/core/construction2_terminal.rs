//! Shared Construction-2 terminal committed-step circuit mechanics.
//!
//! This module owns only relation-neutral plumbing: public `u_i = (C_i, x_i)`
//! boundary allocation, Poseidon2 boundary digest checks, low-norm source-image
//! encoding helpers, and packed Ajtai commitment checks. It does not own any
//! RV32IM or direct-CCS F' semantics.

use bellpepper_core::{num::AllocatedNum, ConstraintSystem, LinearCombination, SynthesisError, Variable};
use neo_math::{balanced::to_balanced_i128, D, F};
use neo_params::NeoParams;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use spartan2::traits::circuit::SpartanCircuit;

use crate::spartan_backend::{NeoFoldDeciderEngine, SpartanF, SplitR1CSShape};
use crate::superneo_circuit::ce_consistency::enforce_ajtai_commitment_linear_consistency;
use crate::superneo_circuit::transcript::hash_field_linear_combinations_raw;
use crate::superneo_circuit::witness::PackedWitnessVar;
use crate::superneo_nifs_circuit::{digest32_as_spartan_fields, enforce_digest_eq};

pub(crate) const U32_BIT_WIDTH: usize = 32;
pub(crate) const U64_BIT_WIDTH: usize = 64;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(crate) enum TerminalPrivateColumnEncoding {
    UnusedPadding,
    Bit,
    U32,
    U64,
}

impl TerminalPrivateColumnEncoding {
    pub(crate) fn limb_count(self) -> usize {
        match self {
            Self::UnusedPadding => 0,
            Self::Bit => 1,
            Self::U32 => U32_BIT_WIDTH,
            Self::U64 => U64_BIT_WIDTH,
        }
    }

    pub(crate) fn limb_label(self, limb_idx: usize) -> String {
        match self {
            Self::UnusedPadding => "padding".to_string(),
            Self::Bit => "bit".to_string(),
            Self::U32 | Self::U64 => format!("bit{limb_idx}"),
        }
    }
}

pub(crate) struct Construction2TerminalBoundaryView<'a> {
    pub(crate) fresh_instance_digest: [u8; 32],
    pub(crate) commitment_digest: [u8; 32],
    pub(crate) commitment_d: u64,
    pub(crate) commitment_kappa: u64,
    pub(crate) commitment_data: &'a [F],
    pub(crate) x_i_bytes: [u8; 32],
}

pub(crate) struct Construction2TerminalBoundaryInputs {
    pub(crate) fresh_instance_digest: [AllocatedNum<SpartanF>; 4],
    pub(crate) commitment_digest: [AllocatedNum<SpartanF>; 4],
    pub(crate) commitment_d: AllocatedNum<SpartanF>,
    pub(crate) commitment_kappa: AllocatedNum<SpartanF>,
    pub(crate) commitment_data: Vec<AllocatedNum<SpartanF>>,
    pub(crate) x_i: [AllocatedNum<SpartanF>; 4],
}

pub(crate) fn terminal_boundary_public_values(boundary: &Construction2TerminalBoundaryView<'_>) -> Vec<SpartanF> {
    let mut values = Vec::with_capacity(14 + boundary.commitment_data.len());
    values.extend(digest32_as_spartan_fields(boundary.fresh_instance_digest));
    values.extend(digest32_as_spartan_fields(boundary.commitment_digest));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_d));
    values.push(SpartanF::from_canonical_u64(boundary.commitment_kappa));
    values.extend(boundary.commitment_data.iter().map(native_to_spartan));
    values.extend(digest32_as_spartan_fields(boundary.x_i_bytes));
    values
}

pub(crate) fn alloc_terminal_boundary_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label_prefix: &str,
    boundary: &Construction2TerminalBoundaryView<'_>,
) -> Result<Construction2TerminalBoundaryInputs, SynthesisError> {
    let fresh_instance_digest = alloc_digest_public_inputs(
        cs,
        &format!("{label_prefix}_fresh_instance_digest"),
        boundary.fresh_instance_digest,
    )?;
    let commitment_digest = alloc_digest_public_inputs(
        cs,
        &format!("{label_prefix}_commitment_digest"),
        boundary.commitment_digest,
    )?;
    let commitment_d = AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_d")), || {
        Ok(SpartanF::from_canonical_u64(boundary.commitment_d))
    })?;
    let commitment_kappa =
        AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_kappa")), || {
            Ok(SpartanF::from_canonical_u64(boundary.commitment_kappa))
        })?;
    let commitment_data = boundary
        .commitment_data
        .iter()
        .enumerate()
        .map(|(idx, value)| {
            AllocatedNum::alloc_input(cs.namespace(|| format!("{label_prefix}_commitment_data_{idx}")), || {
                Ok(native_to_spartan(value))
            })
        })
        .collect::<Result<Vec<_>, _>>()?;
    let x_i = alloc_digest_public_inputs(cs, &format!("{label_prefix}_x_i"), boundary.x_i_bytes)?;
    Ok(Construction2TerminalBoundaryInputs {
        fresh_instance_digest,
        commitment_digest,
        commitment_d,
        commitment_kappa,
        commitment_data,
        x_i,
    })
}

pub(crate) fn enforce_terminal_boundary_digests<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    boundary: &Construction2TerminalBoundaryInputs,
    commitment_raw_tag: u64,
    public_boundary_raw_tag: u64,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_d_eq")),
        &boundary.commitment_d,
        SpartanF::from_canonical_u64(D as u64),
        &format!("{label_prefix}_commitment_d_eq"),
    );
    let expected_commitment_digest = construction2_commitment_digest_circuit(
        &mut cs.namespace(|| format!("{label_prefix}_expected_commitment_digest")),
        commitment_raw_tag,
        &boundary.commitment_d,
        &boundary.commitment_kappa,
        &boundary.commitment_data,
        &format!("{label_prefix}_expected_commitment_digest"),
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_digest_eq")),
        &boundary.commitment_digest,
        &expected_commitment_digest,
        &format!("{label_prefix}_commitment_digest_eq"),
    )?;
    let expected_fresh_instance_digest = construction2_public_boundary_digest_circuit(
        &mut cs.namespace(|| format!("{label_prefix}_expected_fresh_instance_digest")),
        public_boundary_raw_tag,
        &boundary.commitment_digest,
        &boundary.x_i,
        &format!("{label_prefix}_expected_fresh_instance_digest"),
    )?;
    enforce_digest_eq(
        &mut cs.namespace(|| format!("{label_prefix}_fresh_instance_digest_eq")),
        &boundary.fresh_instance_digest,
        &expected_fresh_instance_digest,
        &format!("{label_prefix}_fresh_instance_digest_eq"),
    )
}

pub(crate) fn enforce_public_commitment_shape<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    boundary: &Construction2TerminalBoundaryInputs,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
    if packed_z.rows() != D || boundary.commitment_data.len() % D != 0 {
        return Err(SynthesisError::Unsatisfiable);
    }
    let expected_kappa = boundary.commitment_data.len() / D;
    enforce_allocated_num_eq_constant(
        &mut cs.namespace(|| format!("{label_prefix}_commitment_kappa_matches_data_len")),
        &boundary.commitment_kappa,
        SpartanF::from_canonical_u64(expected_kappa as u64),
        &format!("{label_prefix}_commitment_kappa_matches_data_len"),
    );
    Ok(())
}

pub(crate) fn enforce_packed_padding_zero<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    committed_width: usize,
    label_prefix: &str,
) -> Result<(), SynthesisError> {
    for row in 0..packed_z.rows() {
        for col in 0..packed_z.cols() {
            let logical_col = col
                .checked_mul(D)
                .and_then(|base| base.checked_add(row))
                .ok_or(SynthesisError::Unsatisfiable)?;
            if logical_col < committed_width {
                continue;
            }
            let padding = packed_z.entry(row, col)?;
            cs.enforce(
                || format!("{label_prefix}_{row}_{col}"),
                |lc| lc + padding.get_variable(),
                |lc| lc + CS::one(),
                |lc| lc,
            );
        }
    }
    Ok(())
}

pub(crate) fn enforce_terminal_ajtai_commitment<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    packed_z: &PackedWitnessVar,
    commitment_inputs: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<(), SynthesisError> {
    if packed_z.rows() != D {
        return Err(SynthesisError::Unsatisfiable);
    }
    let packed_entries = packed_z
        .row_major_values()
        .iter()
        .map(|entry| LinearCombination::<SpartanF>::zero() + entry.get_variable())
        .collect::<Vec<_>>();
    enforce_ajtai_commitment_linear_consistency(
        cs,
        packed_z.rows(),
        packed_z.cols(),
        &packed_entries,
        commitment_inputs,
        label,
    )
}

pub(crate) fn collect_private_witness_labels<C>(circuit: &C, context: &str) -> Result<Vec<String>, String>
where
    C: SpartanCircuit<NeoFoldDeciderEngine>,
{
    let mut cs = LabelOnlyConstraintSystem::new();
    let shared = circuit
        .shared(&mut cs)
        .map_err(|err| format!("{context} label shared allocation failed: {err}"))?;
    let precommitted = circuit
        .precommitted(&mut cs, &shared)
        .map_err(|err| format!("{context} label precommitted allocation failed: {err}"))?;
    circuit
        .synthesize(&mut cs, &shared, &precommitted, None)
        .map_err(|err| format!("{context} label synthesis failed: {err}"))?;
    Ok(cs.aux_labels())
}

pub(crate) fn padded_private_witness_labels(
    split_shape: &SplitR1CSShape<NeoFoldDeciderEngine>,
    private_witness_labels: &[String],
    context: &str,
) -> Result<Vec<Option<String>>, String> {
    if private_witness_labels.len() != split_shape.num_variables_unpadded() {
        return Err(format!(
            "{context} unpadded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables_unpadded(),
            private_witness_labels.len()
        ));
    }

    let mut padded = Vec::with_capacity(split_shape.num_variables());
    let mut cursor = 0usize;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_shared_unpadded(),
        split_shape.num_shared(),
        context,
        "shared",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_precommitted_unpadded(),
        split_shape.num_precommitted(),
        context,
        "precommitted",
    )?;
    push_padded_witness_label_segment(
        &mut padded,
        private_witness_labels,
        &mut cursor,
        split_shape.num_rest_unpadded(),
        split_shape.num_rest(),
        context,
        "rest",
    )?;

    if cursor != private_witness_labels.len() {
        return Err(format!(
            "{context} witness label padding consumed {cursor} labels but {} were supplied",
            private_witness_labels.len()
        ));
    }
    if padded.len() != split_shape.num_variables() {
        return Err(format!(
            "{context} padded witness label count mismatch: expected {}, got {}",
            split_shape.num_variables(),
            padded.len()
        ));
    }
    Ok(padded)
}

pub(crate) fn committed_nc_range_error(
    params: &NeoParams,
    full_vector: &[F],
    mut committed_index_label: impl FnMut(usize) -> String,
    context: &str,
) -> Option<String> {
    for (idx, value) in full_vector.iter().copied().enumerate() {
        if is_superneo_digit_representable(value, params.b) {
            continue;
        }
        return Some(format!(
            "{context} committed value at {} is not representable in D={} balanced base-{} digits (centered value {})",
            committed_index_label(idx),
            D,
            params.b,
            to_balanced_i128(value),
        ));
    }
    None
}

pub(crate) fn low_norm_encoded_values(
    value: F,
    encoding: TerminalPrivateColumnEncoding,
    context: &str,
) -> Result<Vec<F>, String> {
    match encoding {
        TerminalPrivateColumnEncoding::UnusedPadding => {
            if value != F::ZERO {
                return Err(format!(
                    "{context} padded witness value is non-zero: {}",
                    value.as_canonical_u64()
                ));
            }
            Ok(Vec::new())
        }
        TerminalPrivateColumnEncoding::Bit => {
            let canonical = value.as_canonical_u64();
            if canonical > 1 {
                return Err(format!("{context} boolean witness value is not binary: {canonical}"));
            }
            Ok(vec![value])
        }
        TerminalPrivateColumnEncoding::U32 => low_norm_bit_values(value, U32_BIT_WIDTH, context),
        TerminalPrivateColumnEncoding::U64 => low_norm_bit_values(value, U64_BIT_WIDTH, context),
    }
}

pub(crate) fn native_to_spartan(value: &F) -> SpartanF {
    SpartanF::from_canonical_u64(value.as_canonical_u64())
}

pub(crate) fn enforce_boolean_allocated<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    bit: &AllocatedNum<SpartanF>,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + bit.get_variable(),
        |lc| lc + bit.get_variable() - CS::one(),
        |lc| lc,
    );
}

pub(crate) fn enforce_allocated_num_eq_constant<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    value: &AllocatedNum<SpartanF>,
    expected: SpartanF,
    label: &str,
) {
    cs.enforce(
        || label,
        |lc| lc + value.get_variable(),
        |lc| lc + CS::one(),
        |lc| lc + (expected, CS::one()),
    );
}

fn low_norm_bit_values(value: F, bit_width: usize, context: &str) -> Result<Vec<F>, String> {
    let canonical = value.as_canonical_u64();
    if bit_width < U64_BIT_WIDTH && (canonical >> bit_width) != 0 {
        return Err(format!(
            "{context} witness value {canonical} does not fit in {bit_width} base-2 digits"
        ));
    }
    Ok((0..bit_width)
        .map(|bit_idx| F::from_u64((canonical >> bit_idx) & 1))
        .collect())
}

fn push_padded_witness_label_segment(
    padded: &mut Vec<Option<String>>,
    labels: &[String],
    cursor: &mut usize,
    unpadded_len: usize,
    padded_len: usize,
    context: &str,
    segment_name: &str,
) -> Result<(), String> {
    if padded_len < unpadded_len {
        return Err(format!(
            "{context} {segment_name} witness segment has padded length {padded_len} below unpadded length {unpadded_len}"
        ));
    }
    let end = cursor
        .checked_add(unpadded_len)
        .ok_or_else(|| format!("{context} witness label cursor overflow"))?;
    if end > labels.len() {
        return Err(format!(
            "{context} {segment_name} witness labels exceed collected label count"
        ));
    }
    padded.extend(labels[*cursor..end].iter().cloned().map(Some));
    padded.resize(padded.len() + (padded_len - unpadded_len), None);
    *cursor = end;
    Ok(())
}

fn construction2_public_boundary_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    raw_tag: u64,
    commitment_digest: &[AllocatedNum<SpartanF>; 4],
    x_i: &[AllocatedNum<SpartanF>; 4],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::with_capacity(9);
    let mut field_constants = Vec::with_capacity(9);
    let mut field_values = Vec::with_capacity(9);

    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(raw_tag));
    field_values.push(SpartanF::from_canonical_u64(raw_tag));
    for lane in commitment_digest.iter().chain(x_i.iter()) {
        field_terms.push(vec![(lane.get_variable(), SpartanF::from_canonical_u64(1))]);
        field_constants.push(SpartanF::from_canonical_u64(0));
        field_values.push(lane.get_value().unwrap_or(SpartanF::from_canonical_u64(0)));
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

fn construction2_commitment_digest_circuit<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    raw_tag: u64,
    d: &AllocatedNum<SpartanF>,
    kappa: &AllocatedNum<SpartanF>,
    data: &[AllocatedNum<SpartanF>],
    label: &str,
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let mut field_terms = Vec::with_capacity(3 + data.len());
    let mut field_constants = Vec::with_capacity(3 + data.len());
    let mut field_values = Vec::with_capacity(3 + data.len());

    field_terms.push(Vec::new());
    field_constants.push(SpartanF::from_canonical_u64(raw_tag));
    field_values.push(SpartanF::from_canonical_u64(raw_tag));
    for value in [d, kappa].into_iter().chain(data.iter()) {
        field_terms.push(vec![(value.get_variable(), SpartanF::from_canonical_u64(1))]);
        field_constants.push(SpartanF::from_canonical_u64(0));
        field_values.push(value.get_value().unwrap_or(SpartanF::from_canonical_u64(0)));
    }

    hash_field_linear_combinations_raw(
        cs.namespace(|| format!("{label}_hash")),
        &field_terms,
        &field_constants,
        &field_values,
    )
}

fn alloc_digest_public_inputs<CS: ConstraintSystem<SpartanF>>(
    cs: &mut CS,
    label: &str,
    digest: [u8; 32],
) -> Result<[AllocatedNum<SpartanF>; 4], SynthesisError> {
    let fields = digest32_as_spartan_fields(digest);
    let values = fields
        .into_iter()
        .enumerate()
        .map(|(idx, value)| AllocatedNum::alloc_input(cs.namespace(|| format!("{label}_{idx}")), || Ok(value)))
        .collect::<Result<Vec<_>, _>>()?;
    values.try_into().map_err(|_| SynthesisError::Unsatisfiable)
}

fn is_superneo_digit_representable(value: F, base: u32) -> bool {
    if base < 2 {
        return false;
    }
    let mut remainder = to_balanced_i128(value);
    let base = base as i128;
    for _ in 0..D {
        let (_, quotient) = balanced_divrem(remainder, base);
        remainder = quotient;
    }
    remainder == 0
}

fn balanced_divrem(value: i128, base: i128) -> (i128, i128) {
    debug_assert!(base >= 2);
    let mut remainder = value % base;
    let mut quotient = (value - remainder) / base;
    let half = base / 2;
    if remainder > half {
        remainder -= base;
        quotient += 1;
    } else if remainder < -half {
        remainder += base;
        quotient -= 1;
    }
    (remainder, quotient)
}

#[derive(Clone, Debug)]
struct LabelOnlyConstraintSystem {
    current_namespace: Vec<String>,
    inputs: usize,
    aux_labels: Vec<String>,
}

impl LabelOnlyConstraintSystem {
    fn new() -> Self {
        Self {
            current_namespace: Vec::new(),
            inputs: 1,
            aux_labels: Vec::new(),
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

    fn aux_labels(self) -> Vec<String> {
        self.aux_labels
    }
}

impl ConstraintSystem<SpartanF> for LabelOnlyConstraintSystem {
    type Root = Self;

    fn alloc<FN, A, AR>(&mut self, annotation: A, _: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let var = Variable::new_unchecked(bellpepper_core::Index::Aux(self.aux_labels.len()));
        self.aux_labels.push(self.alloc_path(&annotation().into()));
        Ok(var)
    }

    fn alloc_input<FN, A, AR>(&mut self, _: A, _: FN) -> Result<Variable, SynthesisError>
    where
        FN: FnOnce() -> Result<SpartanF, SynthesisError>,
        A: FnOnce() -> AR,
        AR: Into<String>,
    {
        let var = Variable::new_unchecked(bellpepper_core::Index::Input(self.inputs));
        self.inputs = self
            .inputs
            .checked_add(1)
            .ok_or(SynthesisError::Unsatisfiable)?;
        Ok(var)
    }

    fn enforce<A, AR, LA, LB, LC>(&mut self, _: A, _: LA, _: LB, _: LC)
    where
        A: FnOnce() -> AR,
        AR: Into<String>,
        LA: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LB: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
        LC: FnOnce(LinearCombination<SpartanF>) -> LinearCombination<SpartanF>,
    {
    }

    fn push_namespace<NR, N>(&mut self, name_fn: N)
    where
        NR: Into<String>,
        N: FnOnce() -> NR,
    {
        self.current_namespace.push(name_fn().into());
    }

    fn pop_namespace(&mut self) {
        assert!(self.current_namespace.pop().is_some());
    }

    fn get_root(&mut self) -> &mut Self::Root {
        self
    }
}
