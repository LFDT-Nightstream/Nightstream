//! Low-norm lowering for direct R1CS frontends.
//!
//! Direct SuperNeo CCS steps require the full witness to fit the small
//! coefficient bound. This module rewrites an arbitrary sparse R1CS assignment
//! into an equivalent bit/limb R1CS whose witness is binary, so the result can
//! pass through the direct CCS adapter without relying on VM-specific code.

use neo_ccs::{CcsMatrix, CscMat};
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::ivc::{DirectCcsFPrimeSnarkError, DirectCcsProgram, DirectCcsStep};
use super::r1cs_export::DirectSparseR1csExport;
use neo_ajtai::Commitment;
use neo_ccs::traits::SModuleHomomorphism;

const FIELD_BITS: usize = 64;
const U32_BITS: usize = 32;
const GOLDILOCKS_LOW_BITS: usize = 32;
const GOLDILOCKS_HIGH_BITS: usize = 32;
const GOLDILOCKS_CANONICAL_AUX_BITS: usize = GOLDILOCKS_HIGH_BITS - 1;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DirectLowNormLaneKind {
    Bit,
    U32,
    Field,
}

#[derive(Clone, Debug, PartialEq, Eq)]
pub struct DirectR1csLowNormLayout {
    kinds: Vec<DirectLowNormLaneKind>,
    public_input_len: usize,
}

#[derive(Clone, Debug, PartialEq, Eq)]
struct LaneMap {
    bits_start_col: usize,
    bit_len: usize,
    canonical_aux_start_col: Option<usize>,
}

impl DirectLowNormLaneKind {
    fn bit_len(self) -> usize {
        match self {
            Self::Bit => 1,
            Self::U32 => U32_BITS,
            Self::Field => FIELD_BITS,
        }
    }

    fn needs_canonical_field_check(self) -> bool {
        matches!(self, Self::Field)
    }
}

impl DirectR1csLowNormLayout {
    pub fn new(public_input_len: usize, kinds: Vec<DirectLowNormLaneKind>) -> Result<Self, DirectCcsFPrimeSnarkError> {
        if public_input_len > kinds.len() {
            return Err(DirectCcsFPrimeSnarkError::Input(format!(
                "direct low-norm R1CS layout public input len {public_input_len} exceeds variable count {}",
                kinds.len()
            )));
        }
        Ok(Self {
            kinds,
            public_input_len,
        })
    }

    pub fn conservative_for_export(export: &DirectSparseR1csExport) -> Self {
        let mut kinds = vec![DirectLowNormLaneKind::Field; export.variable_count];
        if !kinds.is_empty() && export.witness.first().copied() == Some(F::ONE) {
            kinds[0] = DirectLowNormLaneKind::Bit;
        }
        Self {
            kinds,
            public_input_len: export.public_input_len,
        }
    }

    pub fn kinds(&self) -> &[DirectLowNormLaneKind] {
        &self.kinds
    }

    pub fn public_input_len(&self) -> usize {
        self.public_input_len
    }
}

pub fn lower_sparse_r1cs_export_to_low_norm(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
) -> Result<DirectSparseR1csExport, DirectCcsFPrimeSnarkError> {
    validate_layout(export, layout)?;

    let mut witness = Vec::new();
    let mut lanes = vec![
        LaneMap {
            bits_start_col: 0,
            bit_len: 0,
            canonical_aux_start_col: None,
        };
        export.variable_count
    ];
    let public_input_len = push_lanes_for_range(export, layout, 0..export.public_input_len, &mut witness, &mut lanes)?;
    push_lanes_for_range(
        export,
        layout,
        export.public_input_len..export.variable_count,
        &mut witness,
        &mut lanes,
    )?;
    push_canonical_aux_bits(export, layout, &mut witness, &mut lanes)?;

    let mut a_trips = Vec::new();
    let mut b_trips = Vec::new();
    let mut c_trips = Vec::new();
    expand_matrix(&export.a, export.constraint_count, &lanes, &mut a_trips)?;
    expand_matrix(&export.b, export.constraint_count, &lanes, &mut b_trips)?;
    expand_matrix(&export.c, export.constraint_count, &lanes, &mut c_trips)?;

    let mut row = export.constraint_count;
    add_bit_constraints(&mut a_trips, &mut b_trips, &mut row, witness.len());
    add_canonical_field_constraints(&mut a_trips, &mut b_trips, &mut c_trips, &mut row, layout, &lanes)?;
    let variable_count = witness.len();

    Ok(DirectSparseR1csExport {
        a: CcsMatrix::Csc(CscMat::from_triplets(a_trips, row, variable_count)),
        b: CcsMatrix::Csc(CscMat::from_triplets(b_trips, row, variable_count)),
        c: CcsMatrix::Csc(CscMat::from_triplets(c_trips, row, variable_count)),
        witness,
        public_input_len,
        constraint_count: row,
        variable_count,
    })
}

/// Lowers an exported sparse R1CS assignment and packages it as one direct CCS
/// step. This is the frontend-neutral bridge for circuits whose native R1CS
/// witness contains full field values instead of SuperNeo small digits.
pub fn lower_sparse_r1cs_export_to_low_norm_program_and_step<L>(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    log: &L,
    label: impl Into<String>,
) -> Result<(DirectSparseR1csExport, DirectCcsProgram, DirectCcsStep), DirectCcsFPrimeSnarkError>
where
    L: SModuleHomomorphism<F, Commitment>,
{
    let lowered = lower_sparse_r1cs_export_to_low_norm(export, layout)?;
    let program = lowered.to_direct_ccs_program()?;
    let step = lowered.clone().into_direct_ccs_step(&program, log, label)?;
    Ok((lowered, program, step))
}

fn validate_layout(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    if export.variable_count != export.witness.len() {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS export variable count {} does not match witness len {}",
            export.variable_count,
            export.witness.len()
        )));
    }
    if layout.kinds.len() != export.variable_count {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS layout has {} lane kinds, expected {}",
            layout.kinds.len(),
            export.variable_count
        )));
    }
    if layout.public_input_len != export.public_input_len {
        return Err(DirectCcsFPrimeSnarkError::Input(format!(
            "direct low-norm R1CS layout public input len {} does not match export public input len {}",
            layout.public_input_len, export.public_input_len
        )));
    }
    if export.public_input_len == 0
        || export.witness.first().copied() != Some(F::ONE)
        || layout.kinds.first().copied() != Some(DirectLowNormLaneKind::Bit)
    {
        return Err(DirectCcsFPrimeSnarkError::Input(
            "direct low-norm R1CS lowering requires public column 0 to be the Bellpepper one input".into(),
        ));
    }
    Ok(())
}

fn push_lanes_for_range(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    range: std::ops::Range<usize>,
    witness: &mut Vec<F>,
    lanes: &mut [LaneMap],
) -> Result<usize, DirectCcsFPrimeSnarkError> {
    for col in range {
        let start = witness.len();
        let kind = layout.kinds[col];
        push_lane_bits(witness, export.witness[col], kind, col)?;
        lanes[col].bits_start_col = start;
        lanes[col].bit_len = kind.bit_len();
    }
    Ok(witness.len())
}

fn push_lane_bits(
    witness: &mut Vec<F>,
    value: F,
    kind: DirectLowNormLaneKind,
    original_col: usize,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    let raw = value.as_canonical_u64();
    match kind {
        DirectLowNormLaneKind::Bit => {
            if raw > 1 {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct low-norm R1CS bit lane at original column {original_col} has value {raw}"
                )));
            }
            witness.push(F::from_u64(raw));
        }
        DirectLowNormLaneKind::U32 => {
            if raw > u32::MAX as u64 {
                return Err(DirectCcsFPrimeSnarkError::Input(format!(
                    "direct low-norm R1CS u32 lane at original column {original_col} has value {raw}"
                )));
            }
            push_bits(witness, raw, U32_BITS);
        }
        DirectLowNormLaneKind::Field => push_bits(witness, raw, FIELD_BITS),
    }
    Ok(())
}

fn push_bits(witness: &mut Vec<F>, value: u64, bit_len: usize) {
    for bit_index in 0..bit_len {
        witness.push(F::from_u64((value >> bit_index) & 1));
    }
}

fn push_canonical_aux_bits(
    export: &DirectSparseR1csExport,
    layout: &DirectR1csLowNormLayout,
    witness: &mut Vec<F>,
    lanes: &mut [LaneMap],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (col, &kind) in layout.kinds.iter().enumerate() {
        if !kind.needs_canonical_field_check() {
            continue;
        }
        let start = witness.len();
        lanes[col].canonical_aux_start_col = Some(start);
        let value = export.witness[col].as_canonical_u64();
        let mut high_all = ((value >> GOLDILOCKS_LOW_BITS) & 1) & ((value >> (GOLDILOCKS_LOW_BITS + 1)) & 1);
        witness.push(F::from_u64(high_all));
        for high_index in 2..GOLDILOCKS_HIGH_BITS {
            high_all &= (value >> (GOLDILOCKS_LOW_BITS + high_index)) & 1;
            witness.push(F::from_u64(high_all));
        }
        if witness.len() != start + GOLDILOCKS_CANONICAL_AUX_BITS {
            return Err(DirectCcsFPrimeSnarkError::Input(
                "direct low-norm R1CS canonical aux layout mismatch".into(),
            ));
        }
    }
    Ok(())
}

fn expand_matrix(
    matrix: &CcsMatrix<F>,
    rows: usize,
    lanes: &[LaneMap],
    out: &mut Vec<(usize, usize, F)>,
) -> Result<(), DirectCcsFPrimeSnarkError> {
    match matrix {
        CcsMatrix::Identity { n } => {
            if *n > lanes.len() {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct low-norm R1CS identity matrix exceeds export variable count".into(),
                ));
            }
            for row in 0..(*n).min(rows) {
                push_expanded_term(row, F::ONE, &lanes[row], out);
            }
        }
        CcsMatrix::Csc(csc) => {
            if csc.ncols > lanes.len() || csc.nrows != rows {
                return Err(DirectCcsFPrimeSnarkError::Input(
                    "direct low-norm R1CS matrix shape does not match export shape".into(),
                ));
            }
            for col in 0..csc.ncols {
                for idx in csc.col_ptr[col]..csc.col_ptr[col + 1] {
                    push_expanded_term(csc.row_idx[idx], csc.vals[idx], &lanes[col], out);
                }
            }
        }
    }
    Ok(())
}

fn push_expanded_term(row: usize, coeff: F, lane: &LaneMap, out: &mut Vec<(usize, usize, F)>) {
    let mut bit_coeff = coeff;
    for bit_index in 0..lane.bit_len {
        out.push((row, lane.bits_start_col + bit_index, bit_coeff));
        bit_coeff += bit_coeff;
    }
}

fn add_bit_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    variable_count: usize,
) {
    for col in 0..variable_count {
        a_trips.push((*row, col, F::ONE));
        b_trips.push((*row, col, F::ONE));
        b_trips.push((*row, 0, -F::ONE));
        *row += 1;
    }
}

fn add_canonical_field_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    layout: &DirectR1csLowNormLayout,
    lanes: &[LaneMap],
) -> Result<(), DirectCcsFPrimeSnarkError> {
    for (col, &kind) in layout.kinds.iter().enumerate() {
        if !kind.needs_canonical_field_check() {
            continue;
        }
        let lane = &lanes[col];
        let aux_start = lane
            .canonical_aux_start_col
            .ok_or_else(|| DirectCcsFPrimeSnarkError::Input("direct low-norm R1CS canonical aux missing".into()))?;
        add_goldilocks_canonical_lane_constraints(a_trips, b_trips, c_trips, row, lane.bits_start_col, aux_start);
    }
    Ok(())
}

fn add_goldilocks_canonical_lane_constraints(
    a_trips: &mut Vec<(usize, usize, F)>,
    b_trips: &mut Vec<(usize, usize, F)>,
    c_trips: &mut Vec<(usize, usize, F)>,
    row: &mut usize,
    lane_start_col: usize,
    aux_start_col: usize,
) {
    a_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS, F::ONE));
    b_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS + 1, F::ONE));
    c_trips.push((*row, aux_start_col, F::ONE));
    *row += 1;

    for high_index in 2..GOLDILOCKS_HIGH_BITS {
        a_trips.push((*row, aux_start_col + high_index - 2, F::ONE));
        b_trips.push((*row, lane_start_col + GOLDILOCKS_LOW_BITS + high_index, F::ONE));
        c_trips.push((*row, aux_start_col + high_index - 1, F::ONE));
        *row += 1;
    }

    let high_all_ones_col = aux_start_col + GOLDILOCKS_CANONICAL_AUX_BITS - 1;
    for low_index in 0..GOLDILOCKS_LOW_BITS {
        a_trips.push((*row, high_all_ones_col, F::ONE));
        b_trips.push((*row, lane_start_col + low_index, F::ONE));
        *row += 1;
    }
}
