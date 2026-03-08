use super::*;
use neo_ccs::Mat;
use neo_memory::riscv::exec_table::{Rv32PoseidonCycleEventRow, Rv32PoseidonPermSlotMetaRow, Rv32PoseidonSidecarTable};
use neo_memory::riscv::lookups::{
    RiscvInstruction, POSEIDON2_ABSORB_FUNCT7, POSEIDON2_CUSTOM_OPCODE, POSEIDON2_FINALIZE_FUNCT7,
    POSEIDON2_SQUEEZE_FUNCT7,
};
use neo_memory::riscv::trace::Rv64TraceLayout;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoseidonCycleTraceLayout {
    pub row_active: usize,
    pub is_first: usize,
    pub is_last: usize,
    pub cycle: usize,
    pub op_absorb: usize,
    pub op_finalize: usize,
    pub op_squeeze: usize,
    pub mode_finalized: usize,
    pub call_ctr: usize,
    pub cursor_before: usize,
    pub cursor_after: usize,
    pub state_pre_start: usize,
    pub state_post_start: usize,
    pub do_perm_slot0: usize,
    pub do_perm_slot1: usize,
    pub absorb_lo32: usize,
    pub absorb_hi32: usize,
    pub squeeze_idx_b0: usize,
    pub squeeze_idx_b1: usize,
    pub squeeze_idx_b2: usize,
    pub squeeze_word_u32: usize,
    pub digest_lo_start: usize,
    pub digest_hi_start: usize,
    pub slot0_in_start: usize,
    pub slot0_out_start: usize,
    pub slot1_in_start: usize,
    pub slot1_out_start: usize,
    pub link_u_slot0: usize,
    pub link_u_slot1: usize,
    pub cont_u_pre: usize,
    pub cont_u_post: usize,
    pub canonical_c0: usize,
    pub canonical_c1: usize,
    pub canonical_lo_sum: usize,
    pub canonical_hi_sum: usize,
}

impl PoseidonCycleTraceLayout {
    pub fn new() -> Self {
        let mut col = 0usize;
        let row_active = col;
        col += 1;
        let is_first = col;
        col += 1;
        let is_last = col;
        col += 1;
        let cycle = col;
        col += 1;
        let op_absorb = col;
        col += 1;
        let op_finalize = col;
        col += 1;
        let op_squeeze = col;
        col += 1;
        let mode_finalized = col;
        col += 1;
        let call_ctr = col;
        col += 1;
        let cursor_before = col;
        col += 1;
        let cursor_after = col;
        col += 1;
        let state_pre_start = col;
        col += 8;
        let state_post_start = col;
        col += 8;
        let do_perm_slot0 = col;
        col += 1;
        let do_perm_slot1 = col;
        col += 1;
        let absorb_lo32 = col;
        col += 1;
        let absorb_hi32 = col;
        col += 1;
        let squeeze_idx_b0 = col;
        col += 1;
        let squeeze_idx_b1 = col;
        col += 1;
        let squeeze_idx_b2 = col;
        col += 1;
        let squeeze_word_u32 = col;
        col += 1;
        let digest_lo_start = col;
        col += 4;
        let digest_hi_start = col;
        col += 4;
        let slot0_in_start = col;
        col += 8;
        let slot0_out_start = col;
        col += 8;
        let slot1_in_start = col;
        col += 8;
        let slot1_out_start = col;
        col += 8;
        let link_u_slot0 = col;
        col += 1;
        let link_u_slot1 = col;
        col += 1;
        let cont_u_pre = col;
        col += 1;
        let cont_u_post = col;
        col += 1;
        let canonical_c0 = col;
        col += 1;
        let canonical_c1 = col;
        col += 1;
        let canonical_lo_sum = col;
        col += 1;
        let canonical_hi_sum = col;

        Self {
            row_active,
            is_first,
            is_last,
            cycle,
            op_absorb,
            op_finalize,
            op_squeeze,
            mode_finalized,
            call_ctr,
            cursor_before,
            cursor_after,
            state_pre_start,
            state_post_start,
            do_perm_slot0,
            do_perm_slot1,
            absorb_lo32,
            absorb_hi32,
            squeeze_idx_b0,
            squeeze_idx_b1,
            squeeze_idx_b2,
            squeeze_word_u32,
            digest_lo_start,
            digest_hi_start,
            slot0_in_start,
            slot0_out_start,
            slot1_in_start,
            slot1_out_start,
            link_u_slot0,
            link_u_slot1,
            cont_u_pre,
            cont_u_post,
            canonical_c0,
            canonical_c1,
            canonical_lo_sum,
            canonical_hi_sum,
        }
    }

    #[inline]
    pub fn state_pre(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.state_pre_start + idx
    }

    #[inline]
    pub fn state_post(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.state_post_start + idx
    }

    #[inline]
    pub fn digest_lo(&self, idx: usize) -> usize {
        debug_assert!(idx < 4);
        self.digest_lo_start + idx
    }

    #[inline]
    pub fn digest_hi(&self, idx: usize) -> usize {
        debug_assert!(idx < 4);
        self.digest_hi_start + idx
    }

    #[inline]
    pub fn slot0_in(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.slot0_in_start + idx
    }

    #[inline]
    pub fn slot0_out(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.slot0_out_start + idx
    }

    #[inline]
    pub fn slot1_in(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.slot1_in_start + idx
    }

    #[inline]
    pub fn slot1_out(&self, idx: usize) -> usize {
        debug_assert!(idx < 8);
        self.slot1_out_start + idx
    }

    #[inline]
    pub fn cols(&self) -> usize {
        self.canonical_hi_sum + 1
    }
}

pub(crate) fn poseidon_cycle_open_col_ids(layout: &PoseidonCycleTraceLayout) -> Vec<usize> {
    // Keep cycle-lane ME openings canonical and complete: prover and verifier both
    // reconstruct the same full logical column vector from this list.
    (0..layout.cols()).collect()
}

pub(crate) type PoseidonCycleClaims = (
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
    Option<(Box<dyn RoundOracle + Send>, K)>,
);

pub(crate) const POSEIDON_IO_LINK_RESIDUAL_COUNT: usize = 11;
pub(crate) const POSEIDON_BITNESS_RESIDUAL_COUNT: usize = 13;
pub(crate) const POSEIDON_CANONICAL_RESIDUAL_COUNT: usize = 10;
pub(crate) const POSEIDON_SIDECAR_LINK_RESIDUAL_COUNT: usize = 21;
pub(crate) const POSEIDON_MODE_RESIDUAL_COUNT: usize = 2;

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn poseidon_cycle_open_cols_cover_full_layout() {
        let layout = PoseidonCycleTraceLayout::new();
        let open_cols = poseidon_cycle_open_col_ids(&layout);
        let expected: Vec<usize> = (0..layout.cols()).collect();
        assert_eq!(
            open_cols, expected,
            "poseidon cycle open columns must cover full layout in order"
        );
    }
}

#[inline]
pub(crate) fn poseidon_io_link_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_5F49_4F4C_494Eu64)
}

#[inline]
pub(crate) fn poseidon_bitness_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_5F42_4954_4E53u64)
}

#[inline]
pub(crate) fn poseidon_canonical_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_5F43_414E_4F4Eu64)
}

#[inline]
pub(crate) fn poseidon_sidecar_link_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_5F53_434C_4E4Bu64)
}

#[inline]
pub(crate) fn poseidon_mode_weight_vector(r_cycle: &[K], len: usize) -> Vec<K> {
    bitness_weights(r_cycle, len, 0x5032_5F4D_4F44_4521u64)
}

#[inline]
pub(crate) fn poseidon_io_link_residuals(
    op_custom: K,
    rd_has_write: K,
    rd_is_zero: K,
    ram_has_read: K,
    ram_has_write: K,
    shout_has_lookup: K,
    funct3_bits: [K; 3],
    funct7_bits: [K; 7],
) -> [K; POSEIDON_IO_LINK_RESIDUAL_COUNT] {
    let f7_hi_sum = funct7_bits[2] + funct7_bits[3] + funct7_bits[4] + funct7_bits[5] + funct7_bits[6];
    [
        op_custom * shout_has_lookup,
        op_custom * ram_has_read,
        op_custom * ram_has_write,
        op_custom * f7_hi_sum,
        op_custom * funct7_bits[0] * funct7_bits[1],
        op_custom * (K::ONE - funct7_bits[1]) * funct3_bits[0],
        op_custom * (K::ONE - funct7_bits[1]) * funct3_bits[1],
        op_custom * (K::ONE - funct7_bits[1]) * funct3_bits[2],
        op_custom * (K::ONE - funct7_bits[1]) * (K::ONE - rd_is_zero),
        op_custom * (K::ONE - funct7_bits[1]) * rd_has_write,
        op_custom * funct7_bits[1] * (rd_has_write + rd_is_zero - K::ONE),
    ]
}

#[inline]
pub(crate) fn poseidon_bitness_residuals(
    op_custom: K,
    rd_has_write: K,
    rd_is_zero: K,
    funct3_bits: [K; 3],
    funct7_bits: [K; 7],
) -> [K; POSEIDON_BITNESS_RESIDUAL_COUNT] {
    [
        w2_bool01(op_custom),
        w2_bool01(rd_has_write),
        w2_bool01(rd_is_zero),
        w2_bool01(funct3_bits[0]),
        w2_bool01(funct3_bits[1]),
        w2_bool01(funct3_bits[2]),
        w2_bool01(funct7_bits[0]),
        w2_bool01(funct7_bits[1]),
        w2_bool01(funct7_bits[2]),
        w2_bool01(funct7_bits[3]),
        w2_bool01(funct7_bits[4]),
        w2_bool01(funct7_bits[5]),
        w2_bool01(funct7_bits[6]),
    ]
}

#[inline]
pub(crate) fn poseidon_canonical_residuals(
    funct3_bits: [K; 3],
    side_op_squeeze: K,
    side_squeeze_word: K,
    side_digest_lo: [K; 4],
    side_digest_hi: [K; 4],
    side_c0: K,
    side_c1: K,
    side_lo_sum: K,
    side_hi_sum: K,
) -> [K; POSEIDON_CANONICAL_RESIDUAL_COUNT] {
    let b0 = funct3_bits[0];
    let b1 = funct3_bits[1];
    let b2 = funct3_bits[2];
    let s0 = (K::ONE - b1) * (K::ONE - b2);
    let s1 = b1 * (K::ONE - b2);
    let s2 = (K::ONE - b1) * b2;
    let s3 = b1 * b2;

    let selected_lo = s0 * side_digest_lo[0] + s1 * side_digest_lo[1] + s2 * side_digest_lo[2] + s3 * side_digest_lo[3];
    let selected_hi = s0 * side_digest_hi[0] + s1 * side_digest_hi[1] + s2 * side_digest_hi[2] + s3 * side_digest_hi[3];
    let selected_word = selected_lo + b0 * (selected_hi - selected_lo);

    let two32 = K::from(F::from_u64(1u64 << 32));
    let mask32 = K::from(F::from_u64(0xFFFF_FFFF));
    [
        side_op_squeeze * (side_squeeze_word - selected_word),
        side_op_squeeze * (selected_lo + mask32 - side_lo_sum - side_c0 * two32),
        side_op_squeeze * (selected_hi + side_c0 - side_hi_sum - side_c1 * two32),
        side_op_squeeze * w2_bool01(side_c0),
        side_op_squeeze * w2_bool01(side_c1),
        side_op_squeeze * side_c1,
        (K::ONE - side_op_squeeze) * side_c0,
        (K::ONE - side_op_squeeze) * side_c1,
        (K::ONE - side_op_squeeze) * side_lo_sum,
        (K::ONE - side_op_squeeze) * side_hi_sum,
    ]
}

#[inline]
pub(crate) fn poseidon_sidecar_link_residuals(
    op_custom: K,
    rd_has_write: K,
    funct3_bits: [K; 3],
    funct7_bits: [K; 7],
    rs1_val: K,
    rs2_val: K,
    rd_val: K,
    side_op_absorb: K,
    side_op_finalize: K,
    side_op_squeeze: K,
    side_squeeze_idx_bits: [K; 3],
    side_absorb_lo: K,
    side_absorb_hi: K,
    side_squeeze_word: K,
) -> [K; POSEIDON_SIDECAR_LINK_RESIDUAL_COUNT] {
    let is_absorb = op_custom * (K::ONE - funct7_bits[0]) * (K::ONE - funct7_bits[1]);
    let is_finalize = op_custom * funct7_bits[0] * (K::ONE - funct7_bits[1]);
    let is_squeeze = op_custom * (K::ONE - funct7_bits[0]) * funct7_bits[1];
    [
        side_op_absorb - is_absorb,
        side_op_finalize - is_finalize,
        side_op_squeeze - is_squeeze,
        side_op_absorb * (side_op_absorb - K::ONE),
        side_op_finalize * (side_op_finalize - K::ONE),
        side_op_squeeze * (side_op_squeeze - K::ONE),
        side_op_absorb * side_op_finalize,
        side_op_absorb * side_op_squeeze,
        side_op_finalize * side_op_squeeze,
        side_op_absorb * (side_absorb_lo - rs1_val),
        side_op_absorb * (side_absorb_hi - rs2_val),
        side_op_squeeze * rd_has_write * (side_squeeze_word - rd_val),
        side_op_squeeze * (side_squeeze_idx_bits[0] - funct3_bits[0]),
        side_op_squeeze * (side_squeeze_idx_bits[1] - funct3_bits[1]),
        side_op_squeeze * (side_squeeze_idx_bits[2] - funct3_bits[2]),
        (K::ONE - side_op_absorb) * side_absorb_lo,
        (K::ONE - side_op_absorb) * side_absorb_hi,
        (K::ONE - side_op_squeeze) * side_squeeze_idx_bits[0],
        (K::ONE - side_op_squeeze) * side_squeeze_idx_bits[1],
        (K::ONE - side_op_squeeze) * side_squeeze_idx_bits[2],
        (K::ONE - side_op_squeeze) * side_squeeze_word,
    ]
}

#[inline]
pub(crate) fn poseidon_mode_residuals(
    side_op_finalize: K,
    side_op_squeeze: K,
    side_mode_finalized: K,
) -> [K; POSEIDON_MODE_RESIDUAL_COUNT] {
    [
        side_op_squeeze * (K::ONE - side_mode_finalized),
        side_op_finalize * side_mode_finalized,
    ]
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum PoseidonSidecarBuildMode {
    Absorbing,
    Finalized,
}

#[derive(Clone, Debug)]
pub(crate) struct PoseidonSidecarCarryState {
    pub mode_finalized: bool,
    pub state: [Goldilocks; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
    pub absorb_cursor: usize,
    pub digest_words: [u32; neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN * 2],
    pub call_ctr: u64,
}

impl PoseidonSidecarCarryState {
    pub(crate) fn new() -> Self {
        Self {
            mode_finalized: false,
            state: [Goldilocks::ZERO; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
            absorb_cursor: 0,
            digest_words: [0u32; neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN * 2],
            call_ctr: 0,
        }
    }
}

#[inline]
pub(crate) fn poseidon_cycle_continuity_break_before(row: &Rv32PoseidonCycleEventRow) -> bool {
    // A new message starts when an absorb executes while the pre-row mode is Finalized.
    // This row resets state/cursor and bumps call_ctr before applying the absorb, so
    // continuity should not link across the boundary.
    row.op_absorb && row.mode_finalized
}

#[inline]
fn poseidon_state_to_u64_local(
    state: &[Goldilocks; neo_ccs::crypto::poseidon2_goldilocks::WIDTH],
) -> [u64; neo_ccs::crypto::poseidon2_goldilocks::WIDTH] {
    let mut out = [0u64; neo_ccs::crypto::poseidon2_goldilocks::WIDTH];
    for (i, x) in state.iter().enumerate() {
        out[i] = x.as_canonical_u64();
    }
    out
}

#[inline]
fn canonical_u64_lt_goldilocks_aux_local(v: u64) -> (u32, u32, u32, u32) {
    let lo = v as u32;
    let hi = (v >> 32) as u32;
    let (lo_sum, c0) = lo.overflowing_add(0xFFFF_FFFF);
    let (hi_sum, c1) = hi.overflowing_add(if c0 { 1 } else { 0 });
    (lo_sum, hi_sum, u32::from(c0), u32::from(c1))
}

#[inline]
fn decode_poseidon_precompile_word(instr_word: u32) -> Option<RiscvInstruction> {
    let opcode = instr_word & 0x7f;
    if opcode != POSEIDON2_CUSTOM_OPCODE {
        return None;
    }
    let rd = ((instr_word >> 7) & 0x1f) as u8;
    let funct3 = ((instr_word >> 12) & 0x07) as u8;
    let rs1 = ((instr_word >> 15) & 0x1f) as u8;
    let rs2 = ((instr_word >> 20) & 0x1f) as u8;
    let funct7 = ((instr_word >> 25) & 0x7f) as u8;
    match funct7 as u32 {
        POSEIDON2_ABSORB_FUNCT7 if funct3 == 0 && rd == 0 => Some(RiscvInstruction::Poseidon2AbsorbElem { rs1, rs2 }),
        POSEIDON2_FINALIZE_FUNCT7 if funct3 == 0 && rd == 0 && rs1 == 0 && rs2 == 0 => {
            Some(RiscvInstruction::Poseidon2Finalize)
        }
        POSEIDON2_SQUEEZE_FUNCT7 if rs1 == 0 && rs2 == 0 && funct3 < 8 => {
            Some(RiscvInstruction::Poseidon2SqueezeWord { rd, idx: funct3 })
        }
        _ => None,
    }
}

#[derive(Clone, Copy, Debug)]
pub(crate) struct PoseidonCpuWordCols {
    pub(crate) active: usize,
    pub(crate) instr_word: usize,
    pub(crate) rs1_word: usize,
    pub(crate) rs2_word: usize,
    pub(crate) rd_word: usize,
    pub(crate) shout_has_lookup: usize,
}

#[inline]
pub(crate) fn poseidon_cpu_word_cols_for_cpu_len(cpu_cols_len: usize) -> PoseidonCpuWordCols {
    let rv32 = Rv32TraceLayout::new();
    if cpu_cols_len >= Rv64TraceLayout::new().cols {
        let rv64 = Rv64TraceLayout::new();
        PoseidonCpuWordCols {
            active: rv64.active,
            instr_word: rv64.instr_word,
            rs1_word: rv64.rs1_val_lo32,
            rs2_word: rv64.rs2_val_lo32,
            rd_word: rv64.rd_val_lo32,
            shout_has_lookup: rv64.shout_has_lookup,
        }
    } else {
        PoseidonCpuWordCols {
            active: rv32.active,
            instr_word: rv32.instr_word,
            rs1_word: rv32.rs1_val,
            rs2_word: rv32.rs2_val,
            rd_word: rv32.rd_val,
            shout_has_lookup: rv32.shout_has_lookup,
        }
    }
}

#[inline]
fn poseidon_cpu_word_cols(step: &StepWitnessBundle<Cmt, F, K>) -> PoseidonCpuWordCols {
    poseidon_cpu_word_cols_for_cpu_len(step.time_columns.cpu_cols.len())
}

pub(crate) fn build_poseidon_sidecar_table_from_step_witness(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    carry: &mut PoseidonSidecarCarryState,
) -> Result<Rv32PoseidonSidecarTable, PiCcsError> {
    const RATE: usize = neo_ccs::crypto::poseidon2_goldilocks::RATE;
    const DIGEST_LEN: usize = neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;

    let trace = Rv32TraceLayout::new();
    let cpu_cols = poseidon_cpu_word_cols(step);
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(
            "poseidon sidecar build: t_len must be >= 1".into(),
        ));
    }

    let col_ids = [
        cpu_cols.active,
        cpu_cols.instr_word,
        cpu_cols.rs1_word,
        cpu_cols.rs2_word,
        cpu_cols.rd_word,
    ];
    let decoded = decode_trace_col_values_batch(params, step, t_len, &col_ids)?;
    let active_vals = decoded
        .get(&cpu_cols.active)
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon sidecar build: missing active column".into()))?;
    let instr_vals = decoded
        .get(&cpu_cols.instr_word)
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon sidecar build: missing instr_word column".into()))?;
    let rs1_vals = decoded
        .get(&cpu_cols.rs1_word)
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon sidecar build: missing rs1_val column".into()))?;
    let rs2_vals = decoded
        .get(&cpu_cols.rs2_word)
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon sidecar build: missing rs2_val column".into()))?;
    let rd_vals = decoded
        .get(&cpu_cols.rd_word)
        .ok_or_else(|| PiCcsError::ProtocolError("poseidon sidecar build: missing rd_val column".into()))?;

    let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();
    let mut mode = if carry.mode_finalized {
        PoseidonSidecarBuildMode::Finalized
    } else {
        PoseidonSidecarBuildMode::Absorbing
    };
    let mut state = carry.state;
    let mut absorb_cursor: usize = carry.absorb_cursor;
    let mut digest_words = carry.digest_words;
    let mut call_ctr = carry.call_ctr;
    if absorb_cursor > RATE {
        return Err(PiCcsError::ProtocolError(format!(
            "poseidon sidecar build: invalid carry absorb_cursor={} (rate={RATE})",
            absorb_cursor
        )));
    }

    let mut cycle_rows: Vec<Rv32PoseidonCycleEventRow> = Vec::new();
    let mut perm_rows: Vec<Rv32PoseidonPermSlotMetaRow> = Vec::new();

    for j in 0..t_len {
        if active_vals[j] == K::ZERO {
            continue;
        }
        let cycle = j as u64;
        let state_pre = poseidon_state_to_u64_local(&state);
        let mut out = Rv32PoseidonCycleEventRow {
            cycle,
            op_absorb: false,
            op_finalize: false,
            op_squeeze: false,
            mode_finalized: mode == PoseidonSidecarBuildMode::Finalized,
            call_ctr,
            cursor_before: absorb_cursor as u8,
            cursor_after: absorb_cursor as u8,
            do_perm_slot0: false,
            do_perm_slot1: false,
            absorb_lo32: 0,
            absorb_hi32: 0,
            squeeze_idx: 0,
            squeeze_word_u32: 0,
            state_pre,
            state_post: state_pre,
            canonical_lo_sum: 0,
            canonical_hi_sum: 0,
            canonical_c0: 0,
            canonical_c1: 0,
        };

        let instr_word = decode_k_to_u32(instr_vals[j], "poseidon sidecar build/instr_word")?;
        match decode_poseidon_precompile_word(instr_word) {
            Some(RiscvInstruction::Poseidon2AbsorbElem { .. }) => {
                out.op_absorb = true;
                if mode == PoseidonSidecarBuildMode::Finalized {
                    state.fill(Goldilocks::ZERO);
                    absorb_cursor = 0;
                    mode = PoseidonSidecarBuildMode::Absorbing;
                    digest_words.fill(0);
                    call_ctr = call_ctr.wrapping_add(1);
                    out.call_ctr = call_ctr;
                    out.cursor_before = 0;
                    out.state_pre = poseidon_state_to_u64_local(&state);
                }
                let rs1 = decode_k_to_u32(rs1_vals[j], "poseidon sidecar build/rs1_val")?;
                let rs2 = decode_k_to_u32(rs2_vals[j], "poseidon sidecar build/rs2_val")?;
                out.absorb_lo32 = rs1;
                out.absorb_hi32 = rs2;

                let elem_u64 = (rs1 as u64) | ((rs2 as u64) << 32);
                state[absorb_cursor] += Goldilocks::from_u64(elem_u64);
                absorb_cursor += 1;
                if absorb_cursor == RATE {
                    out.do_perm_slot0 = true;
                    let in_state = poseidon_state_to_u64_local(&state);
                    state = perm.permute(state);
                    let out_state = poseidon_state_to_u64_local(&state);
                    perm_rows.push(Rv32PoseidonPermSlotMetaRow {
                        cycle,
                        slot: 0,
                        call_ctr,
                        state_in: in_state,
                        state_out: out_state,
                    });
                    absorb_cursor = 0;
                }
                out.cursor_after = absorb_cursor as u8;
            }
            Some(RiscvInstruction::Poseidon2Finalize) => {
                out.op_finalize = true;
                if mode == PoseidonSidecarBuildMode::Finalized {
                    return Err(PiCcsError::ProtocolError(format!(
                        "poseidon sidecar build: finalize called in Finalized mode at cycle {}",
                        cycle
                    )));
                }
                if absorb_cursor > 0 {
                    out.do_perm_slot0 = true;
                    let in_state = poseidon_state_to_u64_local(&state);
                    state = perm.permute(state);
                    let out_state = poseidon_state_to_u64_local(&state);
                    perm_rows.push(Rv32PoseidonPermSlotMetaRow {
                        cycle,
                        slot: 0,
                        call_ctr,
                        state_in: in_state,
                        state_out: out_state,
                    });
                    absorb_cursor = 0;
                }
                state[0] += Goldilocks::ONE;
                out.do_perm_slot1 = true;
                let in_state = poseidon_state_to_u64_local(&state);
                state = perm.permute(state);
                let out_state = poseidon_state_to_u64_local(&state);
                perm_rows.push(Rv32PoseidonPermSlotMetaRow {
                    cycle,
                    slot: 1,
                    call_ctr,
                    state_in: in_state,
                    state_out: out_state,
                });
                for i in 0..DIGEST_LEN {
                    let v = state[i].as_canonical_u64();
                    digest_words[2 * i] = v as u32;
                    digest_words[2 * i + 1] = (v >> 32) as u32;
                }
                mode = PoseidonSidecarBuildMode::Finalized;
                out.cursor_after = 0;
            }
            Some(RiscvInstruction::Poseidon2SqueezeWord { rd, idx }) => {
                out.op_squeeze = true;
                if mode != PoseidonSidecarBuildMode::Finalized {
                    return Err(PiCcsError::ProtocolError(format!(
                        "poseidon sidecar build: squeeze called before finalize at cycle {}",
                        cycle
                    )));
                }
                let idx_usize = idx as usize;
                if idx_usize >= digest_words.len() {
                    return Err(PiCcsError::ProtocolError(format!(
                        "poseidon sidecar build: squeeze idx out of range at cycle {}: idx={idx}",
                        cycle
                    )));
                }
                let word = digest_words[idx_usize];
                out.squeeze_idx = idx;
                out.squeeze_word_u32 = word;
                if rd != 0 {
                    let rd_word = decode_k_to_u32(rd_vals[j], "poseidon sidecar build/rd_val")?;
                    if rd_word != word {
                        return Err(PiCcsError::ProtocolError(format!(
                            "poseidon sidecar build: squeeze word mismatch at cycle {}: got={rd_word:#x}, expected={word:#x}",
                            cycle
                        )));
                    }
                }
                let digest_elem_idx = idx_usize / 2;
                let digest_elem = state[digest_elem_idx].as_canonical_u64();
                let (lo_sum, hi_sum, c0, c1) = canonical_u64_lt_goldilocks_aux_local(digest_elem);
                out.canonical_lo_sum = lo_sum;
                out.canonical_hi_sum = hi_sum;
                out.canonical_c0 = c0;
                out.canonical_c1 = c1;
                out.cursor_after = absorb_cursor as u8;
            }
            _ => {
                out.cursor_after = absorb_cursor as u8;
            }
        }

        out.state_post = poseidon_state_to_u64_local(&state);
        cycle_rows.push(out);
    }

    carry.mode_finalized = mode == PoseidonSidecarBuildMode::Finalized;
    carry.state = state;
    carry.absorb_cursor = absorb_cursor;
    carry.digest_words = digest_words;
    carry.call_ctr = call_ctr;

    Ok(Rv32PoseidonSidecarTable { cycle_rows, perm_rows })
}

fn build_poseidon_cycle_col_values(
    t_len: usize,
    sidecar: &Rv32PoseidonSidecarTable,
) -> Result<BTreeMap<usize, Vec<K>>, PiCcsError> {
    let layout = PoseidonCycleTraceLayout::new();
    let mut by_col: BTreeMap<usize, Vec<K>> = BTreeMap::new();
    for col_id in 0..layout.cols() {
        by_col.insert(col_id, vec![K::ZERO; t_len]);
    }

    let mut perm_by_cycle_slot: BTreeMap<(u64, u8), &Rv32PoseidonPermSlotMetaRow> = BTreeMap::new();
    for perm in sidecar.perm_rows.iter() {
        let key = (perm.cycle, perm.slot);
        if perm_by_cycle_slot.insert(key, perm).is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle sidecar duplicate permutation row: cycle={} slot={}",
                perm.cycle, perm.slot
            )));
        }
    }

    for (idx, row) in sidecar.cycle_rows.iter().enumerate() {
        let j = row.cycle as usize;
        if j >= t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle sidecar row out of range: cycle={} t_len={t_len}",
                row.cycle
            )));
        }
        let set = |map: &mut BTreeMap<usize, Vec<K>>, col: usize, v: K| {
            if let Some(col_vec) = map.get_mut(&col) {
                col_vec[j] = v;
            }
        };
        set(&mut by_col, layout.row_active, K::ONE);
        let is_first = idx == 0 || poseidon_cycle_continuity_break_before(row);
        let is_last =
            idx + 1 == sidecar.cycle_rows.len()
                || poseidon_cycle_continuity_break_before(sidecar.cycle_rows.get(idx + 1).ok_or_else(|| {
                    PiCcsError::ProtocolError("poseidon cycle sidecar next-row lookup failed".into())
                })?);
        set(&mut by_col, layout.is_first, if is_first { K::ONE } else { K::ZERO });
        set(&mut by_col, layout.is_last, if is_last { K::ONE } else { K::ZERO });
        set(&mut by_col, layout.cycle, K::from(F::from_u64(row.cycle)));
        set(
            &mut by_col,
            layout.op_absorb,
            if row.op_absorb { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.op_finalize,
            if row.op_finalize { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.op_squeeze,
            if row.op_squeeze { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.mode_finalized,
            if row.mode_finalized { K::ONE } else { K::ZERO },
        );
        set(&mut by_col, layout.call_ctr, K::from(F::from_u64(row.call_ctr)));
        set(
            &mut by_col,
            layout.cursor_before,
            K::from(F::from_u64(row.cursor_before as u64)),
        );
        set(
            &mut by_col,
            layout.cursor_after,
            K::from(F::from_u64(row.cursor_after as u64)),
        );
        for i in 0..8usize {
            set(&mut by_col, layout.state_pre(i), K::from(F::from_u64(row.state_pre[i])));
            set(
                &mut by_col,
                layout.state_post(i),
                K::from(F::from_u64(row.state_post[i])),
            );
        }
        set(
            &mut by_col,
            layout.do_perm_slot0,
            if row.do_perm_slot0 { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.do_perm_slot1,
            if row.do_perm_slot1 { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.absorb_lo32,
            K::from(F::from_u64(row.absorb_lo32 as u64)),
        );
        set(
            &mut by_col,
            layout.absorb_hi32,
            K::from(F::from_u64(row.absorb_hi32 as u64)),
        );
        set(
            &mut by_col,
            layout.squeeze_idx_b0,
            if (row.squeeze_idx & 1) != 0 { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.squeeze_idx_b1,
            if (row.squeeze_idx & 2) != 0 { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.squeeze_idx_b2,
            if (row.squeeze_idx & 4) != 0 { K::ONE } else { K::ZERO },
        );
        set(
            &mut by_col,
            layout.squeeze_word_u32,
            K::from(F::from_u64(row.squeeze_word_u32 as u64)),
        );
        for i in 0..4usize {
            let digest_word = row.state_post[i];
            set(
                &mut by_col,
                layout.digest_lo(i),
                K::from(F::from_u64((digest_word as u32) as u64)),
            );
            set(
                &mut by_col,
                layout.digest_hi(i),
                K::from(F::from_u64((digest_word >> 32) as u64)),
            );
        }
        set(
            &mut by_col,
            layout.canonical_c0,
            K::from(F::from_u64(row.canonical_c0 as u64)),
        );
        set(
            &mut by_col,
            layout.canonical_c1,
            K::from(F::from_u64(row.canonical_c1 as u64)),
        );
        set(
            &mut by_col,
            layout.canonical_lo_sum,
            K::from(F::from_u64(row.canonical_lo_sum as u64)),
        );
        set(
            &mut by_col,
            layout.canonical_hi_sum,
            K::from(F::from_u64(row.canonical_hi_sum as u64)),
        );

        let slot0 = perm_by_cycle_slot.get(&(row.cycle, 0)).copied();
        let slot1 = perm_by_cycle_slot.get(&(row.cycle, 1)).copied();
        if row.do_perm_slot0 != slot0.is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle sidecar slot0 mismatch at cycle {} (flag={}, perm_row_present={})",
                row.cycle,
                row.do_perm_slot0,
                slot0.is_some()
            )));
        }
        if row.do_perm_slot1 != slot1.is_some() {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle sidecar slot1 mismatch at cycle {} (flag={}, perm_row_present={})",
                row.cycle,
                row.do_perm_slot1,
                slot1.is_some()
            )));
        }
        if let Some(perm) = slot0 {
            for i in 0..8usize {
                set(&mut by_col, layout.slot0_in(i), K::from(F::from_u64(perm.state_in[i])));
                set(
                    &mut by_col,
                    layout.slot0_out(i),
                    K::from(F::from_u64(perm.state_out[i])),
                );
            }
        }
        if let Some(perm) = slot1 {
            for i in 0..8usize {
                set(&mut by_col, layout.slot1_in(i), K::from(F::from_u64(perm.state_in[i])));
                set(
                    &mut by_col,
                    layout.slot1_out(i),
                    K::from(F::from_u64(perm.state_out[i])),
                );
            }
        }
    }

    Ok(by_col)
}

pub(crate) fn build_poseidon_cycle_trace_matrix(
    step: &StepWitnessBundle<Cmt, F, K>,
    sidecar: &Rv32PoseidonSidecarTable,
) -> Result<(Mat<F>, usize, usize, Vec<usize>), PiCcsError> {
    let trace = Rv32TraceLayout::new();
    let layout = PoseidonCycleTraceLayout::new();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(
            "poseidon cycle trace matrix: t_len must be >= 1".into(),
        ));
    }
    let m_in = step.mcs.0.m_in;
    let cols = layout.cols();
    let m = cols
        .checked_mul(t_len)
        .ok_or_else(|| PiCcsError::InvalidInput("poseidon cycle trace matrix: cols * t_len overflow".into()))?;
    let by_col = build_poseidon_cycle_col_values(t_len, sidecar)?;

    let mut data = vec![F::ZERO; neo_math::D * m];
    for col_id in 0..cols {
        let vals = by_col.get(&col_id).ok_or_else(|| {
            PiCcsError::ProtocolError(format!(
                "poseidon cycle trace matrix: missing populated column col_id={col_id}"
            ))
        })?;
        if vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon cycle trace matrix: column length mismatch for col_id={} (got {}, expected {})",
                col_id,
                vals.len(),
                t_len
            )));
        }
        for (j, v) in vals.iter().enumerate() {
            let coeffs = v.as_coeffs();
            if coeffs.iter().skip(1).any(|&c| c != F::ZERO) {
                return Err(PiCcsError::ProtocolError(format!(
                    "poseidon cycle trace matrix: non-base value in col={} row={j}",
                    col_id
                )));
            }
            data[col_id * t_len + j] = coeffs[0];
        }
    }

    let z = Mat::from_row_major(neo_math::D, m, data);
    let open_cols = poseidon_cycle_open_col_ids(&layout);
    Ok((z, m_in, t_len, open_cols))
}

pub(crate) fn build_route_a_poseidon_cycle_claims(
    params: &NeoParams,
    step: &StepWitnessBundle<Cmt, F, K>,
    r_cycle: &[K],
    enabled: bool,
    sidecar: Option<&Rv32PoseidonSidecarTable>,
) -> Result<PoseidonCycleClaims, PiCcsError> {
    if !enabled {
        return Ok((None, None, None, None, None));
    }
    let sidecar =
        sidecar.ok_or_else(|| PiCcsError::ProtocolError("poseidon cycle claims require sidecar table".into()))?;

    let trace = Rv32TraceLayout::new();
    let cpu_cols = poseidon_cpu_word_cols(step);
    let decode = Rv32DecodeSidecarLayout::new();
    let layout = PoseidonCycleTraceLayout::new();
    let m_in = step.mcs.0.m_in;
    let ell_n = r_cycle.len();
    let t_len = infer_rv32_trace_t_len_for_wb_wp(step, &trace)?;
    if t_len == 0 {
        return Err(PiCcsError::InvalidInput(
            "poseidon cycle stage: t_len must be >= 1".into(),
        ));
    }

    let main_col_ids = vec![
        cpu_cols.active,
        cpu_cols.instr_word,
        cpu_cols.rs1_word,
        cpu_cols.rs2_word,
        cpu_cols.rd_word,
        cpu_cols.shout_has_lookup,
    ];
    let main_decoded = decode_trace_col_values_batch(params, step, t_len, &main_col_ids)?;

    let decode_col_ids = vec![
        decode.op_custom,
        decode.rd_has_write,
        decode.rd_is_zero,
        decode.ram_has_read,
        decode.ram_has_write,
        decode.funct3_bit[0],
        decode.funct3_bit[1],
        decode.funct3_bit[2],
        decode.funct7_bit[0],
        decode.funct7_bit[1],
        decode.funct7_bit[2],
        decode.funct7_bit[3],
        decode.funct7_bit[4],
        decode.funct7_bit[5],
        decode.funct7_bit[6],
    ];
    let decode_decoded = {
        let instr_vals = main_decoded
            .get(&cpu_cols.instr_word)
            .ok_or_else(|| PiCcsError::ProtocolError("poseidon(shared): missing instr_word decode column".into()))?;
        let active_vals = main_decoded
            .get(&cpu_cols.active)
            .ok_or_else(|| PiCcsError::ProtocolError("poseidon(shared): missing active decode column".into()))?;
        if instr_vals.len() != t_len || active_vals.len() != t_len {
            return Err(PiCcsError::ProtocolError(format!(
                "poseidon(shared): decoded CPU column lengths drift (instr={}, active={}, t_len={t_len})",
                instr_vals.len(),
                active_vals.len()
            )));
        }
        let mut decoded = BTreeMap::<usize, Vec<K>>::new();
        for &col_id in decode_col_ids.iter() {
            decoded.insert(col_id, Vec::with_capacity(t_len));
        }
        for j in 0..t_len {
            let instr_word = decode_k_to_u32(instr_vals[j], "poseidon(shared)/instr_word")?;
            let active = active_vals[j] != K::ZERO;
            let mut row = rv32_decode_lookup_backed_row_from_instr_word(&decode, instr_word, active);
            if !active {
                row.fill(F::ZERO);
            }
            for &col_id in decode_col_ids.iter() {
                decoded
                    .get_mut(&col_id)
                    .ok_or_else(|| PiCcsError::ProtocolError("poseidon(shared): decode map build failed".into()))?
                    .push(K::from(row[col_id]));
            }
        }
        decoded
    };

    let side_vals = build_poseidon_cycle_col_values(t_len, sidecar)?;

    let mut main_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in main_col_ids.iter() {
        let vals = main_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing main decoded column {col_id}")))?;
        main_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut decode_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for &col_id in decode_col_ids.iter() {
        let vals = decode_decoded
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing decode decoded column {col_id}")))?;
        decode_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }
    let mut side_sparse = BTreeMap::<usize, SparseIdxVec<K>>::new();
    for col_id in 0..layout.cols() {
        let vals = side_vals
            .get(&col_id)
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing sidecar decoded column {col_id}")))?;
        side_sparse.insert(col_id, sparse_trace_col_from_values(m_in, ell_n, vals)?);
    }

    let main_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        main_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing main sparse column {col_id}")))
    };
    let decode_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        decode_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing decode sparse column {col_id}")))
    };
    let side_col = |col_id: usize| -> Result<SparseIdxVec<K>, PiCcsError> {
        side_sparse
            .get(&col_id)
            .cloned()
            .ok_or_else(|| PiCcsError::ProtocolError(format!("poseidon missing side sparse column {col_id}")))
    };

    let io_weights = poseidon_io_link_weight_vector(r_cycle, POSEIDON_IO_LINK_RESIDUAL_COUNT);
    let io_oracle = FormulaOracleSparseTime::new(
        vec![
            decode_col(decode.op_custom)?,
            decode_col(decode.rd_has_write)?,
            decode_col(decode.rd_is_zero)?,
            decode_col(decode.ram_has_read)?,
            decode_col(decode.ram_has_write)?,
            main_col(cpu_cols.shout_has_lookup)?,
            decode_col(decode.funct3_bit[0])?,
            decode_col(decode.funct3_bit[1])?,
            decode_col(decode.funct3_bit[2])?,
            decode_col(decode.funct7_bit[0])?,
            decode_col(decode.funct7_bit[1])?,
            decode_col(decode.funct7_bit[2])?,
            decode_col(decode.funct7_bit[3])?,
            decode_col(decode.funct7_bit[4])?,
            decode_col(decode.funct7_bit[5])?,
            decode_col(decode.funct7_bit[6])?,
        ],
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = poseidon_io_link_residuals(
                vals[0],
                vals[1],
                vals[2],
                vals[3],
                vals[4],
                vals[5],
                [vals[6], vals[7], vals[8]],
                [vals[9], vals[10], vals[11], vals[12], vals[13], vals[14], vals[15]],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(io_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let bitness_weights = poseidon_bitness_weight_vector(r_cycle, POSEIDON_BITNESS_RESIDUAL_COUNT);
    let bitness_oracle = FormulaOracleSparseTime::new(
        vec![
            decode_col(decode.op_custom)?,
            decode_col(decode.rd_has_write)?,
            decode_col(decode.rd_is_zero)?,
            decode_col(decode.funct3_bit[0])?,
            decode_col(decode.funct3_bit[1])?,
            decode_col(decode.funct3_bit[2])?,
            decode_col(decode.funct7_bit[0])?,
            decode_col(decode.funct7_bit[1])?,
            decode_col(decode.funct7_bit[2])?,
            decode_col(decode.funct7_bit[3])?,
            decode_col(decode.funct7_bit[4])?,
            decode_col(decode.funct7_bit[5])?,
            decode_col(decode.funct7_bit[6])?,
        ],
        3,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = poseidon_bitness_residuals(
                vals[0],
                vals[1],
                vals[2],
                [vals[3], vals[4], vals[5]],
                [vals[6], vals[7], vals[8], vals[9], vals[10], vals[11], vals[12]],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(bitness_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let canonical_weights = poseidon_canonical_weight_vector(r_cycle, POSEIDON_CANONICAL_RESIDUAL_COUNT);
    let canonical_oracle = FormulaOracleSparseTime::new(
        vec![
            decode_col(decode.funct3_bit[0])?,
            decode_col(decode.funct3_bit[1])?,
            decode_col(decode.funct3_bit[2])?,
            side_col(layout.op_squeeze)?,
            side_col(layout.squeeze_word_u32)?,
            side_col(layout.digest_lo(0))?,
            side_col(layout.digest_lo(1))?,
            side_col(layout.digest_lo(2))?,
            side_col(layout.digest_lo(3))?,
            side_col(layout.digest_hi(0))?,
            side_col(layout.digest_hi(1))?,
            side_col(layout.digest_hi(2))?,
            side_col(layout.digest_hi(3))?,
            side_col(layout.canonical_c0)?,
            side_col(layout.canonical_c1)?,
            side_col(layout.canonical_lo_sum)?,
            side_col(layout.canonical_hi_sum)?,
        ],
        6,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = poseidon_canonical_residuals(
                [vals[0], vals[1], vals[2]],
                vals[3],
                vals[4],
                [vals[5], vals[6], vals[7], vals[8]],
                [vals[9], vals[10], vals[11], vals[12]],
                vals[13],
                vals[14],
                vals[15],
                vals[16],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(canonical_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let sidecar_link_weights = poseidon_sidecar_link_weight_vector(r_cycle, POSEIDON_SIDECAR_LINK_RESIDUAL_COUNT);
    let sidecar_link_oracle = FormulaOracleSparseTime::new(
        vec![
            decode_col(decode.op_custom)?,
            decode_col(decode.rd_has_write)?,
            decode_col(decode.funct3_bit[0])?,
            decode_col(decode.funct3_bit[1])?,
            decode_col(decode.funct3_bit[2])?,
            decode_col(decode.funct7_bit[0])?,
            decode_col(decode.funct7_bit[1])?,
            decode_col(decode.funct7_bit[2])?,
            decode_col(decode.funct7_bit[3])?,
            decode_col(decode.funct7_bit[4])?,
            decode_col(decode.funct7_bit[5])?,
            decode_col(decode.funct7_bit[6])?,
            main_col(cpu_cols.rs1_word)?,
            main_col(cpu_cols.rs2_word)?,
            main_col(cpu_cols.rd_word)?,
            side_col(layout.op_absorb)?,
            side_col(layout.op_finalize)?,
            side_col(layout.op_squeeze)?,
            side_col(layout.squeeze_idx_b0)?,
            side_col(layout.squeeze_idx_b1)?,
            side_col(layout.squeeze_idx_b2)?,
            side_col(layout.absorb_lo32)?,
            side_col(layout.absorb_hi32)?,
            side_col(layout.squeeze_word_u32)?,
        ],
        4,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = poseidon_sidecar_link_residuals(
                vals[0],
                vals[1],
                [vals[2], vals[3], vals[4]],
                [vals[5], vals[6], vals[7], vals[8], vals[9], vals[10], vals[11]],
                vals[12],
                vals[13],
                vals[14],
                vals[15],
                vals[16],
                vals[17],
                [vals[18], vals[19], vals[20]],
                vals[21],
                vals[22],
                vals[23],
            );
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(sidecar_link_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    let mode_weights = poseidon_mode_weight_vector(r_cycle, POSEIDON_MODE_RESIDUAL_COUNT);
    let mode_oracle = FormulaOracleSparseTime::new(
        vec![
            side_col(layout.op_finalize)?,
            side_col(layout.op_squeeze)?,
            side_col(layout.mode_finalized)?,
        ],
        3,
        r_cycle,
        Box::new(move |vals: &[K]| {
            let residuals = poseidon_mode_residuals(vals[0], vals[1], vals[2]);
            let mut weighted = K::ZERO;
            for (r, w) in residuals.iter().zip(mode_weights.iter()) {
                weighted += *w * *r;
            }
            weighted
        }),
    );

    Ok((
        Some((Box::new(io_oracle), K::ZERO)),
        Some((Box::new(bitness_oracle), K::ZERO)),
        Some((Box::new(canonical_oracle), K::ZERO)),
        Some((Box::new(sidecar_link_oracle), K::ZERO)),
        Some((Box::new(mode_oracle), K::ZERO)),
    ))
}
