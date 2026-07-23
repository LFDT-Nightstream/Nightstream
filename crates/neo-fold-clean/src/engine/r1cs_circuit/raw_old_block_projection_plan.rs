//! Compact index plan for the terminal raw-witness old-block projection.
//!
//! The plan describes the emitted row/column schedule without constructing
//! the production profile's millions of sparse rows.  The circuit gadget and
//! the Lean artifact exporter share these exact formulas.

use neo_math::D;
use neo_reductions::block_projection::{BLOCK_PROJECTION_DOMAIN_SIZE, BLOCK_PROJECTION_POINT_LEN};

/// Ordered children in the active F' Π_DEC split.
pub const RAW_OLD_BLOCK_CHILD_COUNT: usize = 14;

/// R1CS rows (and allocated columns) emitted by one quadratic-extension
/// Karatsuba multiplication.
pub const RAW_OLD_BLOCK_K_MUL_ROWS: usize = 5;

/// Two base-field limbs represent one quadratic-extension value.
pub const RAW_OLD_BLOCK_K_LIMBS: usize = 2;

/// Schema tag for the direct dataflow join between one
/// `PendingProjectionWires` value, the raw projection, and terminal CE.
pub const RAW_OLD_BLOCK_PENDING_JOIN_ID: usize = 1;

/// A proof-free description of the raw-old-block projection program.
#[doc(hidden)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct RawOldBlockProjectionPlan {
    logical_columns: usize,
    packed_columns: usize,
    child_count: usize,
}

impl RawOldBlockProjectionPlan {
    /// Construct the exact fixed-19-variable program geometry.
    pub fn new(logical_columns: usize, child_count: usize) -> Result<Self, &'static str> {
        if logical_columns == 0 {
            return Err("raw old-block projection needs at least one logical column");
        }
        if child_count == 0 {
            return Err("raw old-block projection needs at least one child");
        }
        let packed_columns = logical_columns.div_ceil(D);
        if packed_columns > BLOCK_PROJECTION_DOMAIN_SIZE {
            return Err("raw old-block projection exceeds the fixed block domain");
        }
        Ok(Self {
            logical_columns,
            packed_columns,
            child_count,
        })
    }

    pub fn logical_columns(self) -> usize {
        self.logical_columns
    }

    pub fn packed_rows(self) -> usize {
        D
    }

    pub fn packed_columns(self) -> usize {
        self.packed_columns
    }

    pub fn block_variables(self) -> usize {
        BLOCK_PROJECTION_POINT_LEN
    }

    /// Tensor coordinates materialized before the optional common final
    /// factor is applied once per active lane.
    pub fn tensor_variables(self) -> usize {
        self.block_variables() - usize::from(self.factor_final_round())
    }

    /// Whether every live block has zero final bit and factoring that common
    /// `(1 - old_block[last])` scalar reduces the emitted row count.
    pub fn factor_final_round(self) -> bool {
        let final_low_domain = 1usize << (self.block_variables() - 1);
        self.packed_columns <= final_low_domain && self.packed_columns > self.active_lanes()
    }

    /// The coordinate factored after the per-block products, when present.
    pub fn factored_variable(self) -> Option<usize> {
        self.factor_final_round().then_some(self.tensor_variables())
    }

    pub fn block_domain_size(self) -> usize {
        BLOCK_PROJECTION_DOMAIN_SIZE
    }

    pub fn child_count(self) -> usize {
        self.child_count
    }

    pub fn active_lanes(self) -> usize {
        D
    }

    pub fn padded_lanes(self) -> usize {
        D.next_power_of_two()
    }

    pub fn virtual_zero_lanes(self) -> usize {
        self.padded_lanes() - self.active_lanes()
    }

    /// Live prefix parents at tensor round `round`.
    pub fn tensor_round_mul_count(self, round: usize) -> Option<usize> {
        (round < self.tensor_variables()).then(|| self.packed_columns.min(1usize << round))
    }

    /// Number of high children retained at tensor round `round`.
    ///
    /// Parents below this boundary need both children.  The gadget computes
    /// the high child and represents the low child as `parent - high`.
    /// Remaining parents need only their low child.
    pub fn tensor_round_high_count(self, round: usize) -> Option<usize> {
        if round >= self.tensor_variables() {
            return None;
        }
        let half = 1usize << round;
        Some(self.packed_columns.saturating_sub(half).min(half))
    }

    pub fn tensor_mul_count(self) -> usize {
        (0..self.tensor_variables())
            .map(|round| {
                self.tensor_round_mul_count(round)
                    .expect("round is in range")
            })
            .sum()
    }

    pub fn tensor_rows(self) -> usize {
        RAW_OLD_BLOCK_K_MUL_ROWS * self.tensor_mul_count()
    }

    pub fn projection_product_rows(self) -> usize {
        RAW_OLD_BLOCK_K_LIMBS * self.active_lanes() * self.packed_columns
    }

    pub fn final_scale_mul_count(self) -> usize {
        usize::from(self.factor_final_round()) * self.active_lanes()
    }

    pub fn final_scale_rows(self) -> usize {
        RAW_OLD_BLOCK_K_MUL_ROWS * self.final_scale_mul_count()
    }

    pub fn terminal_rows(self) -> usize {
        RAW_OLD_BLOCK_K_LIMBS * self.active_lanes()
    }

    pub fn total_rows(self) -> usize {
        self.tensor_rows() + self.projection_product_rows() + self.final_scale_rows() + self.terminal_rows()
    }

    /// Row-major offset of `witness[lane, block]` in one allocated child.
    pub fn witness_flat_index(self, lane: usize, block: usize) -> Option<usize> {
        (lane < self.active_lanes() && block < self.packed_columns).then_some(lane * self.packed_columns + block)
    }

    /// Offset of a child witness in a contiguous ordered witness family.
    pub fn child_witness_flat_index(self, child: usize, lane: usize, block: usize) -> Option<usize> {
        let within_child = self.witness_flat_index(lane, block)?;
        (child < self.child_count).then_some(child * self.active_lanes() * self.packed_columns + within_child)
    }

    /// Tensor multiplication's zero-based ordinal in round-major order.
    pub fn tensor_mul_ordinal(self, round: usize, parent: usize) -> Option<usize> {
        let round_count = self.tensor_round_mul_count(round)?;
        if parent >= round_count {
            return None;
        }
        Some(
            (0..round)
                .map(|prior| {
                    self.tensor_round_mul_count(prior)
                        .expect("prior round is in range")
                })
                .sum::<usize>()
                + parent,
        )
    }

    /// Relative row of one of the five rows owned by a tensor K-multiply.
    pub fn tensor_row_offset(self, round: usize, parent: usize, k_row: usize) -> Option<usize> {
        if k_row >= RAW_OLD_BLOCK_K_MUL_ROWS {
            return None;
        }
        Some(RAW_OLD_BLOCK_K_MUL_ROWS * self.tensor_mul_ordinal(round, parent)? + k_row)
    }

    /// First absolute column allocated by one tensor K multiplication.
    pub fn tensor_mul_first_column(self, tensor_first_column: usize, round: usize, parent: usize) -> Option<usize> {
        Some(tensor_first_column + RAW_OLD_BLOCK_K_MUL_ROWS * self.tensor_mul_ordinal(round, parent)?)
    }

    /// Absolute output columns of one tensor K multiplication.  Its three
    /// Karatsuba intermediates occupy the preceding columns.
    pub fn tensor_mul_output_columns(
        self,
        tensor_first_column: usize,
        round: usize,
        parent: usize,
    ) -> Option<[usize; RAW_OLD_BLOCK_K_LIMBS]> {
        let first = self.tensor_mul_first_column(tensor_first_column, round, parent)?;
        Some([first + 3, first + 4])
    }

    /// Relative row of a base×K product, after the tensor rows.
    pub fn projection_product_row_offset(self, lane: usize, block: usize, limb: usize) -> Option<usize> {
        if lane >= self.active_lanes() || block >= self.packed_columns || limb >= RAW_OLD_BLOCK_K_LIMBS {
            return None;
        }
        Some(self.tensor_rows() + RAW_OLD_BLOCK_K_LIMBS * (lane * self.packed_columns + block) + limb)
    }

    /// Absolute output column of one base-field projection product.
    pub fn projection_product_column(
        self,
        projection_product_first_column: usize,
        lane: usize,
        block: usize,
        limb: usize,
    ) -> Option<usize> {
        let within = self.witness_flat_index(lane, block)?;
        (limb < RAW_OLD_BLOCK_K_LIMBS)
            .then_some(projection_product_first_column + RAW_OLD_BLOCK_K_LIMBS * within + limb)
    }

    /// Relative row of one of the five rows owned by the common final-round
    /// K multiplication for `lane`.
    pub fn final_scale_row_offset(self, lane: usize, k_row: usize) -> Option<usize> {
        if !self.factor_final_round() || lane >= self.active_lanes() || k_row >= RAW_OLD_BLOCK_K_MUL_ROWS {
            return None;
        }
        Some(self.tensor_rows() + self.projection_product_rows() + RAW_OLD_BLOCK_K_MUL_ROWS * lane + k_row)
    }

    /// First column allocated by one lane's common final-round K multiply.
    pub fn final_scale_mul_first_column(self, final_scale_first_column: usize, lane: usize) -> Option<usize> {
        (self.factor_final_round() && lane < self.active_lanes())
            .then_some(final_scale_first_column + RAW_OLD_BLOCK_K_MUL_ROWS * lane)
    }

    /// Output columns of one lane's common final-round K multiply.
    pub fn final_scale_output_columns(
        self,
        final_scale_first_column: usize,
        lane: usize,
    ) -> Option<[usize; RAW_OLD_BLOCK_K_LIMBS]> {
        let first = self.final_scale_mul_first_column(final_scale_first_column, lane)?;
        Some([first + 3, first + 4])
    }

    /// Relative row of a terminal parent equality.
    pub fn terminal_row_offset(self, lane: usize, limb: usize) -> Option<usize> {
        if lane >= self.active_lanes() || limb >= RAW_OLD_BLOCK_K_LIMBS {
            return None;
        }
        Some(
            self.tensor_rows()
                + self.projection_product_rows()
                + self.final_scale_rows()
                + RAW_OLD_BLOCK_K_LIMBS * lane
                + limb,
        )
    }
}
