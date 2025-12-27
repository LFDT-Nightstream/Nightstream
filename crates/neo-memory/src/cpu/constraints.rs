//! CPU constraint builder for Neo.
//!
//! This module provides R1CS/CCS constraints that bind CPU semantics to the shared
//! memory/lookup bus. These constraints are **critical for security** - without them,
//! a malicious prover can create divergent CPU and memory states.
//!
//! # Credits
//!
//! The constraint logic in this module is ported from the Jolt zkVM project:
//! - Repository: https://github.com/a16z/jolt
//! - Original file: `jolt-core/src/zkvm/r1cs/constraints.rs`
//! - License: Apache-2.0 / MIT
//!
//! Jolt's R1CS constraint system defines the binding between CPU instruction semantics
//! and memory/lookup values. We adapt this for Neo's CCS-based proving system and
//! shared CPU bus architecture.
//!
//! # Architecture
//!
//! Neo's shared CPU bus places memory/lookup columns in the tail of the CPU witness:
//!
//! ```text
//! z = [x | w_cpu | ... | bus_tail ]
//!      └─┬─┘     └──────┬────────┘
//!     public       private witness
//!
//! bus_tail layout (per step):
//!   - Shout instances: [addr_bits..., has_lookup, val]
//!   - Twist instances: [ra_bits..., wa_bits..., has_read, has_write, wv, rv, inc]
//! ```
//!
//! The constraints in this module enforce that CPU semantic columns (e.g., the value
//! a load instruction writes to a register) are equal to the corresponding bus columns
//! (e.g., `rv` from Twist).

use core::ops::Range;
use neo_ccs::matrix::Mat;
use neo_ccs::poly::{SparsePoly, Term};
use neo_ccs::relations::CcsStructure;
use p3_field::Field;

use crate::cpu::bus_layout::{build_bus_layout_for_instances, BusLayout, ShoutCols, TwistCols};
use crate::witness::{LutInstance, MemInstance};

/// CPU column layout for binding to the bus.
///
/// This mirrors Jolt's `JoltR1CSInputs` but is specific to Neo's witness structure.
///
/// # Credits
/// Adapted from Jolt's `JoltR1CSInputs` enum in `jolt-core/src/zkvm/r1cs/inputs.rs`.
#[derive(Clone, Debug)]
pub struct CpuColumnLayout {
    /// Column index of the CPU's "is_load" selector (1 when executing a load instruction).
    pub is_load: usize,
    /// Column index of the CPU's "is_store" selector (1 when executing a store instruction).
    pub is_store: usize,
    /// Column index of the CPU's effective address (rs1 + imm for RISC-V).
    pub effective_addr: usize,
    /// Column index of the CPU's destination register write value (for loads).
    pub rd_write_value: usize,
    /// Column index of the CPU's source register value (for stores, typically rs2).
    pub rs2_value: usize,
    /// Column index for the CPU's lookup selector (1 when executing a lookup).
    pub is_lookup: usize,
    /// Column index for the CPU's lookup key (the table index / operand).
    pub lookup_key: usize,
    /// Column index for the CPU's lookup output value.
    pub lookup_output: usize,
}

/// Per-instance CPU→bus binding for a Twist (memory) bus slice.
///
/// This is intentionally generic: the CPU may have multiple memory units/ports, so each Twist
/// instance must bind to its own selector/address/value columns.
#[derive(Clone, Debug)]
pub struct TwistCpuBinding {
    /// CPU selector column for a read op (must equal bus `has_read`).
    pub has_read: usize,
    /// CPU selector column for a write op (must equal bus `has_write`).
    pub has_write: usize,

    /// Packed integer read address column (must equal packed bus `ra_bits` when `has_read=1`).
    pub read_addr: usize,
    /// Packed integer write address column (must equal packed bus `wa_bits` when `has_write=1`).
    pub write_addr: usize,

    /// CPU read value column (must equal bus `rv` when `has_read=1`).
    pub rv: usize,
    /// CPU write value column (must equal bus `wv` when `has_write=1`).
    pub wv: usize,

    /// Optional CPU-side increment column (must equal bus `inc_at_write_addr` when `has_write=1`).
    pub inc: Option<usize>,
}

/// Per-instance CPU→bus binding for a Shout (lookup) bus slice.
#[derive(Clone, Debug)]
pub struct ShoutCpuBinding {
    /// CPU selector column for a lookup op (must equal bus `has_lookup`).
    pub has_lookup: usize,
    /// Packed integer lookup key/address column (must equal packed bus `addr_bits` when `has_lookup=1`).
    pub addr: usize,
    /// CPU lookup output/value column (must equal bus `val` when `has_lookup=1`).
    pub val: usize,
}

/// Constraint label for debugging and logging.
///
/// # Credits
/// Adapted from Jolt's `R1CSConstraintLabel` in `jolt-core/src/zkvm/r1cs/constraints.rs`.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub enum CpuConstraintLabel {
    /// Load: RamReadValue == RdWriteValue (when Load flag is set)
    LoadValueBinding,
    /// Store: Rs2Value == RamWriteValue (when Store flag is set)
    StoreValueBinding,
    /// Load: Packed bus `ra_bits` matches CPU effective address.
    LoadAddressBinding,
    /// Store: Packed bus `wa_bits` matches CPU effective address.
    StoreAddressBinding,
    /// Padding: rv == 0 (when NOT has_read)
    ReadValueZeroPadding,
    /// Padding: wv == 0 (when NOT has_write)
    WriteValueZeroPadding,
    /// Padding: ra_bits == 0 (when NOT has_read)
    ReadAddressBitsZeroPadding,
    /// Padding: wa_bits == 0 (when NOT has_write)
    WriteAddressBitsZeroPadding,
    /// Padding: inc_at_write_addr == 0 (when NOT has_write)
    IncrementZeroPadding,
    /// Write: cpu_inc == bus_inc (when has_write)
    IncrementBinding,
    /// Lookup: val == lookup_output (when has_lookup)
    LookupValueBinding,
    /// Lookup: packed bus `addr_bits` matches CPU lookup key.
    LookupKeyBinding,
    /// Padding: lookup val == 0 (when NOT has_lookup)
    LookupValueZeroPadding,
    /// Padding: lookup addr_bits == 0 (when NOT has_lookup)
    LookupAddressBitsZeroPadding,
    /// Selector binding: is_load == has_read
    LoadSelectorBinding,
    /// Selector binding: is_store == has_write
    StoreSelectorBinding,
    /// Selector binding: is_lookup == has_lookup
    LookupSelectorBinding,
}

/// A single constraint in the CPU binding system.
///
/// Represents: `condition * (left - right) = 0`
///
/// # Credits
/// Adapted from Jolt's `R1CSConstraint` and `r1cs_eq_conditional!` macro.
#[derive(Clone, Debug)]
pub struct CpuConstraint<F> {
    /// Human-readable label for debugging.
    pub label: CpuConstraintLabel,
    /// Condition column index (selector).
    pub condition_col: usize,
    /// Whether to negate the condition (1 - condition).
    pub negate_condition: bool,
    /// Optional: additional columns to add to the condition (for OR-style conditions).
    pub additional_condition_cols: Vec<usize>,
    /// Linear expression for the B side: Σ coeffᵢ · z[colᵢ].
    ///
    /// The constraint enforced is:
    ///   A(z) * B(z) = 0
    /// where A(z) is the (possibly negated / OR'ed) condition expression.
    pub b_terms: Vec<(usize, F)>,
}

impl<F: Field> CpuConstraint<F> {
    /// Create a constraint: `condition * (left - right) = 0`.
    pub fn new_eq(label: CpuConstraintLabel, condition_col: usize, left_col: usize, right_col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint: `(1 - condition) * (left - right) = 0`.
    pub fn new_eq_negated(label: CpuConstraintLabel, condition_col: usize, left_col: usize, right_col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint with OR condition: `(cond1 + cond2 + ...) * (left - right) = 0`.
    pub fn new_with_or_eq(
        label: CpuConstraintLabel,
        condition_cols: &[usize],
        left_col: usize,
        right_col: usize,
    ) -> Self {
        assert!(!condition_cols.is_empty(), "need at least one condition");
        Self {
            label,
            condition_col: condition_cols[0],
            negate_condition: false,
            additional_condition_cols: condition_cols[1..].to_vec(),
            b_terms: vec![(left_col, F::ONE), (right_col, -F::ONE)],
        }
    }

    /// Create a constraint: `condition * (col) = 0`.
    pub fn new_zero(label: CpuConstraintLabel, condition_col: usize, col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: false,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, F::ONE)],
        }
    }

    /// Create a constraint: `(1 - condition) * (col) = 0`.
    pub fn new_zero_negated(label: CpuConstraintLabel, condition_col: usize, col: usize) -> Self {
        Self {
            label,
            condition_col,
            negate_condition: true,
            additional_condition_cols: Vec::new(),
            b_terms: vec![(col, F::ONE)],
        }
    }

    /// Create a constraint with an arbitrary B-side linear expression:
    /// `condition * (Σ coeffᵢ · z[colᵢ]) = 0`.
    pub fn new_terms(
        label: CpuConstraintLabel,
        condition_col: usize,
        negate_condition: bool,
        b_terms: Vec<(usize, F)>,
    ) -> Self {
        Self {
            label,
            condition_col,
            negate_condition,
            additional_condition_cols: Vec::new(),
            b_terms,
        }
    }
}

/// Builder for CPU-to-bus binding constraints.
///
/// Generates CCS constraints that enforce the binding between CPU semantics
/// and the shared memory/lookup bus.
///
/// # Credits
/// The constraint logic is adapted from Jolt's `R1CS_CONSTRAINTS` static array
/// in `jolt-core/src/zkvm/r1cs/constraints.rs`.
///
/// # Example
/// ```ignore
/// let mut builder = CpuConstraintBuilder::new(n, m, bus_base, const_one_col);
/// builder.add_twist_instance(twist_cfg, cpu_layout);
/// builder.add_shout_instance(shout_cfg, cpu_layout);
/// let ccs = builder.build()?;
/// ```
#[derive(Clone, Debug)]
pub struct CpuConstraintBuilder<F: Field> {
    /// Number of constraint rows.
    pub n: usize,
    /// Number of witness columns.
    pub m: usize,
    /// Column index that must be fixed to 1 (typically the R1CS constant-one column).
    pub const_one_col: usize,
    /// Accumulated constraints.
    constraints: Vec<CpuConstraint<F>>,
}

impl<F: Field> CpuConstraintBuilder<F> {
    /// Create a new constraint builder.
    ///
    /// # Arguments
    /// * `n` - Number of constraint rows
    /// * `m` - Number of witness columns (including bus tail)
    /// * `const_one_col` - Column index that is fixed to 1 (public or otherwise constrained)
    pub fn new(n: usize, m: usize, const_one_col: usize) -> Self {
        Self {
            n,
            m,
            const_one_col,
            constraints: Vec::new(),
        }
    }

    /// Add constraints for a Twist (memory) instance.
    ///
    /// # Constraints Added (from Jolt)
    ///
    /// **Value Binding:**
    /// - `is_load * (bus_rv - rd_write_value) = 0` (Load: memory value → register)
    /// - `is_store * (bus_wv - rs2_value) = 0` (Store: register value → memory)
    ///
    /// **Selector Binding:**
    /// - `is_load - has_read = 0` (CPU load flag must match bus has_read)
    /// - `is_store - has_write = 0` (CPU store flag must match bus has_write)
    ///
    /// **Padding Constraints:**
    /// - `(1 - has_read) * rv = 0`
    /// - `(1 - has_write) * wv = 0`
    /// - `(1 - has_read) * ra_bits[i] = 0` for all i
    /// - `(1 - has_write) * wa_bits[i] = 0` for all i
    /// - `(1 - has_write) * inc_at_write_addr = 0`
    ///
    /// # Credits
    /// Constraints adapted from:
    /// - `RamReadEqRdWriteIfLoad` in Jolt
    /// - `Rs2EqRamWriteIfStore` in Jolt
    /// - `RamAddrEqZeroIfNotLoadStore` in Jolt
    pub fn add_twist_instance(&mut self, layout: &BusLayout, twist: &TwistCols, cpu_layout: &CpuColumnLayout) {
        let cpu = TwistCpuBinding {
            has_read: cpu_layout.is_load,
            has_write: cpu_layout.is_store,
            read_addr: cpu_layout.effective_addr,
            write_addr: cpu_layout.effective_addr,
            rv: cpu_layout.rd_write_value,
            wv: cpu_layout.rs2_value,
            inc: None,
        };
        self.add_twist_instance_bound(layout, twist, &cpu);
    }

    /// Add constraints for a Twist (memory) instance using an explicit per-instance CPU binding.
    pub fn add_twist_instance_bound(&mut self, layout: &BusLayout, twist: &TwistCols, cpu: &TwistCpuBinding) {
        assert_eq!(layout.chunk_size, 1, "CpuConstraintBuilder only supports chunk_size==1");

        // Bus column indices (absolute in witness)
        let bus_has_read = layout.bus_cell(twist.has_read, 0);
        let bus_has_write = layout.bus_cell(twist.has_write, 0);
        let bus_rv = layout.bus_cell(twist.rv, 0);
        let bus_wv = layout.bus_cell(twist.wv, 0);
        let bus_inc = layout.bus_cell(twist.inc, 0);

        // Value binding constraints
        // has_read * (rv_cpu - bus_rv) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::LoadValueBinding,
            cpu.has_read,
            cpu.rv,
            bus_rv,
        ));

        // has_write * (wv_cpu - bus_wv) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::StoreValueBinding,
            cpu.has_write,
            cpu.wv,
            bus_wv,
        ));

        // Selector binding: cpu_has_* == bus_has_*
        self.add_equality_constraint(CpuConstraintLabel::LoadSelectorBinding, cpu.has_read, bus_has_read);
        self.add_equality_constraint(CpuConstraintLabel::StoreSelectorBinding, cpu.has_write, bus_has_write);

        // Address binding (bit-pack):
        // - has_read  * (read_addr - pack(ra_bits)) = 0
        // - has_write * (write_addr - pack(wa_bits)) = 0
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::LoadAddressBinding,
            cpu.has_read,
            false,
            pack_addr_bits::<F>(cpu.read_addr, twist.ra_bits.clone(), layout),
        ));
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::StoreAddressBinding,
            cpu.has_write,
            false,
            pack_addr_bits::<F>(cpu.write_addr, twist.wa_bits.clone(), layout),
        ));

        // Optional: bind CPU increment semantics if provided.
        if let Some(cpu_inc) = cpu.inc {
            self.constraints.push(CpuConstraint::new_eq(
                CpuConstraintLabel::IncrementBinding,
                cpu.has_write,
                cpu_inc,
                bus_inc,
            ));
        }

        // Padding: (1 - has_read) * rv = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::ReadValueZeroPadding,
            bus_has_read,
            bus_rv,
        ));

        // Padding: (1 - has_write) * wv = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::WriteValueZeroPadding,
            bus_has_write,
            bus_wv,
        ));

        // Padding: (1 - has_write) * inc_at_write_addr = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::IncrementZeroPadding,
            bus_has_write,
            bus_inc,
        ));

        // Padding: (1 - has_read) * ra_bits[i] = 0 for all i
        for col_id in twist.ra_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::ReadAddressBitsZeroPadding,
                bus_has_read,
                layout.bus_cell(col_id, 0),
            ));
        }

        // Padding: (1 - has_write) * wa_bits[i] = 0 for all i
        for col_id in twist.wa_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::WriteAddressBitsZeroPadding,
                bus_has_write,
                layout.bus_cell(col_id, 0),
            ));
        }
    }

    /// Add constraints for a Shout (lookup) instance.
    ///
    /// # Constraints Added
    ///
    /// **Value Binding:**
    /// - `is_lookup * (lookup_output - bus_val) = 0`
    ///
    /// **Padding Constraints:**
    /// - `(1 - has_lookup) * val = 0`
    /// - `(1 - has_lookup) * addr_bits[i] = 0` for all i
    ///
    /// # Credits
    /// Constraints adapted from Jolt's `RdWriteEqLookupIfWriteLookupToRd`.
    pub fn add_shout_instance(&mut self, layout: &BusLayout, shout: &ShoutCols, cpu_layout: &CpuColumnLayout) {
        let cpu = ShoutCpuBinding {
            has_lookup: cpu_layout.is_lookup,
            addr: cpu_layout.lookup_key,
            val: cpu_layout.lookup_output,
        };
        self.add_shout_instance_bound(layout, shout, &cpu);
    }

    /// Add constraints for a Shout (lookup) instance using an explicit per-instance CPU binding.
    pub fn add_shout_instance_bound(&mut self, layout: &BusLayout, shout: &ShoutCols, cpu: &ShoutCpuBinding) {
        assert_eq!(layout.chunk_size, 1, "CpuConstraintBuilder only supports chunk_size==1");

        // Bus column indices
        let bus_has_lookup = layout.bus_cell(shout.has_lookup, 0);
        let bus_val = layout.bus_cell(shout.val, 0);

        // Value binding: is_lookup * (lookup_output - bus_val) = 0
        self.constraints.push(CpuConstraint::new_eq(
            CpuConstraintLabel::LookupValueBinding,
            cpu.has_lookup,
            cpu.val,
            bus_val,
        ));

        // Selector binding: cpu_has_lookup == bus_has_lookup
        self.add_equality_constraint(
            CpuConstraintLabel::LookupSelectorBinding,
            cpu.has_lookup,
            bus_has_lookup,
        );

        // Key binding (bit-pack): is_lookup * (lookup_key - pack(addr_bits)) = 0
        self.constraints.push(CpuConstraint::new_terms(
            CpuConstraintLabel::LookupKeyBinding,
            cpu.has_lookup,
            false,
            pack_addr_bits::<F>(cpu.addr, shout.addr_bits.clone(), layout),
        ));

        // Padding: (1 - has_lookup) * val = 0
        self.constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::LookupValueZeroPadding,
            bus_has_lookup,
            bus_val,
        ));

        // Padding: (1 - has_lookup) * addr_bits[i] = 0 for all i
        for col_id in shout.addr_bits.clone() {
            self.constraints.push(CpuConstraint::new_zero_negated(
                CpuConstraintLabel::LookupAddressBitsZeroPadding,
                bus_has_lookup,
                layout.bus_cell(col_id, 0),
            ));
        }
    }

    /// Add an unconditional equality constraint: `left == right` (always).
    ///
    /// This is used for selector binding (is_load == has_read, etc.).
    /// Internally represented as: `1 * (left - right) = 0`
    pub fn add_equality_constraint(&mut self, label: CpuConstraintLabel, left_col: usize, right_col: usize) {
        self.constraints
            .push(CpuConstraint::new_eq(label, self.const_one_col, left_col, right_col));
    }

    /// Get the accumulated constraints.
    pub fn constraints(&self) -> &[CpuConstraint<F>] {
        &self.constraints
    }

    /// Build the CCS structure from accumulated constraints.
    ///
    /// Each constraint `A(z) * B(z) = 0` becomes an R1CS constraint:
    /// - A matrix: the condition linear form
    /// - B matrix: the linear expression to gate
    /// - C matrix: all zeros
    ///
    /// The resulting CCS uses f(x1, x2, x3) = x1 * x2 - x3 (standard R1CS embedding).
    pub fn build(&self) -> Result<CcsStructure<F>, String> {
        if self.constraints.is_empty() {
            return Err("no constraints added".to_string());
        }

        let n = self.n;
        let m = self.m;
        let num_constraints = self.constraints.len();

        // We need n >= num_constraints for square CCS
        if num_constraints > n {
            return Err(format!(
                "too many constraints ({}) for CCS with n={}",
                num_constraints, n
            ));
        }

        // Build A, B, C matrices for R1CS
        let mut a_data = vec![F::ZERO; n * m];
        let mut b_data = vec![F::ZERO; n * m];
        let c_data = vec![F::ZERO; n * m]; // C is all zeros for our constraints

        for (row, constraint) in self.constraints.iter().enumerate() {
            // A matrix: condition column(s)
            if constraint.negate_condition {
                // (1 - condition)
                a_data[row * m + self.const_one_col] = F::ONE;
                a_data[row * m + constraint.condition_col] = -F::ONE;
                for &col in &constraint.additional_condition_cols {
                    a_data[row * m + col] = -F::ONE;
                }
            } else {
                // Just condition
                a_data[row * m + constraint.condition_col] = F::ONE;
                for &col in &constraint.additional_condition_cols {
                    a_data[row * m + col] = F::ONE;
                }
            }

            // B matrix: Σ coeffᵢ · z[colᵢ]
            for &(col, coeff) in &constraint.b_terms {
                b_data[row * m + col] += coeff;
            }
        }

        // Create matrices
        let a = Mat::from_row_major(n, m, a_data);
        let b = Mat::from_row_major(n, m, b_data);
        let c = Mat::from_row_major(n, m, c_data);

        // Convert to CCS: f(x1, x2, x3) = x1 * x2 - x3
        // For identity-first CCS (square), we add I_n as M_0
        if n == m {
            let i_n = Mat::identity(n);
            let f = SparsePoly::new(
                4,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![0, 1, 1, 0], // x1 * x2
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![0, 0, 0, 1], // -x3
                    },
                ],
            );
            CcsStructure::new(vec![i_n, a, b, c], f).map_err(|e| format!("failed to create CCS: {:?}", e))
        } else {
            let f = SparsePoly::new(
                3,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![1, 1, 0], // x1 * x2
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![0, 0, 1], // -x3
                    },
                ],
            );
            CcsStructure::new(vec![a, b, c], f).map_err(|e| format!("failed to create CCS: {:?}", e))
        }
    }

    /// Extend an existing CCS with bus binding constraints.
    ///
    /// This is the recommended way to add bus constraints to an existing CPU CCS.
    /// The function merges the constraint matrices while preserving the original
    /// CPU constraints.
    ///
    /// # Arguments
    /// * `base_ccs` - The original CPU CCS
    /// * `constraint_rows` - Which rows in the CCS to place the new constraints
    ///
    /// # Returns
    /// Extended CCS with bus binding constraints added.
    pub fn extend_ccs(
        &self,
        base_ccs: &CcsStructure<F>,
        constraint_start_row: usize,
    ) -> Result<CcsStructure<F>, String> {
        if self.constraints.is_empty() {
            return Ok(base_ccs.clone());
        }

        let num_constraints = self.constraints.len();
        if constraint_start_row + num_constraints > base_ccs.n {
            return Err(format!(
                "constraint rows {} + {} exceed CCS n={}",
                constraint_start_row, num_constraints, base_ccs.n
            ));
        }

        // Ensure base CCS has at least 3 matrices (A, B, C from R1CS)
        if base_ccs.matrices.len() < 3 {
            return Err("base CCS must have at least 3 matrices (R1CS structure)".to_string());
        }

        let _m = base_ccs.m;
        let _n = base_ccs.n;

        // Clone base matrices
        let mut matrices: Vec<Mat<F>> = base_ccs.matrices.clone();

        // Determine which matrix indices are A, B, C
        // For identity-first CCS: M_0 = I_n, M_1 = A, M_2 = B, M_3 = C
        // For non-identity-first: M_0 = A, M_1 = B, M_2 = C
        let (a_idx, b_idx, _c_idx) = if base_ccs.matrices.len() >= 4 && base_ccs.matrices[0].is_identity() {
            (1, 2, 3)
        } else {
            (0, 1, 2)
        };

        // Add constraints to A and B matrices
        for (i, constraint) in self.constraints.iter().enumerate() {
            let row = constraint_start_row + i;

            // A matrix: condition
            if constraint.negate_condition {
                let current = matrices[a_idx][(row, self.const_one_col)];
                matrices[a_idx].set(row, self.const_one_col, current + F::ONE);
                let current = matrices[a_idx][(row, constraint.condition_col)];
                matrices[a_idx].set(row, constraint.condition_col, current - F::ONE);
                for &col in &constraint.additional_condition_cols {
                    let current = matrices[a_idx][(row, col)];
                    matrices[a_idx].set(row, col, current - F::ONE);
                }
            } else {
                let current = matrices[a_idx][(row, constraint.condition_col)];
                matrices[a_idx].set(row, constraint.condition_col, current + F::ONE);
                for &col in &constraint.additional_condition_cols {
                    let current = matrices[a_idx][(row, col)];
                    matrices[a_idx].set(row, col, current + F::ONE);
                }
            }

            // B matrix: Σ coeffᵢ · z[colᵢ]
            for &(col, coeff) in &constraint.b_terms {
                let current = matrices[b_idx][(row, col)];
                matrices[b_idx].set(row, col, current + coeff);
            }
        }

        CcsStructure::new(matrices, base_ccs.f.clone()).map_err(|e| format!("failed to extend CCS: {:?}", e))
    }
}

/// Extend an existing **CPU CCS** with the canonical shared-bus binding + padding constraints.
///
/// This is the "no footguns" helper: it computes the shared-bus tail layout from the provided
/// instance metadata, builds the Jolt-derived constraints via `CpuConstraintBuilder`, and injects
/// them into the **last** rows of the base CCS (which must be reserved / unused).
///
/// ## Requirements
/// - The base CCS must be R1CS-embedded (A/B/C matrices, optionally identity-first).
/// - The base CCS must reserve enough trailing **all-zero** rows in A/B/C to fit the injected constraints.
/// - Only `chunk_size == 1` is supported (i.e., all instances must have `steps == 1`).
/// - `shout_cpu` and `twist_cpu` must be aligned 1:1 with the provided instances.
///
/// ## Bus Ordering
/// This function assumes the caller passes instances in canonical bus order:
/// 1) all Shout instances (table_id-sorted order upstream),
/// 2) all Twist instances (mem_id-sorted order upstream).
pub fn extend_ccs_with_shared_cpu_bus_constraints<F: Field + Copy, Cmt>(
    base_ccs: &CcsStructure<F>,
    m_in: usize,
    const_one_col: usize,
    shout_cpu: &[ShoutCpuBinding],
    twist_cpu: &[TwistCpuBinding],
    lut_insts: &[LutInstance<Cmt, F>],
    mem_insts: &[MemInstance<Cmt, F>],
) -> Result<CcsStructure<F>, String> {
    if shout_cpu.len() != lut_insts.len() {
        return Err(format!(
            "shout_cpu.len()={} != lut_insts.len()={}",
            shout_cpu.len(),
            lut_insts.len()
        ));
    }
    if twist_cpu.len() != mem_insts.len() {
        return Err(format!(
            "twist_cpu.len()={} != mem_insts.len()={}",
            twist_cpu.len(),
            mem_insts.len()
        ));
    }

    // This helper is for the current shared-bus layout in `R1csCpu`, which is chunk_size==1.
    for (i, inst) in lut_insts.iter().enumerate() {
        if inst.steps != 1 {
            return Err(format!(
                "extend_ccs_with_shared_cpu_bus_constraints only supports chunk_size=1 (got lut_insts[{i}].steps={})",
                inst.steps
            ));
        }
    }
    for (i, inst) in mem_insts.iter().enumerate() {
        if inst.steps != 1 {
            return Err(format!(
                "extend_ccs_with_shared_cpu_bus_constraints only supports chunk_size=1 (got mem_insts[{i}].steps={})",
                inst.steps
            ));
        }
    }

    let layout = build_bus_layout_for_instances(
        base_ccs.m,
        m_in,
        1,
        lut_insts.iter().map(|inst| inst.d * inst.ell),
        mem_insts.iter().map(|inst| inst.d * inst.ell),
    )?;

    if layout.bus_cols == 0 {
        return Ok(base_ccs.clone());
    }

    let mut builder = CpuConstraintBuilder::<F>::new(base_ccs.n, base_ccs.m, const_one_col);
    for (i, cpu) in shout_cpu.iter().enumerate() {
        builder.add_shout_instance_bound(&layout, &layout.shout_cols[i], cpu);
    }
    for (i, cpu) in twist_cpu.iter().enumerate() {
        builder.add_twist_instance_bound(&layout, &layout.twist_cols[i], cpu);
    }

    let needed = builder.constraints().len();
    if needed == 0 {
        return Ok(base_ccs.clone());
    }
    if needed > base_ccs.n {
        return Err(format!(
            "not enough CCS rows to inject shared-bus constraints: need {needed}, have n={}",
            base_ccs.n
        ));
    }
    let start = base_ccs.n - needed;

    // Determine which matrix indices are A/B/C.
    let (a_idx, b_idx, c_idx) = if base_ccs.matrices.len() >= 4 && base_ccs.matrices[0].is_identity() {
        (1usize, 2usize, 3usize)
    } else if base_ccs.matrices.len() >= 3 {
        (0usize, 1usize, 2usize)
    } else {
        return Err("base CCS must have at least 3 matrices (R1CS A/B/C)".to_string());
    };

    // Refuse to overwrite any existing constraints: require trailing rows are empty in A/B/C.
    for row in start..base_ccs.n {
        for col in 0..base_ccs.m {
            if base_ccs.matrices[a_idx][(row, col)] != F::ZERO
                || base_ccs.matrices[b_idx][(row, col)] != F::ZERO
                || base_ccs.matrices[c_idx][(row, col)] != F::ZERO
            {
                return Err(format!(
                    "cannot inject shared-bus constraints: base CCS row {row} is not empty in A/B/C.\n\
                     Fix: reserve the LAST {needed} rows of the base CPU R1CS as padding rows (all-zero in A/B/C)."
                ));
            }
        }
    }

    builder.extend_ccs(base_ccs, start)
}

/// Create padding constraints for all bus columns.
///
/// This is a convenience function that adds all necessary padding constraints
/// to ensure inactive bus fields are zero.
///
/// # Credits
/// Implements the "padding row constraints" recommended by the Jolt team:
/// - `(1 - has_read) * rv = 0`
/// - `(1 - has_write) * wv = 0`
/// - etc.
pub fn create_twist_padding_constraints<F: Field>(layout: &BusLayout, twist: &TwistCols) -> Vec<CpuConstraint<F>> {
    assert_eq!(
        layout.chunk_size, 1,
        "create_twist_padding_constraints only supports chunk_size==1"
    );
    let bus_has_read = layout.bus_cell(twist.has_read, 0);
    let bus_has_write = layout.bus_cell(twist.has_write, 0);
    let bus_rv = layout.bus_cell(twist.rv, 0);
    let bus_wv = layout.bus_cell(twist.wv, 0);
    let bus_inc = layout.bus_cell(twist.inc, 0);

    let mut constraints = Vec::new();

    // (1 - has_read) * rv = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::ReadValueZeroPadding,
        bus_has_read,
        bus_rv,
    ));

    // (1 - has_write) * wv = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::WriteValueZeroPadding,
        bus_has_write,
        bus_wv,
    ));

    // (1 - has_write) * inc_at_write_addr = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::IncrementZeroPadding,
        bus_has_write,
        bus_inc,
    ));

    // (1 - has_read) * ra_bits[i] = 0 for all i
    for col_id in twist.ra_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::ReadAddressBitsZeroPadding,
            bus_has_read,
            layout.bus_cell(col_id, 0),
        ));
    }

    // (1 - has_write) * wa_bits[i] = 0 for all i
    for col_id in twist.wa_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::WriteAddressBitsZeroPadding,
            bus_has_write,
            layout.bus_cell(col_id, 0),
        ));
    }

    constraints
}

/// Create padding constraints for Shout (lookup) bus columns.
pub fn create_shout_padding_constraints<F: Field>(layout: &BusLayout, shout: &ShoutCols) -> Vec<CpuConstraint<F>> {
    assert_eq!(
        layout.chunk_size, 1,
        "create_shout_padding_constraints only supports chunk_size==1"
    );
    let bus_has_lookup = layout.bus_cell(shout.has_lookup, 0);
    let bus_val = layout.bus_cell(shout.val, 0);

    let mut constraints = Vec::new();

    // (1 - has_lookup) * val = 0
    constraints.push(CpuConstraint::new_zero_negated(
        CpuConstraintLabel::LookupValueZeroPadding,
        bus_has_lookup,
        bus_val,
    ));

    // (1 - has_lookup) * addr_bits[i] = 0 for all i
    for col_id in shout.addr_bits.clone() {
        constraints.push(CpuConstraint::new_zero_negated(
            CpuConstraintLabel::LookupAddressBitsZeroPadding,
            bus_has_lookup,
            layout.bus_cell(col_id, 0),
        ));
    }

    constraints
}

fn pack_addr_bits<F: Field>(addr_col: usize, bit_cols: Range<usize>, layout: &BusLayout) -> Vec<(usize, F)> {
    assert_eq!(layout.chunk_size, 1, "pack_addr_bits only supports chunk_size==1");
    let len = bit_cols.end.saturating_sub(bit_cols.start);
    let mut terms = Vec::with_capacity(1 + len);
    terms.push((addr_col, F::ONE));
    let mut weight = F::ONE;
    for col_id in bit_cols {
        terms.push((layout.bus_cell(col_id, 0), -weight));
        weight = weight + weight; // *= 2
    }
    terms
}
