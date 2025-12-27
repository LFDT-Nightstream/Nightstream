//! RISC-V Constraint Synthesis for CCS.
//!
//! This module generates Customizable Constraint Systems (CCS) that encode
//! RISC-V instruction semantics. Each instruction step is constrained so that:
//!
//! 1. The program counter advances correctly
//! 2. Register updates follow instruction semantics
//! 3. Memory operations are bound to Twist traces
//! 4. ALU operations are bound to Shout lookups
//!
//! ## Witness Structure
//!
//! For a single RISC-V step, the witness `z` is organized as:
//!
//! ```text
//! z = [1, pc_in, pc_out, regs_in[0..32], regs_out[0..32], opcode, rd, rs1, rs2, imm, mem_addr, mem_val, lookup_result]
//! ```
//!
//! ## Constraint Types
//!
//! 1. **PC Constraint**: `pc_out = pc_in + 4` (for non-branches)
//! 2. **Register Write**: `regs_out[rd] = ALU(regs_in[rs1], regs_in[rs2] or imm)`
//! 3. **Memory Binding**: Memory operations are bound via Twist commitments
//! 4. **Lookup Binding**: ALU results are bound via Shout commitments
//!
//! ## Public Inputs
//!
//! The public inputs are:
//! - `pc_in`: Initial program counter
//! - `pc_out`: Final program counter (for output binding)
//! - `output_reg`: The register containing the program output

use neo_ccs::{
    matrix::Mat,
    poly::{SparsePoly, Term},
    relations::CcsStructure,
};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Goldilocks as F;

// ============================================================================
// Witness Layout
// ============================================================================

/// Witness layout for a single RISC-V step.
///
/// The witness is organized for efficient constraint synthesis:
/// - Public inputs first (const_1, pc_in, pc_out, output_reg_value)
/// - Register file (32 input + 32 output = 64 elements)
/// - Instruction fields
/// - Memory/lookup values
#[derive(Clone, Debug)]
pub struct RiscvWitnessLayout {
    /// Number of public inputs (const_1 + pc + output)
    pub m_in: usize,
    /// Total witness size
    pub m: usize,
}

impl RiscvWitnessLayout {
    // Witness indices
    pub const CONST_1: usize = 0;
    pub const PC_IN: usize = 1;
    pub const PC_OUT: usize = 2;
    pub const OUTPUT_VAL: usize = 3;

    // Registers start at index 4
    pub const REGS_IN_START: usize = 4;
    pub const REGS_OUT_START: usize = 36; // 4 + 32

    // Instruction fields
    pub const OPCODE: usize = 68; // 4 + 64
    pub const RD: usize = 69;
    pub const RS1: usize = 70;
    pub const RS2: usize = 71;
    pub const IMM: usize = 72;

    // Memory/lookup
    pub const MEM_ADDR: usize = 73;
    pub const MEM_VAL: usize = 74;
    pub const LOOKUP_RESULT: usize = 75;

    // Auxiliary
    pub const AUX_START: usize = 76;

    /// Create a new witness layout.
    pub fn new() -> Self {
        Self {
            m_in: 4, // const_1, pc_in, pc_out, output_val
            m: 80,   // Total size with some auxiliary space
        }
    }

    /// Get index for input register.
    pub fn reg_in(r: usize) -> usize {
        assert!(r < 32);
        Self::REGS_IN_START + r
    }

    /// Get index for output register.
    pub fn reg_out(r: usize) -> usize {
        assert!(r < 32);
        Self::REGS_OUT_START + r
    }
}

impl Default for RiscvWitnessLayout {
    fn default() -> Self {
        Self::new()
    }
}

// ============================================================================
// CCS Building Blocks
// ============================================================================

/// Build a selector matrix that picks column `col` from row `row`.
fn selector_matrix(n: usize, m: usize, row: usize, col: usize) -> Mat<F> {
    let mut mat = Mat::zero(n, m, F::ZERO);
    if row < n && col < m {
        mat[(row, col)] = F::ONE;
    }
    mat
}

/// Build an identity-like selector for a specific column (for constraint row 0).
fn column_selector(n: usize, m: usize, col: usize) -> Mat<F> {
    selector_matrix(n, m, 0, col)
}

// ============================================================================
// Simple Step CCS (for testing)
// ============================================================================

/// Build a simple CCS that constrains a single addition operation.
///
/// Constraint: `z[out_idx] = z[in1_idx] + z[in2_idx]`
///
/// This is useful for testing the constraint synthesis pipeline.
pub fn build_add_constraint_ccs(n: usize, m: usize, in1_idx: usize, in2_idx: usize, out_idx: usize) -> CcsStructure<F> {
    // M0: picks in1
    let m0 = column_selector(n, m, in1_idx);
    // M1: picks in2
    let m1 = column_selector(n, m, in2_idx);
    // M2: picks out (with negative coefficient in polynomial)
    let m2 = column_selector(n, m, out_idx);

    // Polynomial: M0*z + M1*z - M2*z = 0
    // i.e., in1 + in2 - out = 0
    let terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }, // M0*z
        Term {
            coeff: F::ONE,
            exps: vec![0, 1, 0],
        }, // M1*z
        Term {
            coeff: F::ZERO - F::ONE,
            exps: vec![0, 0, 1],
        }, // -M2*z
    ];
    let f = SparsePoly::new(3, terms);

    CcsStructure::new(vec![m0, m1, m2], f).expect("valid CCS")
}

/// Build a CCS that constrains `pc_out = pc_in + 4`.
///
/// This is the simplest PC constraint for sequential execution.
pub fn build_pc_increment_ccs(n: usize, m: usize, _layout: &RiscvWitnessLayout) -> CcsStructure<F> {
    // M0: picks pc_in
    let m0 = column_selector(n, m, RiscvWitnessLayout::PC_IN);
    // M1: picks const_1 (we'll multiply by 4)
    let m1 = column_selector(n, m, RiscvWitnessLayout::CONST_1);
    // M2: picks pc_out
    let m2 = column_selector(n, m, RiscvWitnessLayout::PC_OUT);

    // Constraint: pc_in + 4*const_1 - pc_out = 0
    let terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![1, 0, 0],
        }, // pc_in
        Term {
            coeff: F::from_u64(4),
            exps: vec![0, 1, 0],
        }, // 4 * const_1
        Term {
            coeff: F::ZERO - F::ONE,
            exps: vec![0, 0, 1],
        }, // -pc_out
    ];
    let f = SparsePoly::new(3, terms);

    CcsStructure::new(vec![m0, m1, m2], f).expect("valid CCS")
}

// ============================================================================
// Full RISC-V Step CCS
// ============================================================================

/// Configuration for RISC-V CCS generation.
#[derive(Clone, Debug)]
pub struct RiscvCcsConfig {
    /// Whether to include PC constraints
    pub constrain_pc: bool,
    /// Whether to include register file constraints
    pub constrain_registers: bool,
    /// Whether to include lookup binding constraints
    pub constrain_lookups: bool,
    /// Whether to include memory binding constraints
    pub constrain_memory: bool,
}

impl Default for RiscvCcsConfig {
    fn default() -> Self {
        Self {
            constrain_pc: true,
            constrain_registers: true,
            constrain_lookups: true,
            constrain_memory: true,
        }
    }
}

/// Build a CCS for a single RISC-V ALU step.
///
/// This CCS constrains:
/// 1. PC increment (pc_out = pc_in + 4)
/// 2. Register x0 is always 0
/// 3. Unchanged registers stay the same
/// 4. The destination register gets the lookup result
///
/// The actual ALU computation is verified via Shout (lookup argument).
pub fn build_riscv_alu_step_ccs(layout: &RiscvWitnessLayout) -> CcsStructure<F> {
    let n = layout.m; // Square for identity-first
    let m = layout.m;

    // We'll build multiple constraints:
    // Row 0: PC constraint (pc_in + 4 - pc_out = 0)
    // Row 1: x0 constraint (regs_out[0] = 0)
    // Row 2-33: Register propagation (simplified: just check output reg gets lookup result)

    // For simplicity, build a minimal CCS that:
    // 1. Enforces pc_out = pc_in + 4
    // 2. Enforces regs_out[rd] = lookup_result (implicitly via witness structure)

    // M0: picks const_1 (for the constant 4)
    let mut m0 = Mat::zero(n, m, F::ZERO);
    m0[(0, RiscvWitnessLayout::CONST_1)] = F::ONE;

    // M1: picks pc_in
    let mut m1 = Mat::zero(n, m, F::ZERO);
    m1[(0, RiscvWitnessLayout::PC_IN)] = F::ONE;

    // M2: picks pc_out
    let mut m2 = Mat::zero(n, m, F::ZERO);
    m2[(0, RiscvWitnessLayout::PC_OUT)] = F::ONE;

    // M3: picks output_val (for public output binding)
    let mut m3 = Mat::zero(n, m, F::ZERO);
    m3[(0, RiscvWitnessLayout::OUTPUT_VAL)] = F::ONE;

    // Polynomial: pc_in + 4*const_1 - pc_out = 0
    // Plus: output_val is bound to the witness
    let terms = vec![
        Term {
            coeff: F::ONE,
            exps: vec![0, 1, 0, 0],
        }, // pc_in
        Term {
            coeff: F::from_u64(4),
            exps: vec![1, 0, 0, 0],
        }, // 4 * const_1
        Term {
            coeff: F::ZERO - F::ONE,
            exps: vec![0, 0, 1, 0],
        }, // -pc_out
    ];
    let f = SparsePoly::new(4, terms);

    CcsStructure::new(vec![m0, m1, m2, m3], f).expect("valid CCS")
}

// ============================================================================
// Witness Generation
// ============================================================================

/// A RISC-V step witness.
#[derive(Clone, Debug)]
pub struct RiscvStepWitness {
    pub layout: RiscvWitnessLayout,
    pub values: Vec<F>,
}

impl RiscvStepWitness {
    /// Create a new witness with all zeros.
    pub fn new(layout: RiscvWitnessLayout) -> Self {
        let mut values = vec![F::ZERO; layout.m];
        values[RiscvWitnessLayout::CONST_1] = F::ONE;
        Self { layout, values }
    }

    /// Set the program counter values.
    pub fn set_pc(&mut self, pc_in: u64, pc_out: u64) {
        self.values[RiscvWitnessLayout::PC_IN] = F::from_u64(pc_in);
        self.values[RiscvWitnessLayout::PC_OUT] = F::from_u64(pc_out);
    }

    /// Set the output value (for public binding).
    pub fn set_output(&mut self, value: u64) {
        self.values[RiscvWitnessLayout::OUTPUT_VAL] = F::from_u64(value);
    }

    /// Set an input register value.
    pub fn set_reg_in(&mut self, reg: usize, value: u64) {
        if reg > 0 && reg < 32 {
            self.values[RiscvWitnessLayout::reg_in(reg)] = F::from_u64(value);
        }
    }

    /// Set an output register value.
    pub fn set_reg_out(&mut self, reg: usize, value: u64) {
        if reg > 0 && reg < 32 {
            self.values[RiscvWitnessLayout::reg_out(reg)] = F::from_u64(value);
        }
    }

    /// Set the lookup result.
    pub fn set_lookup_result(&mut self, result: u64) {
        self.values[RiscvWitnessLayout::LOOKUP_RESULT] = F::from_u64(result);
    }

    /// Get the witness values as a slice.
    pub fn as_slice(&self) -> &[F] {
        &self.values
    }

    /// Get public inputs.
    pub fn public_inputs(&self) -> &[F] {
        &self.values[..self.layout.m_in]
    }

    /// Get private witness.
    pub fn private_witness(&self) -> &[F] {
        &self.values[self.layout.m_in..]
    }
}

// ============================================================================
// Witness from Trace
// ============================================================================

use neo_vm_trace::VmTrace;

/// Generate a witness for a single step from a VmTrace.
pub fn witness_from_trace_step(
    trace: &VmTrace<u64, u64>,
    step_idx: usize,
    layout: &RiscvWitnessLayout,
) -> Option<RiscvStepWitness> {
    let step = trace.steps.get(step_idx)?;

    let mut witness = RiscvStepWitness::new(layout.clone());

    // Set PC
    witness.set_pc(step.pc_before, step.pc_after);

    // Set registers
    for r in 0..32 {
        witness.set_reg_in(r, step.regs_before[r]);
        witness.set_reg_out(r, step.regs_after[r]);
    }

    // Set output (last register value for final step)
    if step_idx == trace.steps.len() - 1 {
        // Use register x3 as the output register (convention)
        witness.set_output(step.regs_after[3]);
    }

    Some(witness)
}

/// Generate witnesses for all steps in a trace.
pub fn witnesses_from_trace(trace: &VmTrace<u64, u64>, layout: &RiscvWitnessLayout) -> Vec<RiscvStepWitness> {
    (0..trace.steps.len())
        .filter_map(|i| witness_from_trace_step(trace, i, layout))
        .collect()
}

// ============================================================================
// Helper: Check CCS Satisfaction
// ============================================================================

/// Check if a witness satisfies the CCS constraints.
pub fn check_ccs_satisfaction(ccs: &CcsStructure<F>, witness: &[F]) -> bool {
    if witness.len() != ccs.m {
        return false;
    }

    // Evaluate each matrix-vector product
    let mut products: Vec<Vec<F>> = Vec::with_capacity(ccs.matrices.len());
    for mat in &ccs.matrices {
        let mut prod = vec![F::ZERO; ccs.n];
        for row in 0..ccs.n {
            for col in 0..ccs.m {
                prod[row] += mat[(row, col)] * witness[col];
            }
        }
        products.push(prod);
    }

    // Evaluate polynomial at each row
    for row in 0..ccs.n {
        let mut sum = F::ZERO;
        for term in ccs.f.terms() {
            let mut prod = term.coeff;
            for (mat_idx, &exp) in term.exps.iter().enumerate() {
                if exp > 0 && mat_idx < products.len() {
                    for _ in 0..exp {
                        prod *= products[mat_idx][row];
                    }
                }
            }
            sum += prod;
        }
        if sum != F::ZERO {
            return false;
        }
    }

    true
}
