//! R1CS Integration Tests for neo-vm-trace.
//!
//! These tests verify that the VM tracing infrastructure integrates correctly
//! with the R1CS/CCS constraint system. Each CPU step produces:
//! 1. A trace (via `VmCpu` trait)
//! 2. R1CS constraints that the witness must satisfy
//!
//! This ensures the tracing layer produces traces that can be converted to
//! valid CCS witnesses for the Neo proving pipeline.

use neo_ccs::{check_ccs_rowwise_zero, r1cs_to_ccs, Mat};
use neo_math::F;
use neo_vm_trace::*;
use p3_field::PrimeCharacteristicRing;

// ============================================================================
// Fibonacci CPU with R1CS Constraints
// ============================================================================

/// A Fibonacci CPU that computes F_{n+2} = F_n + F_{n+1}.
///
/// Each step has R1CS constraints:
///   A * z ∘ B * z = C * z
/// where z = [1, F_n, F_{n+1}, F_{n+2}]
///
/// The constraint encodes: (F_n + F_{n+1}) * 1 = F_{n+2}
pub struct FibonacciCpu {
    /// Current Fibonacci value F_n
    pub f_curr: u64,
    /// Next Fibonacci value F_{n+1}
    pub f_next: u64,
    /// Current step index
    pub step: u64,
    /// Maximum steps before halt
    pub max_steps: u64,
    /// Whether the CPU has halted
    pub halted: bool,
}

impl FibonacciCpu {
    pub fn new(max_steps: u64) -> Self {
        Self {
            f_curr: 0,
            f_next: 1,
            step: 0,
            max_steps,
            halted: false,
        }
    }

    /// Build the R1CS matrices for a Fibonacci step.
    ///
    /// Constraint: (F_n + F_{n+1}) * 1 = F_{n+2}
    /// Variables: z = [1, F_n, F_{n+1}, F_{n+2}]
    ///
    /// A selects (F_n + F_{n+1}): A * z = z[1] + z[2]
    /// B selects 1: B * z = z[0] = 1
    /// C selects F_{n+2}: C * z = z[3]
    pub fn build_r1cs_matrices() -> (Mat<F>, Mat<F>, Mat<F>) {
        let n = 4; // 4x4 matrices (padded for identity-first CCS)

        let mut a = Mat::zero(n, n, F::ZERO);
        let mut b = Mat::zero(n, n, F::ZERO);
        let mut c = Mat::zero(n, n, F::ZERO);

        // Row 0: Main Fibonacci constraint
        // A: select z[1] + z[2] (F_n + F_{n+1})
        a[(0, 1)] = F::ONE;
        a[(0, 2)] = F::ONE;
        // B: select z[0] (the constant 1)
        b[(0, 0)] = F::ONE;
        // C: select z[3] (F_{n+2})
        c[(0, 3)] = F::ONE;

        // Rows 1-3: Padding with trivial constraints 0 * 0 = 0
        // (identity-first CCS requires square matrices)

        (a, b, c)
    }

    /// Build the witness vector for the current step.
    ///
    /// z = [1, F_n, F_{n+1}, F_{n+2}]
    pub fn build_witness(&self) -> Vec<F> {
        let f_n = F::from_u64(self.f_curr);
        let f_n1 = F::from_u64(self.f_next);
        let f_n2 = f_n + f_n1;

        vec![F::ONE, f_n, f_n1, f_n2]
    }
}

impl VmCpu<u64, u64> for FibonacciCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.f_curr, self.f_next]
    }

    fn pc(&self) -> u64 {
        self.step
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<T, S>(&mut self, _twist: &mut T, _shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        let f_n2 = self.f_curr.wrapping_add(self.f_next);

        // Advance state
        self.f_curr = self.f_next;
        self.f_next = f_n2;
        self.step += 1;

        if self.step >= self.max_steps {
            self.halted = true;
        }

        Ok(StepMeta {
            pc_after: self.step,
            opcode: 1, // FIB opcode
        })
    }
}

// ============================================================================
// ALU CPU with R1CS Constraints
// ============================================================================

/// An ALU CPU that performs arithmetic operations with R1CS constraints.
///
/// Each instruction type has its own constraint:
/// - ADD: a + b = c  →  (a + b) * 1 = c
/// - MUL: a * b = c  →  a * b = c
/// - SUB: a - b = c  →  (c + b) * 1 = a
pub struct AluCpu {
    /// Accumulator register
    pub acc: u64,
    /// Program counter
    pub pc: u64,
    /// Program (opcode, operand)
    pub program: Vec<(u32, u64)>,
    /// Whether halted
    pub halted: bool,
}

/// ALU opcodes
pub mod alu_ops {
    pub const HALT: u32 = 0;
    pub const LOAD: u32 = 1; // acc = operand
    pub const ADD: u32 = 2; // acc = acc + operand
    pub const MUL: u32 = 3; // acc = acc * operand
    pub const SUB: u32 = 4; // acc = acc - operand
}

impl AluCpu {
    pub fn new(program: Vec<(u32, u64)>) -> Self {
        Self {
            acc: 0,
            pc: 0,
            program,
            halted: false,
        }
    }

    /// Build R1CS matrices for an ADD operation: acc_new = acc_old + operand
    ///
    /// z = [1, acc_old, operand, acc_new]
    /// Constraint: (acc_old + operand) * 1 = acc_new
    pub fn build_add_r1cs(acc_old: u64, operand: u64, acc_new: u64) -> (Mat<F>, Mat<F>, Mat<F>, Vec<F>) {
        let n = 4;

        let mut a = Mat::zero(n, n, F::ZERO);
        let mut b = Mat::zero(n, n, F::ZERO);
        let mut c = Mat::zero(n, n, F::ZERO);

        // (acc_old + operand) * 1 = acc_new
        a[(0, 1)] = F::ONE; // acc_old
        a[(0, 2)] = F::ONE; // operand
        b[(0, 0)] = F::ONE; // 1
        c[(0, 3)] = F::ONE; // acc_new

        let z = vec![F::ONE, F::from_u64(acc_old), F::from_u64(operand), F::from_u64(acc_new)];

        (a, b, c, z)
    }

    /// Build R1CS matrices for a MUL operation: acc_new = acc_old * operand
    ///
    /// z = [1, acc_old, operand, acc_new]
    /// Constraint: acc_old * operand = acc_new
    pub fn build_mul_r1cs(acc_old: u64, operand: u64, acc_new: u64) -> (Mat<F>, Mat<F>, Mat<F>, Vec<F>) {
        let n = 4;

        let mut a = Mat::zero(n, n, F::ZERO);
        let mut b = Mat::zero(n, n, F::ZERO);
        let mut c = Mat::zero(n, n, F::ZERO);

        // acc_old * operand = acc_new
        a[(0, 1)] = F::ONE; // acc_old
        b[(0, 2)] = F::ONE; // operand
        c[(0, 3)] = F::ONE; // acc_new

        let z = vec![F::ONE, F::from_u64(acc_old), F::from_u64(operand), F::from_u64(acc_new)];

        (a, b, c, z)
    }
}

impl VmCpu<u64, u64> for AluCpu {
    type Error = String;

    fn snapshot_regs(&self) -> Vec<u64> {
        vec![self.acc]
    }

    fn pc(&self) -> u64 {
        self.pc
    }

    fn halted(&self) -> bool {
        self.halted
    }

    fn step<T, S>(&mut self, _twist: &mut T, _shout: &mut S) -> Result<StepMeta<u64>, Self::Error>
    where
        T: Twist<u64, u64>,
        S: Shout<u64>,
    {
        if self.pc as usize >= self.program.len() {
            self.halted = true;
            return Ok(StepMeta {
                pc_after: self.pc,
                opcode: alu_ops::HALT,
            });
        }

        let (opcode, operand) = self.program[self.pc as usize];

        match opcode {
            alu_ops::HALT => {
                self.halted = true;
            }
            alu_ops::LOAD => {
                self.acc = operand;
            }
            alu_ops::ADD => {
                self.acc = self.acc.wrapping_add(operand);
            }
            alu_ops::MUL => {
                self.acc = self.acc.wrapping_mul(operand);
            }
            alu_ops::SUB => {
                self.acc = self.acc.wrapping_sub(operand);
            }
            _ => return Err(format!("Unknown opcode: {}", opcode)),
        }

        self.pc += 1;

        Ok(StepMeta {
            pc_after: self.pc,
            opcode,
        })
    }
}

// ============================================================================
// Simple Twist/Shout implementations for testing
// ============================================================================

struct EmptyTwist;

impl Twist<u64, u64> for EmptyTwist {
    fn load(&mut self, _twist_id: TwistId, _addr: u64) -> u64 {
        0
    }
    fn store(&mut self, _twist_id: TwistId, _addr: u64, _value: u64) {}
}

struct EmptyShout;

impl Shout<u64> for EmptyShout {
    fn lookup(&mut self, _shout_id: ShoutId, _key: u64) -> u64 {
        0
    }
}

// ============================================================================
// Tests
// ============================================================================

#[test]
fn test_fibonacci_cpu_r1cs_constraints_satisfied() {
    // Run the Fibonacci CPU for 10 steps
    let cpu = FibonacciCpu::new(10);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();

    assert_eq!(trace.len(), 10);
    assert!(trace.did_halt());

    // Build R1CS matrices (same for all Fibonacci steps)
    let (a, b, c) = FibonacciCpu::build_r1cs_matrices();
    let ccs = r1cs_to_ccs(a, b, c);

    // Verify each step satisfies the CCS constraints
    let mut f_curr = 0u64;
    let mut f_next = 1u64;

    for (i, step) in trace.steps.iter().enumerate() {
        // Build witness for this step
        let f_n2 = f_curr.wrapping_add(f_next);
        let z = vec![F::ONE, F::from_u64(f_curr), F::from_u64(f_next), F::from_u64(f_n2)];

        // Check CCS satisfaction (all variables are witness, no public input)
        let public: Vec<F> = vec![];
        let result = check_ccs_rowwise_zero(&ccs, &public, &z);
        assert!(result.is_ok(), "Step {} failed CCS check: {:?}", i, result.err());

        // Verify trace matches expected values
        assert_eq!(
            step.regs_before,
            vec![f_curr, f_next],
            "Step {} regs_before mismatch",
            i
        );

        // Advance to next Fibonacci number
        f_curr = f_next;
        f_next = f_n2;

        assert_eq!(step.regs_after, vec![f_curr, f_next], "Step {} regs_after mismatch", i);
    }

    // Verify final Fibonacci values: F_10 = 55, F_11 = 89
    assert_eq!(f_curr, 55);
    assert_eq!(f_next, 89);
}

#[test]
fn test_fibonacci_cpu_invalid_witness_fails_ccs() {
    // Build R1CS matrices
    let (a, b, c) = FibonacciCpu::build_r1cs_matrices();
    let ccs = r1cs_to_ccs(a, b, c);

    // Valid witness: F_0=0, F_1=1, F_2=1 → (0+1)*1 = 1 ✓
    let valid_z = vec![F::ONE, F::ZERO, F::ONE, F::ONE];
    let empty: Vec<F> = vec![];
    assert!(check_ccs_rowwise_zero(&ccs, &empty, &valid_z).is_ok());

    // Invalid witness: F_0=0, F_1=1, F_2=5 → (0+1)*1 = 5 ✗
    let invalid_z = vec![F::ONE, F::ZERO, F::ONE, F::from_u64(5)];
    assert!(check_ccs_rowwise_zero(&ccs, &empty, &invalid_z).is_err());
}

#[test]
fn test_alu_add_r1cs_constraints_satisfied() {
    // Program: load 10, add 5, add 3, halt
    // Expected: 0 → 10 → 15 → 18
    let program = vec![
        (alu_ops::LOAD, 10),
        (alu_ops::ADD, 5),
        (alu_ops::ADD, 3),
        (alu_ops::HALT, 0),
    ];

    let cpu = AluCpu::new(program);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();

    assert_eq!(trace.len(), 4);
    assert!(trace.did_halt());

    // Verify ADD steps satisfy R1CS constraints
    let empty: Vec<F> = vec![];

    // Step 1: 10 + 5 = 15
    let (a, b, c, z) = AluCpu::build_add_r1cs(10, 5, 15);
    let ccs = r1cs_to_ccs(a, b, c);
    assert!(
        check_ccs_rowwise_zero(&ccs, &empty, &z).is_ok(),
        "ADD 10+5=15 should satisfy CCS"
    );

    // Step 2: 15 + 3 = 18
    let (a, b, c, z) = AluCpu::build_add_r1cs(15, 3, 18);
    let ccs = r1cs_to_ccs(a, b, c);
    assert!(
        check_ccs_rowwise_zero(&ccs, &empty, &z).is_ok(),
        "ADD 15+3=18 should satisfy CCS"
    );

    // Verify trace captured correct accumulator values
    assert_eq!(trace.steps[0].regs_after, vec![10]); // LOAD 10
    assert_eq!(trace.steps[1].regs_after, vec![15]); // ADD 5
    assert_eq!(trace.steps[2].regs_after, vec![18]); // ADD 3
}

#[test]
fn test_alu_mul_r1cs_constraints_satisfied() {
    // Program: load 3, mul 4, mul 2, halt
    // Expected: 0 → 3 → 12 → 24
    let program = vec![
        (alu_ops::LOAD, 3),
        (alu_ops::MUL, 4),
        (alu_ops::MUL, 2),
        (alu_ops::HALT, 0),
    ];

    let cpu = AluCpu::new(program);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();

    assert_eq!(trace.len(), 4);
    assert!(trace.did_halt());

    // Verify MUL steps satisfy R1CS constraints
    let empty: Vec<F> = vec![];

    // Step 1: 3 * 4 = 12
    let (a, b, c, z) = AluCpu::build_mul_r1cs(3, 4, 12);
    let ccs = r1cs_to_ccs(a, b, c);
    assert!(
        check_ccs_rowwise_zero(&ccs, &empty, &z).is_ok(),
        "MUL 3*4=12 should satisfy CCS"
    );

    // Step 2: 12 * 2 = 24
    let (a, b, c, z) = AluCpu::build_mul_r1cs(12, 2, 24);
    let ccs = r1cs_to_ccs(a, b, c);
    assert!(
        check_ccs_rowwise_zero(&ccs, &empty, &z).is_ok(),
        "MUL 12*2=24 should satisfy CCS"
    );

    // Verify trace captured correct accumulator values
    assert_eq!(trace.steps[0].regs_after, vec![3]); // LOAD 3
    assert_eq!(trace.steps[1].regs_after, vec![12]); // MUL 4
    assert_eq!(trace.steps[2].regs_after, vec![24]); // MUL 2
}

#[test]
fn test_alu_invalid_mul_fails_ccs() {
    // Invalid: 3 * 4 ≠ 15
    let (a, b, c, _) = AluCpu::build_mul_r1cs(3, 4, 12);
    let ccs = r1cs_to_ccs(a, b, c);
    let empty: Vec<F> = vec![];

    // Wrong result
    let bad_z = vec![F::ONE, F::from_u64(3), F::from_u64(4), F::from_u64(15)];
    assert!(
        check_ccs_rowwise_zero(&ccs, &empty, &bad_z).is_err(),
        "3*4≠15 should fail CCS"
    );
}

#[test]
fn test_trace_to_witness_consistency() {
    // This test verifies that trace values can be converted to witnesses
    // that satisfy the constraints.

    let cpu = FibonacciCpu::new(5);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();
    let (a, b, c) = FibonacciCpu::build_r1cs_matrices();
    let ccs = r1cs_to_ccs(a, b, c);
    let empty: Vec<F> = vec![];

    // For each step, reconstruct witness from trace and verify
    for step in &trace.steps {
        // The trace captures [f_curr, f_next] before and after
        let f_curr_before = step.regs_before[0];
        let f_next_before = step.regs_before[1];

        // Witness: [1, f_curr, f_next, f_curr + f_next]
        let f_sum = f_curr_before.wrapping_add(f_next_before);
        let z = vec![
            F::ONE,
            F::from_u64(f_curr_before),
            F::from_u64(f_next_before),
            F::from_u64(f_sum),
        ];

        // This witness should satisfy the constraint
        assert!(check_ccs_rowwise_zero(&ccs, &empty, &z).is_ok());
    }
}

#[test]
fn test_mixed_alu_operations() {
    // Program: load 5, add 3, mul 2, sub 6, halt
    // Expected: 0 → 5 → 8 → 16 → 10
    let program = vec![
        (alu_ops::LOAD, 5),
        (alu_ops::ADD, 3),
        (alu_ops::MUL, 2),
        (alu_ops::SUB, 6),
        (alu_ops::HALT, 0),
    ];

    let cpu = AluCpu::new(program);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();

    assert_eq!(trace.len(), 5);
    assert!(trace.did_halt());

    // Verify intermediate values in trace
    assert_eq!(trace.steps[0].regs_after, vec![5]); // LOAD 5
    assert_eq!(trace.steps[1].regs_after, vec![8]); // 5 + 3 = 8
    assert_eq!(trace.steps[2].regs_after, vec![16]); // 8 * 2 = 16
    assert_eq!(trace.steps[3].regs_after, vec![10]); // 16 - 6 = 10
}

#[test]
fn test_fibonacci_sequence_correctness() {
    // Run Fibonacci for 20 steps and verify the sequence
    let cpu = FibonacciCpu::new(20);
    let twist = EmptyTwist;
    let shout = EmptyShout;

    let trace = trace_program(cpu, twist, shout, 100).unwrap();

    // Expected Fibonacci sequence: 0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, ...
    let expected_fib = [
        0, 1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610, 987, 1597, 2584, 4181, 6765,
    ];

    for (i, step) in trace.steps.iter().enumerate() {
        assert_eq!(
            step.regs_before[0], expected_fib[i],
            "Step {} f_curr mismatch: expected F_{} = {}",
            i, i, expected_fib[i]
        );
        assert_eq!(
            step.regs_before[1],
            expected_fib[i + 1],
            "Step {} f_next mismatch: expected F_{} = {}",
            i,
            i + 1,
            expected_fib[i + 1]
        );
    }
}
