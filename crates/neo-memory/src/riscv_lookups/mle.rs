use p3_field::Field;

use super::alu::lookup_entry;
use super::isa::RiscvOpcode;

/// Evaluate the MLE of the AND operation at a random point.
///
/// For AND, the MLE has a simple form:
/// `AND~(r) = Σ_{i=0}^{n-1} 2^i * r_{2i} * r_{2i+1}`
///
/// where r is a vector of length 2*XLEN with interleaved x and y bits.
/// Position 2i contains the i-th bit of x, position 2i+1 contains the i-th bit of y.
pub fn evaluate_and_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        result += coeff * x_i * y_i;
    }
    result
}

/// Evaluate the MLE of the XOR operation at a random point.
///
/// For XOR, the MLE is:
/// `XOR~(r) = Σ_{i=0}^{n-1} 2^i * (r_{2i}(1-r_{2i+1}) + (1-r_{2i})r_{2i+1})`
pub fn evaluate_xor_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        // XOR: x(1-y) + (1-x)y = x + y - 2xy
        result += coeff * (x_i * (F::ONE - y_i) + (F::ONE - x_i) * y_i);
    }
    result
}

/// Evaluate the MLE of the OR operation at a random point.
///
/// For OR, the MLE is:
/// `OR~(r) = Σ_{i=0}^{n-1} 2^i * (r_{2i} + r_{2i+1} - r_{2i}*r_{2i+1})`
pub fn evaluate_or_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);
        // OR: x + y - xy
        result += coeff * (x_i + y_i - x_i * y_i);
    }
    result
}

/// Evaluate the MLE of EQ (equality predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x == y`, else 0.
pub fn evaluate_eq_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ONE;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        // eq(x_i, y_i) = x_i y_i + (1-x_i)(1-y_i)
        result *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }
    result
}

/// Evaluate the MLE of NEQ (inequality predicate) at a random point.
pub fn evaluate_neq_mle<F: Field>(r: &[F]) -> F {
    F::ONE - evaluate_eq_mle(r)
}

/// Evaluate the MLE of SLTU (unsigned less-than predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x < y` as unsigned `xlen`-bit integers.
pub fn evaluate_sltu_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    // Interpret r as interleaved bits (x_0,y_0,x_1,y_1,...) with LSB-first.
    // To compare numbers, scan from MSB → LSB.
    let mut lt = F::ZERO;
    let mut eq_prefix = F::ONE;
    for bit in (0..xlen).rev() {
        let x_i = r[2 * bit];
        let y_i = r[2 * bit + 1];
        lt += (F::ONE - x_i) * y_i * eq_prefix;
        eq_prefix *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }
    lt
}

/// Evaluate the MLE of SLT (signed less-than predicate) at a random point.
///
/// Returns 1 on Boolean points iff `x < y` as signed two's-complement `xlen`-bit integers.
pub fn evaluate_slt_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;
    debug_assert!(xlen > 0);

    // Sign bits (MSB) for our LSB-first encoding.
    let x_sign = r[2 * (xlen - 1)];
    let y_sign = r[2 * (xlen - 1) + 1];

    // Unsigned less-than over the full bitstring (including sign bit), then adjust by sign bits.
    let mut lt = F::ZERO;
    let mut eq_prefix = F::ONE;
    for bit in (0..xlen).rev() {
        let x_i = r[2 * bit];
        let y_i = r[2 * bit + 1];
        lt += (F::ONE - x_i) * y_i * eq_prefix;
        eq_prefix *= x_i * y_i + (F::ONE - x_i) * (F::ONE - y_i);
    }

    x_sign - y_sign + lt
}

/// Evaluate the MLE of ADD at a random point.
///
/// For ADD, we use the decomposition: result = x + y (mod 2^xlen)
/// The MLE can be computed as: ADD~(r) = Σ x_bits + Σ y_bits + carry propagation
///
/// However, for simplicity, we use a different approach inspired by Jolt:
/// We verify ADD using a range check on the result. The MLE returns
/// the lower word (second operand bits in the interleaved representation).
pub fn evaluate_add_mle<F: Field>(r: &[F]) -> F {
    // ADD is verified via decomposition: result = x + y (mod 2^xlen)
    // For the MLE, we compute the sum at the evaluation point.
    // This works because at boolean points, it equals the table value.
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    // The direct polynomial for ADD is complex due to carry propagation.
    // We use the identity: x + y = x ^ y + 2 * (x & y)
    // But more accurately, we need the full ripple-carry:
    // result_i = x_i ^ y_i ^ c_{i-1}
    // c_i = (x_i & y_i) | (x_i & c_{i-1}) | (y_i & c_{i-1})

    // For efficiency, compute iteratively:
    let mut result = F::ZERO;
    let mut carry = F::ZERO;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let coeff = F::from_u64(1u64 << i);

        // result_i = x_i ⊕ y_i ⊕ carry
        // In multilinear form: x + y + c - 2*x*y - 2*x*c - 2*y*c + 4*x*y*c
        let sum_bit = x_i + y_i + carry
            - x_i * y_i * F::from_u64(2)
            - x_i * carry * F::from_u64(2)
            - y_i * carry * F::from_u64(2)
            + x_i * y_i * carry * F::from_u64(4);

        result += coeff * sum_bit;

        // carry_i = (x_i & y_i) | (x_i & c_{i-1}) | (y_i & c_{i-1})
        // In multilinear: xy + xc + yc - 2xyc
        carry = x_i * y_i + x_i * carry + y_i * carry - x_i * y_i * carry * F::from_u64(2);
    }

    result
}

/// Evaluate the MLE of SUB at a random point.
///
/// Computes `x - y (mod 2^xlen)` using ripple-carry addition of `x + (~y + 1)`.
pub fn evaluate_sub_mle<F: Field>(r: &[F]) -> F {
    debug_assert!(r.len() % 2 == 0);
    let xlen = r.len() / 2;

    let mut result = F::ZERO;
    // Initial carry-in of 1 accounts for the "+ 1" in two's complement (~y + 1).
    let mut carry = F::ONE;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        let y_comp = F::ONE - y_i;
        let coeff = F::from_u64(1u64 << i);

        // sum_bit = x_i ⊕ y_comp ⊕ carry
        let sum_bit = x_i + y_comp + carry
            - x_i * y_comp * F::from_u64(2)
            - x_i * carry * F::from_u64(2)
            - y_comp * carry * F::from_u64(2)
            + x_i * y_comp * carry * F::from_u64(4);

        result += coeff * sum_bit;

        // carry = majority(x_i, y_comp, carry) in multilinear form.
        carry = x_i * y_comp + x_i * carry + y_comp * carry - x_i * y_comp * carry * F::from_u64(2);
    }

    result
}

/// Evaluate the MLE of SLL (Shift Left Logical) at a random point.
///
/// For shift operations, Jolt uses a "virtual table" approach where the
/// MLE is computed using products over bit positions.
pub fn evaluate_sll_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // SLL: result_i = x_{i-shamt} if i >= shamt, else 0
    // The MLE is: Σ_i 2^i * Σ_{s=0}^{i} eq(y, s) * x_{i-s}
    // For simplicity, use naive evaluation for now
    evaluate_mle_naive(RiscvOpcode::Sll, r, xlen)
}

/// Evaluate the MLE of SRL (Shift Right Logical) at a random point.
///
/// Following Jolt's virtual SRL table approach.
pub fn evaluate_srl_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // Jolt's SRL formula: iteratively compute result *= (1 + y_i); result += x_i * y_i
    // This works because for each bit position, if y_i=1, we're selecting x_i,
    // otherwise we're shifting (multiplying by 1 + y_i = 2 when y_i=1 at boolean points)
    let mut result = F::ZERO;
    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        result = result * (F::ONE + y_i) + x_i * y_i;
    }
    result
}

/// Evaluate the MLE of SRA (Shift Right Arithmetic) at a random point.
///
/// Following Jolt's virtual SRA table approach.
pub fn evaluate_sra_mle<F: Field>(r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    // SRA is like SRL but with sign extension
    // Jolt's formula adds a sign_extension term based on the MSB
    let mut result = F::ZERO;
    let mut sign_extension = F::ZERO;

    for i in 0..xlen {
        let x_i = r[2 * i];
        let y_i = r[2 * i + 1];
        result = result * (F::ONE + y_i) + x_i * y_i;
        if i != 0 {
            sign_extension += F::from_u64(1 << i) * (F::ONE - y_i);
        }
    }

    // Add sign extension: MSB * sign_extension_mask
    let msb = r[0]; // x_0 is the MSB in interleaved representation
    result + msb * sign_extension
}

/// Evaluate the MLE of a RISC-V opcode at a random point.
///
/// This dispatches to the appropriate MLE evaluation function based on the opcode.
/// For opcodes without closed-form MLEs, this falls back to the naive computation.
///
/// # Note on Shift Operations
///
/// Jolt uses "virtual tables" for shift operations with specialized MLE formulas
/// (see `evaluate_srl_mle` and `evaluate_sra_mle`). These virtual tables encode
/// the shift amount as a bitmask rather than a direct value, which allows for
/// efficient MLE evaluation. Our standard lookup tables use direct shift amounts,
/// so we use naive MLE evaluation for consistency.
pub fn evaluate_opcode_mle<F: Field>(op: RiscvOpcode, r: &[F], xlen: usize) -> F {
    debug_assert_eq!(r.len(), 2 * xlen);

    match op {
        RiscvOpcode::And => evaluate_and_mle(r),
        RiscvOpcode::Xor => evaluate_xor_mle(r),
        RiscvOpcode::Or => evaluate_or_mle(r),
        RiscvOpcode::Add => evaluate_add_mle(r),
        RiscvOpcode::Sub => evaluate_sub_mle(r),
        RiscvOpcode::Eq => evaluate_eq_mle(r),
        RiscvOpcode::Neq => evaluate_neq_mle(r),
        RiscvOpcode::Slt => evaluate_slt_mle(r),
        RiscvOpcode::Sltu => evaluate_sltu_mle(r),
        // For shift and other opcodes, use the naive MLE evaluation when available.
        // Note: naive evaluation is O(2^{2*xlen}) and intentionally limited to tiny xlen.
        _ => {
            if xlen <= 8 {
                evaluate_mle_naive(op, r, xlen)
            } else {
                panic!("evaluate_opcode_mle: closed-form MLE not implemented for opcode {op:?} at xlen={xlen}");
            }
        }
    }
}

/// Naive MLE evaluation by summing over the Boolean hypercube.
///
/// This is O(2^{2*xlen}) and should only be used for testing or small tables.
fn evaluate_mle_naive<F: Field>(op: RiscvOpcode, r: &[F], xlen: usize) -> F {
    assert!(xlen <= 8, "Naive MLE evaluation only supports xlen <= 8");

    let table_size = 1usize << (2 * xlen);
    let mut result = F::ZERO;

    for idx in 0..table_size {
        // Compute χ_idx(r) = Π_k (idx_k * r_k + (1-idx_k)(1-r_k))
        // With LSB-aligned indexing, bit k of idx corresponds to r[k]
        let mut chi = F::ONE;
        for k in 0..(2 * xlen) {
            let bit = ((idx >> k) & 1) as u64;
            let r_k = r[k];
            if bit == 1 {
                chi *= r_k;
            } else {
                chi *= F::ONE - r_k;
            }
        }

        // Add contribution: χ_idx(r) * table[idx]
        let entry = lookup_entry(op, idx as u128, xlen);
        result += chi * F::from_u64(entry);
    }

    result
}
