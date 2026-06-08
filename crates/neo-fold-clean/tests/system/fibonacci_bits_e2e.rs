//! End-to-end direct-CCS test for a real bit-encoded Fibonacci circuit.
//!
//! Each row proves one transition `next = prev + curr` using an R1CS over
//! low-norm `0/1` witnesses:
//!
//! - public bits for `prev`, `curr`, and `next`,
//! - private carry bits,
//! - booleanity constraints for every bit and carry,
//! - ripple-carry addition constraints.
//!
//! This is not the Construction-2 F' encoder. It proves real CCS/R1CS
//! instances and folds them through the IVC chain; the future decider still
//! owns proving that each instance is the recursive F' encoding.

use neo_ccs::matrix::Mat as NeoMat;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use neo_fold_clean::frontends::direct_ccs::{self, FrontendError, R1cs};
use neo_fold_clean::{finish_uncompressed_with_audit, prove, verify_uncompressed_audit, FoldSchedule};

const LIMBS: usize = 5;
const ONE: usize = 0;

fn a_bit(j: usize) -> usize {
    1 + j
}

fn b_bit(j: usize) -> usize {
    1 + LIMBS + j
}

fn c_bit(j: usize) -> usize {
    1 + 2 * LIMBS + j
}

fn carry_bit(j: usize) -> usize {
    debug_assert!(j < LIMBS - 1);
    1 + 3 * LIMBS + j
}

fn public_width() -> usize {
    1 + 3 * LIMBS
}

fn private_width() -> usize {
    LIMBS - 1
}

fn circuit_width() -> usize {
    public_width() + private_width()
}

fn fibonacci_addition_r1cs() -> R1cs {
    assert!(circuit_width() <= D);

    let rows = 1 + 3 * LIMBS + private_width() + LIMBS;
    let mut a = NeoMat::zero(rows, D, F::default());
    let mut b = NeoMat::zero(rows, D, F::default());
    let mut c = NeoMat::zero(rows, D, F::default());

    let mut row = 0;

    // Public constant convention: the first public variable is expected to
    // be one by the verifier. This row only enforces booleanity; the value
    // itself is visible in the public input.
    a[(row, ONE)] = F::ONE;
    b[(row, ONE)] = F::ONE;
    c[(row, ONE)] = F::ONE;
    row += 1;

    for idx in public_bit_indices().chain(carry_bit_indices()) {
        add_boolean_constraint(row, idx, &mut a, &mut b);
        row += 1;
    }

    for j in 0..LIMBS {
        // a_j + b_j + carry_in = next_j + 2 * carry_out.
        //
        // carry_0 is implicit zero. The final carry_out is constrained to
        // zero by omitting it on the RHS of the most-significant limb row.
        a[(row, a_bit(j))] = F::ONE;
        a[(row, b_bit(j))] = F::ONE;
        if j > 0 {
            a[(row, carry_bit(j - 1))] = F::ONE;
        }

        b[(row, ONE)] = F::ONE;
        c[(row, c_bit(j))] = F::ONE;
        if j + 1 < LIMBS {
            c[(row, carry_bit(j))] = F::from_u64(2);
        }
        row += 1;
    }

    debug_assert_eq!(row, rows);

    R1cs {
        a,
        b,
        c,
        m_in: public_width(),
    }
}

fn add_boolean_constraint(row: usize, idx: usize, a: &mut NeoMat<F>, b: &mut NeoMat<F>) {
    // bit * (1 - bit) = 0.
    a[(row, idx)] = F::ONE;
    b[(row, ONE)] = F::ONE;
    b[(row, idx)] = -F::ONE;
}

fn public_bit_indices() -> impl Iterator<Item = usize> {
    (0..LIMBS)
        .map(a_bit)
        .chain((0..LIMBS).map(b_bit))
        .chain((0..LIMBS).map(c_bit))
}

fn carry_bit_indices() -> impl Iterator<Item = usize> {
    (0..LIMBS - 1).map(carry_bit)
}

fn fibonacci_transition_assignment(prev: u64, curr: u64, next: u64) -> Vec<F> {
    assert_eq!(prev + curr, next, "test fixture must be a Fibonacci transition");
    assert!(next < (1 << LIMBS), "increase LIMBS for this trace");

    let mut z = vec![F::ZERO; D];
    z[ONE] = F::ONE;

    for j in 0..LIMBS {
        z[a_bit(j)] = bit(prev, j);
        z[b_bit(j)] = bit(curr, j);
        z[c_bit(j)] = bit(next, j);
    }

    let mut carry = 0;
    for j in 0..LIMBS {
        let sum = bit_u64(prev, j) + bit_u64(curr, j) + carry;
        assert_eq!(bit_u64(next, j), sum & 1);
        carry = sum >> 1;
        if j + 1 < LIMBS {
            z[carry_bit(j)] = F::from_u64(carry);
        } else {
            assert_eq!(carry, 0, "trace overflows LIMBS");
        }
    }

    z
}

fn bit(value: u64, idx: usize) -> F {
    F::from_u64(bit_u64(value, idx))
}

fn bit_u64(value: u64, idx: usize) -> u64 {
    (value >> idx) & 1
}

#[test]
fn rejects_bad_fibonacci_transition() {
    let r1cs = fibonacci_addition_r1cs();
    let mut z = fibonacci_transition_assignment(3, 5, 8);
    z[c_bit(0)] = F::ONE - z[c_bit(0)];

    match r1cs.is_satisfied_by(&z) {
        Err(FrontendError::Unsatisfied { .. }) => {}
        other => panic!("expected Unsatisfied for tampered Fibonacci transition, got {other:?}"),
    }
}

#[test]
fn fibonacci_bits_prove_verify_uncompressed() {
    let r1cs = fibonacci_addition_r1cs();
    let prep = direct_ccs::preprocess_seeded(&r1cs, /* seed = */ 0xF1B0).expect("preprocess");

    let trace = [1_u64, 1, 2, 3, 5, 8, 13, 21];
    let instances = trace
        .windows(3)
        .map(|w| {
            let z = fibonacci_transition_assignment(w[0], w[1], w[2]);
            direct_ccs::build_instance(&prep, &r1cs, &z).expect("build Fibonacci transition instance")
        })
        .collect::<Vec<_>>();

    let batches = FoldSchedule::RowsPerStep(1)
        .partition(instances)
        .expect("partition");

    let proof = prove(&prep, batches).expect("prove");
    let finished = finish_uncompressed_with_audit(&prep, proof).expect("finish_uncompressed_with_audit");
    verify_uncompressed_audit(&prep, &finished).expect("verify_uncompressed_audit");
}
