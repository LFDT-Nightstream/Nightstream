use super::*;
use serde_json::json;

const BITS: usize = 3 * PHI81_RING_DEGREE;
const PRIVATE: usize = BITS + PHI81_RING_DEGREE;

fn run(first: usize, count: usize) -> Run {
    Run {
        first,
        step: 1,
        count,
        last: first + count - 1,
    }
}

/// One source, one block, one cell: 162 challenge bits, then 54 right-operand values.
fn one_ring_plan() -> Plan {
    Plan {
        blocks: Vec::new(),
        families: vec![Phi81FamilyShape {
            source_count: 1,
            block_count: 1,
            cell_count: 1,
            first_invocation: 0,
        }],
        values: vec![run(BITS, PHI81_RING_DEGREE)],
        challenges: vec![run(0, BITS)],
        digest: Vec::new(),
        physical_width: PRIVATE + 1,
        public_width: 0,
        logical_width: 0,
    }
}

fn layout() -> Layout {
    Layout {
        row_count: 0,
        private_column_count: PRIVATE,
        constant_column: PRIVATE,
        public_column_count: 0,
        total_column_count: PRIVATE + 1,
        private_segments: Vec::new(),
        public_segments: Vec::new(),
    }
}

/// Every digit is 2 (centered 0) except `digits`; bits are little-endian.
fn private_values(digits: &[(usize, [u64; 3])], right: &[(usize, u64)]) -> Vec<u64> {
    let mut values = vec![0; PRIVATE];
    for lane in 0..PHI81_RING_DEGREE {
        values[3 * lane + 1] = 1;
    }
    for &(lane, bits) in digits {
        values[3 * lane..3 * lane + 3].copy_from_slice(&bits);
    }
    for &(lane, value) in right {
        values[BITS + lane] = value;
    }
    values
}

fn quotients(values: Vec<u64>) -> Result<Vec<u64>, PackageError> {
    let assignment = WitnessAssignment {
        private_values: values,
        public_values: Vec::new(),
    };
    let layout = layout();
    let plan = one_ring_plan();
    plan.quotients(&PhysicalAssignment::new(&layout, &assignment, plan.physical_width)?)
}

#[test]
fn quotient_reads_little_endian_bits_and_centers_digits() {
    // Digit 4 = bits (0, 0, 1) is the centered coefficient 2.
    // 2 X^53 * X^53 = 2 X^25 + Phi81 * 2 (X^52 - X^25).
    let quotient = quotients(private_values(&[(53, [0, 0, 1])], &[(53, 1)])).unwrap();
    for (degree, &value) in quotient[..PHI81_RING_DEGREE].iter().enumerate() {
        let expected = match degree {
            52 => 2,
            25 => GOLDILOCKS_MODULUS - 2,
            _ => 0,
        };
        assert_eq!(value, expected, "quotient degree {degree}");
    }
    assert!(quotient[PHI81_RING_DEGREE..]
        .iter()
        .all(|&value| value == 0));
}

#[test]
fn quotient_rejects_a_challenge_bit_above_one() {
    let mut values = private_values(&[], &[]);
    values[3 * 7] = 2;
    assert!(matches!(
        quotients(values),
        Err(PackageError::Invalid("wide challenge bit"))
    ));
}

#[test]
fn quotient_rejects_a_challenge_digit_above_four() {
    let values = private_values(&[(9, [1, 0, 1])], &[]);
    assert!(matches!(
        quotients(values),
        Err(PackageError::Invalid("wide challenge digit"))
    ));
}

#[test]
fn decoder_pins_the_production_family_order() {
    let plan = |shapes| json!([4, [], [shapes, [], []], []]);
    let decode_shapes = |shapes| decode(&plan(shapes), 1, 0, 0).map(|_| ());
    // Same total and source count as production, but the first two families swap.
    assert!(matches!(
        decode_shapes(json!([[17, 5, 1], [17, 22, 1], [17, 1, 2], [17, 14, 2]])),
        Err(PackageError::Invalid("wide quotient family shape"))
    ));
    assert!(matches!(
        decode_shapes(json!([[17, 22, 1], [17, 5, 1], [17, 1, 2]])),
        Err(PackageError::Invalid(_))
    ));
    // The production order passes the shape check and fails later on the empty runs.
    assert!(matches!(
        decode_shapes(json!([[17, 22, 1], [17, 5, 1], [17, 1, 2], [17, 14, 2]])),
        Err(PackageError::Invalid("assignment source run coverage"))
    ));
}
