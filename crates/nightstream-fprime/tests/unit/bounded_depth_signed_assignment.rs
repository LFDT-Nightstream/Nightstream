use super::*;

#[test]
fn packed_assignment_has_the_same_digits_across_blocks_and_tasks() {
    let size = PUBLIC + 251;
    let mut columns = vec![Fq::ZERO; size];
    let mut centered = vec![0; size.div_ceil(64)];
    let mut run_starts = Vec::new();
    let mut previous = None;
    let mut spans = SlotSpans {
        starts: vec![0; size],
        widths: vec![0; size],
        ratio: Fq::from_u64(FIELD_RATIO),
    };
    let mut width = PUBLIC;
    for (column, value) in columns[..PUBLIC].iter_mut().enumerate() {
        spans.starts[column] = column as u32;
        spans.widths[column] = 1;
        *value = Fq::from_usize(column % 3) - Fq::ONE;
    }
    for column in PUBLIC..columns.len() {
        if column >= PUBLIC + 128 && column % 3 == 0 {
            continue;
        }
        let (kind, value) = match column {
            c if c < PUBLIC + 64 => (Kind::Field, Fq::from_usize(c)),
            c if c < PUBLIC + 128 => (Kind::Bit, Fq::from_bool(c % 2 == 1)),
            c => match c % 5 {
                0 => (Kind::Bit, Fq::ONE),
                1 => (Kind::Centered, Fq::NEG_ONE),
                2 => (Kind::Field, Fq::new(Fq::ORDER_U64 / 2)),
                3 => (Kind::Field, Fq::new(Fq::ORDER_U64 / 2 + 1)),
                _ => (Kind::Field, Fq::ZERO),
            },
        };
        if kind == Kind::Centered {
            centered[column / 64] |= 1u64 << (column % 64);
        }
        spans.starts[column] = width as u32;
        spans.widths[column] = kind.width() as u8;
        if previous != Some((column - 1, kind)) {
            run_starts.push(column as u32);
        }
        previous = Some((column, kind));
        width += kind.width();
        columns[column] = value;
    }
    let layout = LogicalLayout {
        spans,
        centered,
        run_starts: run_starts.into_boxed_slice(),
        width,
    };
    let logical = layout.assignment(&columns).unwrap();
    for threads in [1, 3] {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .build()
            .unwrap();
        let packed = pool.install(|| layout.signed_assignment(&columns)).unwrap();
        assert!(packed.is_packed_signed_unit());
        for index in 0..packed.rows() * packed.cols() {
            let expected = logical.get(index).copied().unwrap_or(Fq::ZERO);
            assert_eq!(packed[(index % RING_DEGREE, index / RING_DEGREE)], expected);
        }
    }
    columns[0] = Fq::TWO;
    assert!(layout.signed_assignment(&columns).is_err());
}

#[test]
fn ternary_masks_match_individual_digits_at_carries_and_centered_limits() {
    let check = |value: u64| {
        let mut digits = [0i8; FIELD_DIGITS];
        balanced_ternary_digits(value, &mut digits);
        let mut expected = (0u64, 0u64);
        for (lane, digit) in digits.into_iter().enumerate() {
            match digit {
                1 => expected.0 |= 1 << lane,
                -1 => expected.1 |= 1 << lane,
                0 => {}
                _ => unreachable!(),
            }
        }
        assert_eq!(balanced_ternary_masks(value), expected, "canonical field word {value}");
    };
    let both_signs = |magnitude| {
        check(magnitude);
        if magnitude != 0 {
            check(Fq::ORDER_U64 - magnitude);
        }
    };
    for residue in 0..TERNARY_RADIX as u64 {
        both_signs(residue);
    }
    let mut power = TERNARY_RADIX as u64;
    while power <= Fq::ORDER_U64 / 2 {
        for magnitude in [power / 2, power / 2 + 1, power - 1, power, power + 1] {
            both_signs(magnitude);
        }
        let Some(next) = power.checked_mul(TERNARY_RADIX as u64) else {
            break;
        };
        power = next;
    }
    for value in [
        Fq::ORDER_U64 / 2 - 1,
        Fq::ORDER_U64 / 2,
        Fq::ORDER_U64 / 2 + 1,
        Fq::ORDER_U64 - 1,
    ] {
        check(value);
    }
}
