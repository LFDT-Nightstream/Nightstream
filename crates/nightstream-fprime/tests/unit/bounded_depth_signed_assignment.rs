use super::*;

#[test]
fn packed_assignment_has_the_same_digits_across_blocks_and_tasks() {
    let mut layout = LogicalLayout {
        offsets: vec![NO_SLOT; PUBLIC + 251],
        kinds: vec![Kind::Field; PUBLIC + 251],
        slots: Vec::new(),
        width: PUBLIC,
    };
    let mut columns = vec![Fq::ZERO; layout.offsets.len()];
    for (column, value) in columns[..PUBLIC].iter_mut().enumerate() {
        *value = Fq::from_usize(column % 3) - Fq::ONE;
    }
    for column in PUBLIC..columns.len() {
        if column % 3 == 0 {
            continue;
        }
        let (kind, value) = match column % 5 {
            0 => (Kind::Bit, Fq::ONE),
            1 => (Kind::Centered, Fq::NEG_ONE),
            2 => (Kind::Field, Fq::new(Fq::ORDER_U64 / 2)),
            3 => (Kind::Field, Fq::new(Fq::ORDER_U64 / 2 + 1)),
            _ => (Kind::Field, Fq::ZERO),
        };
        layout.kinds[column] = kind;
        layout.offsets[column] = layout.width as u32;
        layout.slots.push(column);
        layout.width += kind.width();
        columns[column] = value;
    }
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
