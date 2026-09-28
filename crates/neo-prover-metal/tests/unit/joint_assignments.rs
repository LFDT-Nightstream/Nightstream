use super::*;
use neo_math::{KExtensions, D};
use objc2_metal::MTLBuffer;

#[test]
fn encoded_device_prefix_matches_dense_cpu_through_every_fold() {
    let session = MetalSession::new().unwrap();
    let width = 2 * D + 5; // Odd length, including a partially used ring block.
    let blocks = width.div_ceil(D);
    let mut words = vec![0u64; 3 * blocks * 2];
    let mut expected = Vec::new();
    for source in [0, 2] {
        let values = (0..width)
            .map(|column| {
                let digit = ((column * (source + 1) + source) % 3) as i32 - 1;
                if digit != 0 {
                    words[2 * (source * blocks + column / D) + usize::from(digit < 0)] |= 1 << (column % D);
                }
                K::from(if digit < 0 { -F::ONE } else { F::from_u64(digit as u64) })
            })
            .collect::<Vec<_>>();
        // A nonzero completion tail must not enter this logical prefix.
        words[2 * (source * blocks + blocks - 1)] |= 1 << (D - 1);
        expected.push(values);
    }
    let masks = session
        .prepare_witness_digit_masks(&words, 3, blocks, 1, width)
        .unwrap();
    let mut tables = AssignmentTables::new(&session, &masks, width).unwrap();
    assert!(matches!(tables.prefix, Prefix::Masks(_)));
    assert_eq!(session.read_buffer::<u64>(&tables.sources, 2), vec![0, 2]);
    for round in 0..width.next_power_of_two().ilog2() as usize {
        let challenge = match round {
            1 => K::ZERO,
            2 => K::ONE,
            _ => K::from_coeffs([F::from_usize(round + 2), F::from_usize(round + 3)]),
        };
        for prefix in &mut expected {
            *prefix = prefix
                .chunks(2)
                .map(|pair| pair[0] + (pair.get(1).copied().unwrap_or(K::ZERO) - pair[0]) * challenge)
                .collect();
        }
        tables.fold(&session, &masks, challenge).unwrap();
        assert_eq!(tables.encoding(), [1, 1, 2, 16, 16, 16, 16][round]);
        assert_eq!(tables.len, expected[0].len());
        let read_k = |buffer: &Buffer, len| {
            session
                .read_buffer::<u64>(buffer, 2 * len)
                .chunks_exact(2)
                .map(|words| K::from_coeffs([F::from_u64(words[0]), F::from_u64(words[1])]))
                .collect::<Vec<_>>()
        };
        let entries = tables.count * tables.len;
        let actual = match &tables.prefix {
            Prefix::Codes {
                data,
                values,
                alphabet,
                zero,
            } => {
                let lookup = read_k(values, *alphabet);
                assert_eq!(lookup[*zero], K::ZERO);
                let codes = if tables.encoding() == 1 {
                    session
                        .read_buffer::<u8>(data, entries)
                        .into_iter()
                        .map(usize::from)
                        .collect::<Vec<_>>()
                } else {
                    session
                        .read_buffer::<u16>(data, entries)
                        .into_iter()
                        .map(usize::from)
                        .collect::<Vec<_>>()
                };
                assert_eq!(data.length(), entries * tables.encoding());
                codes
                    .into_iter()
                    .map(|code| lookup[code])
                    .collect::<Vec<_>>()
            }
            Prefix::Dense(data) => {
                assert_eq!(data.length(), entries * size_of::<K>());
                read_k(data, entries)
            }
            Prefix::Masks(_) => panic!("nonzero prefixes must advance"),
        };
        assert_eq!(
            actual,
            expected.iter().flatten().copied().collect::<Vec<_>>(),
            "round {round}"
        );
    }
}

#[test]
fn zero_witnesses_need_no_folded_assignment_buffer() {
    let session = MetalSession::new().unwrap();
    let masks = session
        .prepare_witness_digit_masks(&[0; 6], 3, 1, 1, D)
        .unwrap();
    let mut tables = AssignmentTables::new(&session, &masks, D).unwrap();
    let before = session.activity();
    tables.fold(&session, &masks, K::ONE).unwrap();
    assert_eq!(tables.count, 0);
    assert!(matches!(tables.prefix, Prefix::Masks(_)));
    assert_eq!(session.activity().allocated_bytes, before.allocated_bytes);
    assert_eq!(session.activity().dispatches, before.dispatches);
}
