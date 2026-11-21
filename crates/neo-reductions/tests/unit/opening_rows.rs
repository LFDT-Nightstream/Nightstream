use super::PackedRows;

#[test]
fn packed_rows_read_every_width_and_stop_at_the_range_boundary() {
    for highest in [0, 1, 255, 256, 65_535, 65_536, 0x00ff_ffff, 0x0100_0000, u32::MAX - 1] {
        let rows = [highest, highest / 2, 0, highest];
        let mut table = PackedRows::new(highest as usize + 1, rows.len()).unwrap();
        for (i, &row) in rows.iter().enumerate() {
            table.set(i, row);
        }
        for lo in 0..=rows.len() {
            for hi in lo..=rows.len() {
                assert!(table.iter(lo..hi).eq(rows[lo..hi].iter().copied()));
            }
        }
    }
}

#[test]
fn translated_columns_preserve_every_term_and_device_task() {
    use super::{Columns, F};
    use crate::superneo_eval::{FactoredSlots, OpeningTiles};
    use p3_field::PrimeCharacteristicRing;
    let count = 4200;
    let mut offsets = vec![0u32];
    let mut expected = Vec::new();
    for slot in 0..count {
        let len = if slot % 499 == 0 { 513 } else { 7 };
        for i in 0..len {
            expected.push((slot * 2 + i + u32::from(i == 2 && (slot / 8) % 2 == 1), (i % 3) as u16));
        }
        offsets.push(expected.len() as u32);
    }
    let mut rows = PackedRows::new(2 * count as usize + 513, expected.len()).unwrap();
    for (i, &(row, _)) in expected.iter().enumerate() {
        rows.set(i, row);
    }
    let mut columns = Columns {
        row_count: 2 * count as usize + 513,
        tiles: Default::default(),
        slots: (0..count).collect(),
        offsets: offsets.clone(),
        rows,
        tags: expected.iter().map(|p| p.1).collect(),
        factors: vec![F::ONE, F::NEG_ONE, F::TWO],
    };
    columns.share_translated_terms(usize::MAX);
    assert_eq!(columns.translations().len(), count as usize);
    for (slot, span) in offsets.windows(2).enumerate() {
        let (base, terms) = columns.terms(slot);
        assert!(terms
            .map(|(r, t)| (base + r, t))
            .eq(expected[span[0] as usize..span[1] as usize].iter().copied()));
    }
    let widths = vec![1; count as usize];
    let layout = FactoredSlots {
        starts: &columns.slots,
        widths: &widths,
        ratio: F::from_u64(3),
    };
    let blocks = (count as usize).div_ceil(64);
    let (_, heavy, tasks) = OpeningTiles::sizes(blocks, &offsets).unwrap();
    let tiles = OpeningTiles::prepare(layout, blocks, &columns.slots, &columns.offsets, (heavy, tasks)).unwrap();
    assert!(!tiles.heavy.is_empty());
    for &[slot, first, end, _] in &tiles.heavy {
        let expected = &expected[offsets[slot as usize] as usize..offsets[slot as usize + 1] as usize];
        let mut actual = Vec::new();
        for &[from, packed] in &tiles.tasks[first as usize..end as usize] {
            let len = 1 + (packed & 255) as usize;
            let range = from as usize..from as usize + len;
            actual.extend(
                columns
                    .rows
                    .iter(range.clone())
                    .zip(&columns.tags[range])
                    .map(|(r, &t)| (r + (packed >> 8), t)),
            );
        }
        assert_eq!(actual, expected);
    }
}
