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
