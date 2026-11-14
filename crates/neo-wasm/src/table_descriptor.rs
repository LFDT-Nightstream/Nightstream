//! Rocq `TableDescriptor.full_table_descriptor`: one PC-ROM word binds a table and an expected type.

pub(crate) const TABLE_ID_BITS: u8 = 8;
pub(crate) const EXPECTED_TYPE_ID_BITS: u8 = 24;
pub(crate) const TABLE_TYPE_FACTOR: u32 = 1 << TABLE_ID_BITS;

pub(crate) fn pack_table_descriptor(table: u32, expected_type: u32) -> Result<u32, String> {
    for (name, value, bits) in [
        ("table id", table, TABLE_ID_BITS),
        ("expected type id", expected_type, EXPECTED_TYPE_ID_BITS),
    ] {
        if value >= 1 << bits {
            return Err(format!("{name} {value} exceeds the {bits}-bit table descriptor limit"));
        }
    }
    Ok(table + TABLE_TYPE_FACTOR * expected_type)
}
