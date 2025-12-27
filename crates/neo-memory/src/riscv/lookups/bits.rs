/// Interleave the bits of two operands into a single lookup index.
///
/// For n-bit operands x and y, produces a 2n-bit index where:
/// - Bit positions 2i contain x_i
/// - Bit positions 2i+1 contain y_i
///
/// This matches Jolt's interleaving convention for lookup tables.
///
/// # Example
/// For x = 0b10 and y = 0b01:
/// - x_0 = 0, x_1 = 1
/// - y_0 = 1, y_1 = 0
/// - Result: bits at pos 0,1,2,3 = x0,y0,x1,y1 = 0,1,1,0 = 0b0110 = 6
pub fn interleave_bits(x: u64, y: u64) -> u128 {
    let mut result = 0u128;
    for i in 0..64 {
        let x_bit = ((x >> i) & 1) as u128;
        let y_bit = ((y >> i) & 1) as u128;
        result |= x_bit << (2 * i);
        result |= y_bit << (2 * i + 1);
    }
    result
}

/// Uninterleave bits from a lookup index back to two operands.
///
/// Inverse of `interleave_bits`.
pub fn uninterleave_bits(index: u128) -> (u64, u64) {
    let mut x = 0u64;
    let mut y = 0u64;
    for i in 0..64 {
        x |= (((index >> (2 * i)) & 1) as u64) << i;
        y |= (((index >> (2 * i + 1)) & 1) as u64) << i;
    }
    (x, y)
}
