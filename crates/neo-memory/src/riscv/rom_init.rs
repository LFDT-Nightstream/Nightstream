use std::collections::HashMap;

use neo_vm_trace::TwistId;
use p3_field::PrimeCharacteristicRing;

use crate::plain::PlainMemLayout;

/// Build a `PlainMemLayout` for a byte-addressed, word-valued program ROM.
///
/// The resulting layout is **bit-addressed** (`n_side = 2`) with `k` chosen as the next
/// power-of-two that covers `[0 .. base + program_bytes.len())` so `d = log2(k)`.
pub fn prog_rom_layout(base: u64, program_bytes: &[u8]) -> Result<PlainMemLayout, String> {
    if base % 4 != 0 {
        return Err("base must be 4-byte aligned".into());
    }
    if program_bytes.len() % 4 != 0 {
        return Err("program must be 4-byte aligned".into());
    }

    let len_u64 = u64::try_from(program_bytes.len()).map_err(|_| "program length overflow".to_string())?;
    let end_excl = base
        .checked_add(len_u64)
        .ok_or_else(|| "program address range overflow".to_string())?;
    let min_k = usize::try_from(end_excl).map_err(|_| "program address range does not fit usize".to_string())?;
    let k = min_k.next_power_of_two().max(4); // ensure d>=2 for RV32 alignment constraints
    let d = k.trailing_zeros() as usize;
    Ok(PlainMemLayout { k, d, n_side: 2 })
}

/// Checked version of [`prog_init_words`] (does not panic).
pub fn prog_init_words_checked<F: PrimeCharacteristicRing>(
    prog_id: TwistId,
    base: u64,
    program_bytes: &[u8],
) -> Result<HashMap<(u32, u64), F>, String> {
    if base % 4 != 0 {
        return Err("base must be 4-byte aligned".into());
    }
    if program_bytes.len() % 4 != 0 {
        return Err("program must be 4-byte aligned".into());
    }

    let mut out = HashMap::new();
    for (i, chunk) in program_bytes.chunks_exact(4).enumerate() {
        let addr = base
            .checked_add((i as u64) * 4)
            .ok_or_else(|| "program address overflow".to_string())?;
        let w = u32::from_le_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]) as u64;
        if w != 0 {
            out.insert((prog_id.0, addr), F::from_u64(w));
        }
    }
    Ok(out)
}

/// Build a ROM layout and its sparse init map in one shot.
pub fn prog_rom_layout_and_init_words<F: PrimeCharacteristicRing>(
    prog_id: TwistId,
    base: u64,
    program_bytes: &[u8],
) -> Result<(PlainMemLayout, HashMap<(u32, u64), F>), String> {
    let layout = prog_rom_layout(base, program_bytes)?;
    let init = prog_init_words_checked(prog_id, base, program_bytes)?;
    Ok((layout, init))
}

/// Build sparse Twist initial memory for a byte-addressed, word-valued ROM.
///
/// The ROM value at address `base + 4*i` is the little-endian `u32` formed from
/// `program_bytes[4*i..4*i+4]`.
///
/// This matches the RV32 B1 step circuit convention:
/// - instruction fetch address is the architectural PC (byte address),
/// - instruction fetch value is a 32-bit word.
pub fn prog_init_words<F: PrimeCharacteristicRing>(
    prog_id: TwistId,
    base: u64,
    program_bytes: &[u8],
) -> HashMap<(u32, u64), F> {
    prog_init_words_checked(prog_id, base, program_bytes).expect("prog_init_words: invalid input")
}
