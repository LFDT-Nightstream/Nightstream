use super::*;

pub(super) fn extend_packed_bytes_as_fields(dst: &mut Vec<F>, bytes: &[u8]) {
    dst.push(F::from_u64(bytes.len() as u64));
    for chunk in bytes.chunks(PACKED_BYTES_PER_LIMB) {
        let mut limb = [0u8; 8];
        limb[..chunk.len()].copy_from_slice(chunk);
        dst.push(F::from_u64(u64::from_le_bytes(limb)));
    }
}

const PACKED_BYTES_PER_LIMB: usize = 7;
const SPARTAN_GOLDILOCKS_MODULUS: u64 = 0xFFFF_FFFF_0000_0001;

pub(super) fn packed_bytes_field_len(bytes_len: usize) -> usize {
    1 + bytes_len.div_ceil(PACKED_BYTES_PER_LIMB)
}

pub(super) fn spartan2_chunk_summary_field_len() -> usize {
    FixedShapeChunkSummary::packed_field_len() + FIXED_SHAPE_DIGEST_FIELD_LEN
}

pub(super) fn spartan2_chunk_summary_terminal_relation_digest_field_offset() -> usize {
    FixedShapeChunkSummary::packed_field_len()
}

pub(super) fn extend_spartan2_chunk_summary_fields(dst: &mut Vec<F>, summary: &FixedShapeChunkSummary) {
    dst.extend(summary.packed_fields());
    dst.extend(digest32_as_fields(summary.chunk_relation_digest));
}

fn spartan_pow(mut base: SpartanF, mut exp: u64) -> SpartanF {
    let mut acc = SpartanF::from_canonical_u64(1);
    while exp != 0 {
        if (exp & 1) == 1 {
            acc = acc * base;
        }
        base = base * base;
        exp >>= 1;
    }
    acc
}

pub(super) fn spartan_inverse(value: SpartanF) -> Option<SpartanF> {
    if value.to_canonical_u64() == 0 {
        return None;
    }
    Some(spartan_pow(value, SPARTAN_GOLDILOCKS_MODULUS - 2))
}
