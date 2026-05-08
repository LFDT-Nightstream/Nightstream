use crate::spartan_backend::SpartanF;
use bellpepper_core::{boolean::Boolean, num::AllocatedNum, ConstraintSystem, SynthesisError};

use super::U16S_PER_DIGEST32;

pub(super) fn digest_u16_words<CS: ConstraintSystem<SpartanF>>(
    mut cs: CS,
    digest: &[AllocatedNum<SpartanF>; 4],
    digest_values: &[SpartanF; 4],
    label: &str,
) -> Result<Vec<(Vec<Boolean>, u16)>, SynthesisError> {
    let mut out = Vec::with_capacity(U16S_PER_DIGEST32);
    for (limb_idx, limb) in digest.iter().enumerate() {
        let bits = limb.to_bits_le_strict(cs.namespace(|| format!("{label}_bits_{limb_idx}")))?;
        for word_idx in 0..4 {
            let start = word_idx * 16;
            let end = start + 16;
            let word_bits = bits[start..end].to_vec();
            let limb_value = digest_values[limb_idx].to_canonical_u64();
            let word_value = ((limb_value >> start) & 0xFFFF) as u16;
            out.push((word_bits, word_value));
        }
    }
    Ok(out)
}
