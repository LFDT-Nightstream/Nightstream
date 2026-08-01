//! `EncInst` — Construction 2 step 4 b (bit-decomposition of `x_out`).
//!
//! Decomposes a 32-byte digest into 256 binary low-norm field elements.
//! Norm bound `‖x‖_∞ < 2` by construction.

/// Bit-decomposition of a 32-byte digest, packed for use as F's public input.
#[derive(Clone, Debug, PartialEq, Eq, serde::Deserialize, serde::Serialize)]
pub struct EncInst {
    pub(crate) digest_bytes: [u8; 32],
}

impl EncInst {
    pub fn from_digest(digest_bytes: [u8; 32]) -> Self {
        Self { digest_bytes }
    }

    /// 256 binary field elements `x_j ∈ {0,1}` packed by little-endian bit
    /// order from the digest bytes.
    pub fn bits(&self) -> [u8; 256] {
        let mut out = [0u8; 256];
        for j in 0..256 {
            out[j] = (self.digest_bytes[j / 8] >> (j % 8)) & 1;
        }
        out
    }
}
