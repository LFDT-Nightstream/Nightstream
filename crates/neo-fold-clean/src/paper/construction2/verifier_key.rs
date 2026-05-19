//! `VerifierKey` — vk_fs (Construction 2 verifier key).
//!
//! The fold-scheme verifier key, hashed and re-derivable by the verifier
//! from the `Structure` and parameters. This struct's `digest()` is the
//! only thing the hash chain reads.

use neo_math::F;

use crate::paper::digest;
use crate::paper::params::Params;
use crate::paper::relations::Structure;

/// vk_fs (Construction 2): the fold-scheme verifier key, hashed and
/// re-derivable by the verifier.
///
/// **Auditor**: this struct's `digest()` is the only thing the hash chain
/// reads. Construction goes through [`VerifierKey::derive`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKey {
    pub(crate) digest: [u8; 32],
}

impl VerifierKey {
    /// Derive vk_fs from `(params, structure, public_input_len)`. Wraps
    /// [`digest::vk_fs_digest`]; this is **Soundness Invariant I-5**.
    pub fn derive(pp: &Params, s: &Structure, public_input_len: Option<usize>) -> Self {
        Self::derive_from_structure_digest(pp, &digest::structure_digest(s), public_input_len)
    }

    /// Same as [`VerifierKey::derive`] but with the structure digest
    /// supplied directly. Avoids re-running `digest::structure_digest`
    /// when the caller already computed it (e.g. inside `preprocess`,
    /// which stores the digest on `Preprocessing`).
    pub fn derive_from_structure_digest(
        pp: &Params,
        structure_digest: &[F; 4],
        public_input_len: Option<usize>,
    ) -> Self {
        Self {
            digest: digest::vk_fs_digest(pp.inner(), structure_digest, public_input_len),
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }
}
