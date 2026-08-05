//! `VerifierKey` — vk_fs (Construction 2 verifier key).
//!
//! The fold-scheme verifier key, hashed and re-derivable by the verifier
//! from the `Structure`, parameters, and verifier-owned Ajtai setup. This
//! struct's `digest()` is the only thing the hash chain reads.

use neo_math::F;
use thiserror::Error;

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
    pi_ccs_header_bundle: [F; 4],
}

#[derive(Debug, Error)]
pub enum VerifierKeyError {
    #[error("failed to derive the canonical SuperNeo verifier header: {0}")]
    Header(String),
}

impl VerifierKey {
    /// Derive vk_fs from `(params, structure, ajtai_pp_digest,
    /// public_input_len, initial_semantic_state_digest)`. Wraps
    /// [`digest::vk_fs_digest`]; this is **Soundness Invariant I-5**.
    ///
    /// `initial_semantic_state_digest` is the chain's claimed starting
    /// app-state digest. Stateless chains MUST pass
    /// `digest::empty_semantic_state_digest()` so the seed matches what
    /// the stateless invariant carries natively. Stateful frontends pass
    /// `H(initial_app_state)` per their plan's
    /// `semantic_state_preimage_sources`.
    pub(crate) fn derive(
        pp: &Params,
        s: &Structure,
        ajtai_pp_digest: [F; 4],
        public_input_len: Option<usize>,
        initial_semantic_state_digest: [u8; 32],
    ) -> Result<Self, VerifierKeyError> {
        let pi_ccs_header_bundle = neo_reductions::engines::utils::digest_ccs_matrices(s)
            .try_into()
            .expect("the PiCCS matrix digest has four fields");
        Ok(Self::derive_from_structure_digest(
            pp,
            &digest::structure_digest(s),
            pi_ccs_header_bundle,
            ajtai_pp_digest,
            public_input_len,
            initial_semantic_state_digest,
        ))
    }

    /// Same as [`VerifierKey::derive`] but with the structure digest
    /// supplied directly. Avoids re-running `digest::structure_digest`
    /// when the caller already computed it (e.g. inside `preprocess`,
    /// which stores the digest on `Preprocessing`).
    pub(crate) fn derive_from_structure_digest(
        pp: &Params,
        structure_digest: &[F; 4],
        pi_ccs_header_bundle: [F; 4],
        ajtai_pp_digest: [F; 4],
        public_input_len: Option<usize>,
        initial_semantic_state_digest: [u8; 32],
    ) -> Self {
        Self {
            digest: digest::vk_fs_digest(
                pp.inner(),
                structure_digest,
                &pi_ccs_header_bundle,
                &ajtai_pp_digest,
                public_input_len,
                initial_semantic_state_digest,
            ),
            pi_ccs_header_bundle,
        }
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn pi_ccs_header_bundle(&self) -> [F; 4] {
        self.pi_ccs_header_bundle
    }
}
