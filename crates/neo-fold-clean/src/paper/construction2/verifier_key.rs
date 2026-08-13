//! `VerifierKey` — vk_fs (Construction 2 verifier key).
//!
//! The fold-scheme verifier key, hashed and re-derivable by the verifier
//! from the `Structure`, parameters, and verifier-owned Ajtai setup. This
//! struct's `digest()` is the only thing the hash chain reads.

use neo_math::F;
use thiserror::Error;

use crate::paper::digest;
use crate::paper::params::Params;

/// vk_fs (Construction 2): the fold-scheme verifier key, hashed and
/// re-derivable by the verifier.
///
/// **Auditor**: this struct's `digest()` is the only thing the hash chain
/// reads. Construction goes through [`VerifierKey::derive`].
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct VerifierKey {
    pub(crate) digest: [u8; 32],
    base_digest: [u8; 32],
    pi_ccs_header_bundle: [F; 4],
    initial_boundary_digest: [u8; 32],
    initial_public_trace: [u8; 32],
    initial_semantic_state_digest: [u8; 32],
}

#[derive(Debug, Error)]
pub enum VerifierKeyError {
    #[error("failed to derive the canonical SuperNeo verifier header: {0}")]
    Header(String),
}

impl VerifierKey {
    /// Derive the verifier key with the cached structure digest
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
        let base_digest = digest::vk_fs_digest(
            pp.inner(),
            structure_digest,
            &pi_ccs_header_bundle,
            &ajtai_pp_digest,
            public_input_len,
            initial_semantic_state_digest,
        );
        Self {
            digest: digest::vk_fs_policy_digest(base_digest, false, false, false),
            base_digest,
            pi_ccs_header_bundle,
            initial_boundary_digest: digest::initial_boundary_digest(structure_digest, public_input_len),
            initial_public_trace: digest::public_trace_seed_digest(structure_digest),
            initial_semantic_state_digest,
        }
    }

    pub(crate) fn with_policy(
        mut self,
        stateful: bool,
        f_prime_recursive_link: bool,
        terminal_induction: bool,
    ) -> Self {
        self.digest =
            digest::vk_fs_policy_digest(self.base_digest, stateful, f_prime_recursive_link, terminal_induction);
        self
    }

    pub fn digest(&self) -> [u8; 32] {
        self.digest
    }

    pub fn pi_ccs_header_bundle(&self) -> [F; 4] {
        self.pi_ccs_header_bundle
    }

    pub(crate) fn initial_boundary_digest(&self) -> [u8; 32] {
        self.initial_boundary_digest
    }

    pub(crate) fn initial_public_trace(&self) -> [u8; 32] {
        self.initial_public_trace
    }

    pub(crate) fn initial_semantic_state_digest(&self) -> [u8; 32] {
        self.initial_semantic_state_digest
    }
}
