//! Where the prover's batch loops run.
//!
//! Owns: the `Device` contract, batch operations in `u64` words so that no
//! Plonky3 type crosses, and `Accelerator`, the prover's optional device. A
//! device changes where work runs, never a byte of a proof. Verifiers never
//! use one.

use std::sync::Arc;

use neo_ccs::crypto::poseidon2_goldilocks::DIGEST_LEN;

/// Batch operations that a GPU runs for the prover.
pub trait Device: Send + Sync {
    /// The Merkle leaf digests of the `words.len() / width` rows of `width`
    /// words that sit back to back in `words`: Plonky3's overwrite sponge
    /// (`PaddingFreeSponge`) over the workspace Poseidon2 (width 16, rate
    /// 12, four outputs). Each row starts from the zero state; each block of
    /// up to 12 words overwrites the first lanes and is followed by one
    /// permutation. Input words may be non-canonical; digests are canonical.
    /// `None` when the device cannot run the batch: the caller then hashes
    /// it on the CPU, with the same digests.
    fn hash_rows(&self, words: &[u64], width: usize) -> Option<Vec<[u64; DIGEST_LEN]>>;
}

/// The prover's device, or none: the CPU.
#[derive(Clone, Default)]
pub struct Accelerator(Option<Arc<dyn Device>>);

impl Accelerator {
    pub fn cpu() -> Self {
        Self(None)
    }

    pub fn new(device: Arc<dyn Device>) -> Self {
        Self(Some(device))
    }

    pub(crate) fn device(&self) -> Option<&dyn Device> {
        self.0.as_deref()
    }
}
