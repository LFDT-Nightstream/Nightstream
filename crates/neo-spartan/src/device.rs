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
    /// Every digest layer of the binary Merkle tree over `rows` rows of
    /// `width` words (`rows` a power of two), leaves first and the root
    /// last. A leaf is Plonky3's overwrite sponge (`PaddingFreeSponge`) of
    /// its row over the workspace Poseidon2 (width 16, rate 12, four
    /// outputs): each block of up to 12 words overwrites the first lanes of
    /// the zero state and is followed by one permutation. A node is the same
    /// sponge of its two children's eight words, which is Plonky3's
    /// `TruncatedPermutation` compression. `fill` writes the rows back to
    /// back into a buffer of `rows * width` words; words may be
    /// non-canonical, digests are canonical. `None` when the device cannot
    /// build the tree: the caller builds it on the CPU, with the same
    /// digests.
    fn merkle_layers(
        &self,
        rows: usize,
        width: usize,
        fill: &(dyn Fn(&mut [u64]) + Sync),
    ) -> Option<Vec<Vec<[u64; DIGEST_LEN]>>>;

    /// The leaf digests of the rows of `width` words in `values`: the leaves
    /// of `merkle_layers`, and no layer above them. Words may be
    /// non-canonical, digests are canonical. `None` when the device cannot
    /// hash them: the caller hashes them on the CPU, with the same digests.
    fn merkle_leaves(&self, values: &[u64], width: usize) -> Option<Vec<[u64; DIGEST_LEN]>>;

    /// `Σ_i powers[i]·a_{i,b}` for every block `b < blocks`, with `a_{i,b}`
    /// the `D` coefficients of row `i` of the production commitment key
    /// (`neo_ajtai::nightstream_fprime_setup::coefficient_block(&PRODUCTION_SEED, i, b)`)
    /// and `i < powers.len()`: `D` values per block, block by block. Powers
    /// and values are cubic-extension elements as their three base
    /// coordinates. `None` when the device cannot compute them: the caller
    /// computes them on the CPU, with the same values.
    fn key_weights(&self, powers: &[[u64; 3]], blocks: usize) -> Option<Vec<[u64; 3]>>;

    /// The DFT of every column of the row-major matrix `values` of `width`
    /// columns, in place: row `k` becomes `Σ_i row_i·ω^{ik}` with `ω` =
    /// `generator` of order the row count (a power of two). Inputs may be
    /// non-canonical; outputs are canonical. `Ok(false)` when the device did
    /// not run it, with `values` unchanged: the caller runs it on the CPU.
    /// `Err` when it failed after it started: `values` are undefined and the
    /// proof fails.
    fn dft_columns(&self, values: &mut [u64], width: usize, generator: u64) -> Result<bool, String>;

    /// The rounds of one GKR layer over the caller's tables, in place: per
    /// tree its parts (at most two factors, then the denominator; `4·n`
    /// cubic-extension elements of three words each), and the eq table (`2·n`
    /// elements). After `r` folds, logical entry `j` of a table sits at
    /// element `j·2^r`. `None` when the device cannot run the layer, with
    /// nothing changed: the caller runs it on the CPU.
    fn gkr_layer<'a>(&self, trees: Vec<Vec<&'a mut [u64]>>, eq: &'a mut [u64]) -> Option<Box<dyn GkrLayer + 'a>>;
}

/// The rounds of one batched fraction-sum GKR layer on a device
/// (`gkr::prove` draws every challenge). Values are cubic-extension elements
/// as three base words. A device failure here fails the proof: the
/// transcript has moved on, so no CPU fallback can resume.
pub trait GkrLayer {
    /// The round message at nodes `0, 2, ..., degree`, one weight per tree.
    fn message(&mut self, weights: &[[u64; 3]], mu: [u64; 3], degree: usize) -> Result<Vec<[u64; 3]>, String>;
    /// Bind the lowest variable of every table and the eq table to `r`.
    fn fold(&mut self, r: [u64; 3]) -> Result<(), String>;
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
