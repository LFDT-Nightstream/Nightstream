use std::sync::{Arc, Mutex, PoisonError, Weak};

use neo_math::signed_sums::SplitRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{production_key_pair, D, PRODUCTION_VERIFIER_ROWS};

/// SPEC.md Algorithm 1, SPEC.md Algorithm 11, step 3.
pub struct ExpandedKey {
    blocks: u64,
    words: Vec<u64>,
    #[cfg(not(target_arch = "wasm32"))]
    crt: super::crt::Cache,
}

// SPEC.md Algorithm 11, step 6.
static SHARED: Mutex<Weak<ExpandedKey>> = Mutex::new(Weak::new());

impl ExpandedKey {
    pub fn production(blocks: u64) -> Arc<Self> {
        let key = Arc::new(Self::of_pairs(blocks, |row, pair| {
            production_key_pair(row, [2 * pair, 2 * pair + 1])
        }));
        let mut shared = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        if shared.upgrade().is_none_or(|kept| kept.blocks < key.blocks) {
            *shared = Arc::downgrade(&key);
        }
        key
    }

    pub fn of_pairs(blocks: u64, pair: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync) -> Self {
        let pairs = blocks.div_ceil(2);
        let tasks = 0..PRODUCTION_VERIFIER_ROWS * pairs;
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        let tasks = tasks.into_par_iter();
        let words = tasks
            .map(|task| pair((task / pairs) as u32, task % pairs))
            .collect::<Vec<_>>()
            .into_flattened()
            .into_flattened();
        Self {
            blocks: 2 * pairs,
            words,
            #[cfg(not(target_arch = "wasm32"))]
            crt: super::crt::Cache::new(2 * pairs),
        }
    }

    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub(super) fn shared() -> Option<Arc<Self>> {
        SHARED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .upgrade()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn crt_tiles(&self) -> usize {
        self.crt.tiles()
    }

    #[cfg(not(target_arch = "wasm32"))]
    pub(super) fn crt_tile(&self, index: usize) -> Option<&super::crt::Tile> {
        self.crt.tile(index, &self.words, self.blocks)
    }

    pub(super) fn element(&self, row: u32, block: u64) -> SplitRing {
        let start = (u64::from(row) * self.blocks + block) as usize * D;
        SplitRing::from_u64(
            self.words[start..start + D]
                .try_into()
                .expect("one key element"),
        )
    }
}
