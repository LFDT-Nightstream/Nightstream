use std::collections::BTreeMap;
use std::sync::{Arc, Mutex, PoisonError, Weak};

use neo_math::signed_sums::SplitRing;
#[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
use rayon::prelude::*;

use super::{production_key_pair, D, PRODUCTION_VERIFIER_ROWS};

/// SPEC.md Algorithm 1, SPEC.md Algorithm 11, step 3.
pub struct ExpandedKey {
    blocks: u64,
    words: Arc<Vec<u64>>,
}

#[derive(Default)]
struct SharedWidth {
    ready: Weak<ExpandedKey>,
    building: Weak<()>,
}

// SPEC.md Algorithm 11, step 6.
static SHARED: Mutex<BTreeMap<u64, SharedWidth>> = Mutex::new(BTreeMap::new());

impl ExpandedKey {
    pub fn production(blocks: u64) -> Arc<Self> {
        let rounded = blocks.next_multiple_of(2);
        Self::production_in(rounded, rounded, false).expect("expanded production key allocation")
    }

    pub fn within_budget(blocks: u64, max_bytes: usize) -> Option<Arc<Self>> {
        let rounded = blocks.checked_next_multiple_of(2)?;
        let max_blocks = (max_bytes / (PRODUCTION_VERIFIER_ROWS as usize * D * size_of::<u64>())) as u64;
        if rounded == 0 || rounded > max_blocks {
            return None;
        }
        Self::production_in(rounded, max_blocks, true)
    }

    fn production_in(blocks: u64, max_blocks: u64, allow_streaming: bool) -> Option<Arc<Self>> {
        let find = |keys: &BTreeMap<u64, SharedWidth>| {
            keys.range(blocks..=max_blocks)
                .find_map(|(_, width)| width.ready.upgrade())
        };
        let _building = {
            let mut keys = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
            if let Some(key) = find(&keys) {
                return Some(key);
            }
            if allow_streaming
                && keys
                    .range(blocks..=max_blocks)
                    .any(|(_, width)| width.building.strong_count() != 0)
            {
                return None;
            }
            keys.retain(|_, width| width.ready.strong_count() != 0 || width.building.strong_count() != 0);
            let width = keys.entry(blocks).or_default();
            width.building.upgrade().unwrap_or_else(|| {
                let building = Arc::new(());
                width.building = Arc::downgrade(&building);
                building
            })
        };
        let key = Arc::new(Self::try_of_pairs(blocks, |row, pair| {
            production_key_pair(row, [2 * pair, 2 * pair + 1])
        })?);
        let mut keys = SHARED.lock().unwrap_or_else(PoisonError::into_inner);
        if let Some(existing) = find(&keys) {
            return Some(existing);
        }
        keys.get_mut(&blocks)?.ready = Arc::downgrade(&key);
        Some(key)
    }

    pub fn of_pairs(blocks: u64, pair: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync) -> Self {
        Self::try_of_pairs(blocks, pair).expect("expanded key allocation")
    }

    fn try_of_pairs(blocks: u64, pair: impl Fn(u32, u64) -> [[u64; D]; 2] + Sync) -> Option<Self> {
        let pairs = usize::try_from(blocks.div_ceil(2)).ok()?;
        let count = pairs.checked_mul(PRODUCTION_VERIFIER_ROWS as usize)?;
        let mut words = Vec::new();
        words.try_reserve_exact(count).ok()?;
        let expand = |task: usize| pair((task / pairs) as u32, (task % pairs) as u64);
        #[cfg(any(not(target_arch = "wasm32"), feature = "wasm-threads"))]
        (0..count)
            .into_par_iter()
            .map(expand)
            .collect_into_vec(&mut words);
        #[cfg(all(target_arch = "wasm32", not(feature = "wasm-threads")))]
        words.extend((0..count).map(expand));
        Some(Self {
            blocks: 2 * pairs as u64,
            words: Arc::new(words.into_flattened().into_flattened()),
        })
    }

    pub fn blocks(&self) -> u64 {
        self.blocks
    }

    pub fn words(&self) -> &[u64] {
        &self.words
    }

    pub fn shared_words(&self) -> Arc<Vec<u64>> {
        self.words.clone()
    }

    pub fn shared() -> Option<Arc<Self>> {
        SHARED
            .lock()
            .unwrap_or_else(PoisonError::into_inner)
            .values()
            .rev()
            .find_map(|width| width.ready.upgrade())
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
