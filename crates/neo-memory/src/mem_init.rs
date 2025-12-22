use neo_reductions::error::PiCcsError;
use p3_field::PrimeCharacteristicRing;
use serde::{Deserialize, Serialize};
use std::collections::BTreeSet;

/// Public initial memory state for a Twist instance.
///
/// This is intentionally compact to support large memories without requiring a dense `Vec<F>`.
#[derive(Clone, Debug, Serialize, Deserialize, PartialEq)]
pub enum MemInit<F> {
    /// All cells start at 0.
    Zero,
    /// A small set of non-zero initial cells (addr -> value).
    ///
    /// Canonical form: strictly increasing addresses, no duplicates, and all addresses < `k`.
    Sparse(Vec<(u64, F)>),
}

impl<F> Default for MemInit<F> {
    fn default() -> Self {
        Self::Zero
    }
}

impl<F: PrimeCharacteristicRing> MemInit<F> {
    pub fn validate(&self, k: usize) -> Result<(), PiCcsError> {
        match self {
            MemInit::Zero => Ok(()),
            MemInit::Sparse(pairs) => {
                let mut seen = BTreeSet::<u64>::new();
                let mut prev: Option<u64> = None;
                for (addr, _val) in pairs {
                    if let Some(prev) = prev {
                        if *addr <= prev {
                            return Err(PiCcsError::InvalidInput(
                                "MemInit::Sparse must be strictly increasing by address".into(),
                            ));
                        }
                    }
                    prev = Some(*addr);
                    if (*addr as usize) >= k {
                        return Err(PiCcsError::InvalidInput(format!(
                            "MemInit::Sparse address out of range: addr={} >= k={}",
                            addr, k
                        )));
                    }
                    if !seen.insert(*addr) {
                        return Err(PiCcsError::InvalidInput(
                            "MemInit::Sparse must not contain duplicate addresses".into(),
                        ));
                    }
                }
                Ok(())
            }
        }
    }
}

/// Evaluate the multilinear extension of the initial memory table at `r_addr`.
///
/// For `Sparse`, this runs in `O(nnz * ell_addr)` time.
pub fn eval_init_at_r_addr<F, K>(init: &MemInit<F>, k: usize, r_addr: &[K]) -> Result<K, PiCcsError>
where
    F: PrimeCharacteristicRing + Copy,
    K: PrimeCharacteristicRing + From<F> + Copy,
{
    init.validate(k)?;

    match init {
        MemInit::Zero => Ok(K::ZERO),
        MemInit::Sparse(pairs) => {
            if r_addr.len() > 64 && !pairs.is_empty() {
                return Err(PiCcsError::InvalidInput(format!(
                    "MemInit::Sparse only supports up to 64 address bits (got ell_addr={})",
                    r_addr.len()
                )));
            }

            let one = K::ONE;
            let mut acc = K::ZERO;

            for (addr, val_f) in pairs.iter() {
                let mut chi = one;
                for (bit_idx, &r) in r_addr.iter().enumerate() {
                    let bit = ((*addr >> bit_idx) & 1) as u8;
                    chi *= if bit == 1 { r } else { one - r };
                }
                acc += K::from(*val_f) * chi;
            }

            Ok(acc)
        }
    }
}
