//! The memory carry of spec §11.1 and its 39 words. Mirrors
//! `Lifecycle/Nebula/Carry.lean` (`carryVector`).

use neo_math::K;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing};
use p3_goldilocks::Goldilocks as F;

use super::transcript::Digest;

/// The four running products: read, write, initial, final.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Products {
    pub read: K,
    pub write: K,
    pub initial: K,
    pub final_: K,
}

impl Products {
    pub fn one() -> Self {
        Self {
            read: K::ONE,
            write: K::ONE,
            initial: K::ONE,
            final_: K::ONE,
        }
    }
}

/// Spec §11.1 `MemoryCarry`.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct Carry {
    pub seg_idx: u64,
    pub idx: u64,
    pub ts: u64,
    pub eta: (K, K),
    pub products: Products,
    pub proposed: (Digest, Digest),
    pub seen: [Digest; 3],
    pub mem_root: Digest,
}

pub fn k_words(value: K) -> [F; 2] {
    let coefficients = value.as_basis_coefficients_slice();
    [coefficients[0], coefficients[1]]
}

impl Carry {
    /// The closed start carry of spec §11.1: `idx = N`, every root `D_init`.
    pub fn start(n: usize, d_init: Digest) -> Self {
        Self {
            seg_idx: 0,
            idx: n as u64,
            ts: 0,
            eta: (K::ZERO, K::ZERO),
            products: Products::one(),
            proposed: (d_init, d_init),
            seen: [d_init; 3],
            mem_root: d_init,
        }
    }

    /// The 39 carry words in spec §11.1 order.
    pub fn words(&self) -> [F; 39] {
        let mut words = Vec::with_capacity(39);
        words.push(F::from_u64(self.seg_idx));
        words.push(F::from_u64(self.idx));
        words.push(F::from_u64(self.ts));
        for value in [
            self.eta.0,
            self.eta.1,
            self.products.read,
            self.products.write,
            self.products.initial,
            self.products.final_,
        ] {
            words.extend(k_words(value));
        }
        words.extend(self.proposed.0);
        words.extend(self.proposed.1);
        for digest in self.seen {
            words.extend(digest);
        }
        words.extend(self.mem_root);
        words.try_into().expect("39 carry words")
    }
}
