use rand_chacha::rand_core::{CryptoRng, RngCore};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
use neo_math::F;
use crate::poseidon2::Poseidon2Transcript;
use neo_ccs::crypto::poseidon2_goldilocks as p2;

#[derive(Clone)]
pub struct TranscriptRngBuilder { st: [Goldilocks; p2::WIDTH], rate_idx: usize }
#[derive(Clone)]
pub struct TranscriptRng { st: [Goldilocks; p2::WIDTH] }

impl TranscriptRngBuilder {
    pub fn from_transcript(tr: &Poseidon2Transcript) -> Self {
        Self { st: tr.state(), rate_idx: 0 }
    }
    pub fn from_state(state: [Goldilocks; p2::WIDTH]) -> Self { Self { st: state, rate_idx: 0 } }

    #[inline] fn absorb_elem(&mut self, u: u64) {
        if self.rate_idx >= p2::RATE { self.permute(); }
        self.st[self.rate_idx] = Goldilocks::from_u64(u);
        self.rate_idx += 1;
    }
    #[inline] fn permute(&mut self) {
        self.st = p2::permutation().permute(self.st);
        self.rate_idx = 0;
    }

    pub fn rekey_with_witness_fields(mut self, label: &'static [u8], ws: &[F]) -> Self {
        for &b in label { self.absorb_elem(b as u64); }
        self.absorb_elem(ws.len() as u64);
        for &w in ws { self.absorb_elem(w.as_canonical_u64()); }
        self.permute();
        self
    }
    pub fn finalize<R: RngCore + CryptoRng>(mut self, rng: &mut R) -> TranscriptRng {
        // Mix external entropy as 32 bytes
        let mut seed = [0u8; 32];
        rng.fill_bytes(&mut seed);
        for chunk in seed.chunks(8) {
            let mut b = [0u8; 8]; b[..chunk.len()].copy_from_slice(chunk);
            self.absorb_elem(u64::from_le_bytes(b));
        }
        // Squeeze gate before first RNG output: absorb a sentinel then permute
        self.absorb_elem(1);
        self.permute();
        TranscriptRng { st: self.st }
    }
}

impl TranscriptRng {
    pub fn fill_bytes(&mut self, out: &mut [u8]) {
        let mut produced = 0;
        while produced < out.len() {
            self.st = p2::permutation().permute(self.st);
            for i in 0..4 {
                let limb = self.st[i].as_canonical_u64().to_le_bytes();
                let take = out.len().saturating_sub(produced).min(8);
                out[produced..produced+take].copy_from_slice(&limb[..take]);
                produced += take;
                if produced >= out.len() { break; }
            }
        }
    }
    pub fn field(&mut self) -> F {
        let mut b = [0u8; 8];
        self.fill_bytes(&mut b);
        F::from_u64(u64::from_le_bytes(b))
    }
}
