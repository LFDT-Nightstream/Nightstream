use neo_modint::ModInt;
use neo_ring::RingElement;
use crate::fiat_shamir::{fiat_shamir_challenge, fiat_shamir_challenge_base};
use crate::{ExtF, F};
use p3_field::PrimeField64;

pub struct NeoChallenger {
    transcript: Vec<u8>,
}

impl NeoChallenger {
    pub fn new(protocol_id: &str) -> Self {
        let mut this = Self { transcript: Vec::new() };
        this.observe_bytes("neo_version", b"v1.0");
        this.observe_bytes("protocol_id", protocol_id.as_bytes());
        this.observe_bytes("field_id", b"Goldilocks");
        this
    }

    pub fn observe_field(&mut self, label: &str, x: &F) {
        self.observe_bytes(label, &x.as_canonical_u64().to_be_bytes());
    }

    pub fn observe_ext(&mut self, label: &str, x: &ExtF) {
        let [r, i] = x.to_array();
        let mut bytes = r.as_canonical_u64().to_be_bytes().to_vec();
        bytes.extend(i.as_canonical_u64().to_be_bytes());
        self.observe_bytes(label, &bytes);
    }

    /// Observe arbitrary labeled bytes with length prefixing for unambiguous framing.
    pub fn observe_bytes(&mut self, label: &str, bytes: &[u8]) {
        let mut framed = label.as_bytes().to_vec();
        framed.extend((bytes.len() as u64).to_be_bytes());
        framed.extend_from_slice(bytes);
        self.transcript.extend_from_slice(&framed);
        // Hash framed for FS derivation (stateless helper); keeps semantics aligned with diff
        let _ = fiat_shamir_challenge_base(&framed);
    }

    /// Base-field challenge derived by hashing current transcript.
    pub fn challenge_base(&mut self, label: &str) -> F {
        self.observe_bytes("challenge_label", label.as_bytes());
        fiat_shamir_challenge_base(&self.transcript)
    }

    /// Extension-field challenge derived by hashing current transcript.
    pub fn challenge_ext(&mut self, label: &str) -> ExtF {
        self.observe_bytes("challenge_label", label.as_bytes());
        fiat_shamir_challenge(&self.transcript)
    }

    /// Squeeze a vector of extension-field challenges.
    pub fn challenge_vec_in_k(&mut self, label: &str, len: usize) -> Vec<ExtF> {
        self.observe_bytes("vec_label", label.as_bytes());
        (0..len).map(|_| self.challenge_ext("vec_elem")).collect()
    }

    /// Deterministically derive an invertible "rotation" element ρ ∈ R = Z_q[X]/(X^n+1)
    /// from the transcript. We return ±X^j, which is always invertible in R.
    pub fn challenge_rotation(&mut self, label: &str, n: usize) -> RingElement<ModInt> {
        // Domain-separate and derive one base-field limb
        self.observe_bytes("rotation_label", label.as_bytes());
        let limb = self.challenge_base(&format!("{label}|rot_j")).as_canonical_u64() as usize;

        // Map into {0, …, 2n-1}. Indices in [n, 2n) correspond to a negative sign due to X^n ≡ -1.
        let m = 2 * n;
        let j = if m > 0 { limb % m } else { 0 };

        // Build ±X^j as a ring element
        let mut coeffs = vec![ModInt::from_u64(0); n];
        if j < n {
            coeffs[j] = ModInt::from_u64(1);                 //  +X^j
        } else {
            coeffs[j - n] = ModInt::from_u64(ModInt::Q - 1); //  -X^{j-n}
        }
        let rot = RingElement::from_coeffs(coeffs, n);

        debug_assert!(rot.is_invertible(), "constructed rotation should be invertible");
        rot
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn challenge_rotation_is_always_invertible() {
        let mut ch = NeoChallenger::new("test_rot");
        ch.observe_bytes("prefix", b"unit");
        for n in [2usize, 4, 8, 64] {
            for _ in 0..10 {
                let rho = ch.challenge_rotation("rlc_rho", n);
                assert!(rho.is_invertible(), "ρ must be invertible for n={}", n);
            }
        }
    }
}

