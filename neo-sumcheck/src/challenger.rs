use neo_modint::ModInt;
use neo_ring::RingElement;
use crate::fiat_shamir::{fiat_shamir_challenge, fiat_shamir_challenge_base};
use crate::{ExtF, F};
use p3_field::PrimeField64;

const SMALL_COEFF_BOUND: i64 = 2; // [-2..2]

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

    /// Squeeze a rotation element in the cyclotomic ring with small coefficients.
    pub fn challenge_rotation(&mut self, label: &str, n: usize) -> RingElement<ModInt> {
        self.observe_bytes("rotation_label", label.as_bytes());
        let squeezed_bytes = (0..n)
            .map(|_| self.challenge_base("rotation_byte").as_canonical_u64() as u8)
            .collect::<Vec<_>>();

        let mut coeffs = vec![ModInt::from_u64(0); n];
        for (i, &byte) in squeezed_bytes.iter().enumerate() {
            let signed = (byte as i64 % (2 * SMALL_COEFF_BOUND + 1)) - SMALL_COEFF_BOUND;
            coeffs[i] = ModInt::from(signed as i128);
        }

        let a = RingElement::from_coeffs(coeffs, n);
        for _ in 0..10 {
            if a.is_invertible() {
                return a.rotate(1);
            }
            let _ = self.challenge_ext("retry_invert");
        }
        panic!("Failed to sample invertible rotation");
    }
}


