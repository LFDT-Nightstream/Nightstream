use neo_modint::ModInt;
use neo_ring::RingElement;
use crate::{ExtF, F};
use p3_field::{PrimeField64, PrimeCharacteristicRing};
use p3_challenger::{DuplexChallenger, CanObserve, CanSample, FieldChallenger};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};

// Type alias for our specific Poseidon2 configuration
type NeoDuplexChallenger = DuplexChallenger<Goldilocks, Poseidon2Goldilocks<16>, 16, 15>;

pub struct NeoChallenger {
    challenger: NeoDuplexChallenger,
}

impl NeoChallenger {
    pub fn new(protocol_id: &str) -> Self {
        // Use deterministic seed for reproducible Poseidon2 parameters
        use rand::SeedableRng;
        let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
        let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
        let mut challenger = DuplexChallenger::new(poseidon2);
        
        // Domain separation: absorb protocol metadata as field elements
        let protocol_fields = Self::bytes_to_fields(protocol_id.as_bytes());
        challenger.observe_slice(&protocol_fields);
        
        let version_fields = Self::bytes_to_fields(b"neo_v1.0");
        challenger.observe_slice(&version_fields);
        
        let field_id_fields = Self::bytes_to_fields(b"Goldilocks");
        challenger.observe_slice(&field_id_fields);
        
        Self { challenger }
    }
    
    /// Helper: Convert bytes to field elements (8 bytes per element)
    fn bytes_to_fields(bytes: &[u8]) -> Vec<F> {
        bytes.chunks(8).map(|chunk| {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            F::from_u64(u64::from_le_bytes(buf))
        }).collect()
    }

    pub fn observe_field(&mut self, label: &str, x: &F) {
        // Absorb label first for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        // Then absorb the field element
        self.challenger.observe(*x);
    }

    pub fn observe_ext(&mut self, label: &str, x: &ExtF) {
        // Absorb label first for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        // Then absorb the extension field element
        self.challenger.observe_algebra_element(*x);
    }

    /// Observe arbitrary labeled bytes with structured framing
    pub fn observe_bytes(&mut self, label: &str, bytes: &[u8]) {
        // Absorb label first for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        // Then absorb the bytes as field elements
        let byte_fields = Self::bytes_to_fields(bytes);
        self.challenger.observe_slice(&byte_fields);
    }

    /// Base-field challenge derived from current challenger state
    pub fn challenge_base(&mut self, label: &str) -> F {
        // Absorb challenge label for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        // Sample base field element
        self.challenger.sample()
    }

    /// Extension-field challenge derived from current challenger state
    pub fn challenge_ext(&mut self, label: &str) -> ExtF {
        // Absorb challenge label for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        // Sample extension field element
        self.challenger.sample_algebra_element()
    }

    /// Squeeze a vector of extension-field challenges.
    pub fn challenge_vec_in_k(&mut self, label: &str, len: usize) -> Vec<ExtF> {
        // Absorb label and length for domain separation
        let label_fields = Self::bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        self.challenger.observe(F::from_u64(len as u64));
        
        // Sample the requested number of extension field elements
        (0..len).map(|_| self.challenger.sample_algebra_element()).collect()
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



