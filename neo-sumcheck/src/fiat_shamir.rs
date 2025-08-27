use crate::{ExtF, F, Polynomial};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_challenger::{DuplexChallenger, CanObserve, CanSample, FieldChallenger};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use rand_chacha::{ChaCha20Rng, rand_core::SeedableRng};
use neo_modint::{Coeff, ModInt};

/// Type aliases for p3-challenger with Poseidon2
type FiatShamirChallenger = DuplexChallenger<Goldilocks, Poseidon2Goldilocks<16>, 16, 15>;

/// Helper: Convert bytes to field elements for p3-challenger
fn bytes_to_fields(bytes: &[u8]) -> Vec<F> {
    bytes.chunks(8).map(|chunk| {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        F::from_u64(u64::from_le_bytes(buf))
    }).collect()
}

/// Create a fresh challenger for Fiat-Shamir
fn create_challenger() -> FiatShamirChallenger {
    // Use deterministic seed for reproducible Poseidon2 parameters
    use rand::SeedableRng;
    let mut rng = rand_chacha::ChaCha20Rng::from_seed([0u8; 32]);
    let poseidon2 = Poseidon2Goldilocks::<16>::new_from_rng_128(&mut rng);
    DuplexChallenger::new(poseidon2)
}

/// Generate a cryptographic challenge from the current transcript using Fiat-Shamir
///
/// # Arguments
/// * `transcript` - Byte array containing all protocol messages so far
///
/// # Returns
/// A pseudo-random field element derived from the transcript
pub fn fiat_shamir_challenge(transcript: &[u8]) -> ExtF {
    let mut challenger = create_challenger();
    
    // Absorb transcript as field elements
    let field_elems = bytes_to_fields(transcript);
    challenger.observe_slice(&field_elems);
    
    // Sample extension field element
    challenger.sample_algebra_element()
}

/// Generate a base-field Fiat-Shamir challenge from the transcript.
pub fn fiat_shamir_challenge_base(transcript: &[u8]) -> F {
    let mut challenger = create_challenger();
    
    // Absorb transcript as field elements
    let field_elems = bytes_to_fields(transcript);
    challenger.observe_slice(&field_elems);
    
    // Sample base field element
    challenger.sample()
}

/// Combine multiple univariate polynomials into a single batched polynomial
/// using a random linear combination with powers of ρ
///
/// This is a key optimization that allows proving multiple sum-check instances
/// simultaneously rather than proving each one individually.
///
/// # Arguments
/// * `unis` - Array of univariate polynomials to batch
/// * `rho` - Random batching coefficient
///
/// # Returns
/// Batched polynomial = Σᵢ ρⁱ · unis[i]
pub fn batch_unis(unis: &[Polynomial<ExtF>], rho: ExtF) -> Polynomial<ExtF> {
    let max_deg = unis.iter().map(|u| u.degree()).max().unwrap_or(0);
    let mut batched = Polynomial::new(vec![ExtF::ZERO; max_deg + 1]);
    let mut current_rho = ExtF::ONE;

    // Linear combination: batched = Σᵢ ρⁱ · unis[i]
    for uni in unis {
        let scaled = uni.clone() * Polynomial::new(vec![current_rho]);
        batched = batched + scaled;
        current_rho *= rho;
    }
    batched
}

// --- NEW: Structured, domain-separated FS absorption helpers ---

#[inline]
fn fs_write_u32_be(buf: &mut Vec<u8>, v: u32) { buf.extend_from_slice(&v.to_be_bytes()); }

#[inline]
fn fs_begin_domain(buf: &mut Vec<u8>, label: &[u8]) {
    // Frame: "FS" | len(label) | label
    buf.extend_from_slice(b"FS");
    fs_write_u32_be(buf, label.len() as u32);
    buf.extend_from_slice(label);
}

pub fn fs_absorb_bytes(transcript: &mut Vec<u8>, label: &[u8], bytes: &[u8]) {
    fs_begin_domain(transcript, label);
    fs_write_u32_be(transcript, bytes.len() as u32);
    transcript.extend_from_slice(bytes);
}

pub fn fs_absorb_u64(transcript: &mut Vec<u8>, label: &[u8], x: u64) {
    fs_begin_domain(transcript, label);
    fs_write_u32_be(transcript, 8);
    transcript.extend_from_slice(&x.to_be_bytes());
}

pub fn fs_absorb_extf(transcript: &mut Vec<u8>, label: &[u8], x: ExtF) {
    fs_begin_domain(transcript, label);
    // two limbs (real, imag) as u64 be
    fs_write_u32_be(transcript, 16);
    let a = x.to_array();
    transcript.extend_from_slice(&a[0].as_canonical_u64().to_be_bytes());
    transcript.extend_from_slice(&a[1].as_canonical_u64().to_be_bytes());
}

pub fn fs_absorb_poly(transcript: &mut Vec<u8>, label: &[u8], p: &Polynomial<ExtF>) {
    fs_begin_domain(transcript, label);
    // encode degree & coeff count for unambiguous framing
    let deg = p.degree() as u32;
    let coeffs = p.coeffs();
    let count = coeffs.len() as u32;
    // header: deg | count | payload_size
    fs_write_u32_be(transcript, 12); // 12 bytes follow in "payload header"
    fs_write_u32_be(transcript, deg);
    fs_write_u32_be(transcript, count);
    // then each coefficient (two limbs) - use checked_mul to prevent overflow
    let payload_size = count.checked_mul(16).unwrap_or_else(|| {
        // Cap at max u32 for very large polynomials (extremely unlikely in practice)
        eprintln!("Warning: polynomial too large for u32 framing, capping payload size");
        u32::MAX
    });
    fs_write_u32_be(transcript, payload_size);
    for &c in coeffs {
        let a = c.to_array();
        transcript.extend_from_slice(&a[0].as_canonical_u64().to_be_bytes());
        transcript.extend_from_slice(&a[1].as_canonical_u64().to_be_bytes());
    }
}

pub fn fs_challenge_ext(transcript: &mut Vec<u8>, label: &[u8]) -> ExtF {
    fs_begin_domain(transcript, b"challenge");
    fs_absorb_bytes(transcript, b"dst", label);
    // Hash **all** accumulated bytes (including the structured frame we just wrote)
    fiat_shamir_challenge(transcript)
}

pub fn fs_challenge_base(transcript: &mut Vec<u8>, label: &[u8]) -> F {
    let e = fs_challenge_ext(transcript, label);
    e.to_array()[0]
}

/// Canonical domain separator for all FS in this workspace.
pub const NEO_FS_DOMAIN: &[u8] = b"NEO_FS_V1";

/// Canonical: challenge in base field `F` from (domain,label,transcript).
/// Stateless (Sponge is fed from bytes); use when you don't have a live challenger object.
pub fn fs_challenge_base_labeled(transcript: &[u8], label: &str) -> F {
    // Keep the construction consistent everywhere:
    // input := NEO_FS_V1 || label || transcript
    let mut buf = Vec::with_capacity(NEO_FS_DOMAIN.len() + label.len() + transcript.len());
    buf.extend_from_slice(NEO_FS_DOMAIN);
    buf.extend_from_slice(label.as_bytes());
    buf.extend_from_slice(transcript);
    fiat_shamir_challenge_base(&buf)
}

/// Canonical: challenge in extension field `ExtF` from (domain,label,transcript).
pub fn fs_challenge_ext_labeled(transcript: &[u8], label: &str) -> ExtF {
    let mut buf = Vec::with_capacity(NEO_FS_DOMAIN.len() + label.len() + transcript.len());
    buf.extend_from_slice(NEO_FS_DOMAIN);
    buf.extend_from_slice(label.as_bytes());
    buf.extend_from_slice(transcript);
    fiat_shamir_challenge(&buf)
}

/// Canonical: derive a u64 from the base-field challenge (useful for seeds etc.).
pub fn fs_challenge_u64_labeled(transcript: &[u8], label: &str) -> u64 {
    fs_challenge_base_labeled(transcript, label).as_canonical_u64()
}

/// Unified Transcript facade for consistent Fiat-Shamir transcript construction.
/// 
/// This provides a single, canonical API for all transcript operations, eliminating
/// the dual-surface problem of mixing typed absorb/challenge with ad-hoc byte manipulation.
/// All domain separation follows the NEO/V1/<module>/<phase>/<name> scheme.
#[derive(Clone)]
pub struct Transcript {
    /// Internal p3-challenger state
    challenger: FiatShamirChallenger,
}

impl Transcript {
    /// Create a new transcript with protocol identifier
    pub fn new(protocol: &str) -> Self {
        let mut challenger = create_challenger();
        
        // Domain separation with protocol identifier
        let domain_fields = bytes_to_fields(b"NEO/V1/");
        challenger.observe_slice(&domain_fields);
        
        let protocol_fields = bytes_to_fields(protocol.as_bytes());
        challenger.observe_slice(&protocol_fields);
        
        Self { challenger }
    }

    /// Absorb a domain separation tag (zero-copy marker)
    pub fn absorb_tag(&mut self, tag: &str) {
        self.absorb_bytes("tag", tag.as_bytes());
    }

    /// Absorb bytes with structured framing to prevent concatenation ambiguities
    pub fn absorb_bytes(&mut self, label: &str, bytes: &[u8]) {
        // Absorb label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Absorb the actual bytes
        let byte_fields = bytes_to_fields(bytes);
        self.challenger.observe_slice(&byte_fields);
    }

    /// Absorb a u64 value
    pub fn absorb_u64(&mut self, label: &str, value: u64) {
        self.absorb_bytes(label, &value.to_be_bytes());
    }

    /// Absorb an extension field element
    pub fn absorb_extf(&mut self, label: &str, value: ExtF) {
        // Absorb label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Absorb the extension field element directly
        self.challenger.observe_algebra_element(value);
    }

    /// Absorb a polynomial
    pub fn absorb_poly(&mut self, label: &str, poly: &Polynomial<ExtF>) {
        // Absorb label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Absorb polynomial degree and coefficient count
        let deg = poly.degree() as u64;
        let coeffs = poly.coeffs();
        let count = coeffs.len() as u64;
        
        self.challenger.observe(F::from_u64(deg));
        self.challenger.observe(F::from_u64(count));
        
        // Absorb coefficients directly as extension field elements
        for &coeff in coeffs {
            self.challenger.observe_algebra_element(coeff);
        }
    }

    /// Generate a base field challenge
    pub fn challenge_base(&mut self, label: &str) -> F {
        // Absorb challenge label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Sample base field element
        self.challenger.sample()
    }

    /// Generate an extension field challenge
    pub fn challenge_ext(&mut self, label: &str) -> ExtF {
        // Absorb challenge label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Sample extension field element
        self.challenger.sample_algebra_element()
    }

    /// Generate a ModInt challenge with bias-free rejection sampling
    pub fn challenge_modint(&mut self, label: &str) -> ModInt {
        let q = <ModInt as Coeff>::modulus() as u128;
        let max_attempts = 1000; // Prevent infinite loops
        
        for _ in 0..max_attempts {
            // Squeeze 128 bits for better rejection sampling
            let wide_bytes = self.challenge_wide(label);
            let x = u128::from_be_bytes([
                wide_bytes[0], wide_bytes[1], wide_bytes[2], wide_bytes[3],
                wide_bytes[4], wide_bytes[5], wide_bytes[6], wide_bytes[7],
                wide_bytes[8], wide_bytes[9], wide_bytes[10], wide_bytes[11],
                wide_bytes[12], wide_bytes[13], wide_bytes[14], wide_bytes[15],
            ]);
            
            // Rejection sampling to avoid bias
            let k = (u128::MAX / q) * q; // Largest multiple of q ≤ 2^128
            if x < k {
                return ModInt::from_u64((x % q) as u64);
            }
            
            // Modify state slightly for next attempt
            self.absorb_bytes("retry", &[0u8]);
        }
        
        // Fallback (should be extremely rare)
        ModInt::from_u64(0)
    }

    /// Generate 32 bytes of randomness
    pub fn challenge_wide(&mut self, label: &str) -> [u8; 32] {
        // Absorb label for domain separation
        let label_fields = bytes_to_fields(label.as_bytes());
        self.challenger.observe_slice(&label_fields);
        
        // Generate two extension field elements for 32 bytes total
        let h0: ExtF = self.challenger.sample_algebra_element();
        let h1: ExtF = self.challenger.sample_algebra_element();
        
        let a0 = h0.to_array();
        let a1 = h1.to_array();
        let mut result = [0u8; 32];
        result[0..8].copy_from_slice(&a0[0].as_canonical_u64().to_be_bytes());
        result[8..16].copy_from_slice(&a0[1].as_canonical_u64().to_be_bytes());
        result[16..24].copy_from_slice(&a1[0].as_canonical_u64().to_be_bytes());
        result[24..32].copy_from_slice(&a1[1].as_canonical_u64().to_be_bytes());
        result
    }

    /// Create a properly seeded ChaCha20Rng with 256-bit entropy
    pub fn rng(&mut self, label: &str) -> ChaCha20Rng {
        let seed = self.challenge_wide(label);
        ChaCha20Rng::from_seed(seed)
    }

    /// Create an immutable fork of the transcript
    pub fn fork(&self, fork_label: &str) -> Self {
        let mut forked = self.clone();
        forked.absorb_tag("fork");
        forked.absorb_bytes("fork_label", fork_label.as_bytes());
        forked
    }


}

// --- Tests for structured FS ---



