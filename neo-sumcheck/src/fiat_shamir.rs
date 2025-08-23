use crate::{ExtF, F, Polynomial};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Poseidon2Goldilocks;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_chacha::ChaCha20Rng;
use neo_modint::{Coeff, ModInt};

/// Poseidon2 parameters: 16-element state, 15-element rate, 2-element output for ExtF
const WIDTH: usize = 16;
const RATE: usize = WIDTH - 1;
const OUT: usize = 2;

/// Type alias for Poseidon2 permutation over Goldilocks field
pub(crate) type Perm = Poseidon2Goldilocks<WIDTH>;

/// Get a deterministic Poseidon2 permutation using fixed, recommended parameters.
fn get_perm() -> Perm {
    // Use a fixed seed to deterministically construct the permutation parameters.
    // This yields reproducible, interoperable Poseidon2 parameters.
    let mut rng = StdRng::seed_from_u64(0);
    Perm::new_from_rng_128(&mut rng)
}

/// Type alias for the sponge construction used in Fiat-Shamir
pub(crate) type SpongeType = PaddingFreeSponge<Perm, WIDTH, RATE, OUT>;

/// Generate a cryptographic challenge from the current transcript using Fiat-Shamir
///
/// # Arguments
/// * `transcript` - Byte array containing all protocol messages so far
///
/// # Returns
/// A pseudo-random field element derived from the transcript
pub fn fiat_shamir_challenge(transcript: &[u8]) -> ExtF {
    let mut field_elems = vec![];

    // Convert transcript bytes to field elements (8 bytes per element),
    // padding the final chunk to avoid dropping bytes.
    for chunk in transcript.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let val = u64::from_be_bytes(buf);
        field_elems.push(F::from_u64(val));
    }

    // Hash the field elements using Poseidon2 sponge
    let perm = get_perm();
    let sponge = SpongeType::new(perm);
    let output = sponge.hash_iter(field_elems);

    // Convert two base field outputs to extension field (real + imag parts)
    ExtF::new_complex(output[0], output[1])
}

/// Generate a base-field Fiat-Shamir challenge from the transcript.
pub fn fiat_shamir_challenge_base(transcript: &[u8]) -> F {
    let mut field_elems = vec![];
    for chunk in transcript.chunks(8) {
        let mut buf = [0u8; 8];
        buf[..chunk.len()].copy_from_slice(chunk);
        let val = u64::from_be_bytes(buf);
        field_elems.push(F::from_u64(val));
    }
    let perm = get_perm();
    let sponge = SpongeType::new(perm);
    let output = sponge.hash_iter(field_elems);
    output[0]
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
    // header: deg | count
    fs_write_u32_be(transcript, 8); // 8 bytes follow in "payload header"
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
    /// Internal transcript state as bytes
    state: Vec<u8>,
}

impl Transcript {
    /// Create a new transcript with protocol identifier
    pub fn new(protocol: &str) -> Self {
        let mut state = Vec::new();
        state.extend_from_slice(b"NEO/V1/");
        state.extend_from_slice(protocol.as_bytes());
        Self { state }
    }

    /// Absorb a domain separation tag (zero-copy marker)
    pub fn absorb_tag(&mut self, tag: &str) {
        self.absorb_bytes("tag", tag.as_bytes());
    }

    /// Absorb bytes with structured framing to prevent concatenation ambiguities
    pub fn absorb_bytes(&mut self, label: &str, bytes: &[u8]) {
        // Frame: "FS" | len(label) | label | len(bytes) | bytes
        self.state.extend_from_slice(b"FS");
        self.state.extend_from_slice(&(label.len() as u32).to_be_bytes());
        self.state.extend_from_slice(label.as_bytes());
        self.state.extend_from_slice(&(bytes.len() as u32).to_be_bytes());
        self.state.extend_from_slice(bytes);
    }

    /// Absorb a u64 value
    pub fn absorb_u64(&mut self, label: &str, value: u64) {
        self.absorb_bytes(label, &value.to_be_bytes());
    }

    /// Absorb an extension field element
    pub fn absorb_extf(&mut self, label: &str, value: ExtF) {
        let arr = value.to_array();
        let mut bytes = Vec::with_capacity(16);
        bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
        bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
        self.absorb_bytes(label, &bytes);
    }

    /// Absorb a polynomial
    pub fn absorb_poly(&mut self, label: &str, poly: &Polynomial<ExtF>) {
        let mut bytes = Vec::new();
        let deg = poly.degree() as u32;
        let coeffs = poly.coeffs();
        let count = coeffs.len() as u32;
        
        // Header: degree | coefficient count
        bytes.extend_from_slice(&deg.to_be_bytes());
        bytes.extend_from_slice(&count.to_be_bytes());
        
        // Coefficients
        for &coeff in coeffs {
            let arr = coeff.to_array();
            bytes.extend_from_slice(&arr[0].as_canonical_u64().to_be_bytes());
            bytes.extend_from_slice(&arr[1].as_canonical_u64().to_be_bytes());
        }
        
        self.absorb_bytes(label, &bytes);
    }

    /// Generate a base field challenge
    pub fn challenge_base(&mut self, label: &str) -> F {
        self.absorb_tag("challenge");
        self.absorb_bytes("challenge_label", label.as_bytes());
        fiat_shamir_challenge_base(&self.state)
    }

    /// Generate an extension field challenge
    pub fn challenge_ext(&mut self, label: &str) -> ExtF {
        self.absorb_tag("challenge");
        self.absorb_bytes("challenge_label", label.as_bytes());
        fiat_shamir_challenge(&self.state)
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
        self.absorb_tag("wide_challenge");
        self.absorb_bytes("wide_label", label.as_bytes());
        
        // Generate two extension field elements for 32 bytes total
        let h0 = fiat_shamir_challenge(&self.state);
        
        // Modify state for second challenge
        self.absorb_bytes("wide_second", &[1u8]);
        let h1 = fiat_shamir_challenge(&self.state);
        
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

    /// Get the current transcript state (for compatibility with existing code)
    pub fn state(&self) -> &[u8] {
        &self.state
    }

    /// Get a mutable reference to the state (for compatibility, use sparingly)
    pub fn state_mut(&mut self) -> &mut Vec<u8> {
        &mut self.state
    }
}

// --- Tests for structured FS ---



