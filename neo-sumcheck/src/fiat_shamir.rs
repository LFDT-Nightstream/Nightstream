use crate::{ExtF, F, Polynomial};
use p3_field::PrimeCharacteristicRing;
use p3_goldilocks::Poseidon2Goldilocks;
use p3_symmetric::{CryptographicHasher, PaddingFreeSponge};
use rand::rngs::StdRng;
use rand::SeedableRng;

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

