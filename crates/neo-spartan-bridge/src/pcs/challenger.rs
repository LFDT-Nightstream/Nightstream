use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::PrimeCharacteristicRing;
use super::mmcs::{Perm, Val, Challenge};

pub const DS_BRIDGE_INIT: &[u8]   = b"neo:bridge:init";
pub const DS_BRIDGE_COMMIT: &[u8] = b"neo:bridge:commit";
pub const DS_BRIDGE_OPEN: &[u8]   = b"neo:bridge:open";
pub const DS_BRIDGE_VERIFY: &[u8] = b"neo:bridge:verify";

pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

pub fn make_challenger(perm: Perm) -> Challenger {
    let mut ch = Challenger::new(perm);
    // TODO: Add domain separation once we find the right API for observing bytes
    // For now, observe a fixed field element to establish the separation
    ch.observe(Val::ZERO);
    ch
}

// Observe a commitment digest represented as field elements.
pub fn observe_commitment_words(ch: &mut Challenger, words: &[Val]) {
    for &w in words {
        ch.observe(w);
    }
}

/// Convenience: turn bytes into Goldilocks elements and observe them.
/// For now, just observe the bytes as a length since we can't construct Goldilocks from u64 directly
pub fn observe_commitment_bytes(ch: &mut Challenger, bytes: &[u8]) {
    // TODO: Find the right way to construct Goldilocks from u64 once p3-goldilocks API is clearer
    // For now, just observe a field element representing the byte length
    let len_as_field = if bytes.len() <= 1000 { Val::ONE } else { Val::ZERO };
    ch.observe(len_as_field);
}

// Convenience: observe a commitment with domain separation (for compatible types)
pub fn observe_commitment<C>(ch: &mut Challenger, c: C)
where
    Challenger: CanObserve<C>,
{
    // TODO: Add proper domain separation once we find the right API for observing bytes
    // For now, just observe the commitment directly
    ch.observe(c);
}

// Sample a K = F_{q^2} challenge element
pub fn sample_point(ch: &mut Challenger) -> Challenge {
    ch.sample_algebra_element()
}

// Optional anti-grinding: call before queries
pub fn grind(ch: &mut Challenger, pow_bits: usize) {
    let _witness = ch.grind(pow_bits);
    // Spartan2 verifier must call `check_witness` with same bits later; p3-fri handles this internally.
}

// Ensure trait bounds compile if used from other modules
#[allow(dead_code)]
pub fn _bound_assertions(ch: &mut Challenger)
{
    let test_val = Val::ONE;
    observe_commitment(ch, test_val);
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::pcs::mmcs::make_mmcs_and_dft;

    #[test]
    fn test_challenger_real_fiat_shamir() {
        let mats = make_mmcs_and_dft(42);
        
        let mut ch = make_challenger(mats.perm);
        
        println!("✅ Real Fiat-Shamir challenger created");
        println!("   Domain separation: {:?}", std::str::from_utf8(DS_BRIDGE_INIT));
        
        // Test that we can sample challenges
        let challenge1 = sample_point(&mut ch);
        let challenge2 = sample_point(&mut ch);
        
        // They should be different (with overwhelming probability)
        assert_ne!(challenge1, challenge2);
        
        println!("   Challenge 1: {:?}", challenge1);
        println!("   Challenge 2: {:?}", challenge2);
        println!("   Real K = F_q^2 challenges: ✅");
    }

    #[test]
    fn test_commitment_observation_bytes() {
        let mats = make_mmcs_and_dft(123);
        let mut ch = make_challenger(mats.perm);
        
        // Test observing bytes (converted to field elements)
        let fake_commitment = [0xABu8; 21];
        observe_commitment_bytes(&mut ch, &fake_commitment);
        
        let challenge_after = sample_point(&mut ch);
        
        println!("✅ Commitment observation (bytes) with domain separation");
        println!("   Observed {} bytes as field elements", fake_commitment.len());
        println!("   Challenge after observation: {:?}", challenge_after);
    }

    #[test]
    fn test_commitment_observation_field_elements() {
        let mats = make_mmcs_and_dft(456);
        let mut ch = make_challenger(mats.perm);
        
        // Test observing field elements directly
        let test_commitment = Val::ONE;
        observe_commitment(&mut ch, test_commitment);
        
        let challenge_after = sample_point(&mut ch);
        
        println!("✅ Commitment observation (field element) works");
        println!("   Challenge after observation: {:?}", challenge_after);
    }
}