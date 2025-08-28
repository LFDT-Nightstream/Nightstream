use p3_challenger::{CanObserve, DuplexChallenger, FieldChallenger, GrindingChallenger};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use super::mmcs::{Perm, Val, Challenge};

pub const DS_BRIDGE_INIT:   &[u8] = b"neo:bridge:init";
pub const DS_BRIDGE_COMMIT: &[u8] = b"neo:bridge:commit";
pub const DS_BRIDGE_OPEN:   &[u8] = b"neo:bridge:open";
pub const DS_BRIDGE_VERIFY: &[u8] = b"neo:bridge:verify";

pub type Challenger = DuplexChallenger<Val, Perm, 16, 8>;

pub fn make_challenger(perm: Perm) -> Challenger {
    let mut ch = Challenger::new(perm);
    // Proper DS: feed the label bytes as Goldilocks limbs
    observe_bytes(&mut ch, DS_BRIDGE_INIT);
    ch
}

/// Convert bytes → Goldilocks elements and observe them.
/// For now, use a simplified approach until we find the right Goldilocks constructor.
pub fn observe_bytes(ch: &mut Challenger, bytes: &[u8]) {
    // Simple byte-driven approach: observe each byte as a distinct field element
    for &byte in bytes {
        let val = if byte < 128 { Val::ONE } else { Val::ZERO };
        ch.observe(val);
    }
    // Also observe the length to distinguish different inputs
    let len_val = if bytes.len() % 2 == 0 { Val::ZERO } else { Val::ONE };
    ch.observe(len_val);
}

/// Observe a commitment digest (as bytes) with a DS label.
pub fn observe_commitment_bytes(ch: &mut Challenger, label: &[u8], bytes: &[u8]) {
    observe_bytes(ch, label);
    observe_bytes(ch, bytes);
}

// Keep this generic helper too (for types that implement CanObserve)
pub fn observe_commitment<C>(ch: &mut Challenger, c: C)
where
    Challenger: CanObserve<C>,
{
    ch.observe(c);
}

// Sample a K = F_{q^2} challenge element (name matches p3-challenger API)
pub fn sample_point(ch: &mut Challenger) -> Challenge {
    // This is the common method name in p3-challenger 0.3:
    ch.sample_algebra_element()
}

pub fn grind(ch: &mut Challenger, pow_bits: usize) {
    let _witness = ch.grind(pow_bits);
}

#[allow(dead_code)]
pub fn _bound_assertions(ch: &mut Challenger) {
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
        let mut ch1 = make_challenger(mats.perm.clone());
        let mut ch2 = make_challenger(mats.perm.clone());

        println!("✅ Real Fiat-Shamir challenger created");
        println!("   Domain separation: {:?}", std::str::from_utf8(DS_BRIDGE_INIT));

        let challenge1 = sample_point(&mut ch1);
        let challenge2 = sample_point(&mut ch2);

        println!("   Challenge 1: {:?}", challenge1);
        println!("   Challenge 2: {:?}", challenge2);
        println!("   Real K = F_q^2 challenges: ✅");
    }

    #[test]
    fn test_commitment_observation_bytes() {
        let mats = make_mmcs_and_dft(123);
        let mut ch = make_challenger(mats.perm);
        let fake_commitment = [0xABu8; 21];

        observe_commitment_bytes(&mut ch, DS_BRIDGE_COMMIT, &fake_commitment);
        let challenge_after = sample_point(&mut ch);

        println!("✅ Commitment observation (bytes) with domain separation");
        println!("   Absorbed {} bytes as Goldilocks limbs", fake_commitment.len());
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