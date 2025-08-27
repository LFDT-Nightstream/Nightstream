use neo_sumcheck::*;
use neo_fields::ExtF;
use p3_field::PrimeCharacteristicRing;

// Test polynomial that's identically zero
struct ZeroPoly { l: usize }

impl UnivPoly for ZeroPoly {
    fn evaluate(&self, _x: &[ExtF]) -> ExtF { ExtF::ZERO }
    fn degree(&self) -> usize { self.l }
    fn max_individual_degree(&self) -> usize { 1 }
}

#[test]
fn batched_sumcheck_roundtrip_with_structured_fs() {
    let polys: Vec<&dyn UnivPoly> = vec![&ZeroPoly { l: 2 }]; // 2 vars, identically 0
    let claims = vec![ExtF::ZERO];

    let mut prover_t = Vec::new();
    let msgs = batched_sumcheck_prover(&claims, &polys, &mut prover_t)
        .expect("prover must succeed");

    // Verifier starts from the same *pre-sumcheck* transcript state (empty here),
    // derives the same rho & round challenges, and accepts.
    let mut verifier_t = Vec::new();
    let res = batched_sumcheck_verifier(&claims, &msgs, &mut verifier_t);
    assert!(res.is_some(), "verifier should accept prover messages");
    let (_r, final_current) = res.unwrap();
    // For identically-0 polynomials, the final 'current' should be 0 after all rounds.
    assert_eq!(final_current, ExtF::ZERO);
}
