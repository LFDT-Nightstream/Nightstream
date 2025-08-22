use neo_fields::ExtF;
use p3_field::PrimeCharacteristicRing;
use neo_poly::Polynomial;
use neo_sumcheck::batched_sumcheck_verifier;

#[test]
fn forged_sumcheck_rejected() {
    // single round message with wrong polynomial
    let honest_poly = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let msgs = vec![(honest_poly.clone(), ExtF::ZERO)];
    let mut bad_msgs = msgs.clone();
    bad_msgs[0].0 = Polynomial::new(vec![ExtF::ONE]);
    let mut transcript = Vec::new();
    let result = batched_sumcheck_verifier(&[ExtF::ZERO], &bad_msgs, &mut transcript);
    assert!(result.is_none());
}
