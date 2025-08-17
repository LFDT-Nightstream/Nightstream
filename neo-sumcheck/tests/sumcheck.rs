use neo_sumcheck::{batched_sumcheck_prover, ExtF, FnOracle, UnivPoly};
use p3_field::PrimeCharacteristicRing;

struct TestPoly;
impl UnivPoly for TestPoly {
    fn evaluate(&self, _point: &[ExtF]) -> ExtF { ExtF::ZERO }
    fn degree(&self) -> usize { 3 }
    fn max_individual_degree(&self) -> usize { 1 }
}

#[test]
fn test_batched_sumcheck_no_clone_panic() {
    let poly: Box<dyn UnivPoly> = Box::new(TestPoly);
    let mut oracle = FnOracle::new(|_| vec![]);
    let mut transcript = vec![];
    let result = batched_sumcheck_prover(&[ExtF::ZERO], &[&*poly], &mut oracle, &mut transcript);
    assert!(result.is_ok());
}
