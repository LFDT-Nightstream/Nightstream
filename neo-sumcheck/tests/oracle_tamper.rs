use neo_sumcheck::{
    batched_sumcheck_prover, batched_sumcheck_verifier, from_base, Commitment, ExtF,
    FriOracle, OpeningProof, PolyOracle, Polynomial, UnivPoly, F,
};
use p3_field::PrimeCharacteristicRing;

struct LinearPoly;

impl UnivPoly for LinearPoly {
    fn evaluate(&self, point: &[ExtF]) -> ExtF {
        point[0]
    }
    fn degree(&self) -> usize {
        1
    }
    fn max_individual_degree(&self) -> usize {
        1
    }
}

fn setup_simple_sumcheck() -> (ExtF, Vec<(Polynomial<ExtF>, ExtF)>, Vec<Commitment>, Polynomial<ExtF>) {
    let poly = LinearPoly;
    let claim = from_base(F::from_u64(1));
    let dense = Polynomial::new(vec![ExtF::ZERO, ExtF::ONE]);
    let mut transcript = vec![];
    let mut oracle = FriOracle::new(vec![dense.clone()], &mut transcript);
    let (msgs, comms) =
        batched_sumcheck_prover(&[claim], &[&poly], &mut oracle, &mut transcript).unwrap();
    (claim, msgs, comms, dense)
}

struct TamperOracle {
    inner: FriOracle,
    tamper_eval: bool,
    tamper_proof: bool,
}

impl PolyOracle for TamperOracle {
    fn commit(&mut self) -> Vec<Commitment> {
        self.inner.commit()
    }

    fn open_at_point(&mut self, point: &[ExtF]) -> (Vec<ExtF>, Vec<OpeningProof>) {
        let (mut evals, mut proofs) = self.inner.open_at_point(point);
        if self.tamper_eval && !evals.is_empty() {
            evals[0] += ExtF::ONE;
        }
        if self.tamper_proof && !proofs.is_empty() && !proofs[0].is_empty() {
            proofs[0][0] ^= 1;
        }
        (evals, proofs)
    }

    fn verify_openings(
        &self,
        comms: &[Commitment],
        point: &[ExtF],
        evals: &[ExtF],
        proofs: &[OpeningProof],
    ) -> bool {
        self.inner.verify_openings(comms, point, evals, proofs)
    }
}

#[test]
fn test_tampered_eval_rejected() {
    let (claim, msgs, comms, dense) = setup_simple_sumcheck();
    let tamper_oracle = {
        let mut t = vec![];
        TamperOracle { inner: FriOracle::new(vec![dense], &mut t), tamper_eval: true, tamper_proof: false }
    };
    let mut oracle = tamper_oracle;
    let mut transcript = vec![];
    let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut oracle, &mut transcript);
    assert!(result.is_none());
}

#[test]
fn test_invalid_proof_rejected() {
    let (claim, msgs, comms, dense) = setup_simple_sumcheck();
    let tamper_oracle = {
        let mut t = vec![];
        TamperOracle { inner: FriOracle::new(vec![dense], &mut t), tamper_eval: false, tamper_proof: true }
    };
    let mut oracle = tamper_oracle;
    let mut transcript = vec![];
    let result = batched_sumcheck_verifier(&[claim], &msgs, &comms, &mut oracle, &mut transcript);
    assert!(result.is_none());
}

