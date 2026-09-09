//! Check the complete-oracle driver boundary against the cached optimized prover.
//! These small cases test the API; selected honest evidence uses the fixture CLI.

#![cfg(feature = "paper-exact")]

use std::sync::Arc;

use neo_ajtai::{setup, AjtaiSModule, Commitment};
use neo_ccs::{traits::SModuleHomomorphism, CcsClaim, CcsStructure, CcsWitness, CeClaim, Mat, SparsePoly, Term};
use neo_math::{D, F, K};
use neo_params::NeoParams;
use neo_reductions::{
    engines::{pi_ccs_joint::build_joint_dims, pi_ccs_joint_protocol::V1_1OutputOpening},
    optimized_engine::{
        canonical_audit::OptimizedPaperJointOracle, optimized_prove_with_cache_and_perf,
        optimized_prove_with_complete_oracle, optimized_verify_with_trace, Challenges, OptimizedStructureCache,
        PaperJointRoundOracle,
    },
    PiCcsError,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand_chacha::{rand_core::SeedableRng, ChaCha8Rng};

type Output = CeClaim<Commitment, F, K>;

fn transcript() -> Poseidon2Transcript {
    Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0)
}

struct Case {
    structure: CcsStructure<F>,
    params: NeoParams,
    commitment: AjtaiSModule,
    cache: OptimizedStructureCache,
    fresh: Vec<CcsClaim<Commitment, F>>,
    witnesses: Vec<CcsWitness<F>>,
    running: Vec<Output>,
    running_witnesses: Vec<Mat<F>>,
}

impl Case {
    fn new() -> Self {
        let columns = D + 1;
        let mut matrix = Mat::zero(D / 2, columns, F::ZERO);
        for column in 0..columns {
            matrix[(column % (D / 2), column)] = F::ONE;
        }
        let structure = CcsStructure::new(
            vec![matrix.clone(), matrix],
            SparsePoly::new(
                2,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![1, 0],
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![0, 1],
                    },
                ],
            ),
        )
        .unwrap();
        let params = NeoParams::goldilocks_auto_r1cs_ccs(columns).unwrap();
        let mut rng = ChaCha8Rng::seed_from_u64(0x636f6d706c657465);
        let commitment = AjtaiSModule::new(Arc::new(
            setup(&mut rng, D, params.kappa as usize, columns.div_ceil(D)).unwrap(),
        ));
        let mut assignment = Mat::zero(D, columns.div_ceil(D), F::ZERO);
        for column in 0..columns {
            assignment[(column % D, column / D)] = match column % 3 {
                0 => F::ONE,
                1 => -F::ONE,
                _ => F::ZERO,
            };
        }
        let fresh = vec![CcsClaim {
            c: commitment.commit(&assignment),
            x: (0..D).map(|column| assignment[(column, 0)]).collect(),
            m_in: D,
            adv: None,
        }];
        let witnesses = vec![CcsWitness {
            w: vec![assignment[(0, 1)]],
            Z: assignment,
        }];
        let dims = build_joint_dims(&params, &structure, 1, 1).unwrap();
        let running = vec![Output {
            c: Commitment::zeros(D, params.kappa as usize),
            X: Mat::zero(D, 1, F::ZERO),
            r: vec![K::ZERO; dims.variables],
            eval_k: vec![K::ZERO; D.next_power_of_two()],
            eval_a: vec![vec![K::ZERO; D.next_power_of_two()]; 2],
            m_in: D,
            fold_digest: [0; 32],
            adv: None,
        }];
        let running_witnesses = vec![Mat::virtual_constant(D, columns.div_ceil(D), F::ZERO)];
        let cache = OptimizedStructureCache::build(&structure).unwrap();
        Self {
            structure,
            params,
            commitment,
            cache,
            fresh,
            witnesses,
            running,
            running_witnesses,
        }
    }

    fn reference(
        &self,
    ) -> (
        Vec<Output>,
        neo_reductions::PiCcsProof,
        neo_reductions::engines::pi_ccs_joint::ProtocolTrace,
    ) {
        let (outputs, proof, _) = optimized_prove_with_cache_and_perf(
            &mut transcript(),
            &self.params,
            &self.structure,
            &self.fresh,
            &self.witnesses,
            &self.running,
            &self.running_witnesses,
            &self.commitment,
            &self.cache,
        )
        .unwrap();
        let (accepted, trace) = optimized_verify_with_trace(
            &mut transcript(),
            &self.params,
            &self.structure,
            &self.fresh,
            &self.running,
            &outputs,
            &proof,
        )
        .unwrap();
        assert!(accepted);
        (outputs, proof, trace)
    }

    fn oracle<'a>(&'a self, challenges: &Challenges, point: Vec<K>, outputs: &[Output]) -> Complete<'a> {
        let dims = build_joint_dims(&self.params, &self.structure, 1, 1).unwrap();
        Complete {
            inner: OptimizedPaperJointOracle::new(
                &self.structure,
                &self.params,
                &self.witnesses,
                &self.running_witnesses,
                challenges.clone(),
                Some(&self.running[0].r),
                dims,
                &self.cache,
            )
            .unwrap(),
            openings: Some(
                outputs
                    .iter()
                    .map(|output| V1_1OutputOpening {
                        eval_k: output.eval_k[..D].to_vec(),
                        eval_a: output
                            .eval_a
                            .iter()
                            .map(|family| family[..D].to_vec())
                            .collect(),
                    })
                    .collect(),
            ),
            point,
            calls: 0,
        }
    }
}

// The inner evaluator is the optimized implementation. The existing audit
// export only makes its type available; no PaperExact engine is executed.
struct Complete<'a> {
    inner: OptimizedPaperJointOracle<'a>,
    openings: Option<Vec<V1_1OutputOpening>>,
    point: Vec<K>,
    calls: usize,
}

impl PaperJointRoundOracle for Complete<'_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError> {
        self.calls += 1;
        self.inner.evals_at(points)
    }
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }
    fn degree_bound(&self) -> usize {
        self.inner.degree_bound()
    }
    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError> {
        self.inner.fold(challenge)
    }
    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_1OutputOpening>>, PiCcsError> {
        assert_eq!(point, self.point);
        Ok(self.openings.take())
    }
}

#[test]
fn complete_oracle_matches_cached_optimized_prover() {
    let case = Case::new();
    let (expected, expected_proof, expected_trace) = case.reference();
    let challenges = Challenges::new(expected_trace.alpha.clone(), expected_trace.gamma);
    let mut oracle = case.oracle(&challenges, expected_trace.round_challenges.clone(), &expected);
    let (outputs, proof, _, trace) = optimized_prove_with_complete_oracle(
        &mut transcript(),
        &case.params,
        &case.structure,
        &case.fresh,
        &case.witnesses,
        &case.running,
        &case.running_witnesses,
        &challenges,
        &mut oracle,
    )
    .unwrap();
    assert_eq!(proof, expected_proof);
    assert_eq!(trace.events, expected_trace.events);
    assert_eq!(trace.outgoing_state, expected_trace.outgoing_state);
    assert_eq!(outputs.len(), expected.len());
    for (actual, expected) in outputs.iter().zip(expected) {
        assert_eq!(actual.c, expected.c);
        assert_eq!(actual.X, expected.X);
        assert_eq!(actual.r, expected.r);
        assert_eq!(actual.eval_k, expected.eval_k);
        assert_eq!(actual.eval_a, expected.eval_a);
        assert_eq!(actual.fold_digest, expected.fold_digest);
    }
}

#[test]
fn complete_oracle_rejects_changed_alpha_or_gamma_before_evaluation() {
    let case = Case::new();
    let (outputs, _, trace) = case.reference();
    let challenges = Challenges::new(trace.alpha.clone(), trace.gamma);
    for alpha in [true, false] {
        let mut changed = challenges.clone();
        if alpha {
            changed.alpha[0] += K::ONE;
        } else {
            changed.gamma += K::ONE;
        }
        let mut oracle = case.oracle(&challenges, trace.round_challenges.clone(), &outputs);
        let error = optimized_prove_with_complete_oracle(
            &mut transcript(),
            &case.params,
            &case.structure,
            &case.fresh,
            &case.witnesses,
            &case.running,
            &case.running_witnesses,
            &changed,
            &mut oracle,
        )
        .unwrap_err();
        assert!(error.to_string().contains("challenges do not match"));
        assert_eq!(oracle.calls, 0);
    }
}

#[test]
fn complete_oracle_rejects_missing_or_incomplete_openings() {
    let case = Case::new();
    let (outputs, _, trace) = case.reference();
    let challenges = Challenges::new(trace.alpha.clone(), trace.gamma);
    for missing in [true, false] {
        let mut oracle = case.oracle(&challenges, trace.round_challenges.clone(), &outputs);
        if missing {
            oracle.openings = None;
        } else {
            oracle.openings.as_mut().unwrap().pop();
        }
        let error = optimized_prove_with_complete_oracle(
            &mut transcript(),
            &case.params,
            &case.structure,
            &case.fresh,
            &case.witnesses,
            &case.running,
            &case.running_witnesses,
            &challenges,
            &mut oracle,
        )
        .unwrap_err();
        assert!(error.to_string().contains(if missing {
            "did not return output openings"
        } else {
            "wrong source count"
        }));
    }
}

#[test]
fn cached_optimized_prover_keeps_its_matrix_binding_check() {
    let case = Case::new();
    let changed = CcsStructure::new(vec![Mat::zero(D / 2, D + 1, F::ONE); 2], case.structure.f.clone()).unwrap();
    let error = optimized_prove_with_cache_and_perf(
        &mut transcript(),
        &case.params,
        &changed,
        &case.fresh,
        &case.witnesses,
        &case.running,
        &case.running_witnesses,
        &case.commitment,
        &case.cache,
    )
    .unwrap_err();
    assert!(error.to_string().contains("matrix digest does not match"));
}
