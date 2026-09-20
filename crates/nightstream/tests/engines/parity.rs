//! Complete C/R/D comparisons: reference to CPU, then CPU to devices.

use super::{Engine, Prover};
use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer,
    transcript::{Poseidon2TranscriptSnapshot, Transcript},
    CcsClaim, CcsInstance, CcsWitness, NifsProof, Params, RunningInstance, Structure,
};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::{poly::Term, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_reductions::{
    paper_exact_engine::PaperMatrixRows,
    superneo_eval::{SuperneoEvalCache, SuperneoEvalCacheBuilder, SuperneoZBlocks},
};
#[cfg(feature = "metal")]
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use std::sync::Arc;

struct Rows {
    matrices: Vec<Vec<(usize, F)>>,
}
impl PaperMatrixRows<F> for Rows {
    fn shape(&self) -> (usize, usize, usize) {
        (1, D + 1, self.matrices.len())
    }
    fn row(&self, matrix: usize, row: usize) -> Vec<(usize, F)> {
        assert_eq!(row, 0);
        self.matrices[matrix].clone()
    }
}

struct Fixture {
    structure: Structure,
    params: Params,
    rows: Rows,
    cache: Arc<SuperneoEvalCache>,
    fresh: CcsInstance,
    running: RunningInstance,
}

impl Fixture {
    fn new(polynomial: SparsePoly<F>, matrices: Vec<Vec<(usize, F)>>) -> Self {
        // One public ring plus one private word gives a non-aligned logical
        // width. The carried source also has a nonzero completion tail.
        // The production decomposition and commitment parameters stay unchanged.
        let rows = Rows { matrices };
        let structure = Structure::new_verifier_artifact_header(1, D + 1, rows.matrices.len(), polynomial).unwrap();
        let params = Params::for_ccs_shape(1, D + 1, structure.t(), structure.max_degree()).unwrap();
        let mut cache = SuperneoEvalCacheBuilder::new(1, D + 1, structure.t()).unwrap();
        for matrix in 0..structure.t() {
            cache.push_row(matrix, 0, rows.row(matrix, 0)).unwrap();
        }
        let cache = Arc::new(cache.finish().unwrap());
        let mut witness = Mat::zero(D, 2, F::ZERO);
        witness[(0, 0)] = F::ONE;
        let commitment = commit_production_signed_unit_prefix_matrix(&witness).unwrap();
        let fresh = CcsInstance {
            claim: CcsClaim {
                c: commitment.clone(),
                x: (0..D).map(|lane| witness[(lane, 0)]).collect(),
                m_in: D,
                adv: None,
            },
            witness: CcsWitness {
                w: vec![],
                Z: witness.clone(),
            },
        };
        witness[(D - 1, 1)] = -F::ONE;
        let mut running = RunningInstance::canonical_zero(&params, &structure, D).unwrap();
        let blocks = SuperneoZBlocks::from_witness_mat(&witness, structure.m).unwrap();
        let opening = cache
            .eval_real_v1_1_openings(&running.claims[0].r, &[blocks])
            .unwrap()
            .remove(0);
        let claim = &mut running.claims[0];
        claim.c = commit_production_signed_unit_prefix_matrix(&witness).unwrap();
        claim.X[(0, 0)] = F::ONE;
        claim.eval_k = opening.eval_k;
        claim.eval_k.resize(D.next_power_of_two(), K::ZERO);
        claim.eval_a = opening.eval_a;
        for values in &mut claim.eval_a {
            values.resize(D.next_power_of_two(), K::ZERO);
        }
        running.parent_authority = Some(claim.clone());
        running.witnesses[0] = witness;
        Self {
            structure,
            params,
            rows,
            cache,
            fresh,
            running,
        }
    }

    fn bit() -> Self {
        Self::new(
            SparsePoly::new(
                1,
                vec![
                    Term {
                        coeff: F::ONE,
                        exps: vec![2],
                    },
                    Term {
                        coeff: -F::ONE,
                        exps: vec![1],
                    },
                ],
            ),
            vec![vec![(0, F::ONE)]],
        )
    }

    #[cfg(feature = "metal")]
    fn selected_polynomial() -> Self {
        let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
        let value: serde_json::Value = serde_json::from_slice(&std::fs::read(path).unwrap()).unwrap();
        let relation = &value[1][4];
        let count = relation[3].as_array().unwrap().len();
        let terms = relation[5]
            .as_array()
            .unwrap()
            .iter()
            .map(|term| Term {
                coeff: F::from_u64(term[0].as_u64().unwrap()),
                exps: serde_json::from_value(term[1].clone()).unwrap(),
            })
            .collect();
        let polynomial = SparsePoly::new(count, terms);
        // Exercise the complete exported polynomial with nonzero input ports.
        // Its C port (slot 4) is affine; solve that coordinate so the fresh row
        // is valid. The declared final zero matrix stays zero.
        assert!(polynomial.terms().iter().all(|term| term.exps[4] <= 1));
        let mut values: Vec<_> = (1..=count).map(F::from_usize).collect();
        values[count - 1] = F::ZERO;
        values[4] = F::ZERO;
        let constant = polynomial.eval(&values);
        values[4] = F::ONE;
        let coefficient = polynomial.eval(&values) - constant;
        assert_ne!(coefficient, F::ZERO);
        values[4] = -constant * coefficient.inverse();
        assert_eq!(polynomial.eval(&values), F::ZERO);
        let matrices = values
            .into_iter()
            .map(|value| if value == F::ZERO { vec![] } else { vec![(0, value)] })
            .collect();
        Self::new(polynomial, matrices)
    }
}

struct Run {
    running: RunningInstance,
    proof: NifsProof,
    transcript: Poseidon2TranscriptSnapshot,
}

fn run(engine: Engine, fixture: &Fixture) -> Result<Run, Box<dyn std::error::Error>> {
    let mut transcript = Transcript::session();
    let (running, proof) = match Prover::new(engine)? {
        Prover::Optimized => folding::prove_owned_with_rows(
            &mut transcript,
            &fixture.params,
            &fixture.structure,
            &fixture.cache,
            vec![fixture.fresh.clone()],
            fixture.running.clone(),
        )?,
        Prover::PaperExact => super::paper_exact::prove(
            &mut transcript,
            &fixture.params,
            &fixture.structure,
            &fixture.rows,
            vec![fixture.fresh.clone()],
            fixture.running.clone(),
        )?,
        #[cfg(feature = "metal")]
        Prover::Metal(device) => {
            let mut device = device.lock().unwrap();
            let result = super::metal::prove(
                &mut device,
                &mut transcript,
                &fixture.params,
                &fixture.structure,
                &fixture.cache,
                vec![fixture.fresh.clone()],
                fixture.running.clone(),
            )?;
            assert!(
                device.activity().dispatches > 0,
                "Metal parity requires real device work"
            );
            result
        }
    };
    let mut verifier = Transcript::session();
    let checked = folding::verify(
        &mut verifier,
        &fixture.params,
        &fixture.structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(&fixture.fresh.claim),
        &fixture.running.claims_only(),
        &proof,
    )?;
    assert_eq!(checked.claims, running.claims);
    assert_eq!(checked.parent_authority, running.parent_authority);
    assert_eq!(verifier.snapshot(), transcript.snapshot());
    let mut changed = proof.clone();
    changed.pi_dec.children[0].eval_k[0] += K::ONE;
    assert!(folding::verify(
        &mut Transcript::session(),
        &fixture.params,
        &fixture.structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(&fixture.fresh.claim),
        &fixture.running.claims_only(),
        &changed
    )
    .is_err());
    Ok(Run {
        running,
        proof,
        transcript: transcript.snapshot(),
    })
}

fn equal(left: Run, right: Run) {
    assert_eq!(left.proof.canonical_bytes(), right.proof.canonical_bytes());
    assert_eq!(left.proof, right.proof);
    assert_eq!(left.running, right.running);
    assert_eq!(left.transcript, right.transcript);
}

#[test]
fn paper_exact_matches_optimized() {
    let fixture = Fixture::bit();
    equal(
        run(Engine::PaperExact, &fixture).unwrap(),
        run(Engine::Optimized, &fixture).unwrap(),
    );
}

#[cfg(feature = "metal")]
#[test]
fn optimized_matches_metal() {
    let fixture = Fixture::bit();
    equal(
        run(Engine::Optimized, &fixture).unwrap(),
        run(Engine::Metal, &fixture).unwrap(),
    );
}

#[cfg(feature = "metal")]
#[test]
fn zero_running_matches_metal() {
    let mut fixture = Fixture::selected_polynomial();
    fixture.running = RunningInstance::canonical_zero(&fixture.params, &fixture.structure, D).unwrap();
    equal(
        run(Engine::Optimized, &fixture).unwrap(),
        run(Engine::Metal, &fixture).unwrap(),
    );
}

#[cfg(feature = "metal")]
#[test]
fn selected_polynomial_matches_metal() {
    let fixture = Fixture::selected_polynomial();
    equal(
        run(Engine::Optimized, &fixture).unwrap(),
        run(Engine::Metal, &fixture).unwrap(),
    );
}

#[cfg(feature = "cuda")]
#[test]
#[ignore = "The canonical CUDA kernel is not implemented; this is an open parity check, not a pass."]
fn optimized_matches_cuda() {
    let fixture = Fixture::bit();
    equal(
        run(Engine::Optimized, &fixture).unwrap(),
        run(Engine::Cuda, &fixture).unwrap(),
    );
}

#[cfg(all(feature = "metal", feature = "cuda"))]
#[test]
#[ignore = "The canonical CUDA kernel is not implemented; this is an open parity check, not a pass."]
fn metal_matches_cuda() {
    let fixture = Fixture::bit();
    equal(
        run(Engine::Metal, &fixture).unwrap(),
        run(Engine::Cuda, &fixture).unwrap(),
    );
}
