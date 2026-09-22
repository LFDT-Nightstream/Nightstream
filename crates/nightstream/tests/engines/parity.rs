//! Complete C/R/D comparisons: reference to CPU, then CPU to devices.

use super::{Backend, Engine, EngineError};
use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer,
    transcript::{Poseidon2TranscriptSnapshot, Transcript},
    CcsClaim, CcsInstance, CcsWitness, NifsProof, Params, RunningInstance, Structure,
};
use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::{poly::Term, Mat, SparsePoly};
use neo_math::{D, F, K};
use neo_reductions::{
    engines::pi_ccs_joint::build_joint_dims,
    paper_exact_engine::PaperMatrixRows,
    superneo_eval::{CachedMatrixRows, MatrixWindow, SuperneoEvalCache, SuperneoEvalCacheBuilder, SuperneoZBlocks},
};
use p3_field::Field;
use p3_field::PrimeCharacteristicRing;
use std::sync::{
    atomic::{AtomicBool, Ordering},
    Arc, OnceLock,
};

struct Rows {
    n: usize,
    matrices: Vec<Vec<Vec<(usize, F)>>>,
}
impl PaperMatrixRows<F> for Rows {
    fn shape(&self) -> (usize, usize, usize) {
        (self.n, D + 1, self.matrices.len())
    }
    fn row(&self, matrix: usize, row: usize) -> Vec<(usize, F)> {
        self.matrices[matrix][row].clone()
    }
}

struct WorkerRows<'a> {
    rows: &'a Rows,
    caller: std::thread::ThreadId,
    called: AtomicBool,
}
impl PaperMatrixRows<F> for WorkerRows<'_> {
    fn shape(&self) -> (usize, usize, usize) {
        self.rows.shape()
    }
    fn row(&self, matrix: usize, row: usize) -> Vec<(usize, F)> {
        assert_ne!(std::thread::current().id(), self.caller);
        self.called.store(true, Ordering::Relaxed);
        self.rows.row(matrix, row)
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
    fn workspace_bytes(&self, rows: &CachedMatrixRows<'_>) -> usize {
        let dims = build_joint_dims(self.params.inner(), &self.structure, 1, self.running.witnesses.len()).unwrap();
        let row_payload = self.structure.t() * size_of::<K>() + size_of::<F>() + size_of::<K>();
        let row_window = MatrixWindow::required_workspace(rows, 0..self.structure.n, row_payload).unwrap();
        let witnesses = (1 + self.running.witnesses.len()) * dims.assignment_width;
        let openings = self.params.k_rho() as usize * (self.structure.t() + 1) * D;
        // The host and device may both hold row metadata. The remaining
        // payload comes from this fixture's witnesses, child openings, and round.
        2 * row_window + (witnesses + openings + dims.degree + 1 + self.structure.t()) * size_of::<K>()
    }

    fn new(polynomial: SparsePoly<F>, matrices: Vec<Vec<Vec<(usize, F)>>>) -> Self {
        // One public ring plus one private word gives a non-aligned logical
        // width. The carried source also has a nonzero completion tail.
        // The production decomposition and commitment parameters stay unchanged.
        let n = matrices[0].len();
        let rows = Rows { n, matrices };
        let structure = Structure::new_verifier_artifact_header(n, D + 1, rows.matrices.len(), polynomial).unwrap();
        let params = Params::for_ccs_shape(n, D + 1, structure.t(), structure.max_degree(), 114).unwrap();
        let mut cache = SuperneoEvalCacheBuilder::new(n, D + 1, structure.t()).unwrap();
        for row in 0..n {
            for matrix in 0..structure.t() {
                cache.push_row(matrix, row, rows.row(matrix, row)).unwrap();
            }
        }
        let cache = Arc::new(cache.finish().unwrap());
        let mut witness = Mat::zero(D, 2, F::ZERO);
        witness[(0, 0)] = F::ONE;
        if n > 1 {
            witness[(0, 1)] = F::ONE;
        }
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
                Z: Mat::compact_signed_unit_from_column_masks(D, 2, &[1, u64::from(n > 1)], &[0, 0]).unwrap(),
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
        running.witnesses[0] =
            Mat::compact_signed_unit_from_column_masks(D, 2, &[1, u64::from(n > 1)], &[0, 1 << (D - 1)]).unwrap();
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
            vec![vec![vec![(0, F::ONE)]]],
        )
    }

    fn selected_polynomial() -> Self {
        static POLYNOMIAL: OnceLock<SparsePoly<F>> = OnceLock::new();
        let polynomial = POLYNOMIAL.get_or_init(|| {
            let path = std::path::Path::new(env!("CARGO_MANIFEST_DIR"))
                .join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json");
            nightstream_fprime::load_poseidon2_hash_chain_v1_package(&std::fs::read(path).unwrap())
                .unwrap()
                .ccs_structure_header()
                .unwrap()
                .f
        });
        let count = polynomial.terms()[0].exps.len();
        // Exercise the complete exported polynomial with nonzero input ports.
        // Its C port (slot 4) is affine; solve that coordinate so the fresh row
        // is valid. The declared final zero matrix stays zero.
        assert!(polynomial.terms().iter().all(|term| term.exps[4] <= 1));
        let mut matrices = vec![Vec::new(); count];
        // Two distinct rows exercise interpolation across rows and two witness
        // columns. This count comes from the requested multi-row parity case.
        for (row, column) in [D, 0].into_iter().enumerate() {
            let mut values: Vec<_> = (1..=count)
                .map(|value| F::from_usize(value + row * count))
                .collect();
            values[count - 1] = F::ZERO;
            values[4] = F::ZERO;
            let constant = polynomial.eval(&values);
            values[4] = F::ONE;
            let coefficient = polynomial.eval(&values) - constant;
            assert_ne!(coefficient, F::ZERO);
            values[4] = -constant * coefficient.inverse();
            assert_eq!(polynomial.eval(&values), F::ZERO);
            for (matrix, value) in matrices.iter_mut().zip(values) {
                matrix.push(if value == F::ZERO {
                    vec![]
                } else {
                    vec![(column, value)]
                });
            }
        }
        Self::new(polynomial.clone(), matrices)
    }
}

#[derive(Clone)]
struct Run {
    running: RunningInstance,
    proof: NifsProof,
    transcript: Poseidon2TranscriptSnapshot,
}

fn run(engine: Engine, fixture: &Fixture) -> Result<Run, Box<dyn std::error::Error>> {
    let mut transcript = Transcript::session();
    let matrix_rows = CachedMatrixRows::new(&fixture.cache)?;
    let workspace_bytes = fixture.workspace_bytes(&matrix_rows);
    let (running, proof) = match Backend::new(engine)? {
        Backend::Optimized => folding::prove_owned_with_rows(
            &mut transcript,
            &fixture.params,
            &fixture.structure,
            &matrix_rows,
            workspace_bytes,
            vec![fixture.fresh.clone()],
            fixture.running.clone(),
        )?,
        Backend::PaperExact => super::paper_exact::prove(
            &mut transcript,
            &fixture.params,
            &fixture.structure,
            &fixture.rows,
            vec![fixture.fresh.clone()],
            fixture.running.clone(),
        )?,
        Backend::Crosscheck => {
            let rows = WorkerRows {
                rows: &fixture.rows,
                caller: std::thread::current().id(),
                called: AtomicBool::new(false),
            };
            let result = super::crosscheck::prove(
                &mut transcript,
                &fixture.params,
                &fixture.structure,
                &matrix_rows,
                workspace_bytes,
                &rows,
                vec![fixture.fresh.clone()],
                fixture.running.clone(),
            )?;
            assert!(
                rows.called.load(Ordering::Relaxed),
                "reference must read the original rows"
            );
            result
        }
        #[cfg(feature = "metal")]
        Backend::Metal(device) => {
            let mut device = device.lock().unwrap();
            let result = super::metal::prove(
                &mut device,
                &mut transcript,
                &fixture.params,
                &fixture.structure,
                &matrix_rows,
                workspace_bytes,
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
fn crosscheck_matches_optimized() {
    let fixture = Fixture::bit();
    equal(
        run(Engine::Crosscheck, &fixture).unwrap(),
        run(Engine::Optimized, &fixture).unwrap(),
    );
}

#[test]
fn selected_polynomial_crosscheck_matches_optimized() {
    let fixture = Fixture::selected_polynomial();
    equal(
        run(Engine::Crosscheck, &fixture).unwrap(),
        run(Engine::Optimized, &fixture).unwrap(),
    );
}

#[test]
fn crosscheck_rejects_changed_results() {
    let original = run(Engine::Optimized, &Fixture::bit()).unwrap();
    let check = |changed: &Run, expected| {
        let result = super::crosscheck::require_match(
            original.transcript,
            &original.running,
            &original.proof,
            changed.transcript,
            &changed.running,
            &changed.proof,
        );
        assert!(matches!(result, Err(EngineError::CrosscheckMismatch { boundary }) if boundary == expected));
    };

    let mut changed = original.clone();
    changed.proof.pi_ccs.sumcheck.sumcheck_rounds[0][0] += K::ONE;
    check(&changed, "proof fields");

    let mut changed = original.clone();
    changed.running.witnesses[0][(0, 0)] += F::ONE;
    check(&changed, "running accumulator");

    let mut changed = original.clone();
    changed.running.claims[0].eval_k[0] += K::ONE;
    check(&changed, "running accumulator");

    let mut changed = original.clone();
    changed.running.parent_authority.as_mut().unwrap().eval_k[0] += K::ONE;
    check(&changed, "running accumulator");

    let mut changed = original.clone();
    let mut state = original.transcript.state();
    state[0] += F::ONE;
    let mut transcript = Transcript::session();
    *transcript.inner_mut() =
        neo_transcript::Poseidon2Transcript::from_state_and_absorbed(state, original.transcript.absorbed());
    changed.transcript = transcript.snapshot();
    check(&changed, "prover transcript");

    assert_eq!(original.transcript.absorbed(), 0);
    *transcript.inner_mut() =
        neo_transcript::Poseidon2Transcript::from_state_and_absorbed(original.transcript.state(), 1);
    changed.transcript = transcript.snapshot();
    check(&changed, "prover transcript");
}

#[test]
fn crosscheck_rejects_different_rows_without_updating_the_transcript() {
    let mut fixture = Fixture::bit();
    // Both x^2 - x relations accept this fresh bit, but their openings differ.
    // Deliberately give the reference different rows from the optimized cache.
    fixture.rows.matrices[0][0].clear();
    fixture.running = RunningInstance::canonical_zero(&fixture.params, &fixture.structure, D).unwrap();
    let matrix_rows = CachedMatrixRows::new(&fixture.cache).unwrap();
    let workspace_bytes = fixture.workspace_bytes(&matrix_rows);
    let mut transcript = Transcript::session();
    let before = transcript.snapshot();
    let result = super::crosscheck::prove(
        &mut transcript,
        &fixture.params,
        &fixture.structure,
        &matrix_rows,
        workspace_bytes,
        &fixture.rows,
        vec![fixture.fresh],
        fixture.running,
    );
    assert!(matches!(result, Err(EngineError::CrosscheckMismatch { .. })));
    assert_eq!(transcript.snapshot(), before);
}

#[test]
fn crosscheck_rejects_a_prover_error_without_updating_the_transcript() {
    let mut fixture = Fixture::bit();
    // Only the reference loses its required matrix. The optimized cache is valid.
    fixture.rows.matrices.clear();
    let matrix_rows = CachedMatrixRows::new(&fixture.cache).unwrap();
    let workspace_bytes = fixture.workspace_bytes(&matrix_rows);
    let mut transcript = Transcript::session();
    let before = transcript.snapshot();
    let result = super::crosscheck::prove(
        &mut transcript,
        &fixture.params,
        &fixture.structure,
        &matrix_rows,
        workspace_bytes,
        &fixture.rows,
        vec![fixture.fresh],
        fixture.running,
    );
    assert!(matches!(
        result,
        Err(EngineError::CrosscheckMismatch { boundary: "acceptance" })
    ));
    assert_eq!(transcript.snapshot(), before);
}

#[test]
fn crosscheck_rejects_a_worker_panic_without_updating_the_transcript() {
    struct PanickingRows;
    impl PaperMatrixRows<F> for PanickingRows {
        fn shape(&self) -> (usize, usize, usize) {
            (1, D + 1, 1)
        }
        fn row(&self, _: usize, _: usize) -> Vec<(usize, F)> {
            panic!("reference worker failed");
        }
    }
    let fixture = Fixture::bit();
    let matrix_rows = CachedMatrixRows::new(&fixture.cache).unwrap();
    let workspace_bytes = fixture.workspace_bytes(&matrix_rows);
    let mut transcript = Transcript::session();
    let before = transcript.snapshot();
    let result = super::crosscheck::prove(
        &mut transcript,
        &fixture.params,
        &fixture.structure,
        &matrix_rows,
        workspace_bytes,
        &PanickingRows,
        vec![fixture.fresh],
        fixture.running,
    );
    assert!(matches!(
        result,
        Err(EngineError::Failure {
            engine: Engine::PaperExact,
            ..
        })
    ));
    assert_eq!(transcript.snapshot(), before);
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
