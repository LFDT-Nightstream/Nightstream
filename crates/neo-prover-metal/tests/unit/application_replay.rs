use super::opening::tests::matrix_workspace;
use super::*;
use neo_ccs::{poly::Term, CcsMatrix, CcsStructure, CcsWitness, CscMat, GeometricRowRun, SparsePoly};
use neo_reductions::{
    engines::{
        pi_ccs_joint::{build_joint_dims, JointDims, ProtocolTrace},
        pi_ccs_joint_protocol::{assemble_proof, prove_phase},
    },
    optimized_engine::{canonical_audit::OptimizedPaperJointOracle, OptimizedStructureCache},
    superneo_eval::{check_ccs_relation_zero_cached_with_blocks, CachedMatrixRows, SuperneoCachedRelationError},
    Challenges,
};
use neo_transcript::Poseidon2Transcript;

// The selected profile has fourteen matrices. A row pair is the input to one
// SumCheck fold; replay can hold a base pair and an extension pair together.
const MATRIX_COUNT: usize = 14;
const APPLICATION_WORKSPACE: usize = 2 * MATRIX_COUNT * (size_of::<F>() + size_of::<K>());

struct Fixture {
    structure: CcsStructure<F>,
    cache: OptimizedStructureCache,
    params: neo_params::NeoParams,
    dims: JointDims,
    fresh: [CcsWitness<F>; 1],
    running: Vec<Mat<F>>,
    challenges: Challenges,
    prior_point: Vec<K>,
}

impl Fixture {
    fn new(rows: usize) -> Self {
        let columns = D + 1;
        let mut entries = vec![Vec::new(); MATRIX_COUNT];
        let mut runs = Vec::new();
        for row in 0..rows {
            entries[0].extend([(row, 0, F::from_usize(row + 1)), (row, D, -F::ONE)]);
            entries[1].push((row, 0, F::ONE));
            // An identity-pattern matrix remains rectangular, like the other
            // matrices in the production row-cache interface.
            entries[3].push((row, row, F::ONE));
            for (matrix, terms) in entries.iter_mut().enumerate().skip(4) {
                terms.push((row, (row + matrix) % columns, F::from_usize(row + matrix + 1)));
            }
            // Cross the ring-block boundary and overlap the explicit D term.
            runs.push(GeometricRowRun::new(
                row,
                D - 2,
                3,
                F::from_usize(row + 2),
                F::from_u64(3),
            ));
        }
        let mut output = entries[0].clone();
        for terms in entries.iter().skip(3) {
            output.extend_from_slice(terms);
        }
        // With z[0] = z[D] = 1, every row satisfies A * B + sum(M_j) = C.
        // Changing only z[D] then breaks exactly the final row.
        output.extend([(rows - 1, D, F::ONE), (rows - 1, 0, -F::ONE)]);
        entries[2] = output;
        let matrices = entries
            .into_iter()
            .enumerate()
            .map(|(matrix, terms)| {
                CcsMatrix::csc_with_compact_rows(
                    CscMat::from_triplets(terms, rows, columns),
                    Vec::new(),
                    if matrix == 0 || matrix == 2 {
                        runs.clone()
                    } else {
                        Vec::new()
                    },
                )
                .unwrap()
            })
            .collect();
        let mut product = vec![0; MATRIX_COUNT];
        product[0] = 1;
        product[1] = 1;
        let mut terms = vec![Term {
            coeff: F::ONE,
            exps: product,
        }];
        for matrix in 2..MATRIX_COUNT {
            let mut exps = vec![0; MATRIX_COUNT];
            exps[matrix] = 1;
            terms.push(Term {
                coeff: if matrix == 2 { -F::ONE } else { F::ONE },
                exps,
            });
        }
        let structure = CcsStructure::new_sparse(matrices, SparsePoly::new(MATRIX_COUNT, terms)).unwrap();
        let cache = OptimizedStructureCache::build(&structure).unwrap();
        let mut params = neo_params::NeoParams::nightstream_goldilocks_k16();
        let security = params
            .padded_row_security_summary_for_shape(
                rows,
                columns,
                MATRIX_COUNT,
                structure.max_degree(),
                neo_params::goldilocks_paper_b2::CHALLENGE_ALPHABET.len() as u32,
            )
            .unwrap();
        params.lambda = params.lambda.min(security.security_bits);
        let dims = build_joint_dims(&params, &structure, 1, params.k_rho as usize).unwrap();
        let blocks = columns.div_ceil(D);
        let mut fresh = Mat::zero(D, blocks, F::ZERO);
        for column in 0..columns {
            fresh[(column % D, column / D)] = digit(column);
        }
        fresh[(0, 0)] = F::ONE;
        fresh[(0, 1)] = F::ONE;
        let running = (0..params.k_rho as usize)
            .map(|source| {
                let mut witness = Mat::zero(D, blocks, F::ZERO);
                // Carried witnesses include a nonzero completion tail.
                for column in 0..blocks * D {
                    witness[(column % D, column / D)] = digit(source + 2 * column);
                }
                witness
            })
            .collect();
        let alpha = (0..dims.variables)
            .map(|index| extension(index + 3, index + 1))
            .collect();
        let prior_point = (0..dims.variables)
            .map(|index| extension(index + 5, index + 2))
            .collect();
        Self {
            structure,
            cache,
            params,
            dims,
            fresh: [CcsWitness { w: vec![], Z: fresh }],
            running,
            challenges: Challenges::new(alpha, extension(3, 5)),
            prior_point,
        }
    }

    fn input<'a>(&'a self, source: &'a dyn MatrixRows, workspace_bytes: usize) -> PaperJointOracleInput<'a> {
        PaperJointOracleInput {
            structure: &self.structure,
            params: &self.params,
            fresh_witnesses: &self.fresh,
            running_witnesses: &self.running,
            challenges: self.challenges.clone(),
            prior_point: Some(&self.prior_point),
            dims: self.dims,
            rows: source,
            workspace_bytes,
        }
    }

    fn cpu_oracle<'a>(&'a self, source: &'a dyn MatrixRows, workspace_bytes: usize) -> OptimizedPaperJointOracle<'a> {
        OptimizedPaperJointOracle::new(
            &self.structure,
            &self.params,
            &self.fresh,
            &self.running,
            self.challenges.clone(),
            Some(&self.prior_point),
            self.dims,
            source,
            workspace_bytes,
        )
        .unwrap()
    }
}

fn digit(index: usize) -> F {
    match index % 3 {
        0 => F::ZERO,
        1 => F::ONE,
        _ => -F::ONE,
    }
}

fn extension(real: usize, imaginary: usize) -> K {
    K::from_coeffs([F::from_usize(real), F::from_usize(imaginary)])
}

#[derive(Debug, PartialEq, Eq)]
struct PhaseResult {
    proof_bytes: Vec<u8>,
    trace: ProtocolTrace,
    transcript_state: [F; 8],
    transcript_cursor: usize,
    openings: Vec<V1_1Evaluations<K>>,
}

fn run_phase(oracle: &mut dyn PaperJointRoundOracle, initial: K) -> PhaseResult {
    let mut transcript = Poseidon2Transcript::new_v1_1();
    let mut trace = ProtocolTrace::default();
    let (rounds, point, _) = prove_phase(&mut transcript, &mut trace, initial, oracle).unwrap();
    let openings = oracle.output_openings(&point).unwrap().unwrap();
    PhaseResult {
        proof_bytes: assemble_proof(rounds).canonical_bytes(),
        trace,
        transcript_state: transcript.state(),
        transcript_cursor: transcript.absorbed(),
        openings,
    }
}

struct ReplayObserver<'oracle, 'session> {
    oracle: &'oracle mut MetalPaperJointOracle<'session>,
    resident_after_prefix: Vec<bool>,
    peak_bytes: usize,
}

impl<'oracle, 'session> ReplayObserver<'oracle, 'session> {
    fn new(oracle: &'oracle mut MetalPaperJointOracle<'session>) -> Self {
        Self {
            resident_after_prefix: vec![oracle.application_is_resident()],
            peak_bytes: oracle.application_workspace_peak_bytes(),
            oracle,
        }
    }

    fn record_peak(&mut self) {
        self.peak_bytes = self
            .peak_bytes
            .max(self.oracle.application_workspace_peak_bytes());
    }
}

impl PaperJointRoundOracle for ReplayObserver<'_, '_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, neo_reductions::PiCcsError> {
        let values = self.oracle.evals_at(points)?;
        self.record_peak();
        Ok(values)
    }

    fn num_rounds(&self) -> usize {
        self.oracle.num_rounds()
    }

    fn degree_bound(&self) -> usize {
        self.oracle.degree_bound()
    }

    fn fold(&mut self, challenge: K) -> Result<(), neo_reductions::PiCcsError> {
        self.oracle.fold(challenge)?;
        self.record_peak();
        self.resident_after_prefix
            .push(self.oracle.application_is_resident());
        Ok(())
    }

    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_1Evaluations<K>>>, neo_reductions::PiCcsError> {
        self.oracle.output_openings(point)
    }
}

#[test]
fn replay_matches_cpu_and_resident_rounds_transcript_and_complete_openings() {
    let session = MetalSession::new().unwrap();
    // Seven is the first odd row count whose base table exceeds 672 bytes.
    // Thirteen is the first odd count whose twice-folded K table still exceeds
    // that workspace: ceil(13 / 4) * 14 * 16 = 896 bytes.
    // Seventeen exceeds a 16-base-word workspace per table. This selects W=8:
    // base8 + next4 + destination4 = 16 words. Round two returns a window after
    // two folds; round three accumulates source windows after three folds.
    for (rows, workspace, replay_prefix) in [
        (7, APPLICATION_WORKSPACE, 1),
        (13, APPLICATION_WORKSPACE, 2),
        (17, 16 * MATRIX_COUNT * size_of::<F>(), 3),
    ] {
        let fixture = Fixture::new(rows);
        let source = CachedMatrixRows::new(fixture.cache.superneo()).unwrap();
        let metadata_workspace = matrix_workspace(&source, 0..rows);
        let plan = session
            .prepare_joint_matrix_plan(&source, metadata_workspace)
            .unwrap();
        let matrix_window = session.load_matrix_window(&plan, 0..rows, 0).unwrap();
        assert!(matrix_window.matrices[0].geometric_row_offset_width != 0);
        drop(matrix_window);
        assert!(rows * MATRIX_COUNT * size_of::<F>() > workspace);
        let mut cpu = fixture.cpu_oracle(&source, metadata_workspace);
        let endpoints = cpu.evals_at(&[K::ZERO, K::ONE]).unwrap();
        let initial = endpoints[0] + endpoints[1];
        let expected = run_phase(&mut cpu, initial);
        assert!(expected.trace.round_challenges[..replay_prefix.max(2)]
            .iter()
            .all(|challenge| challenge.as_coeffs()[1] != F::ZERO));

        let mut resident =
            MetalPaperJointOracle::new(&session, plan, fixture.input(&source, metadata_workspace)).unwrap();
        assert!(resident.application_is_resident());
        assert_eq!(run_phase(&mut resident, initial), expected, "resident rows={rows}");

        let plan = session
            .prepare_joint_matrix_plan(&source, metadata_workspace)
            .unwrap();
        let mut replay = MetalPaperJointOracle::new_with_application_workspace(
            &session,
            plan,
            fixture.input(&source, metadata_workspace),
            workspace,
        )
        .unwrap();
        let mut observed = ReplayObserver::new(&mut replay);
        assert_eq!(run_phase(&mut observed, initial), expected, "replay rows={rows}");
        assert!(!observed.resident_after_prefix[0]);
        assert!(
            !observed.resident_after_prefix[replay_prefix],
            "rows={rows}: {replay_prefix} prior challenges must still require replay"
        );
        assert_eq!(observed.resident_after_prefix.last(), Some(&true));
        assert!(observed.peak_bytes > 0);
        assert!(
            observed.peak_bytes <= workspace,
            "rows={rows}: {} bytes exceeds {workspace}",
            observed.peak_bytes
        );
    }
}

#[test]
fn terminal_replay_reports_the_global_row_in_the_final_window() {
    let fixture = Fixture::new(7);
    let session = MetalSession::new().unwrap();
    let source = CachedMatrixRows::new(fixture.cache.superneo()).unwrap();
    let metadata_workspace = matrix_workspace(&source, 0..fixture.structure.n);
    let plan = session
        .prepare_joint_matrix_plan(&source, metadata_workspace)
        .unwrap();
    let mut witness = fixture.fresh[0].Z.clone();
    for expected in [None, Some(6)] {
        let blocks = SuperneoZBlocks::from_witness_mat(&witness, fixture.structure.m).unwrap();
        let cpu =
            match check_ccs_relation_zero_cached_with_blocks(fixture.cache.superneo(), &fixture.structure.f, &blocks) {
                Ok(()) => None,
                Err(SuperneoCachedRelationError::UnsatisfiedRow { row }) => Some(row),
                Err(error) => panic!("unexpected CPU relation error: {error}"),
            };
        assert_eq!(cpu, expected);
        assert_eq!(
            session
                .first_unsatisfied_row(&plan, &fixture.structure, &witness)
                .unwrap(),
            expected
        );
        assert_eq!(
            session
                .first_unsatisfied_row_with_application_workspace(
                    &plan,
                    &fixture.structure,
                    &witness,
                    APPLICATION_WORKSPACE,
                )
                .unwrap(),
            expected,
        );
        witness[(0, 1)] = -F::ONE;
    }
}
