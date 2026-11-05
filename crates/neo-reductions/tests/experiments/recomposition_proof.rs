//! Exact complete PiCCS comparison with a factored linear-map backend.
//!
//! Both variants enter through `PaperJointOracleBackend`, so both get the same
//! phase timers and matrix-row counter. The production variant builds the
//! unchanged `OptimizedPaperJointOracle::new`.
use super::*;
use crate::engines::pi_ccs_joint_protocol::V1_2OutputOpening;
use crate::optimized_engine::{
    optimized_prove_with_matrix_rows, optimized_verify_with_trace, PaperJointOracleBackend, PaperJointOracleInput,
};
use neo_ajtai::Commitment;
use neo_ccs::{CcsClaim, CeClaim};
use neo_transcript::Poseidon2Transcript;
use std::{
    cell::RefCell,
    collections::BTreeMap,
    sync::atomic::{AtomicUsize, Ordering},
};

/// Wall-clock seconds and matrix rows visited for each named phase of one proof.
#[derive(Default)]
struct Phases {
    rows_visited: AtomicUsize,
    entries: RefCell<BTreeMap<&'static str, (f64, usize)>>,
}
impl Phases {
    fn time<T>(&self, name: &'static str, f: impl FnOnce() -> T) -> T {
        let rows = self.rows_visited.load(Ordering::Relaxed);
        let (value, elapsed) = seconds(f);
        let visited = self.rows_visited.load(Ordering::Relaxed) - rows;
        let mut entries = self.entries.borrow_mut();
        let entry = entries.entry(name).or_default();
        entry.0 += elapsed;
        entry.1 += visited;
        value
    }
    /// Phases as JSON; `other` is the proof time outside the oracle phases.
    /// `matrix_passes` counts complete visits of all matrix rows.
    fn report(&self, total: f64, rows: usize) -> serde_json::Value {
        let entries = self.entries.borrow();
        let mut report: serde_json::Map<_, _> = entries
            .iter()
            .map(|(name, &(seconds, visited))| {
                let passes = visited as f64 / rows as f64;
                (name.to_string(), json!({"seconds": seconds, "matrix_passes": passes}))
            })
            .collect();
        let inside: f64 = entries.values().map(|&(seconds, _)| seconds).sum();
        let outside = self.rows_visited.load(Ordering::Relaxed) - entries.values().map(|&(_, v)| v).sum::<usize>();
        report.insert(
            "other".into(),
            json!({"seconds": total - inside, "matrix_passes": outside as f64 / rows as f64}),
        );
        report.into()
    }
}

/// Counts the matrix rows that each window build or row pass requests.
struct Counted<'a> {
    inner: &'a dyn MatrixRows,
    rows_visited: &'a AtomicUsize,
}
impl MatrixRows for Counted<'_> {
    fn shape(&self) -> MatrixShape {
        self.inner.shape()
    }
    fn retained_window(&self) -> Option<&RetainedMatrixWindow> {
        self.inner.retained_window()
    }
    fn visit_rows(&self, range: Range<usize>, sink: &mut dyn MatrixRowSink) -> Result<(), PiCcsError> {
        self.rows_visited.fetch_add(range.len(), Ordering::Relaxed);
        self.inner.visit_rows(range, sink)
    }
}

/// Times each oracle call. Round 0 is separate because the production oracle
/// builds its fresh tables lazily in the first round.
struct Timed<'a> {
    inner: Box<dyn PaperJointRoundOracle + 'a>,
    phases: &'a Phases,
    round: usize,
}
impl PaperJointRoundOracle for Timed<'_> {
    fn evals_at(&mut self, points: &[K]) -> Result<Vec<K>, PiCcsError> {
        let name = if self.round == 0 { "round_0" } else { "rounds" };
        self.phases.time(name, || self.inner.evals_at(points))
    }
    fn num_rounds(&self) -> usize {
        self.inner.num_rounds()
    }
    fn degree_bound(&self) -> usize {
        self.inner.degree_bound()
    }
    fn fold(&mut self, challenge: K) -> Result<(), PiCcsError> {
        self.round += 1;
        self.phases.time("folds", || self.inner.fold(challenge))
    }
    fn output_openings(&mut self, point: &[K]) -> Result<Option<Vec<V1_2OutputOpening>>, PiCcsError> {
        self.phases
            .time("openings", || self.inner.output_openings(point))
    }
}

/// Builds the production oracle when `plan` is `None`, else the factored one.
struct Backend<'a> {
    plan: Option<&'a Plan>,
    phases: &'a Phases,
}
impl PaperJointOracleBackend for Backend<'_> {
    fn create<'a>(
        &'a mut self,
        input: PaperJointOracleInput<'a>,
    ) -> Result<Box<dyn PaperJointRoundOracle + 'a>, PiCcsError> {
        let plan = self.plan;
        let inner = self.phases.time("create", || match plan {
            Some(plan) => factored(plan, input),
            None => OptimizedPaperJointOracle::new(
                input.structure,
                input.params,
                input.fresh_witnesses,
                input.running_witnesses,
                input.challenges,
                input.prior_point,
                input.dims,
                input.rows,
                input.workspace_bytes,
            ),
        })?;
        Ok(Box::new(Timed {
            inner: Box::new(inner),
            phases: self.phases,
            round: 0,
        }))
    }
}

fn factored<'a>(plan: &Plan, input: PaperJointOracleInput<'a>) -> Result<OptimizedPaperJointOracle<'a>, PiCcsError> {
    assert_eq!(input.fresh_witnesses.len(), 1);
    let witnesses: Vec<_> = input
        .fresh_witnesses
        .iter()
        .map(|w| &w.Z)
        .chain(input.running_witnesses)
        .collect();
    let witness_blocks: Vec<_> = witnesses
        .iter()
        .map(|w| SuperneoZBlocks::from_witness_mat(w, input.structure.m()))
        .collect::<Result<_, _>>()?;
    let assignments = witnesses
        .iter()
        .map(|w| prefix::Assignment::new(w, input.dims.assignment_width))
        .collect();
    let mut fresh_tables = ApplicationTables::new(input.rows, &witness_blocks[..1], input.workspace_bytes)?;
    fresh_tables.resident = Some(Values {
        words: plan.fresh(&input.fresh_witnesses[0].Z),
        rows: plan.rows,
    });
    let running = &witness_blocks[1..];
    let gamma = input.challenges.gamma;
    let width = input.dims.assignment_width;
    let evaluation_table = if running.iter().all(SuperneoZBlocks::is_zero) {
        Vec::new()
    } else {
        let powers: Vec<_> = (0..running.len()).map(|i| gamma_power(gamma, i)).collect();
        let weights = std::array::from_fn(|i| gamma_power(gamma, running.len() * 4 * i));
        let coefficients: Vec<_> = (0..4)
            .map(|i| gamma_power(gamma, running.len() * i))
            .collect();
        let mut result = vec![K::ZERO; width.max(plan.rows)];
        fill_combined_projection(running, &powers, &weights, &mut result[..width]);
        let matrix = plan.weighted(&result[..width], &coefficients);
        let weights = std::array::from_fn(|i| gamma_power(gamma, running.len() * i));
        fill_combined_projection(running, &powers, &weights, &mut result[..width]);
        let shift = gamma_power(gamma, running.len() * D);
        for (out, value) in result.iter_mut().zip(matrix) {
            *out += shift * value;
        }
        while result.last() == Some(&K::ZERO) {
            result.pop();
        }
        result
    };
    Ok(OptimizedPaperJointOracle {
        structure: input.structure,
        source: input.rows,
        workspace_bytes: input.workspace_bytes,
        base: input.params.b(),
        challenges: input.challenges.clone(),
        dims: input.dims,
        point: Vec::new(),
        fixed_equality: K::ONE,
        prior_point: input.prior_point.map(<[K]>::to_vec),
        fixed_prior_equality: K::ONE,
        fresh_tables,
        assignments,
        witness_blocks,
        evaluation_table,
        constraint_shift: gamma_power(gamma, input.running_witnesses.len() * D * (input.structure.t() + 1)),
    })
}

#[test]
#[ignore = "reads a JSON request on stdin; run through experiments/recomposition/run.py"]
fn recomposition_proofs() {
    let mut request = String::new();
    std::io::stdin().read_to_string(&mut request).unwrap();
    let request: Request = serde_json::from_str(&request).unwrap();
    let pool = rayon::ThreadPoolBuilder::new()
        .num_threads(THREADS)
        .build()
        .unwrap();
    pool.install(|| {
        let directory = &request.directory;
        let step = request.step;
        let package =
            nightstream_fprime::load_prepared_package(fs::File::open(directory.join("package.bin")).unwrap()).unwrap();
        let witnesses = read_witnesses(directory.join(format!("witnesses-{step}.bin")));
        let claims = fs::read(directory.join(format!("claims-{step}.json"))).unwrap();
        let (claim, running): (CcsClaim<Commitment, F>, Vec<CeClaim<Commitment, F, K>>) =
            serde_json::from_slice(&claims).unwrap();
        // One kept window for all proofs of this process, as `Circuit` shares
        // one across its lifecycles; the warmup pair builds it.
        let retained = RetainedMatrixWindow::default();
        let rows = Rows(&package, Some(&retained));
        let workspace: serde_json::Value =
            serde_json::from_slice(&fs::read(directory.join("workspace.json")).unwrap()).unwrap();
        let workspace_bytes = workspace["matrix_workspace_bytes"].as_u64().unwrap() as usize;
        let row_count = rows.shape().rows;
        let structure = package.ccs_structure_header().unwrap();
        let params = neo_params::NeoParams::nightstream_goldilocks_k16()
            .with_lambda(114)
            .unwrap();
        let (plan, setup_seconds) = seconds(|| Plan::new(&package));
        let fresh = vec![CcsWitness {
            Z: witnesses[0].clone(),
        }];
        let claims = vec![claim];
        // Trial 0 is a warmup pair. Alternate the order inside each pair.
        for trial in 0..=request.pairs() {
            let mut reference = None;
            let order = if trial % 2 == 0 { [false, true] } else { [true, false] };
            for factor in order {
                let phases = Phases::default();
                let source = Counted {
                    inner: &rows,
                    rows_visited: &phases.rows_visited,
                };
                let mut backend = Backend {
                    plan: factor.then_some(&plan),
                    phases: &phases,
                };
                let (result, elapsed) = seconds(|| {
                    optimized_prove_with_matrix_rows(
                        &mut Poseidon2Transcript::new_v1_2(),
                        &params,
                        &structure,
                        &claims,
                        &fresh,
                        &running,
                        &witnesses[1..],
                        &source,
                        workspace_bytes,
                        Some(&mut backend),
                    )
                    .unwrap()
                });
                let report = phases.report(elapsed, row_count);
                let (accepted, _) = optimized_verify_with_trace(
                    &mut Poseidon2Transcript::new_v1_2(),
                    &params,
                    &structure,
                    &claims,
                    &running,
                    &result.0,
                    &result.1,
                )
                .unwrap();
                assert!(accepted);
                let serialized = serde_json::to_vec(&(&result.0, &result.1.sumcheck_rounds)).unwrap();
                match &reference {
                    Some(expected) => assert_eq!(&serialized, expected),
                    None => reference = Some(serialized),
                }
                if trial != 0 {
                    let variant = if factor { "factor" } else { "production" };
                    let record = json!({
                        "event": "proof",
                        "step": step,
                        "trial": trial,
                        "variant": variant,
                        "seconds": elapsed,
                        "phases": report,
                        "setup_seconds": setup_seconds,
                        "workspace_bytes": workspace_bytes,
                        "verified": true,
                        "exact_equal": true,
                    });
                    println!("EXPERIMENT {record}");
                }
            }
        }
    });
}
