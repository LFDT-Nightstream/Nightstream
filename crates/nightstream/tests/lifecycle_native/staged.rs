//! Capped recursive test phases. Checkpoints are prover data, never authority.
//! Record shapes and replay checks are adapted from neo-fold-clean's
//! tests/nifs/{stage1_actual,stage1_parent,stage1_assemble}.rs at 73b6e2de.

use super::{artifact, field, fields, output, read};
use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_dec, pi_rlc, transcript::Transcript, CcsClaim, CcsInstance,
    CcsWitness, CeClaim, NifsProof, Params, RunningInstance,
};
use crate::lifecycle::{extend::prepare_running, PreparedLifecycle, Stage1Envelope, Stage1State};
use neo_ajtai::{
    nightstream_fprime_setup::{
        commit_production_signed_unit_prefix_matrices, commit_production_signed_unit_prefix_matrix,
    },
    Commitment,
};
use neo_ccs::{Mat, V1_1Evaluations};
use neo_math::{D, F, K};
use neo_reductions::common::{project_x_from_witness_mat, validate_fresh_witness_tail_zero};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{de::DeserializeOwned, Deserialize, Serialize};
use serde_json::{json, Value};
use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
    time::Instant,
};

#[path = "staged_fold.rs"]
mod fold;
#[path = "staged_terminal.rs"]
mod terminal;

#[derive(Clone, Copy, Debug, Default, Deserialize)]
#[serde(rename_all = "snake_case")]
enum EvaluationEngine {
    #[default]
    Optimized,
    #[cfg(feature = "metal")]
    Metal,
}

#[derive(Debug, Deserialize)]
#[serde(tag = "phase", rename_all = "snake_case", deny_unknown_fields)]
enum Request {
    Base {
        directory: PathBuf,
        #[serde(default)]
        engine: EvaluationEngine,
    },
    Sources {
        directory: PathBuf,
        step: u64,
        #[serde(default)]
        engine: EvaluationEngine,
    },
    Ccs {
        directory: PathBuf,
        step: u64,
        #[serde(default)]
        engine: EvaluationEngine,
        cpu_reference: Option<PathBuf>,
    },
    Rlc {
        directory: PathBuf,
        step: u64,
    },
    Split {
        directory: PathBuf,
        step: u64,
    },
    Openings {
        directory: PathBuf,
        step: u64,
    },
    Child {
        directory: PathBuf,
        step: u64,
        child: usize,
        #[serde(default)]
        engine: EvaluationEngine,
    },
    Nifs {
        directory: PathBuf,
        step: u64,
    },
    Prove {
        directory: PathBuf,
        step: u64,
        engine: EvaluationEngine,
        reference_proof: PathBuf,
    },
    Successor {
        directory: PathBuf,
        step: u64,
        #[serde(default)]
        engine: EvaluationEngine,
    },
    Terminal {
        directory: PathBuf,
        #[serde(default)]
        engine: EvaluationEngine,
    },
    Mutation {
        directory: PathBuf,
    },
    Reject {
        directory: PathBuf,
        #[serde(default)]
        engine: EvaluationEngine,
    },
}

#[test]
#[ignore = "Explicit staged full-profile execution; provide one JSON request on stdin and apply the project 300-second process cap."]
fn run_phase() {
    let request: Request = serde_json::from_reader(std::io::stdin().lock()).expect("strict phase request JSON");
    let started = Instant::now();
    eprintln!("staged request={request:?}");
    match request {
        Request::Base { directory, engine } => base(&directory, engine),
        Request::Sources {
            directory,
            step,
            engine,
        } => sources(&directory, step, engine),
        Request::Ccs {
            directory,
            step,
            engine,
            cpu_reference,
        } => fold::ccs(&directory, step, engine, cpu_reference.as_deref()),
        Request::Rlc { directory, step } => fold::rlc(&directory, step),
        Request::Split { directory, step } => fold::split(&directory, step),
        Request::Openings { directory, step } => fold::openings(&directory, step),
        Request::Child {
            directory,
            step,
            child,
            engine,
        } => fold::child(&directory, step, child, engine),
        Request::Nifs { directory, step } => fold::nifs(&directory, step),
        Request::Prove {
            directory,
            step,
            engine,
            reference_proof,
        } => fold::prove(&directory, step, engine, &reference_proof),
        Request::Successor {
            directory,
            step,
            engine,
        } => terminal::successor(&directory, step, engine),
        Request::Terminal { directory, engine } => terminal::accept(&directory, engine),
        Request::Mutation { directory } => terminal::mutation(&directory),
        Request::Reject { directory, engine } => terminal::reject(&directory, engine),
    }
    eprintln!("staged phase passed elapsed={:?}", started.elapsed());
}

fn save<T: Serialize>(path: &Path, value: &T) {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("fresh checkpoint file");
    let mut output = BufWriter::new(file);
    serde_json::to_writer(&mut output, value).expect("checkpoint serialization");
    output.write_all(b"\n").unwrap();
    output.flush().unwrap();
}
fn save_bytes(path: &Path, bytes: &[u8]) {
    let mut file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("fresh proof bytes");
    file.write_all(bytes).unwrap();
    file.flush().unwrap();
}
fn load<T: DeserializeOwned>(path: &Path) -> T {
    serde_json::from_reader(BufReader::new(File::open(path).expect("checkpoint input"))).expect("typed checkpoint data")
}
fn fold_dir(root: &Path, step: u64) -> PathBuf {
    assert!(matches!(step, 1 | 2), "this test covers the two requested fresh folds");
    root.join(format!("fold-{step}"))
}
fn step_dir(root: &Path, step: u64) -> PathBuf {
    root.join(format!("step-{step}"))
}
fn prepare() -> PreparedLifecycle {
    prepare_with_engine(EvaluationEngine::Optimized)
}
fn prepare_with_engine(engine: EvaluationEngine) -> PreparedLifecycle {
    let started = Instant::now();
    let prover = match engine {
        EvaluationEngine::Optimized => crate::engine::Prover::Optimized,
        #[cfg(feature = "metal")]
        EvaluationEngine::Metal => crate::engine::Prover::new(crate::Engine::Metal).unwrap(),
    };
    let application = crate::application::poseidon2_hash_chain_v1().unwrap();
    let reference = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let (package, binding) = crate::assembly::prepare(&reference, &application).unwrap();
    let package = PreparedLifecycle::from_package(package, binding, prover).unwrap();
    eprintln!("staged circuit preparation elapsed={:?}", started.elapsed());
    package
}
fn params(package: &PreparedLifecycle) -> Params {
    Params::for_ccs_shape(
        package.structure.n,
        package.structure.m,
        package.structure.t(),
        package.structure.max_degree(),
    )
    .unwrap()
}
fn message() -> [F; 4] {
    let request: Value = load(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_recursive_states/nonzero-running.json"),
    );
    fields(&request[3]).try_into().unwrap()
}
fn expected_state(step: u64) -> Stage1State {
    let request: Value = load(
        &PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_recursive_states/nonzero-running.json"),
    );
    let initial: [F; 4] = fields(&request[1]).try_into().unwrap();
    let second: [F; 4] = fields(&request[2]).try_into().unwrap();
    let current = match step {
        1 => {
            let base = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
            fields(&base[4][0]).try_into().unwrap()
        }
        2 => second,
        3 => output(second, message()),
        _ => panic!("expected base and two recursive outputs"),
    };
    Stage1State::new(step, initial, current)
}

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedEnvelope {
    schema: u64,
    package_identity: [u64; 4],
    iteration: u64,
    z0: [u64; 4],
    current: [u64; 4],
    child_witness_count: usize,
    running_claims: Vec<CeClaim>,
    running_parent: Option<CeClaim>,
}
struct Sources {
    state: Stage1State,
    fresh: CcsInstance,
    running: RunningInstance,
}
fn load_claims(package: &PreparedLifecycle, directory: &Path, step: u64) -> (Stage1State, CcsClaim, RunningInstance) {
    let saved: SavedEnvelope = load(&directory.join("envelope.json"));
    assert_eq!(saved.schema, 1);
    assert_eq!(saved.package_identity, package.package_identity());
    assert_eq!(saved.child_witness_count, 16);
    assert_eq!(saved.running_claims.len(), 16);
    let state = Stage1State::new(saved.iteration, saved.z0.map(field), saved.current.map(field));
    assert_eq!(state, expected_state(step), "external source state");
    let fresh: CcsClaim = load(&directory.join("fresh-claim.json"));
    let mut running = RunningInstance::new(saved.running_claims, Vec::new(), saved.running_parent);
    let (_, digest) = package
        .checked_prior_state(&state, &running, &fresh)
        .unwrap();
    // Use the production normalization; supplied parent/frame caches are not authority.
    let params = params(package);
    prepare_running(&mut running, &params, digest);
    folding::validate_running_parent_authority(&params, &package.structure, ajtai_dec_mixer, &running).unwrap();
    (state, fresh, running)
}
fn load_sources(package: &PreparedLifecycle, directory: &Path, step: u64) -> Sources {
    let (state, claim, mut running) = load_claims(package, directory, step);
    let witness: Mat<F> = load(&directory.join("fresh-witness.json"));
    validate_fresh_witness_tail_zero(&witness, package.structure.m, "staged fresh source").unwrap();
    let projected = project_x_from_witness_mat(&witness, package.structure.m, claim.m_in).unwrap();
    for (column, value) in claim.x.iter().enumerate() {
        assert_eq!(projected[(column % D, column / D)], *value);
    }
    for child in 0..16 {
        let witness: Mat<F> = load(&directory.join(format!("digit-{child}.json")));
        let claim = &running.claims[child];
        assert_eq!(
            project_x_from_witness_mat(&witness, package.structure.m, claim.m_in).unwrap(),
            claim.X
        );
        running.witnesses.push(witness);
    }
    Sources {
        state,
        fresh: CcsInstance {
            claim,
            witness: CcsWitness {
                w: Vec::new(),
                Z: witness,
            },
        },
        running,
    }
}
fn load_envelope(package: &PreparedLifecycle, directory: &Path, step: u64) -> Stage1Envelope {
    let source = load_sources(package, directory, step);
    Stage1Envelope::from_parts(source.state, source.running, source.fresh)
}
fn save_envelope(
    package: &PreparedLifecycle,
    envelope: &Stage1Envelope,
    directory: &Path,
    digit_directory: Option<&Path>,
) {
    fs::create_dir(directory).expect("fresh envelope directory");
    let running = envelope.running().unwrap();
    assert_eq!(running.claims.len(), 16);
    assert_eq!(running.witnesses.len(), 16);
    for (child, witness) in running.witnesses.iter().enumerate() {
        let name = format!("digit-{child}.json");
        if let Some(source) = digit_directory {
            // These are exactly the checked digit objects moved into complete_step.
            // The link operation, like create_new, rejects an existing destination.
            fs::hard_link(source.join(&name), directory.join(name)).expect("new link to checked digit");
        } else {
            save(&directory.join(name), witness);
        }
    }
    let fresh = envelope.fresh().unwrap();
    save(&directory.join("fresh-claim.json"), &fresh.claim);
    save(&directory.join("fresh-witness.json"), &fresh.witness.Z);
    let state = envelope.state();
    save(
        &directory.join("envelope.json"),
        &SavedEnvelope {
            schema: 1,
            package_identity: package.package_identity(),
            iteration: state.iteration(),
            z0: state.z0().map(|value| value.as_canonical_u64()),
            current: state.current().map(|value| value.as_canonical_u64()),
            child_witness_count: running.witnesses.len(),
            running_claims: running.claims.clone(),
            running_parent: running.parent_authority.clone(),
        },
    );
}
fn base(root: &Path, engine: EvaluationEngine) {
    let package = prepare_with_engine(engine);
    let expected = expected_state(1);
    let message = message();
    let output = output(expected.z0(), message);
    let envelope = package
        .extend_with_output(
            Stage1Envelope::initial(expected.z0()),
            &message.map(|value| value.as_canonical_u64()),
            output,
        )
        .unwrap();
    assert_eq!(envelope.state(), &expected);
    save_envelope(&package, &envelope, &step_dir(root, 1), None);
}
fn sources(root: &Path, step: u64, engine: EvaluationEngine) {
    let destination = fold_dir(root, step);
    fs::create_dir_all(&destination).unwrap();
    let package = prepare_with_engine(engine);
    let source = load_sources(&package, &step_dir(root, step), step);
    let started = Instant::now();
    assert_eq!(
        package
            .prover
            .commit(std::slice::from_ref(&source.fresh.witness.Z))
            .unwrap()
            .remove(0),
        source.fresh.claim.c
    );
    eprintln!(
        "fresh source commitment engine={engine:?} elapsed={:?}",
        started.elapsed()
    );
    let commitments = package.prover.commit(&source.running.witnesses).unwrap();
    for (claim, commitment) in source.running.claims.iter().zip(commitments) {
        assert_eq!(commitment, claim.c);
    }
    #[cfg(feature = "metal")]
    if let crate::engine::Prover::Metal(device) = &package.prover {
        eprintln!(
            "source commitment device activity={:?}",
            device.lock().unwrap().activity()
        );
    }
    save(
        &destination.join("sources-checked.json"),
        &json!({"schema":1,"package_identity":package.package_identity(),"step":step}),
    );
}
