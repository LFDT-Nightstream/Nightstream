//! The first Nebula memory application: native segment runs and spec §13
//! terminal checks, and (ignored, production profile) proofs over the
//! Lean-emitted package.

use nightstream::nebula::{state_words, Context, MachineState, Plan, Run, Step, TerminalError};
use p3_goldilocks::Goldilocks as F;

/// The Stage 1 input state that an invocation's own words open to.
fn input_state(words: &[F]) -> [F; 4] {
    state_words(&words[0..2], &words[2..41])
}

fn program_run() -> Run {
    let context = Context::new(Plan::first()).expect("first plan");
    Run::new(context, MachineState::default())
}

#[test]
fn native_run_closes_every_segment_and_passes_the_terminal() {
    let mut run = program_run();
    assert_eq!(run.context().witness_word_count(), 600);
    let mut previous = run.initial_state();
    let segments = [
        [Step::Execute, Step::Execute],
        [Step::Execute, Step::Execute],
        [Step::Idle, Step::Idle],
        [Step::Idle, Step::Idle],
    ];
    for steps in segments {
        let invocations = run.segment(&steps).expect("segment runs");
        assert_eq!(invocations.len(), 2);
        for invocation in invocations {
            assert_eq!(invocation.words.len(), 600);
            assert_eq!(input_state(&invocation.words), previous, "state link");
            previous = invocation.output;
        }
        assert_eq!(run.carry().idx, 2, "every segment closes");
    }
    let statement = run.statement();
    assert_eq!(statement.steps, 8);
    assert_eq!(statement.segments, 4);
    assert_eq!(statement.final_, MachineState { pc: 3, acc: 5 });
    assert_eq!(statement.final_ts, 6, "loadi 1, store 2, load 2, halt 1 accesses");
    let (initial, last) = run
        .context()
        .terminal_states(&statement, run.carry())
        .expect("terminal checks pass");
    assert_eq!(initial, run.initial_state());
    assert_eq!(last, previous);

    let mut wrong = statement.clone();
    wrong.segments = 3;
    assert_eq!(
        run.context().terminal_states(&wrong, run.carry()),
        Err(TerminalError::StepCount)
    );
    let mut wrong = statement;
    wrong.final_ts += 1;
    assert_eq!(
        run.context().terminal_states(&wrong, run.carry()),
        Err(TerminalError::Field)
    );
}

fn memory_package() -> Vec<u8> {
    std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts/nightstream-fprime-stage2-nebula-memory-v1.json"),
    )
    .expect("saved Lean memory package")
}

/// One complete memory proof: one segment of two invocations, then the spec
/// §13 terminal checks and the Stage 1 terminal verification.
fn one_segment_proof(engine: nightstream::Engine) {
    use nightstream::{Circuit, State, Verifier};
    let started = std::time::Instant::now();
    let circuit = Circuit::load_package(&memory_package(), nightstream::nebula::PACKAGE_STRUCTURAL_IDENTIFIER)
        .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, engine, 114).unwrap();
    eprintln!("memory preparation elapsed={:?}", started.elapsed());

    let mut run = program_run();
    let z0 = run.initial_state();
    let invocations = run.segment(&[Step::Execute, Step::Execute]).unwrap();
    let proving = std::time::Instant::now();
    let mut proof = prover
        .prove_with_output(z0, &invocations[0].words, invocations[0].output)
        .unwrap();
    eprintln!("memory base proving elapsed={:?}", proving.elapsed());
    for invocation in &invocations[1..] {
        let proving = std::time::Instant::now();
        proof = prover
            .extend_with_output(&proof, &invocation.words, invocation.output)
            .unwrap();
        eprintln!("memory fold proving elapsed={:?}", proving.elapsed());
    }
    let statement = run.statement();
    let (initial, last) = run
        .context()
        .terminal_states(&statement, run.carry())
        .unwrap();
    let expected = State::new(statement.steps, initial, last);
    assert_eq!(proof.state(), &expected);
    let verifying = std::time::Instant::now();
    verifier.verify(&expected, &proof).unwrap();
    eprintln!("memory verification elapsed={:?}", verifying.elapsed());
}

#[test]
#[ignore = "Full production-profile memory proof; run separately under the 300-second cap."]
fn memory_segment_proves_and_verifies() {
    one_segment_proof(nightstream::Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production-profile memory proof on Metal; run separately under the 300-second cap."]
fn memory_segment_proves_and_verifies_on_metal() {
    one_segment_proof(nightstream::Engine::Metal);
}

#[test]
#[ignore = "Prints the saved memory package's structural identifier for pinning."]
fn print_memory_package_identifier() {
    let value: serde_json::Value = serde_json::from_slice(&memory_package()).unwrap();
    let package = nightstream_fprime::load_prepared_application_value(value).unwrap();
    println!("structural_identifier={:?}", package.structural_identifier());
}
