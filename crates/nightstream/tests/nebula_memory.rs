//! The first Nebula memory application: native segment runs and spec §13
//! terminal checks, and (ignored, production profile) spec §14 conformance
//! proofs over the Lean-emitted package.

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

/// The spec §14 accept cases that the first plan reaches, and the rejections
/// that need no invalid witness, through `Context::verify`. One run of four
/// segments (`S = S_max`) has continue arms (`N = 2`), writes RAM in the first
/// segment and reads it in the second, and ends with an idle segment.
fn memory_conformance(engine: nightstream::Engine) {
    use nightstream::nebula::VerifyError;
    use nightstream::{Circuit, Verifier};
    let started = std::time::Instant::now();
    let circuit = Circuit::load_package(&memory_package(), nightstream::nebula::PACKAGE_STRUCTURAL_IDENTIFIER)
        .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = Verifier::from_package(&circuit, engine, 114).unwrap();
    eprintln!("memory preparation elapsed={:?}", started.elapsed());

    let mut run = program_run();
    let z0 = run.initial_state();
    let mut invocations = Vec::new();
    for steps in [
        [Step::Execute, Step::Execute],
        [Step::Execute, Step::Execute],
        [Step::Idle, Step::Idle],
        [Step::Idle, Step::Idle],
    ] {
        invocations.extend(run.segment(&steps).unwrap());
    }
    let proving = std::time::Instant::now();
    let first = prover
        .prove_with_output(z0, &invocations[0].words, invocations[0].output)
        .unwrap();
    let mut proof = prover
        .extend_with_output(&first, &invocations[1].words, invocations[1].output)
        .unwrap();
    let segment_proof = prover
        .extend_with_output(&first, &invocations[1].words, invocations[1].output)
        .unwrap();
    for invocation in &invocations[2..] {
        proof = prover
            .extend_with_output(&proof, &invocation.words, invocation.output)
            .unwrap();
    }
    eprintln!(
        "memory proving elapsed={:?} invocations={}",
        proving.elapsed(),
        invocations.len()
    );

    let context = run.context();
    let statement = run.statement();
    assert_eq!(statement.segments, context.plan().s_max as u64, "S = S_max");
    assert_eq!(
        statement.final_.acc, 5,
        "the second segment reads the first segment's write"
    );
    let verifying = std::time::Instant::now();
    context
        .verify(&verifier, &statement, run.carry(), &proof)
        .unwrap();
    eprintln!("memory verification elapsed={:?}", verifying.elapsed());

    // One closed segment is a complete proof of its own.
    let mut one = statement.clone();
    one.steps = 2;
    one.segments = 1;
    one.final_ = MachineState { pc: 2, acc: 5 };
    one.final_ts = invocations[1].carry.ts;
    one.final_root = invocations[1].carry.mem_root;
    context
        .verify(&verifier, &one, &invocations[1].carry, &segment_proof)
        .unwrap();

    // End inside a segment: reject at the terminal `idx = N` check.
    let mut open = one.clone();
    open.steps = 1;
    open.final_ = MachineState { pc: 1, acc: 5 };
    open.final_ts = invocations[0].carry.ts;
    assert!(matches!(
        context.verify(&verifier, &open, &invocations[0].carry, &first),
        Err(VerifyError::Terminal(TerminalError::OpenCarry))
    ));

    // The Stage 1 initial envelope (`T = 0`): reject (spec §13).
    let mut empty = statement.clone();
    empty.steps = 0;
    empty.segments = 0;
    assert!(matches!(
        context.verify(&verifier, &empty, run.carry(), &proof),
        Err(VerifyError::Terminal(TerminalError::SegmentRange))
    ));

    // An envelope carry that does not open the final state: reject at the
    // Stage 1 state link.
    let mut carry = *run.carry();
    carry.seen.swap(0, 1);
    assert_ne!(carry, *run.carry());
    assert!(matches!(
        context.verify(&verifier, &statement, &carry, &proof),
        Err(VerifyError::Proof(_))
    ));
}

#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn memory_conformance_on_cpu() {
    memory_conformance(nightstream::Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production-profile memory proofs on Metal; run separately under the 300-second cap."]
fn memory_conformance_on_metal() {
    memory_conformance(nightstream::Engine::Metal);
}

#[test]
#[ignore = "Prints the saved memory package's structural identifier for pinning."]
fn print_memory_package_identifier() {
    let value: serde_json::Value = serde_json::from_slice(&memory_package()).unwrap();
    let package = nightstream_fprime::load_prepared_application_value(value).unwrap();
    println!("structural_identifier={:?}", package.structural_identifier());
}
