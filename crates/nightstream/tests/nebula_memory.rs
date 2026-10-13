//! The first Nebula memory application: native segment runs and spec §13
//! terminal checks, and (ignored, production profile) spec §14 conformance
//! proofs over the Lean-emitted package.

use nightstream::nebula::{state_words, Context, MachineState, Package, Run, Step, TerminalError};
use p3_field::PrimeField64;
use p3_goldilocks::Goldilocks as F;

/// The Stage 1 input state that an invocation's own words open to. The carry
/// words come first; the machine's five words come last (input state, output
/// state, idle flag).
fn input_state(words: &[F]) -> [F; 4] {
    let app_in = words.len() - 5;
    state_words(&words[app_in..app_in + 2], &words[0..39])
}

fn program_run() -> Run {
    Run::new(Context::new(Package::V1), MachineState::default())
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

/// Ob1 (security note §3): an integer of the statement that becomes a field
/// word must be below `q`. Otherwise `v` and `v + q` give one word, and the
/// proof of the honest statement also verifies the second one.
#[test]
fn terminal_rejects_statement_integers_at_or_above_q() {
    let mut run = program_run();
    for steps in [
        [Step::Execute, Step::Execute],
        [Step::Execute, Step::Execute],
        [Step::Idle, Step::Idle],
        [Step::Idle, Step::Idle],
    ] {
        run.segment(&steps).expect("segment runs");
    }
    let (honest, carry, q) = (run.statement(), *run.carry(), F::ORDER_U64);
    let context = run.context();
    assert!(context.terminal_states(&honest, &carry).is_ok());

    let mut cases = Vec::new();
    let mut statement = honest.clone();
    statement.initial.pc += q;
    cases.push(("initial pc", statement, carry));
    let mut statement = honest.clone();
    statement.initial.acc += q;
    cases.push(("initial acc", statement, carry));
    let mut statement = honest.clone();
    statement.final_.pc += q;
    cases.push(("final pc", statement, carry));
    let mut statement = honest.clone();
    statement.final_.acc += q;
    cases.push(("final acc", statement, carry));
    let mut statement = honest.clone();
    statement.final_ts += q;
    let mut aliased = carry;
    aliased.ts += q;
    cases.push(("final timestamp", statement, aliased));

    let accepted: Vec<&str> = cases
        .iter()
        .filter(|(_, statement, carry)| context.terminal_states(statement, carry) != Err(TerminalError::NonCanonical))
        .map(|(name, _, _)| *name)
        .collect();
    assert!(accepted.is_empty(), "accepted non-canonical integers: {accepted:?}");
}

/// The verifier's `D_init` (spec §9.2), from which it computes the start
/// carry and `z0`, equals the value that Lean computes for the same plan
/// (`emit --application-start`). It binds the plan digest and the images.
#[test]
fn d_init_matches_lean() {
    for (package, fixture) in [
        (Package::V1, include_str!("fixtures/nebula-memory-v1-start.json")),
        (Package::N1, include_str!("fixtures/nebula-memory-n1-start.json")),
    ] {
        let lean: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let lean: Vec<u64> = serde_json::from_value(lean["d_init"].clone()).unwrap();
        let d_init = Context::new(package).d_init();
        assert_eq!(d_init.map(|word| word.as_canonical_u64()).to_vec(), lean);
    }
}

fn package(name: &str) -> Vec<u8> {
    std::fs::read(
        std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
            .join("artifacts")
            .join(name),
    )
    .expect("saved Lean memory package")
}

/// A memory verifier accepts only its own package: the second package's
/// circuit is not the first package.
#[test]
fn memory_verifier_rejects_another_package() {
    let circuit = nightstream::Circuit::load_package(
        &package("nightstream-fprime-stage2-nebula-memory-n1.json"),
        Package::N1.structural_identifier(),
    )
    .unwrap();
    assert!(matches!(
        nightstream::nebula::Verifier::new(Package::V1, &circuit, nightstream::Engine::Optimized, 114),
        Err(nightstream::nebula::VerifyError::Package)
    ));
}

fn memory_package() -> Vec<u8> {
    package("nightstream-fprime-stage2-nebula-memory-v1.json")
}

fn n1_run() -> Run {
    Run::new(Context::new(Package::N1), MachineState::default())
}

/// The second plan (`N = 1`): every invocation opens and closes a segment,
/// every step uses both ports, and `S_max` segments end at the largest
/// reachable timestamp `S_max · N · B_ops`.
#[test]
fn n1_native_run_reaches_every_counter_maximum() {
    let mut run = n1_run();
    let plan = run.context().plan().clone();
    for _ in 0..plan.s_max {
        assert_eq!(run.segment(&[Step::Execute]).unwrap().len(), 1);
        assert_eq!(run.carry().idx, 1, "every invocation closes its segment");
    }
    let statement = run.statement();
    assert_eq!(statement.segments, plan.s_max as u64);
    assert_eq!(statement.final_ts, (plan.s_max * plan.n * plan.b_ops) as u64);
    assert_eq!(statement.final_, MachineState { pc: 4, acc: 7 });
    let (initial, _) = run
        .context()
        .terminal_states(&statement, run.carry())
        .expect("terminal checks pass");
    assert_eq!(initial, run.initial_state());
}

/// The spec §14 accept cases of the second plan through the memory verifier:
/// `N = 1`, and `S = S_max` with every counter at its largest value.
fn n1_accept(engine: nightstream::Engine) {
    use nightstream::{nebula, Circuit};
    let circuit = Circuit::load_package(
        &package("nightstream-fprime-stage2-nebula-memory-n1.json"),
        Package::N1.structural_identifier(),
    )
    .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = nebula::Verifier::new(Package::N1, &circuit, engine, 114).unwrap();
    let mut run = n1_run();
    let z0 = run.initial_state();
    let mut proof: Option<nightstream::Proof> = None;
    for _ in 0..run.context().plan().s_max {
        let invocation = run.segment(&[Step::Execute]).unwrap().remove(0);
        proof = Some(match &proof {
            None => prover
                .prove_with_output(z0, &invocation.words, invocation.output)
                .unwrap(),
            Some(proof) => prover
                .extend_with_output(proof, &invocation.words, invocation.output)
                .unwrap(),
        });
    }
    verifier
        .verify(&run.statement(), run.carry(), proof.as_ref().unwrap())
        .unwrap();
}

#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn n1_accept_on_cpu() {
    n1_accept(nightstream::Engine::Optimized);
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Full production-profile memory proofs on Metal; run separately under the 300-second cap."]
fn n1_accept_on_metal() {
    n1_accept(nightstream::Engine::Metal);
}

/// The spec §14 accept cases that the first plan reaches, and the rejections
/// that need no invalid witness, through the memory verifier. One run of four
/// segments (`S = S_max`) has continue arms (`N = 2`), writes RAM in the first
/// segment and reads it in the second, and ends with an idle segment.
fn memory_conformance(engine: nightstream::Engine) {
    use nightstream::nebula::{self, VerifyError};
    use nightstream::Circuit;
    let started = std::time::Instant::now();
    let circuit =
        Circuit::load_package(&memory_package(), Package::V1.structural_identifier()).expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = nebula::Verifier::new(Package::V1, &circuit, engine, 114).unwrap();
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
    verifier.verify(&statement, run.carry(), &proof).unwrap();
    eprintln!("memory verification elapsed={:?}", verifying.elapsed());

    // Ob1: an application integer at or above `q` has the word of a smaller
    // integer, so the honest proof would verify it. Reject it at the terminal.
    let mut alias = statement.clone();
    alias.final_.acc += F::ORDER_U64;
    assert!(matches!(
        verifier.verify(&alias, run.carry(), &proof),
        Err(VerifyError::Terminal(TerminalError::NonCanonical))
    ));

    // One closed segment is a complete proof of its own.
    let mut one = statement.clone();
    one.steps = 2;
    one.segments = 1;
    one.final_ = MachineState { pc: 2, acc: 5 };
    one.final_ts = invocations[1].carry.ts;
    one.final_root = invocations[1].carry.mem_root;
    verifier
        .verify(&one, &invocations[1].carry, &segment_proof)
        .unwrap();

    // End inside a segment: reject at the terminal `idx = N` check.
    let mut open = one.clone();
    open.steps = 1;
    open.final_ = MachineState { pc: 1, acc: 5 };
    open.final_ts = invocations[0].carry.ts;
    assert!(matches!(
        verifier.verify(&open, &invocations[0].carry, &first),
        Err(VerifyError::Terminal(TerminalError::OpenCarry))
    ));

    // The Stage 1 initial envelope (`T = 0`): reject (spec §13).
    let mut empty = statement.clone();
    empty.steps = 0;
    empty.segments = 0;
    assert!(matches!(
        verifier.verify(&empty, run.carry(), &proof),
        Err(VerifyError::Terminal(TerminalError::SegmentRange))
    ));

    // A fold with more than one fresh claim: the proof format has one fresh
    // slot, so the decoder rejects any further payload.
    let bytes = prover.encode_proof(&proof).unwrap();
    assert!(verifier.decode_proof(&bytes).is_ok());
    let mut two = bytes.clone();
    two.extend_from_slice(&bytes[bytes.len() / 2..]);
    assert!(verifier.decode_proof(&two).is_err());

    // An envelope carry that does not open the final state: reject at the
    // Stage 1 state link.
    let mut carry = *run.carry();
    carry.seen.swap(0, 1);
    assert_ne!(carry, *run.carry());
    assert!(matches!(
        verifier.verify(&statement, &carry, &proof),
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
#[ignore = "Prints the saved memory packages' structural identifiers for pinning."]
fn print_memory_package_identifier() {
    for name in [
        "nightstream-fprime-stage2-nebula-memory-v1.json",
        "nightstream-fprime-stage2-nebula-memory-n1.json",
    ] {
        let value: serde_json::Value = serde_json::from_slice(&package(name)).unwrap();
        let package = nightstream_fprime::load_prepared_application_value(value).unwrap();
        println!("{name} structural_identifier={:?}", package.structural_identifier());
    }
}
