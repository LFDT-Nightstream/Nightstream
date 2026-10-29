//! The first Nebula memory application: native segment runs and spec §13
//! terminal checks, and (ignored, production profile) spec §14 conformance
//! proofs over the Lean-emitted package.

use neo_math::K;
use nightstream::nebula::{
    state_words, Carry, Context, MachineState, Package, Products, Program, ProgramError, Run, Statement, Step,
    TerminalError,
};
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks as F;

/// The Stage 1 input state that an invocation's own words open to. The carry
/// words come first; the machine's five words come last (input state, output
/// state, idle flag).
fn input_state(words: &[F]) -> [F; 4] {
    let app_in = words.len() - 5;
    state_words(&words[app_in..app_in + 2], &words[0..39])
}

fn program_run() -> Run {
    Run::new(Context::new(&Package::V1.program()), MachineState::default())
}

/// `loadi 7; store 1; load 1; halt` with RAM `[0, 0, 0, 9]`: a second
/// program of the first package's geometry.
fn other_program() -> Program {
    Program::new(Package::V1, vec![31, 6, 5, 0], vec![0, 0, 0, 9]).unwrap()
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

/// A program's images must fit its package's geometry, as in Lean
/// `Plan.Valid`: `2^r` ROM words and `2^μ` RAM words, each below `2^32`.
#[test]
fn program_rejects_images_outside_the_geometry() {
    let (rom, ram) = (vec![0; 4], vec![0; 4]);
    assert!(Program::new(Package::V1, rom.clone(), ram.clone()).is_ok());
    assert_eq!(
        Program::new(Package::V1, vec![0; 3], ram.clone()),
        Err(ProgramError::Size)
    );
    assert_eq!(
        Program::new(Package::V1, rom.clone(), vec![0; 8]),
        Err(ProgramError::Size)
    );
    assert_eq!(
        Program::new(Package::N1, rom.clone(), ram.clone()),
        Err(ProgramError::Size)
    );
    let mut wide = ram.clone();
    wide[3] = 1 << 32;
    assert_eq!(Program::new(Package::V1, rom.clone(), wide), Err(ProgramError::Word));
    let mut widest = rom;
    widest[0] = u64::from(u32::MAX);
    assert!(Program::new(Package::V1, widest, ram).is_ok());
}

/// One package, two programs: each run passes the terminal checks of its own
/// program. The programs have different `D_init`, so one statement opens a
/// different initial state under each program, and the Stage 1 state link
/// rejects a proof of one program for the other (`wrong_program_rejected_on_cpu`).
#[test]
fn programs_of_one_package_have_their_own_initial_state() {
    let contexts = [Context::new(&Package::V1.program()), Context::new(&other_program())];
    let runs = contexts.clone().map(|context| {
        let mut run = Run::new(context, MachineState::default());
        for steps in [
            [Step::Execute, Step::Execute],
            [Step::Execute, Step::Execute],
            [Step::Idle, Step::Idle],
            [Step::Idle, Step::Idle],
        ] {
            run.segment(&steps).expect("segment runs");
        }
        run
    });
    assert_eq!(runs[0].statement().final_.acc, 5);
    assert_eq!(runs[1].statement().final_.acc, 7, "the second program loads 7");
    assert_ne!(contexts[0].d_init(), contexts[1].d_init());
    for run in &runs {
        let (initial, _) = run
            .context()
            .terminal_states(&run.statement(), run.carry())
            .expect("terminal checks pass");
        assert_eq!(initial, run.initial_state());
    }
    let (other, _) = contexts[1]
        .terminal_states(&runs[0].statement(), runs[0].carry())
        .expect("the terminal checks read no image");
    assert_ne!(other, runs[0].initial_state());
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
        let d_init = Context::new(&package.program()).d_init();
        assert_eq!(d_init.map(|word| word.as_canonical_u64()).to_vec(), lean);
    }
}

#[derive(serde::Deserialize)]
struct TerminalFixture {
    cases: Vec<TerminalCase>,
}

#[derive(serde::Deserialize)]
struct TerminalCase {
    name: String,
    statement: FixtureStatement,
    carry: Vec<u64>,
    canonical: bool,
    checks: TerminalChecks,
    final_state: Vec<u64>,
}

#[derive(serde::Deserialize)]
struct FixtureStatement {
    steps: u64,
    initial: [u64; 2],
    #[serde(rename = "final")]
    final_: [u64; 2],
    segments: u64,
    final_ts: u64,
    final_root: [u64; 4],
}

/// The fields of Lean's `TerminalChecks` after `state`, as Lean decides them.
#[derive(serde::Deserialize)]
struct TerminalChecks {
    closed: bool,
    segments_range: bool,
    step_count: bool,
    seg_idx: bool,
    ts: bool,
    root: bool,
}

/// The carry of 39 words in spec §11.1 order (Lean `decodeCarry`).
fn decode_carry(words: &[u64]) -> Carry {
    let f = |i: usize| F::from_u64(words[i]);
    let k = |i: usize| K::from_basis_coefficients_slice(&[f(i), f(i + 1)]).unwrap();
    let digest = |i: usize| [f(i), f(i + 1), f(i + 2), f(i + 3)];
    Carry {
        seg_idx: words[0],
        idx: words[1],
        ts: words[2],
        eta: (k(3), k(5)),
        products: Products {
            read: k(7),
            write: k(9),
            initial: k(11),
            final_: k(13),
        },
        proposed: (digest(15), digest(19)),
        seen: [digest(23), digest(27), digest(31)],
        mem_root: digest(35),
    }
}

fn canonical_words(words: &[F]) -> Vec<u64> {
    words.iter().map(|word| word.as_canonical_u64()).collect()
}

/// The spec §13 terminal check agrees with Lean's `TerminalChecks` on every
/// case of `emit --application-terminal`. It accepts exactly when every
/// statement integer is below `q` and every check holds, with Lean's final
/// state words. Otherwise it fails at the first false check in its own order:
/// Ob1, closed carry, segment range, step count, then the carry fields.
#[test]
fn terminal_checks_match_lean() {
    for (package, fixture) in [
        (Package::V1, include_str!("fixtures/nebula-memory-v1-terminal.json")),
        (Package::N1, include_str!("fixtures/nebula-memory-n1-terminal.json")),
    ] {
        let context = Context::new(&package.program());
        let fixture: TerminalFixture = serde_json::from_str(fixture).unwrap();
        for case in fixture.cases {
            let carry = decode_carry(&case.carry);
            assert_eq!(
                canonical_words(&carry.words()),
                case.carry,
                "{package:?} {}: carry words",
                case.name
            );
            let fields = case.statement;
            let statement = Statement {
                steps: fields.steps,
                initial: MachineState {
                    pc: fields.initial[0],
                    acc: fields.initial[1],
                },
                final_: MachineState {
                    pc: fields.final_[0],
                    acc: fields.final_[1],
                },
                segments: fields.segments,
                final_ts: fields.final_ts,
                final_root: fields.final_root.map(F::from_u64),
            };
            let checks = case.checks;
            let expected = if !case.canonical {
                Err(TerminalError::NonCanonical)
            } else if !checks.closed {
                Err(TerminalError::OpenCarry)
            } else if !checks.segments_range {
                Err(TerminalError::SegmentRange)
            } else if !checks.step_count {
                Err(TerminalError::StepCount)
            } else if !(checks.seg_idx && checks.ts && checks.root) {
                Err(TerminalError::Field)
            } else {
                Ok(case.final_state)
            };
            let actual = context
                .terminal_states(&statement, &carry)
                .map(|(_, last)| canonical_words(&last));
            assert_eq!(actual, expected, "{package:?} {}", case.name);
        }
    }
}

/// Spec §4.2 rule 6: the memory fingerprint error `2·m_mem/q²` is at most
/// `2^−109.91` exactly when `m_mem ≤ 139,509` (Lean `Plan.secure_iff`).
#[test]
fn memory_security_bits_match_the_lean_setup_boundary() {
    let bits = |max_tuples: usize| {
        let mut plan = Package::V1.plan();
        plan.n = 1;
        plan.b_ops = max_tuples - plan.cells();
        assert_eq!(plan.max_tuples(), max_tuples);
        plan.memory_security_bits()
    };
    assert!(bits(139_509) >= 109.91);
    assert!(109.91 > bits(139_510));
    for package in [Package::V1, Package::N1] {
        assert!(package.plan().memory_security_bits() > 115.0, "{package:?}");
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

/// A package loads only from its own bytes: the second package's bytes fail
/// as the first package at the pinned structural identifier.
#[test]
fn package_loads_only_its_own_bytes() {
    assert!(matches!(
        Package::V1.load(&package("nightstream-fprime-stage2-nebula-memory-n1.json")),
        Err(nightstream::Error::Package(
            nightstream_fprime::PackageError::IdentityMismatch { .. }
        ))
    ));
}

fn memory_package() -> Vec<u8> {
    package("nightstream-fprime-stage2-nebula-memory-v1.json")
}

fn n1_run() -> Run {
    Run::new(Context::new(&Package::N1.program()), MachineState::default())
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
    let circuit = Package::N1
        .load(&package("nightstream-fprime-stage2-nebula-memory-n1.json"))
        .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = nightstream::nebula::Verifier::new(&circuit, engine, 114).unwrap();
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
        .verify(
            &Package::N1.program(),
            &run.statement(),
            run.carry(),
            proof.as_ref().unwrap(),
        )
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
    let started = std::time::Instant::now();
    let circuit = Package::V1
        .load(&memory_package())
        .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = nebula::Verifier::new(&circuit, engine, 114).unwrap();
    let program = Package::V1.program();
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
    verifier
        .verify(&program, &statement, run.carry(), &proof)
        .unwrap();
    eprintln!("memory verification elapsed={:?}", verifying.elapsed());

    // Ob1: an application integer at or above `q` has the word of a smaller
    // integer, so the honest proof would verify it. Reject it at the terminal.
    let mut alias = statement.clone();
    alias.final_.acc += F::ORDER_U64;
    assert!(matches!(
        verifier.verify(&program, &alias, run.carry(), &proof),
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
        .verify(&program, &one, &invocations[1].carry, &segment_proof)
        .unwrap();

    // End inside a segment: reject at the terminal `idx = N` check.
    let mut open = one.clone();
    open.steps = 1;
    open.final_ = MachineState { pc: 1, acc: 5 };
    open.final_ts = invocations[0].carry.ts;
    assert!(matches!(
        verifier.verify(&program, &open, &invocations[0].carry, &first),
        Err(VerifyError::Terminal(TerminalError::OpenCarry))
    ));

    // The Stage 1 initial envelope (`T = 0`): reject (spec §13).
    let mut empty = statement.clone();
    empty.steps = 0;
    empty.segments = 0;
    assert!(matches!(
        verifier.verify(&program, &empty, run.carry(), &proof),
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
        verifier.verify(&program, &statement, &carry, &proof),
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

/// A verifier given another program rejects an honest proof of the default
/// program: the other `D_init` opens another initial state, so the Stage 1
/// state link fails. A program of another package fails before any check.
#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn wrong_program_rejected_on_cpu() {
    use nightstream::nebula::{self, VerifyError};
    let engine = nightstream::Engine::Optimized;
    let started = std::time::Instant::now();
    let circuit = Package::V1
        .load(&memory_package())
        .expect("pinned memory package");
    let prover = circuit.prover(engine, 114).unwrap();
    let verifier = nebula::Verifier::new(&circuit, engine, 114).unwrap();
    let mut run = program_run();
    let z0 = run.initial_state();
    let invocations = run.segment(&[Step::Execute, Step::Execute]).unwrap();
    let first = prover
        .prove_with_output(z0, &invocations[0].words, invocations[0].output)
        .unwrap();
    let proof = prover
        .extend_with_output(&first, &invocations[1].words, invocations[1].output)
        .unwrap();
    let statement = run.statement();
    verifier
        .verify(&Package::V1.program(), &statement, run.carry(), &proof)
        .unwrap();
    assert!(matches!(
        verifier.verify(&other_program(), &statement, run.carry(), &proof),
        Err(VerifyError::Proof(_))
    ));
    assert!(matches!(
        verifier.verify(&Package::N1.program(), &statement, run.carry(), &proof),
        Err(VerifyError::Package)
    ));
    eprintln!("wrong program elapsed={:?}", started.elapsed());
}

#[test]
#[ignore = "Prints the saved memory packages' structural identifiers for pinning."]
fn print_memory_package_identifier() {
    for name in [
        "nightstream-fprime-stage2-nebula-memory-v1.json",
        "nightstream-fprime-stage2-nebula-memory-n1.json",
    ] {
        let placeholder = nightstream_fprime::StructuralIdentifier::from_words([0; 4]);
        let computed = match nightstream_fprime::load_per_application_package(&package(name), placeholder) {
            Ok(loaded) => loaded.structural_identifier().words(),
            Err(nightstream_fprime::PackageError::IdentityMismatch { computed, .. }) => computed,
            Err(error) => panic!("{name}: {error}"),
        };
        println!("{name} structural_identifier={computed:?}");
    }
}
