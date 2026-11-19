//! SPEC.md Algorithm 8 and section 9.

use neo_math::K;

use super::carry::{Carry, Products};
use super::transcript::state_words;
use super::{
    Application, Context, Expr, Machine, MachineError, MachineState, Memory, Package, Program, ProgramError, Row, Run,
    SegmentError, Statement, Step, TerminalError, Transition, View,
};
use neo_math::F;
use p3_field::{BasedVectorSpace, PrimeCharacteristicRing, PrimeField64};

fn input_state(words: &[F]) -> [F; 4] {
    let app_in = words.len() - 5;
    state_words(&words[app_in..app_in + 2], &words[0..39])
}

fn program_run() -> Run<Machine> {
    Run::new(Machine, &Package::V1.program(), MachineState::default())
}

fn other_program() -> Program {
    Program::new(Package::V1.geometry(), vec![31, 6, 5, 0], vec![0, 0, 0, 9]).unwrap()
}

#[test]
fn native_run_closes_every_segment_and_passes_the_terminal() {
    let mut run = program_run();
    assert_eq!(run.context().witness_word_count(), 664);
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
            assert_eq!(invocation.words.len(), 664);
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

/// SPEC.md Algorithm 8, step 2.
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

/// Lean `Plan.Valid`.
#[test]
fn program_rejects_images_outside_the_geometry() {
    let (rom, ram) = (vec![0; 4], vec![0; 4]);
    let (v1, n1) = (Package::V1.geometry(), Package::N1.geometry());
    assert!(Program::new(v1, rom.clone(), ram.clone()).is_ok());
    assert_eq!(Program::new(v1, vec![0; 3], ram.clone()), Err(ProgramError::Size));
    assert_eq!(Program::new(v1, rom.clone(), vec![0; 8]), Err(ProgramError::Size));
    assert_eq!(Program::new(n1, rom.clone(), ram.clone()), Err(ProgramError::Size));
    let mut widest = rom;
    widest[0] = u32::MAX;
    assert!(Program::new(v1, widest, ram).is_ok());
}

#[test]
fn programs_of_one_package_have_their_own_initial_state() {
    let contexts = [
        Context::new(Machine, &Package::V1.program()),
        Context::new(Machine, &other_program()),
    ];
    let runs = contexts.clone().map(|context| {
        let mut run = Run::new(Machine, context.program(), MachineState::default());
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
    assert_ne!(contexts[0].program().d_init(), contexts[1].program().d_init());
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

/// SPEC.md section 7.4; Lean `FirstPlan.otherPlan`.
#[test]
fn d_init_matches_lean() {
    let other = Program::new(
        Package::V1.geometry(),
        vec![4294967295, 2147483649, 305419896, 1],
        vec![2, 3735928559, 2147483647, 3405691582],
    )
    .unwrap();
    for (program, fixture) in [
        (
            Package::V1.program(),
            include_str!("../fixtures/nebula-memory-v1-start.json"),
        ),
        (
            Package::N1.program(),
            include_str!("../fixtures/nebula-memory-n1-start.json"),
        ),
        (other, include_str!("../fixtures/nebula-memory-v1-other-start.json")),
    ] {
        let lean: serde_json::Value = serde_json::from_str(fixture).unwrap();
        let lean: Vec<u64> = serde_json::from_value(lean["d_init"].clone()).unwrap();
        assert_eq!(
            program
                .d_init()
                .map(|word| word.as_canonical_u64())
                .to_vec(),
            lean
        );
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

/// Lean `TerminalChecks`.
#[derive(serde::Deserialize)]
struct TerminalChecks {
    closed: bool,
    segments_range: bool,
    step_count: bool,
    seg_idx: bool,
    ts: bool,
    root: bool,
}

/// SPEC.md section 7.5; Lean `decodeCarry`.
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

/// SPEC.md Algorithm 8.
#[test]
fn terminal_checks_match_lean() {
    for (package, fixture) in [
        (Package::V1, include_str!("../fixtures/nebula-memory-v1-terminal.json")),
        (Package::N1, include_str!("../fixtures/nebula-memory-n1-terminal.json")),
    ] {
        let context = Context::new(Machine, &package.program());
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

/// SPEC.md section 7.1, rule 6; Lean `Geometry.secure_iff`.
#[test]
fn memory_security_bits_match_the_lean_setup_boundary() {
    let bits = |max_tuples: usize| {
        let mut geometry = Package::V1.geometry();
        geometry.n = 1;
        geometry.b_ops = max_tuples - geometry.cells();
        assert_eq!(geometry.max_tuples(), max_tuples);
        geometry.memory_security_bits()
    };
    assert!(bits(139_509) >= 109.91);
    assert!(109.91 > bits(139_510));
    for package in [Package::V1, Package::N1] {
        assert!(package.geometry().memory_security_bits() > 115.0, "{package:?}");
    }
}

/// SPEC.md section 7.1.
#[test]
fn the_memory_gate_charges_every_segment_exactly() {
    use super::{memory_security, Geometry, Prover};
    use crate::{Engine, Error};
    use neo_params::ParamsError;

    let q = u128::from(F::ORDER_U64);
    let edge = Geometry::new(17, 12, 4341, 1, 4).expect("a secure geometry at the boundary");
    assert_eq!(edge.max_tuples(), 139_509);
    let mut geometries = vec![edge, Package::V1.geometry(), Package::N1.geometry()];
    for s_max in [1, 2, 3, 7, 64, 1000, 65_536] {
        geometries.extend(Geometry::new(10, 8, 2, 1, s_max));
        geometries.extend(Geometry::new(14, 10, 40, 8, s_max));
    }
    assert!(geometries.len() > 10);
    for geometry in &geometries {
        let charge = 2 * geometry.max_tuples() as u128 * geometry.s_max as u128;
        let bits = geometry.segments_security_bits();
        assert!(
            charge
                .checked_mul(1 << bits)
                .is_some_and(|error| error <= q * q),
            "{geometry:?}"
        );
        assert!(
            charge
                .checked_mul(1 << (bits + 1))
                .is_none_or(|error| error > q * q),
            "{geometry:?}"
        );
    }

    assert_eq!(edge.segments_security_bits(), 107);
    assert!(memory_security(&edge, 107).is_ok());
    let rejected = |result: Result<(), Error>| {
        matches!(
            result,
            Err(Error::Parameters(ParamsError::InsufficientStatisticalSecurity {
                required: 108,
                available: 107
            }))
        )
    };
    assert!(rejected(memory_security(&edge, 108)));
    let load = Prover::load("", edge, Machine, Engine::Optimized, 108).map(|_| ());
    assert!(rejected(load), "the gate must reject before it reads a saved circuit");
}

struct Packages {
    light: Vec<u8>,
    refresh: Vec<u8>,
}

impl Packages {
    fn read(name: &str) -> Vec<u8> {
        std::fs::read(
            std::path::PathBuf::from(env!("CARGO_MANIFEST_DIR"))
                .join("../../formal/nightstream-fprime/artifacts")
                .join(name),
        )
        .expect("Lean bounded-depth package")
    }

    fn of(stem: &str) -> Self {
        Self {
            light: Self::read(&format!("nightstream-fprime-bounded-depth-{stem}light-v1.json")),
            refresh: Self::read(&format!("nightstream-fprime-bounded-depth-{stem}refresh-v1.json")),
        }
    }

    fn references(&self) -> crate::References<'_> {
        crate::References {
            light: &self.light,
            refresh: &self.refresh,
        }
    }
}

#[test]
fn package_loads_only_its_own_bytes() {
    assert!(matches!(
        Package::V1.load(Packages::of("nebula-n1-").references()),
        Err(crate::Error::Package(
            nightstream_fprime::PackageError::IdentityMismatch { .. }
        ))
    ));
}

fn n1_run() -> Run<Machine> {
    Run::new(Machine, &Package::N1.program(), MachineState::default())
}

#[test]
fn n1_native_run_reaches_every_counter_maximum() {
    let mut run = n1_run();
    let geometry = run.context().geometry();
    for _ in 0..geometry.s_max {
        assert_eq!(run.segment(&[Step::Execute]).unwrap().len(), 1);
        assert_eq!(run.carry().idx, 1, "every invocation closes its segment");
    }
    let statement = run.statement();
    assert_eq!(statement.segments, geometry.s_max as u64);
    assert_eq!(
        statement.final_ts,
        (geometry.s_max * geometry.n * geometry.b_ops) as u64
    );
    assert_eq!(statement.final_, MachineState { pc: 4, acc: 7 });
    let (initial, _) = run
        .context()
        .terminal_states(&statement, run.carry())
        .expect("terminal checks pass");
    assert_eq!(initial, run.initial_state());
}

/// SPEC.md section 9.
fn n1_accept(engine: crate::Engine) {
    let circuit = Package::N1
        .load(Packages::of("nebula-n1-").references())
        .expect("pinned memory package");
    let prover = circuit.circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let mut run = n1_run();
    let z0 = run.initial_state();
    let mut proof: Option<crate::Proof> = None;
    for _ in 0..run.context().geometry().s_max {
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
        .verify_parts(
            &Package::N1.program(),
            &run.statement(),
            run.carry(),
            proof.as_ref().unwrap(),
        )
        .unwrap();
    let memory = circuit.prover(engine, 109).unwrap();
    let proof = super::proving::Proof::from_parts(proof.unwrap(), *run.carry());
    let key = verifier.compression_key().unwrap();
    let setup = memory.compression_setup().unwrap();
    let compressed = memory.compress(&proof, &setup).unwrap();
    let statement = run.statement();
    let bytes = compressed.to_bytes().unwrap();
    let decoded = crate::nebula::FinalProof::from_bytes(&bytes).unwrap();
    verifier
        .verify_final(
            &Package::N1.program(),
            &statement.initial,
            &statement.final_,
            &key,
            &decoded,
        )
        .unwrap();
    assert!(crate::nebula::FinalProof::from_bytes(&bytes[..bytes.len() - 1]).is_err());
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(crate::nebula::FinalProof::from_bytes(&longer).is_err());
    let mut aliased = bytes.clone();
    aliased[..8].copy_from_slice(&u64::MAX.to_le_bytes());
    assert!(crate::nebula::FinalProof::from_bytes(&aliased).is_err());
    let mut changed = bytes;
    changed[0] ^= 1;
    let changed = crate::nebula::FinalProof::from_bytes(&changed).unwrap();
    assert!(verifier
        .verify_final(
            &Package::N1.program(),
            &statement.initial,
            &statement.final_,
            &key,
            &changed,
        )
        .is_err());
}

#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn n1_accept_on_cpu() {
    n1_accept(crate::Engine::Optimized);
}

/// SPEC.md section 9.
fn memory_conformance(engine: crate::Engine) {
    use crate::nebula::VerifyError;
    let started = std::time::Instant::now();
    let circuit = Package::V1
        .load(Packages::of("nebula-v1-").references())
        .expect("pinned memory package");
    let prover = circuit.circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
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
    assert_eq!(statement.segments, context.geometry().s_max as u64, "S = S_max");
    assert_eq!(
        statement.final_.acc, 5,
        "the second segment reads the first segment's write"
    );
    let verifying = std::time::Instant::now();
    verifier
        .verify_parts(&program, &statement, run.carry(), &proof)
        .unwrap();
    eprintln!("memory verification elapsed={:?}", verifying.elapsed());

    let mut alias = statement.clone();
    alias.final_.acc += F::ORDER_U64;
    assert!(matches!(
        verifier.verify_parts(&program, &alias, run.carry(), &proof),
        Err(VerifyError::Terminal(TerminalError::NonCanonical))
    ));

    let mut one = statement.clone();
    one.steps = 2;
    one.segments = 1;
    one.final_ = MachineState { pc: 2, acc: 5 };
    one.final_ts = invocations[1].carry.ts;
    one.final_root = invocations[1].carry.mem_root;
    verifier
        .verify_parts(&program, &one, &invocations[1].carry, &segment_proof)
        .unwrap();

    let mut open = one.clone();
    open.steps = 1;
    open.final_ = MachineState { pc: 1, acc: 5 };
    open.final_ts = invocations[0].carry.ts;
    assert!(matches!(
        verifier.verify_parts(&program, &open, &invocations[0].carry, &first),
        Err(VerifyError::Terminal(TerminalError::OpenCarry))
    ));

    // SPEC.md Algorithm 8.
    let mut empty = statement.clone();
    empty.steps = 0;
    empty.segments = 0;
    assert!(matches!(
        verifier.verify_parts(&program, &empty, run.carry(), &proof),
        Err(VerifyError::Terminal(TerminalError::SegmentRange))
    ));

    let bytes = circuit
        .prover(engine, 109)
        .unwrap()
        .encode_proof(&crate::nebula::Proof::from_parts(proof.clone(), *run.carry()))
        .unwrap();
    assert!(verifier.decode_proof(&bytes).is_ok());
    let mut two = bytes.clone();
    two.extend_from_slice(&bytes[bytes.len() / 2..]);
    assert!(verifier.decode_proof(&two).is_err());

    let mut carry = *run.carry();
    carry.seen.swap(0, 1);
    assert_ne!(carry, *run.carry());
    assert!(matches!(
        verifier.verify_parts(&program, &statement, &carry, &proof),
        Err(VerifyError::Proof(_))
    ));
}

#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn memory_conformance_on_cpu() {
    memory_conformance(crate::Engine::Optimized);
}

/// SPEC.md section 9.
#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn each_program_verifies_only_its_own_proof_on_cpu() {
    use crate::nebula::VerifyError;
    let engine = crate::Engine::Optimized;
    let started = std::time::Instant::now();
    let circuit = Package::V1
        .load(Packages::of("nebula-v1-").references())
        .expect("pinned memory package");
    let prover = circuit.circuit.prover(engine, 109).unwrap();
    let verifier = circuit.verifier(engine, 109).unwrap();
    let prove = |program: &Program| {
        let mut run = Run::new(Machine, program, MachineState::default());
        let z0 = run.initial_state();
        let invocations = run.segment(&[Step::Execute, Step::Execute]).unwrap();
        let first = prover
            .prove_with_output(z0, &invocations[0].words, invocations[0].output)
            .unwrap();
        let proof = prover
            .extend_with_output(&first, &invocations[1].words, invocations[1].output)
            .unwrap();
        (run.statement(), *run.carry(), proof)
    };
    let (default, other) = (Package::V1.program(), other_program());
    let (statement, carry, proof) = prove(&default);
    let (other_statement, other_carry, other_proof) = prove(&other);
    verifier
        .verify_parts(&default, &statement, &carry, &proof)
        .unwrap();
    verifier
        .verify_parts(&other, &other_statement, &other_carry, &other_proof)
        .unwrap();
    assert!(matches!(
        verifier.verify_parts(&other, &statement, &carry, &proof),
        Err(VerifyError::Proof(_))
    ));
    assert!(matches!(
        verifier.verify_parts(&default, &other_statement, &other_carry, &other_proof),
        Err(VerifyError::Proof(_))
    ));
    assert!(matches!(
        verifier.verify_parts(&Package::N1.program(), &statement, &carry, &proof),
        Err(VerifyError::Geometry)
    ));
    eprintln!("two programs elapsed={:?}", started.elapsed());
}

#[test]
#[ignore = "Prints the memory programs' bounded-depth package identities for pinning."]
fn print_memory_package_identifier() {
    for stem in ["nebula-v1-", "nebula-n1-"] {
        let packages = Packages::of(stem);
        let computed = match nightstream_fprime::load_bounded_depth_packages(&packages.light, &packages.refresh, [0; 4])
        {
            Ok(loaded) => loaded.identity(),
            Err(nightstream_fprime::PackageError::IdentityMismatch { computed, .. }) => computed,
            Err(error) => panic!("{stem}: {error}"),
        };
        println!("{stem} identity={computed:?}");
    }
}

#[derive(Clone, Copy, Debug)]
enum Changed {
    Row,
    Read,
    Ports,
}

impl Application for Changed {
    type Input = Step;
    type State = MachineState;
    type Error = MachineError;

    fn width(&self) -> usize {
        Machine.width()
    }

    fn aux(&self) -> usize {
        Machine.aux()
    }

    fn port_count(&self) -> usize {
        Machine.port_count()
    }

    fn words(&self, state: &MachineState) -> Vec<u64> {
        Machine.words(state)
    }

    fn rows(&self, view: &View) -> Result<Vec<Row>, crate::Error> {
        let mut rows = Machine.rows(view)?;
        if let Changed::Row = self {
            rows[2] = (rows[2].expr() * Expr::constant(2)).into();
        }
        Ok(rows)
    }

    fn step(
        &self,
        state: &MachineState,
        input: &Step,
        memory: &Memory,
    ) -> Result<Transition<MachineState>, MachineError> {
        let mut transition = Machine.step(state, input, memory)?;
        match self {
            Changed::Row => {}
            Changed::Read => {
                let fetch = transition.ports[0]
                    .as_mut()
                    .expect("an executed step fetches");
                fetch.vr += 1;
                fetch.vw += 1;
            }
            Changed::Ports => transition.ports.truncate(1),
        }
        Ok(transition)
    }
}

/// Lean `Executes`.
#[test]
fn runner_rejects_a_misread_port_and_a_short_port_list() {
    for (app, expected) in [(Changed::Read, "Access"), (Changed::Ports, "Shape")] {
        let mut run = Run::new(app, &Package::V1.program(), MachineState::default());
        let error = run.segment(&[Step::Execute, Step::Execute]).unwrap_err();
        let name = match error {
            SegmentError::Access => "Access",
            SegmentError::Shape => "Shape",
            _ => "other",
        };
        assert_eq!(name, expected, "{app:?}: {error}");
    }
}

/// Lean `Geometry.Valid`.
#[test]
fn invalid_geometries_make_no_program_and_no_circuit() {
    let valid = Package::V1.geometry();
    assert!(valid.is_valid() && Package::N1.geometry().is_valid());
    let cases: [(&str, fn(&mut crate::nebula::Geometry)); 5] = [
        ("r = 64", |g| g.r = 64),
        ("timestamps", |g| g.w_ts = 4),
        ("exact cover", |g| g.b_scan = 3),
        ("zero ports", |g| g.b_ops = 0),
        ("μ = 64", |g| g.mu = 64),
    ];
    for (name, change) in cases {
        let mut geometry = valid;
        change(&mut geometry);
        assert!(!geometry.is_valid(), "{name}");
        assert_eq!(
            Program::new(geometry, vec![0; 4], vec![0; 4]),
            Err(ProgramError::Geometry),
            "{name}"
        );
        assert!(
            matches!(
                crate::nebula::Circuit::compile(geometry, Machine),
                Err(crate::Error::Circuit(_))
            ),
            "{name}"
        );
    }
}

/// SPEC.md section 7.1, rule 6; Lean `Geometry.Secure`.
#[test]
fn insecure_geometry_makes_no_circuit() {
    let insecure = crate::nebula::Geometry {
        r: 16,
        mu: 16,
        w_ts: 18,
        b_ops: 1,
        b_scan: 1,
        n: 1 << 17,
        s_max: 1,
    };
    assert!(insecure.is_valid());
    assert!(insecure.max_tuples() > 139_509);
    assert!(matches!(
        crate::nebula::Circuit::compile(insecure, Machine),
        Err(crate::Error::Circuit("the memory geometry breaks SPEC.md section 7.1"))
    ));
}

/// Lean `Machine.portApp` and `PortApp.portsLength`.
#[test]
fn machine_makes_no_circuit_with_a_third_port() {
    let mut geometry = Package::V1.geometry();
    geometry.b_ops = 3;
    assert!(geometry.is_valid() && geometry.max_tuples() <= 139_509);
    assert!(matches!(
        crate::nebula::Circuit::compile(geometry, Machine),
        Err(crate::Error::Circuit(
            "the application's port count is not the geometry's b_ops"
        ))
    ));
    assert!(matches!(
        crate::nebula::size_parts(&geometry, &Machine),
        Err(crate::Error::Circuit(
            "the application's port count is not the geometry's b_ops"
        ))
    ));
}

struct ReadsWord(usize);

impl Application for ReadsWord {
    type Input = ();
    type State = ();
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        0
    }

    fn aux(&self) -> usize {
        0
    }

    fn port_count(&self) -> usize {
        2
    }

    fn words(&self, _: &()) -> Vec<u64> {
        Vec::new()
    }

    fn rows(&self, view: &View) -> Result<Vec<Row>, crate::Error> {
        let word = crate::application::ApplicationBuilder::new(self.0 + 1)
            .unwrap()
            .private_inputs()[self.0];
        Ok(vec![(view.pad(0) * (view.vr(0) - Expr::Var(word))).into()])
    }

    fn step(&self, _: &(), _: &(), _: &Memory) -> Result<Transition<()>, Self::Error> {
        Ok(Transition {
            next: (),
            ports: vec![None, None],
            aux: Vec::new(),
        })
    }
}

/// Lean `AppView.env`.
#[test]
fn application_rows_outside_the_view_make_no_circuit() {
    for word in [90, 0] {
        assert!(
            matches!(
                crate::nebula::Circuit::compile(Package::V1.geometry(), ReadsWord(word)),
                Err(crate::Error::Application(
                    crate::application::ApplicationError::VariableOutOfScope(_)
                ))
            ),
            "word {word}"
        );
    }
}

#[derive(Clone)]
struct Listed;

impl Application for Listed {
    type Input = ();
    type State = Vec<u64>;
    type Error = std::convert::Infallible;

    fn width(&self) -> usize {
        2
    }

    fn aux(&self) -> usize {
        0
    }

    fn port_count(&self) -> usize {
        2
    }

    fn words(&self, state: &Vec<u64>) -> Vec<u64> {
        state.clone()
    }

    fn rows(&self, _: &View) -> Result<Vec<Row>, crate::Error> {
        Ok(Vec::new())
    }

    fn step(&self, state: &Vec<u64>, _: &(), _: &Memory) -> Result<Transition<Vec<u64>>, Self::Error> {
        Ok(Transition {
            next: state.clone(),
            ports: vec![None, None],
            aux: Vec::new(),
        })
    }
}

#[test]
fn terminal_checks_reject_a_state_of_another_width() {
    let mut run = Run::new(Listed, &Package::V1.program(), vec![0, 0]);
    run.segment(&vec![(); Package::V1.geometry().n]).unwrap();
    let mut statement = run.statement();
    assert!(run
        .context()
        .terminal_states(&statement, run.carry())
        .is_ok());
    statement.initial = vec![0, 0, 0];
    assert_eq!(
        run.context().terminal_states(&statement, run.carry()),
        Err(TerminalError::Width)
    );
}

fn prove_segment<A: Application<Input = Step, State = MachineState>>(
    prover: &crate::Prover,
    app: A,
) -> (Statement<MachineState>, Carry, crate::Proof) {
    let mut run = Run::new(app, &Package::V1.program(), MachineState::default());
    let z0 = run.initial_state();
    let invocations = run.segment(&[Step::Execute, Step::Execute]).unwrap();
    let mut proof = prover.prove(z0, &invocations[0].words).unwrap();
    assert_eq!(proof.state().current(), invocations[0].output);
    proof = prover.extend(&proof, &invocations[1].words).unwrap();
    assert_eq!(proof.state().current(), invocations[1].output);
    (run.statement(), *run.carry(), proof)
}

fn rust_built(engine: crate::Engine) {
    use crate::nebula::{Circuit, VerifyError};
    let started = std::time::Instant::now();
    let geometry = Package::V1.geometry();
    let path = std::env::temp_dir().join(format!("nightstream-nebula-setup-{}.package", std::process::id()));
    let _ = std::fs::remove_file(&path);
    Circuit::compile(geometry, Machine)
        .unwrap()
        .write(&path)
        .unwrap();
    let prover = crate::Prover::load(&path, engine, 109);
    std::fs::remove_file(&path).unwrap();
    let prover = prover.unwrap();
    let verifier = Circuit::compile(geometry, Machine)
        .unwrap()
        .verifier(engine, 109)
        .unwrap();
    let changed = Circuit::compile(geometry, Changed::Row).unwrap();
    eprintln!("rust-built setup elapsed={:?}", started.elapsed());
    let program = Package::V1.program();

    let (statement, carry, proof) = prove_segment(&prover, Machine);
    verifier
        .verify_parts(&program, &statement, &carry, &proof)
        .unwrap();
    let (statement, carry, proof) = prove_segment(&changed.circuit.prover(engine, 109).unwrap(), Changed::Row);
    assert!(matches!(
        verifier.verify_parts(&program, &statement, &carry, &proof),
        Err(VerifyError::Proof(_))
    ));
    eprintln!("rust-built elapsed={:?}", started.elapsed());
}

#[test]
#[ignore = "Full production-profile memory proofs; run separately under the 300-second cap."]
fn rust_built_verifier_accepts_a_saved_prover_and_rejects_a_changed_row_on_cpu() {
    rust_built(crate::Engine::Optimized);
}
