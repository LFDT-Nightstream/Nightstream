//! Base fixture checks and shared complete caller-assignment validation.
//!
//! The inner zero PiCCS message is a base-only placeholder. This test is not
//! PiCCS conformance or a production proof. It checks the generated base
//! assignment and its canonical bounded carrier against all emitted rows.

use std::{fs, path::PathBuf, time::Instant};

use neo_ajtai::nightstream_fprime_setup::{commit_production_signed_units, PRODUCTION_CARRIER_WIDTH};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use nightstream_fprime::{
    derive_pi_ccs_v1_1_transcript, load_poseidon2_hash_chain_v1_package, LoadedPerApplicationPackage, WitnessAssignment,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{de::IgnoredAny, Deserialize};

#[allow(dead_code, unused_imports)]
#[path = "../src/bin/check_package_conformance/support.rs"]
mod conformance_support;

#[path = "per_application_logical_matrix_conformance/reference/mod.rs"]
mod logical_reference;

// Exact current package and BaseStepFixture schema dimensions.
const PRIVATE_INPUTS: usize = 177_326;
const PUBLIC_INPUTS: usize = 278;
const STATE_WORDS: usize = 49_393;
const PI_CCS_INPUT_END: usize = 128_074;
const CHILD_PUBLIC_START: usize = 173_002;
const CHILD_COUNT: usize = 16;
const PUBLIC_WORDS: usize = 270;
const INITIAL_STATE: [u64; 4] = [202, 203, 204, 205];
const MESSAGE: [u64; 4] = [7, 11, 13, 17];
const APPLICATION_TAG: &[u8] = b"Nightstream/Stage1/Poseidon2HashChain/v1";
const MODULUS: u64 = 0xffff_ffff_0000_0001;

#[derive(Deserialize)]
struct Expected(
    [u64; 4],
    [u64; 4],
    Vec<u64>,
    [[u64; 2]; 28],
    [u64; 8],
    [u64; 8],
    Vec<u64>,
);

#[derive(Deserialize)]
struct Fixture(u64, [u64; 4], Vec<u64>, Vec<u64>, Expected);

#[derive(Deserialize)]
struct Segment(u64, usize, usize);

#[derive(Deserialize)]
struct PhysicalLayout(usize, usize, usize, usize, usize, Vec<Segment>, Vec<Segment>);

#[derive(Deserialize)]
struct CircuitLayout(
    u64,
    IgnoredAny,
    IgnoredAny,
    PhysicalLayout,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
);

#[derive(Deserialize)]
struct SealedLayout(
    u64,
    CircuitLayout,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    IgnoredAny,
    usize,
);

fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

fn hash(words: &[u64]) -> [u64; 4] {
    let values = words
        .iter()
        .copied()
        .map(Goldilocks::from_u64)
        .collect::<Vec<_>>();
    poseidon2_hash(&values).map(|value| value.as_canonical_u64())
}

fn enc_hash(digest: [u64; 4]) -> Vec<u64> {
    let mut values = vec![1];
    for word in digest {
        values.extend((0..u64::BITS).map(|bit| (word >> bit) & 1));
    }
    values.resize(PUBLIC_WORDS, 0);
    values
}

fn check_preimage(words: &[u64], context: [u64; 4], iteration: u64, current: [u64; 4]) {
    assert_eq!(words.len(), STATE_WORDS);
    let domain = b"HyperNova/NIVC/state/v1"
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    assert_eq!(&words[..domain.len()], domain);
    assert_eq!(words[domain.len()], 4);
    assert_eq!(&words[domain.len() + 1..28], context);
    assert_eq!(words[28], iteration);
    assert_eq!(words[29], 4);
    assert_eq!(&words[30..34], INITIAL_STATE);
    assert_eq!(words[34], 4);
    assert_eq!(&words[35..39], current);
    let mut cursor = 39;
    for length in std::iter::once(56).chain((0..CHILD_COUNT).flat_map(|_| [1_188, PUBLIC_WORDS, 1_620])) {
        assert_eq!(words[cursor], length as u64);
        cursor += 1;
        assert!(
            words[cursor..cursor + length].iter().all(|word| *word == 0),
            "default running claim"
        );
        cursor += length;
    }
    assert_eq!(cursor + 1, words.len());
    assert_eq!(words[cursor], 1);
}

fn centered(value: u64) -> i128 {
    if value <= (MODULUS - 1) / 2 {
        i128::from(value)
    } else {
        i128::from(value) - i128::from(MODULUS)
    }
}

fn check_caller_layout(bytes: &[u8], private: &[u64], public: &[u64], assignment: &WitnessAssignment) {
    let SealedLayout(
        outer,
        CircuitLayout(inner, _, _, layout, _, _, _, _, _, _, _, _, _, _),
        _,
        _,
        _,
        _,
        logical_public,
    ) = serde_json::from_slice(bytes).expect("independent caller-layout decode");
    assert_eq!((outer, inner, logical_public), (6, 8, PUBLIC_WORDS));
    assert_eq!(
        (layout.0, layout.1, layout.2, layout.3, layout.4),
        (29_225_729, 29_344_146, 29_344_146, PUBLIC_INPUTS, 29_344_425)
    );
    assert_eq!(assignment.private_values().len(), layout.1);
    assert_eq!(assignment.public_values(), public);
    let mut cursor = 0;
    for Segment(role, start, length) in layout.5 {
        // Schema-8 roles 3,15,16,18 are generated witness intervals.
        match role {
            3 | 15 | 16 | 18 => continue,
            1 | 2 | 6..=14 | 17 => {}
            _ => panic!("unexpected private caller role {role}"),
        }
        assert_eq!(
            &assignment.private_values()[start..start + length],
            &private[cursor..cursor + length],
            "caller role {role} at physical column {start}"
        );
        cursor += length;
    }
    assert_eq!(cursor, private.len());
    let mut physical = layout.2 + 1;
    let mut public_count = 0;
    for (Segment(role, start, length), expected_role) in layout.6.iter().zip([4, 5, 10]) {
        assert_eq!(*role, expected_role);
        assert_eq!(*start, physical);
        physical += length;
        public_count += length;
    }
    assert_eq!(layout.6.len(), 3);
    assert_eq!((physical, public_count), (layout.4, public.len()));
}

#[test]
#[ignore = "complete base witness, production transport, and independent rows; run under the 300-second cap"]
fn base_step_assignment_satisfies_every_canonical_row() {
    let sealed =
        fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).expect("canonical package");
    let bytes =
        fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).expect("Lean base-step fixture");
    let expanded = fs::read(artifact(
        "nightstream-fprime-stage1-poseidon2-hash-chain-v1-expanded.json",
    ))
    .expect("Lean canonical physical expansion");
    let package = load_poseidon2_hash_chain_v1_package(&sealed).expect("verifier-owned package");
    check_base_assignment(package, sealed, bytes, expanded);
}

fn checked_caller_fixture(package: &LoadedPerApplicationPackage, bytes: &[u8]) -> Fixture {
    let Fixture(schema, context, private, public, expected) =
        serde_json::from_slice(bytes).expect("base-step fixture schema");
    assert_eq!(schema, 1);
    assert_eq!((private.len(), public.len()), (PRIVATE_INPUTS, PUBLIC_INPUTS));
    assert_eq!((expected.2.len(), expected.6.len()), (PUBLIC_WORDS, PUBLIC_WORDS));
    assert!(context
        .iter()
        .chain(&private)
        .chain(&public)
        .chain(&expected.0)
        .chain(&expected.1)
        .chain(&expected.2)
        .chain(expected.3.iter().flatten())
        .chain(&expected.4)
        .chain(&expected.5)
        .chain(&expected.6)
        .all(|word| *word < MODULUS));
    assert_eq!(
        package
            .production_verifier_binding()
            .expect("package binding")
            .verifier_context()
            .digest(),
        context
    );
    assert_eq!(
        (package.private_input_count(), package.public_input_count()),
        (private.len(), public.len())
    );
    assert_eq!(package.total_column_count(), 29_344_425);
    assert_eq!(package.physical_row_count(), 29_225_729);
    assert_eq!(package.row_count(), logical_reference::evaluation::ACTIVE_ROWS);
    assert_eq!(
        package.logical_column_count(),
        logical_reference::evaluation::LOGICAL_WIDTH
    );
    assert_eq!(package.logical_public_input_count(), PUBLIC_WORDS);

    Fixture(schema, context, private, public, expected)
}

fn checked_base_fixture(package: &LoadedPerApplicationPackage, bytes: &[u8]) -> Fixture {
    let Fixture(schema, context, private, public, expected) = checked_caller_fixture(package, bytes);

    let prior = &private[..STATE_WORDS];
    let output = &private[STATE_WORDS..2 * STATE_WORDS];
    check_preimage(prior, context, 0, INITIAL_STATE);
    check_preimage(output, context, 1, expected.0);
    let mut application_input = APPLICATION_TAG
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    application_input.extend(INITIAL_STATE);
    application_input.extend(MESSAGE);
    assert_eq!(hash(&application_input), expected.0);
    assert_eq!(hash(output), expected.1);
    assert_eq!(&public[..PUBLIC_WORDS], enc_hash(hash(prior)));
    assert_eq!(&public[PUBLIC_WORDS..PUBLIC_WORDS + 4], expected.1);
    assert_eq!(&public[PUBLIC_WORDS + 4..], context);
    assert_eq!(expected.2, enc_hash(expected.1));
    assert_eq!(&private[PRIVATE_INPUTS - MESSAGE.len()..], MESSAGE);
    assert!(
        private[2 * STATE_WORDS..CHILD_PUBLIC_START]
            .iter()
            .all(|word| *word == 0),
        "base-only zero inner proof and zero child commitment/evaluation messages"
    );
    let children = &private[CHILD_PUBLIC_START..PRIVATE_INPUTS - MESSAGE.len()];
    assert_eq!(children.len(), CHILD_COUNT * PUBLIC_WORDS);
    assert!(
        children.iter().any(|word| *word != 0),
        "checked child digits must not be replaced with zeros"
    );
    for (column, parent) in expected.6.iter().copied().enumerate() {
        let parent = centered(parent);
        assert!(parent.abs() < 1i128 << CHILD_COUNT, "strict parent B = 2^16 bound");
        let mut recombined = 0i128;
        for child in 0..CHILD_COUNT {
            let digit = centered(children[child * PUBLIC_WORDS + column]);
            assert!(digit.abs() < 2, "bounded child public digit");
            recombined += digit * (1i128 << child);
        }
        assert_eq!(
            recombined, parent,
            "checked child recombination at public column {column}"
        );
    }
    let transcript = derive_pi_ccs_v1_1_transcript(
        &[hash(prior).to_vec(), vec![0; 1_188], public[..PUBLIC_WORDS].to_vec()],
        &[vec![0; 28 * 2], vec![0; 16 * 1_620]],
        &vec![vec![[0; 2]; 10]; 28],
        &vec![0; 17 * 1_620],
    )
    .expect("base placeholder transcript replay");
    assert_eq!(transcript.round_point(), expected.3);
    assert_eq!(transcript.outgoing_state(), expected.4);
    Fixture(schema, context, private, public, expected)
}

/// The same complete base-assignment check used by the pinned test and the
/// external candidate CLI. The caller selects the identity-checked package.
pub fn check_base_assignment(package: LoadedPerApplicationPackage, sealed: Vec<u8>, bytes: Vec<u8>, expanded: Vec<u8>) {
    let fixture = checked_base_fixture(&package, &bytes);
    check_assignment(package, sealed, fixture, expanded);
}

/// Check the complete selected caller packet after its caller-specific
/// preimage, proof, and handoff checks. Every canonical row is still checked.
pub fn check_caller_assignment(
    package: LoadedPerApplicationPackage,
    sealed: Vec<u8>,
    bytes: Vec<u8>,
    expanded: Vec<u8>,
) {
    let fixture = checked_caller_fixture(&package, &bytes);
    check_assignment(package, sealed, fixture, expanded);
}

/// Mutate child caller columns only after a valid physical witness exists.
/// The raw evaluator sees each changed assignment without witness execution.
pub fn check_caller_mutations(
    package: LoadedPerApplicationPackage,
    sealed: Vec<u8>,
    bytes: Vec<u8>,
    expanded: Vec<u8>,
) {
    let Fixture(_, _, private, public, _) = checked_caller_fixture(&package, &bytes);
    conformance_support::require_sealed_expansion(&sealed, &expanded);
    let physical = package
        .execute_witness(&private, &public)
        .expect("valid mutation control");
    check_caller_layout(&sealed, &private, &public, &physical);
    assert_eq!(
        conformance_support::evaluate_canonical_assignment(
            &expanded,
            physical.private_values(),
            physical.public_values()
        ),
        Ok(package.physical_row_count()),
        "independent positive control"
    );
    let SealedLayout(_, CircuitLayout(_, _, _, layout, _, _, _, _, _, _, _, _, _, _), _, _, _, _, _) =
        serde_json::from_slice(&sealed).expect("independent child caller layout");
    let fields = [
        ("commitment", PI_CCS_INPUT_END),
        ("Eval_K", PI_CCS_INPUT_END + CHILD_COUNT * 1_188),
        ("Eval_A", PI_CCS_INPUT_END + CHILD_COUNT * (1_188 + 54 * 2)),
    ];
    let mut columns = Vec::new();
    let mut cursor = 0;
    for Segment(role, start, length) in layout.5 {
        if matches!(role, 3 | 15 | 16 | 18) {
            continue;
        }
        for &(label, caller) in &fields {
            if (cursor..cursor + length).contains(&caller) {
                columns.push((label, start + caller - cursor));
            }
        }
        cursor += length;
    }
    assert_eq!(cursor, private.len());
    assert_eq!(
        columns.len(),
        fields.len(),
        "one physical owner per selected child field"
    );
    let mut raw = physical.private_values().to_vec();
    for (label, column) in columns {
        let original = raw[column];
        raw[column] = (original + 1) % MODULUS;
        let row = conformance_support::evaluate_canonical_assignment(&expanded, &raw, physical.public_values())
            .expect_err("independent child assignment mutation must fail");
        assert!(row < package.physical_row_count());
        raw[column] = original;
        println!("independent_child_assignment_mutation={label} column={column} rejected_row={row}");
    }
    println!("independent_child_assignment_mutations=passed cases={}", fields.len());
}

fn check_assignment(package: LoadedPerApplicationPackage, sealed: Vec<u8>, fixture: Fixture, expanded: Vec<u8>) {
    let started = Instant::now();
    let Fixture(_, _, private, public, expected) = fixture;
    conformance_support::require_sealed_expansion(&sealed, &expanded);
    println!("caller fixture and package checks: {:?}", started.elapsed());

    let physical = package
        .execute_witness(&private, &public)
        .expect("complete caller physical assignment");
    check_caller_layout(&sealed, &private, &public, &physical);
    assert_eq!(
        &physical.private_values()[..PI_CCS_INPUT_END],
        &private[..PI_CCS_INPUT_END]
    );
    for (&column, value) in package.application().witness_columns().iter().zip(MESSAGE) {
        assert_eq!(physical.private_values()[column], value);
    }
    for (column, value) in package
        .application()
        .output_columns()
        .into_iter()
        .zip(expected.0)
    {
        assert_eq!(physical.private_values()[column], value);
    }
    println!("caller physical witness and caller layout: {:?}", started.elapsed());
    let physical_rows = conformance_support::evaluate_canonical_assignment(
        &expanded,
        physical.private_values(),
        physical.public_values(),
    )
    .expect("every independent physical caller row holds");
    assert_eq!(physical_rows, package.physical_row_count());
    drop(expanded);
    println!(
        "caller independent physical rows: {physical_rows}, elapsed {:?}",
        started.elapsed()
    );

    let production = package
        .execute_logical_assignment(&physical)
        .expect("production caller logical transport");
    assert_eq!(production.len(), logical_reference::evaluation::LOGICAL_WIDTH);
    assert!(
        production
            .balanced_values()
            .iter()
            .all(|value| (-1..=1).contains(value)),
        "every logical coordinate is bounded"
    );
    assert_eq!(production.balanced_values()[0], 1);
    for (column, expected) in expected.2.iter().copied().enumerate() {
        assert_eq!(
            production
                .value(column)
                .expect("production public coordinate"),
            expected,
            "next fresh public projection at column {column}"
        );
    }
    drop(package);
    println!(
        "caller production logical transport and bounds: {:?}",
        started.elapsed()
    );
    let independent = logical_reference::assignment::LogicalAssignment::decode(
        &sealed,
        physical.private_values(),
        physical.public_values(),
    )
    .expect("independent complete caller logical lift");
    assert_eq!(independent.len(), production.len());
    for (column, (actual, expected)) in production
        .balanced_values()
        .iter()
        .zip(independent.balanced_values())
        .enumerate()
    {
        assert_eq!(actual, expected, "caller logical transport coordinate {column}");
    }
    let alignment = logical_reference::evaluation::CARRIER_WIDTH - production.len();
    assert_eq!(alignment, 37);
    // The public transport returns logical coordinates. The paper carrier
    // extends them with these alignment zeros; no backend allocator is used.
    for column in production.len()..logical_reference::evaluation::CARRIER_WIDTH {
        assert_eq!(
            independent
                .carrier_value(column)
                .expect("canonical carrier alignment"),
            logical_reference::Field::ZERO
        );
    }
    drop(independent);
    drop(physical);
    println!(
        "caller all logical coordinates compared; {alignment} alignment zeros: {:?}",
        started.elapsed()
    );

    let relation = logical_reference::relation::Relation::decode(&sealed).expect("independent final relation");
    let artifact = logical_reference::source::SourcePackage::decode(&sealed).expect("independent source package");
    let program = logical_reference::matrix::MatrixProgram::decode(
        &artifact.matrix_program,
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent canonical matrix program");
    assert_eq!(artifact.sealed_schema, 6);
    assert_eq!(artifact.logical_rows, logical_reference::evaluation::ACTIVE_ROWS);
    assert_eq!(artifact.logical_columns, production.len());
    assert_eq!(artifact.cube_variables, expected.3.len());
    assert_eq!(artifact.logical_public_inputs, expected.2.len());
    println!("caller complete logical row program decoded: {:?}", started.elapsed());
    logical_reference::evaluation::verify_satisfaction_with(&program, &artifact.sources, &relation, |column| {
        let value = if column < production.len() {
            production
                .value(column)
                .expect("production logical coordinate")
        } else {
            assert!(
                column < logical_reference::evaluation::CARRIER_WIDTH,
                "canonical carrier column bound"
            );
            0
        };
        logical_reference::Field::checked(value, "caller production carrier coordinate").expect("canonical field word")
    })
    .expect("every canonical logical row holds on the production caller assignment");
    println!(
        "caller all {} logical rows and zero padding checked: {:?}",
        artifact.logical_rows,
        started.elapsed()
    );
}

#[test]
#[ignore = "commit the complete independently checked base carrier; run under the 300-second cap"]
fn base_step_assignment_has_a_selected_key_commitment() {
    let sealed =
        fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).expect("canonical package");
    let bytes =
        fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).expect("Lean base-step fixture");
    let package = load_poseidon2_hash_chain_v1_package(&sealed).expect("verifier-owned package");
    let words = check_base_commitment(package, bytes);
    println!(
        "base selected-key commitment: {}",
        serde_json::to_string(&words).expect("commitment words")
    );
}

/// Commit the same complete bounded base carrier under the selected key.
/// This computes an input commitment; it does not construct a PiCCS proof.
pub fn check_base_commitment(package: LoadedPerApplicationPackage, bytes: Vec<u8>) -> Vec<u64> {
    let started = Instant::now();
    let Fixture(_, _, private, public, expected) = checked_base_fixture(&package, &bytes);
    let physical = package
        .execute_witness(&private, &public)
        .expect("complete base witness");
    let logical = package
        .execute_logical_assignment(&physical)
        .expect("complete base logical transport");
    assert_eq!(logical.len(), logical_reference::evaluation::LOGICAL_WIDTH);
    assert_eq!(PRODUCTION_CARRIER_WIDTH, logical_reference::evaluation::CARRIER_WIDTH);
    assert_eq!(expected.2.len(), PUBLIC_WORDS);
    for (column, word) in expected.2.iter().copied().enumerate() {
        assert_eq!(logical.value(column).expect("public projection"), word);
    }
    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(PRODUCTION_CARRIER_WIDTH, 0);
    drop(logical);
    drop(physical);
    drop(package);
    println!(
        "base complete carrier prepared for indexed commitment: {:?}",
        started.elapsed()
    );
    let commitment = commit_production_signed_units(&carrier).expect("actual selected-key base commitment");
    assert_eq!((commitment.d, commitment.kappa), (54, 22));
    let words = commitment
        .data
        .iter()
        .map(|word| word.as_canonical_u64())
        .collect::<Vec<_>>();
    assert_eq!(words.len(), 1_188);
    assert!(words.iter().any(|word| *word != 0));
    println!("base full carrier commitment complete: {:?}", started.elapsed());
    words
}

#[test]
#[ignore = "independent arbitrary-assignment application-wiring regression; run under the 300-second cap"]
fn base_step_rows_reject_a_detached_application_output() {
    let sealed =
        fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).expect("canonical package");
    let bytes =
        fs::read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json")).expect("Lean base-step fixture");
    let package = load_poseidon2_hash_chain_v1_package(&sealed).expect("verifier-owned package");
    check_detached_application(package, sealed, bytes);
}

/// Replace only the application witness/local suffix with another valid
/// execution. The independent rows must reject the detached state binding.
pub fn check_detached_application(package: LoadedPerApplicationPackage, sealed: Vec<u8>, bytes: Vec<u8>) {
    let started = Instant::now();
    let Fixture(_, _, private, public, expected) = checked_base_fixture(&package, &bytes);
    let physical = package
        .execute_witness(&private, &public)
        .expect("base witness");
    let original = package
        .execute_logical_assignment(&physical)
        .expect("base logical transport");
    drop(physical);

    let mut changed_private = private.clone();
    let mut changed_message = MESSAGE;
    changed_message[0] += 1;
    changed_private[PRIVATE_INPUTS - MESSAGE.len()..].copy_from_slice(&changed_message);
    let mut step_words = APPLICATION_TAG
        .iter()
        .copied()
        .map(u64::from)
        .collect::<Vec<_>>();
    step_words.extend(INITIAL_STATE);
    step_words.extend(changed_message);
    let changed_output = hash(&step_words);
    assert_ne!(
        changed_output, expected.0,
        "the changed message has a different application output"
    );
    changed_private[STATE_WORDS + 35..STATE_WORDS + 39].copy_from_slice(&changed_output);
    let mut changed_public = public.clone();
    changed_public[PUBLIC_WORDS..PUBLIC_WORDS + 4]
        .copy_from_slice(&hash(&changed_private[STATE_WORDS..2 * STATE_WORDS]));
    let changed_physical = package
        .execute_witness(&changed_private, &changed_public)
        .expect("construct the changed application's local witness");
    let changed = package
        .execute_logical_assignment(&changed_physical)
        .expect("encode the changed application's logical coordinates");
    drop(changed_physical);
    drop(package);

    let relation = logical_reference::relation::Relation::decode(&sealed).expect("independent final relation");
    let artifact = logical_reference::source::SourcePackage::decode(&sealed).expect("independent source package");
    let program = logical_reference::matrix::MatrixProgram::decode(
        &artifact.matrix_program,
        &artifact.sources,
        artifact.logical_columns,
        artifact.logical_rows,
    )
    .expect("independent canonical matrix program");
    // DirectPiRLCSamplerCompletePrefixPlan.plan_rowCount and the concrete
    // application row-count theorem locate these 7,700 canonical rows.
    const APPLICATION_ROW_START: usize = 6_369_850;
    const APPLICATION_ROW_END: usize = APPLICATION_ROW_START + 7_700;
    let checked = logical_reference::evaluation::verify_satisfaction_range_with(
        &program,
        &artifact.sources,
        &relation,
        APPLICATION_ROW_START,
        APPLICATION_ROW_END,
        |column| {
            let value = changed.value(column).map_err(|error| error.to_string())?;
            logical_reference::Field::checked(value, "changed application assignment")
        },
    )
    .expect("the replacement application suffix satisfies its canonical rows with its own state");
    assert_eq!(checked, APPLICATION_ROW_END - APPLICATION_ROW_START);

    // ApplicationRetainedGeometry.witnessStart = 256216447 on this identity.
    // Keep every prefix/hash/public coordinate unchanged and replace only
    // the application witness/local block family. Input/output coordinates
    // belong to the preserved pilot preimage blocks.
    const APPLICATION_START: usize = 256_216_447;
    let mut detached = original.balanced_values().to_vec();
    detached[APPLICATION_START..].copy_from_slice(&changed.balanced_values()[APPLICATION_START..]);
    assert_eq!(
        &detached[..APPLICATION_START],
        &original.balanced_values()[..APPLICATION_START]
    );
    assert_ne!(
        &detached[APPLICATION_START..],
        &original.balanced_values()[APPLICATION_START..]
    );
    assert!(detached.iter().all(|value| (-1..=1).contains(value)));
    drop(original);
    drop(changed);
    println!("detached application assignment prepared: {:?}", started.elapsed());

    let result =
        logical_reference::evaluation::verify_satisfaction_with(&program, &artifact.sources, &relation, |column| {
            assert!(column < logical_reference::evaluation::CARRIER_WIDTH);
            match detached.get(column).copied().unwrap_or(0) {
                -1 => logical_reference::Field::checked(MODULUS - 1, "negative unit").expect("canonical unit"),
                0 => logical_reference::Field::ZERO,
                1 => logical_reference::Field::checked(1, "positive unit").expect("canonical unit"),
                _ => unreachable!("bounded detached assignment"),
            }
        });
    let failure = result.expect_err("canonical rows must reject the detached application output");
    let (row, residual) = failure
        .strip_prefix("logical relation failed at row ")
        .and_then(|detail| detail.split_once(": "))
        .expect("rejection must be a nonzero canonical row residual");
    let row = row.parse::<usize>().expect("rejected row index");
    assert!(
        (APPLICATION_ROW_START..APPLICATION_ROW_END).contains(&row),
        "rejection must occur in the application rows: {failure}"
    );
    let residual = residual.parse::<u64>().expect("canonical residual value");
    assert!(residual != 0 && residual < MODULUS);
    println!(
        "detached application rejected at canonical row {row}: {:?}",
        started.elapsed()
    );
}
