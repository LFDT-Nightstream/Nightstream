//! Candidate metadata checks and dispatch to the existing conformance bodies.
//! This is a test runner; the production loader retains its published pins.

use std::{
    fs,
    path::{Path, PathBuf},
    time::Instant,
};

use neo_ajtai::nightstream_fprime_setup::{
    commit_production_signed_units, PRODUCTION_CARRIER_WIDTH, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED,
    PRODUCTION_VERIFIER_ROWS, SETUP_ID,
};
use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use nightstream_fprime::{load_per_application_package, LoadedPerApplicationPackage};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use serde::{de::IgnoredAny, Deserialize};
use serde_json::Value;

const MODULUS: u64 = 0xffff_ffff_0000_0001;
const PROFILE: [u64; 14] = [4_294_967_295, 1, 2, 16, 65_536, 1, 16, 17, 16, 14, 28, 9, 54, 22];
const SCHEDULE: [u64; 10] = [1, 1, 1, 28, 10, 17, 14, 54, 16, 64];

// Poseidon2HashChainV1BindingParity schema 1 and AjtaiSetupV1Parity schema 3.
#[derive(Deserialize)]
struct LeanBinding(u64, [u64; 4], [u64; 4], Vec<u64>, Vec<u64>, [u64; 4]);

#[derive(Deserialize)]
struct LeanSetup(u64, Vec<u8>, IgnoredAny, IgnoredAny, Vec<u8>, IgnoredAny, Vec<u64>);

#[derive(Deserialize)]
struct LeanSparseCommitment(u64, Vec<u64>, Vec<[u64; 3]>, Vec<Vec<u64>>);

struct Candidate {
    package: LoadedPerApplicationPackage,
    bytes: Vec<u8>,
}

fn numeric_words(value: &Value) {
    match value {
        Value::Number(number) => {
            let word = number.as_u64().expect("canonical metadata integer");
            assert!(word < MODULUS, "canonical metadata Goldilocks word");
        }
        Value::Array(values) => values.iter().for_each(numeric_words),
        _ => panic!("metadata must contain only numeric arrays"),
    }
}

fn read_metadata(path: &Path) -> Value {
    let bytes = fs::read(path).expect("Lean metadata file");
    let value: Value = serde_json::from_slice(&bytes).expect("Lean metadata JSON");
    numeric_words(&value);
    let mut canonical = serde_json::to_vec(&value).expect("canonical Lean metadata");
    canonical.push(b'\n');
    assert_eq!(bytes, canonical, "canonical Lean metadata bytes at {}", path.display());
    value
}

fn words(bytes: &[u8]) -> Vec<u64> {
    bytes.iter().copied().map(u64::from).collect()
}

fn framed(values: &[u64]) -> Vec<u64> {
    let mut result = Vec::with_capacity(values.len() + 1);
    result.push(u64::try_from(values.len()).expect("authority frame length"));
    result.extend_from_slice(values);
    result
}

fn hash(values: &[u64]) -> [u64; 4] {
    assert!(values.iter().all(|word| *word < MODULUS), "canonical hash preimage");
    let fields = values
        .iter()
        .copied()
        .map(Goldilocks::from_u64)
        .collect::<Vec<_>>();
    poseidon2_hash(&fields).map(|word| word.as_canonical_u64())
}

fn component(tag: u64, authority: &[u64]) -> [u64; 4] {
    let mut preimage = words(b"Nightstream/FPrime/context/v1_1");
    preimage.push(tag);
    preimage.extend(framed(authority));
    hash(&preimage)
}

// Same independent numeric-array framing as tests/production_binding.rs.
// Source words come from the complete candidate, not from carried digests.
fn append_value_preimage(value: &Value, output: &mut Vec<u64>) {
    match value {
        Value::Number(number) => {
            let word = number.as_u64().expect("canonical package integer");
            output.extend([0, word & 0xffff_ffff, word >> 32, 0]);
        }
        Value::Array(values) => {
            let length = u64::try_from(values.len()).expect("package array length");
            output.extend([1, length & 0xffff_ffff, length >> 32, 0]);
            for value in values {
                append_value_preimage(value, output);
            }
        }
        _ => panic!("package must contain only numeric arrays"),
    }
}

fn require_words(actual: &[u64], expected: &[u64], label: &str) {
    assert_eq!(actual.len(), expected.len(), "{label} length");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert_eq!(actual, expected, "{label} word {index}");
    }
}

fn setup_authority_words() -> Vec<u64> {
    let mut authority = vec![SETUP_ID.len() as u64];
    authority.extend(words(SETUP_ID));
    authority.extend([
        PRODUCTION_VERIFIER_ROWS,
        PRODUCTION_MESSAGE_COLUMNS,
        PRODUCTION_SEED.len() as u64,
    ]);
    authority.extend(PRODUCTION_SEED.map(u64::from));
    authority
}

/// Execute the full-carrier primitive on the exact sparse input emitted by
/// Lean. Its final coordinate is nonzero; this is not a fresh CCS padding test.
pub fn check_sparse_commitment(path: &Path) {
    let started = Instant::now();
    let LeanSparseCommitment(schema, authority, support, expected_rows) =
        serde_json::from_value(read_metadata(path)).expect("Lean sparse-commitment schema");
    assert_eq!(schema, 1);
    require_words(
        &authority,
        &setup_authority_words(),
        "Lean sparse-commitment raw authority",
    );
    assert_eq!(authority.len(), 73);
    assert_eq!(
        support,
        [
            [0, 0, 1],
            [32_768, 27, MODULUS - 1],
            [PRODUCTION_MESSAGE_COLUMNS - 1, 53, 1],
        ],
        "canonical Lean block/lane/scalar support"
    );
    assert_eq!(expected_rows.len(), PRODUCTION_VERIFIER_ROWS as usize);
    assert!(
        expected_rows.iter().all(|row| row.len() == 54),
        "complete Lean commitment rows"
    );
    let mut carrier = vec![0_i8; PRODUCTION_CARRIER_WIDTH];
    for [block, lane, scalar] in &support {
        assert!(
            *block < PRODUCTION_MESSAGE_COLUMNS && *lane < 54,
            "sparse support range"
        );
        let coordinate = usize::try_from(*block * 54 + *lane).expect("sparse carrier coordinate");
        assert_eq!(carrier[coordinate], 0, "distinct sparse carrier coordinate");
        carrier[coordinate] = match *scalar {
            1 => 1,
            value if value == MODULUS - 1 => -1,
            _ => panic!("sparse support scalar is not a signed unit"),
        };
    }
    let actual = commit_production_signed_units(&carrier).expect("actual full-carrier selected-key commitment");
    assert_eq!((actual.d, actual.kappa), (54, PRODUCTION_VERIFIER_ROWS as usize));
    assert_eq!(actual.data.len(), expected_rows.len() * 54);
    for (row, expected) in expected_rows.iter().enumerate() {
        for (lane, expected) in expected.iter().enumerate() {
            assert_eq!(
                actual.data[row * 54 + lane].as_canonical_u64(),
                *expected,
                "Lean/Rust commitment row {row}, lane {lane}"
            );
        }
    }
    println!("primitive_support_coordinates={}", support.len());
    println!("primitive_commitment_coefficients={}", actual.data.len());
    println!("candidate_primitive_conformance=passed elapsed={:?}", started.elapsed());
}

impl Candidate {
    fn load(path: &Path, binding_path: &Path, setup_path: &Path, expected: [u64; 4]) -> Self {
        assert!(
            expected.iter().all(|word| *word < MODULUS),
            "canonical expected identity"
        );
        let bytes = fs::read(path).expect("Lean candidate sealed package");
        let package = load_per_application_package(&bytes, expected).expect("expected Lean candidate identity");
        let LeanBinding(schema, structural, package_identity, descriptor, key_preimage, key_digest) =
            serde_json::from_value(read_metadata(binding_path)).expect("Lean binding schema");
        assert_eq!(schema, 1);
        assert_eq!(structural, expected, "separately supplied Lean structural identity");
        let LeanSetup(schema, setup_id, _, _, seed, _, authority) =
            serde_json::from_value(read_metadata(setup_path)).expect("Lean setup schema");
        assert_eq!(schema, 3);
        assert_eq!(setup_id, SETUP_ID);
        assert_eq!(seed, PRODUCTION_SEED);
        let carrier_blocks = package.logical_column_count().div_ceil(54);
        assert_eq!(
            u64::try_from(carrier_blocks).expect("carrier block count"),
            PRODUCTION_MESSAGE_COLUMNS
        );
        require_words(&authority, &setup_authority_words(), "Lean raw Ajtai authority");
        assert_eq!(authority.len(), 73);

        let sealed: Value = serde_json::from_slice(&bytes).expect("candidate raw authority");
        let fields = sealed.as_array().expect("candidate envelope");
        let mut relation_words = Vec::new();
        append_value_preimage(&fields[1][4], &mut relation_words);
        relation_words.extend(structural);
        let mut application_words = Vec::new();
        append_value_preimage(&fields[3], &mut application_words);
        let commitment_digest = component(4, &authority);
        let mut nifs_words = words(b"Nightstream/FPrime/nifs-key/v1_1");
        nifs_words.extend(framed(&relation_words));
        nifs_words.extend(framed(&PROFILE));
        nifs_words.extend(framed(&SCHEDULE));
        nifs_words.extend(framed(&commitment_digest));
        let mut expected_descriptor = words(b"Nightstream/FPrime/verifier-context/v1_1");
        expected_descriptor.extend(framed(&PROFILE));
        expected_descriptor.extend(framed(&SCHEDULE));
        for digest in [
            component(1, &relation_words),
            component(2, &application_words),
            component(3, &nifs_words),
            commitment_digest,
        ] {
            expected_descriptor.extend(framed(&digest));
        }
        require_words(
            &descriptor,
            &expected_descriptor,
            "Lean full verifier-context descriptor",
        );
        let mut package_preimage = words(b"Nightstream/FPrime/sealed-package/v2");
        package_preimage.extend(framed(&structural));
        package_preimage.extend(framed(&descriptor));
        assert_eq!(
            hash(&package_preimage),
            package_identity,
            "Lean candidate package identity"
        );
        let mut expected_key_preimage = words(b"Nightstream/FPrime/verifier-key/v1");
        expected_key_preimage.extend(framed(&package_identity));
        expected_key_preimage.extend(framed(&descriptor));
        require_words(
            &key_preimage,
            &expected_key_preimage,
            "Lean full verification-key preimage",
        );
        assert_eq!(
            hash(&key_preimage),
            key_digest,
            "Lean candidate verification-key digest"
        );

        let binding = package
            .production_verifier_binding()
            .expect("candidate binding recomputation");
        assert_eq!(binding.structural_identifier(), structural);
        assert_eq!(binding.package_identity(), package_identity);
        assert_eq!(binding.verification_key_digest(), key_digest);
        require_words(
            binding.verification_key_words(),
            &key_preimage,
            "production verification-key preimage",
        );
        let context = binding.verifier_context();
        require_words(
            context.relation_words(),
            &relation_words,
            "production raw relation authority",
        );
        require_words(
            context.application_words(),
            &application_words,
            "production raw application authority",
        );
        require_words(context.nifs_key_words(), &nifs_words, "production raw NIFS authority");
        require_words(
            context.commitment_key_words(),
            &authority,
            "production raw commitment authority",
        );
        require_words(
            context.descriptor_words(),
            &descriptor,
            "production full context descriptor",
        );
        assert_eq!(context.digest(), hash(&descriptor));
        println!("candidate_structural_identity={structural:?}");
        println!("candidate_package_identity={package_identity:?}");
        println!("candidate_verifier_context={:?}", context.digest());
        println!("candidate_verification_key_digest={key_digest:?}");
        Self { package, bytes }
    }
}

/// Run exactly one independently bounded gate. Paths and expected identity
/// are explicit arguments; no candidate value replaces a production pin.
pub fn run(
    mode: &str,
    candidate_path: &Path,
    binding_path: &Path,
    setup_path: &Path,
    expected: [u64; 4],
    inputs: &[PathBuf],
) {
    let input_count = match mode {
        "physical" | "detached" => 1,
        "logical" | "mutations" => 0,
        "base" | "commitment" => 2,
        "recursive" | "recursive-mutations" => 7,
        _ => panic!(
            "mode must be physical, logical, mutations, base, recursive, recursive-mutations, commitment, or detached"
        ),
    };
    assert_eq!(
        inputs.len(),
        input_count,
        "mode paths: physical=expanded; logical/mutations=none; base=expanded,fixture; recursive/recursive-mutations=expanded,fixture,base,PiCCS-input,children,PiCCS-result,folded-metadata; commitment=fixture,output; detached=fixture"
    );
    let started = Instant::now();
    let Candidate { package, bytes } = Candidate::load(candidate_path, binding_path, setup_path, expected);
    println!("candidate metadata and binding: {:?}", started.elapsed());
    match mode {
        "physical" => {
            let expanded = fs::read(&inputs[0]).expect("Lean physical expansion");
            super::support::require_sealed_expansion(&bytes, &expanded);
            let matrices = package
                .r1cs_matrices()
                .expect("actual candidate physical matrices");
            let nonzeros = super::support::compare_lean_expanded_matrices(&expanded, &matrices);
            println!("candidate_physical_matrix_nonzeros={nonzeros:?}");
        }
        "logical" => super::logical_checks::check_logical_matrices(package, bytes),
        "mutations" => super::logical_checks::check_matrix_mutations(package, bytes),
        "base" => {
            let expanded = fs::read(&inputs[0]).expect("Lean physical expansion");
            let fixture = fs::read(&inputs[1]).expect("Lean base-step fixture");
            super::base_checks::check_base_assignment(package, bytes, fixture, expanded);
        }
        "recursive" | "recursive-mutations" => {
            let expanded = fs::read(&inputs[0]).expect("Lean physical expansion");
            let fixture = fs::read(&inputs[1]).expect("Lean recursive caller fixture");
            let base = fs::read(&inputs[2]).expect("checked base fixture");
            let input = fs::read(&inputs[3]).expect("preceding PiCCS input");
            let children = fs::read(&inputs[4]).expect("checked child claims");
            let result = fs::read(&inputs[5]).expect("preceding PiCCS result");
            let folded = fs::read(&inputs[6]).expect("checked folded metadata");
            super::recursive_checks::check_fixture(&fixture, &base, &input, &children, &result, &folded);
            if mode == "recursive-mutations" {
                super::base_checks::check_caller_mutations(package, bytes, fixture, expanded);
            } else {
                super::base_checks::check_caller_assignment(package, bytes, fixture, expanded);
            }
        }
        "commitment" => {
            let fixture = fs::read(&inputs[0]).expect("Lean base-step fixture");
            drop(bytes);
            let words = super::base_checks::check_base_commitment(package, fixture);
            let mut encoded = serde_json::to_vec(&words).expect("base commitment encoding");
            encoded.push(b'\n');
            fs::write(&inputs[1], encoded).expect("external base commitment output");
            println!("base_commitment_words={}", words.len());
        }
        "detached" => {
            let fixture = fs::read(&inputs[0]).expect("Lean base-step fixture");
            super::base_checks::check_detached_application(package, bytes, fixture);
        }
        _ => unreachable!("validated mode"),
    }
    println!("candidate_{mode}_conformance=passed elapsed={:?}", started.elapsed());
}
