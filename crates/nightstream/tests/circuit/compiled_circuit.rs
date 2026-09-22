use super::*;
use crate::application::{Affine, ApplicationBuilder};
use p3_field::PrimeCharacteristicRing;
#[cfg(feature = "metal")]
use std::{fs::File, io::Read};
use std::{
    fs::{self, OpenOptions},
    io::{Seek, SeekFrom, Write},
    path::{Path, PathBuf},
    sync::{
        atomic::{AtomicU64, Ordering},
        OnceLock,
    },
};

struct TestDirectory(PathBuf);

impl TestDirectory {
    fn new() -> Self {
        static NEXT: AtomicU64 = AtomicU64::new(0);
        loop {
            let ordinal = NEXT.fetch_add(1, Ordering::Relaxed);
            let path = std::env::temp_dir().join(format!(
                "nightstream-compiled-circuit-test-{}-{ordinal}",
                std::process::id()
            ));
            match fs::create_dir(&path) {
                Ok(()) => return Self(path),
                Err(error) if error.kind() == std::io::ErrorKind::AlreadyExists => continue,
                Err(error) => panic!("create test directory: {error}"),
            }
        }
    }

    fn path(&self, name: &str) -> PathBuf {
        self.0.join(name)
    }
}

impl Drop for TestDirectory {
    fn drop(&mut self) {
        let _ = fs::remove_dir_all(&self.0);
    }
}

fn reference() -> Vec<u8> {
    fs::read(
        Path::new(env!("CARGO_MANIFEST_DIR")).join("artifacts/nightstream-fprime-stage1-poseidon2-hash-chain-v1.json"),
    )
    .unwrap()
}

fn compiled_fixture() -> &'static Circuit {
    // Share the full reference compilation across the storage tests. The
    // ignored Metal test has its own original and changed applications.
    static CIRCUIT: OnceLock<Circuit> = OnceLock::new();
    CIRCUIT.get_or_init(|| {
        let mut builder = ApplicationBuilder::new(2).unwrap();
        let input = builder.input_state();
        let private = builder.private_inputs().to_vec();
        let sum = builder
            .affine(Affine::from(input[0]) + Affine::from(private[0]))
            .unwrap();
        let product = builder.multiply(sum.into(), private[1].into()).unwrap();
        let application = builder
            .finish([
                product.into(),
                Affine::from(input[1]) + Affine::constant(F::ONE),
                input[2].into(),
                Affine::constant(F::from_u64(17)),
            ])
            .unwrap();
        Circuit::compile(&reference(), application).unwrap()
    })
}

#[test]
fn security_minimum_is_explicit_and_cannot_be_lowered_by_preparation() {
    let circuit = compiled_fixture();
    for minimum in [0, 125] {
        assert!(matches!(
            circuit.prover(Engine::Optimized, minimum),
            Err(Error::Parameters(_))
        ));
        assert!(matches!(
            Verifier::from_package(circuit, Engine::Optimized, minimum),
            Err(Error::Parameters(_))
        ));
    }
    assert!(circuit.prover(Engine::Optimized, 114).is_ok());
    assert!(Verifier::from_package(circuit, Engine::Optimized, 114).is_ok());
}

#[test]
fn extension_errors_leave_the_supplied_proof_unchanged() {
    let prover = compiled_fixture().prover(Engine::Optimized, 114).unwrap();
    let original = Stage1Envelope::initial([F::ZERO; 4]);
    assert!(matches!(prover.extend(&original, &[]), Err(Error::Application(_))));
    assert!(original.is_initial());
    assert_eq!(original.state().iteration(), 0);
    // This reaches lifecycle validation after valid application execution.
    let invalid = Stage1Envelope::from_parts(
        Stage1State::new(1, [F::ZERO; 4], [F::ZERO; 4]),
        crate::folding::RunningInstance::default(),
        crate::folding::CcsInstance {
            claim: crate::folding::CcsClaim {
                c: neo_ajtai::Commitment::zeros(neo_math::D, 22),
                x: Vec::new(),
                m_in: 0,
                adv: None,
            },
            witness: crate::folding::CcsWitness {
                w: Vec::new(),
                Z: neo_ccs::Mat::virtual_constant(0, 0, F::ZERO),
            },
        },
    );
    let before = format!("{invalid:?}");
    assert!(prover.extend(&invalid, &[F::ONE, F::ONE]).is_err());
    assert_eq!(format!("{invalid:?}"), before);
}

fn assert_same_data(expected: &Circuit, actual: &Circuit) {
    assert_eq!(actual.identity(), expected.identity());
    assert_eq!(actual.compiled.binding, expected.compiled.binding);
    let expected_package = &expected.compiled.package;
    let actual_package = &actual.compiled.package;
    assert_eq!(actual_package.application(), expected_package.application());
    assert_eq!(actual_package.assignment_plan(), expected_package.assignment_plan());
    assert_eq!(actual_package.ccs_relation(), expected_package.ccs_relation());
    assert_eq!(actual_package.row_count(), expected_package.row_count());
    assert_eq!(
        actual_package.logical_column_count(),
        expected_package.logical_column_count()
    );

    let input = [2, 3, 5, 7].map(F::from_u64);
    let private = [11, 13].map(F::from_u64);
    let expected_witness = expected
        .compiled
        .application
        .execute(input, &private)
        .unwrap();
    let actual_witness = actual
        .compiled
        .application
        .execute(input, &private)
        .unwrap();
    assert_eq!(expected_witness.output_state(), [169, 4, 5, 17].map(F::from_u64));
    assert_eq!(actual_witness.values(), expected_witness.values());
    assert_eq!(actual_witness.output_state(), expected_witness.output_state());
    assert_eq!(
        actual.compiled.application.prepared_output_forms().unwrap(),
        [2, 0, 2, 0]
    );

    // Matrix visits use logical rows; application metadata uses source rows.
    let end = expected_package.row_count();
    for range in [0..1, end - 1..end] {
        let mut rows = Vec::new();
        expected_package
            .visit_matrix_rows(range.clone(), |index, row| {
                rows.push((index, row));
                Ok(())
            })
            .unwrap();
        let mut expected_rows = rows.into_iter();
        actual_package
            .visit_matrix_rows(range, |index, row| {
                assert_eq!(Some((index, row)), expected_rows.next());
                Ok(())
            })
            .unwrap();
        assert!(expected_rows.next().is_none());
    }
}

#[test]
fn compiled_roundtrip_preserves_identity_witness_and_selected_matrix_rows() {
    let original = compiled_fixture();
    let directory = TestDirectory::new();
    let path = directory.path("roundtrip.package");
    original.write(&path).unwrap();
    let loaded = Circuit::load(&path).unwrap();
    assert_same_data(original, &loaded);
}

#[test]
fn compiled_load_rejects_bad_root_magic_output_tags_trailing_and_truncated_data() {
    let directory = TestDirectory::new();
    let source = directory.path("valid.package");
    compiled_fixture().write(&source).unwrap();
    assert!(Circuit::load(&source).is_ok(), "valid control package");

    // The outer format is the eight-byte NSCIR version followed by four
    // output-form tags. Only tags A=0 and C=2 are allowed for these forms.
    for (name, offset, byte) in [
        ("bad-magic", 0, b'!'),
        ("output-b", 8, 1),
        ("output-unknown", 11, u8::MAX),
    ] {
        let path = directory.path(name);
        fs::copy(&source, &path).unwrap();
        let mut file = OpenOptions::new().write(true).open(&path).unwrap();
        file.seek(SeekFrom::Start(offset)).unwrap();
        file.write_all(&[byte]).unwrap();
        drop(file);
        assert!(Circuit::load(&path).is_err(), "accepted {name}");
    }

    let path = directory.path("trailing");
    fs::copy(&source, &path).unwrap();
    OpenOptions::new()
        .append(true)
        .open(&path)
        .unwrap()
        .write_all(&[0])
        .unwrap();
    assert!(Circuit::load(&path).is_err(), "accepted trailing data");

    let length = fs::metadata(&source).unwrap().len();
    for (name, end) in [("short-magic", 7), ("short-output", 11), ("short-record", length - 1)] {
        let path = directory.path(name);
        fs::copy(&source, &path).unwrap();
        OpenOptions::new()
            .write(true)
            .open(&path)
            .unwrap()
            .set_len(end)
            .unwrap();
        assert!(Circuit::load(&path).is_err(), "accepted {name}");
    }
}

#[test]
fn loaded_execution_and_resaving_use_private_snapshots_after_external_mutation() {
    let original = compiled_fixture();
    let directory = TestDirectory::new();
    let source = directory.path("source.package");
    original.write(&source).unwrap();
    let loaded = Circuit::load(&source).unwrap();

    // Truncate the same inode, so keeping an open external file would not pass.
    fs::write(&source, b"changed after loading").unwrap();
    assert!(Circuit::load(&source).is_err());
    assert_same_data(original, &loaded);

    // This also reads the retained fixed-envelope snapshot, not just decoded rows.
    let saved = directory.path("resaved.package");
    loaded.write(&saved).unwrap();
    assert_same_data(original, &Circuit::load(&saved).unwrap());
}

#[test]
fn atomic_write_refuses_to_replace_an_existing_destination() {
    let directory = TestDirectory::new();
    let path = directory.path("existing.package");
    let contents = b"existing destination must remain unchanged";
    fs::write(&path, contents).unwrap();
    assert!(matches!(
        compiled_fixture().write(&path),
        Err(Error::Io(error)) if error.kind() == std::io::ErrorKind::AlreadyExists
    ));
    assert_eq!(fs::read(&path).unwrap(), contents);
    let remaining: Vec<_> = fs::read_dir(&directory.0)
        .unwrap()
        .map(|entry| entry.unwrap().path())
        .collect();
    assert_eq!(remaining, vec![path], "failed publication retained a staging file");
}

#[cfg(feature = "metal")]
fn private_zero_application(coefficient: F) -> ApplicationCircuit {
    let mut builder = ApplicationBuilder::new(1).unwrap();
    let input = builder.input_state();
    let private = builder.private_inputs()[0];
    // x + 1 = 1 means x = 0. Both coefficients use the same lowering and
    // retain one raw term; the zero coefficient removes only the constraint.
    builder
        .assert_equal(
            Affine::from(private) * coefficient + Affine::constant(F::ONE),
            Affine::constant(F::ONE),
        )
        .unwrap();
    builder.finish(input.map(Affine::from)).unwrap()
}

#[cfg(feature = "metal")]
fn copy_cached_identity_components(original: &Path, changed: &Path) {
    // Outer NSCIR header: 8-byte magic + 4 output tags. Inner NSFPREP1:
    // 8-byte magic + u64 source length + 4 structural words + 4 application
    // digest words + u64 application word count. Do not replace the source length.
    const CACHE_START: usize = 8 + 4 + 8 + 8;
    const CACHE_END: usize = CACHE_START + (4 + 4 + 1) * 8;
    let prefix = |path| {
        let mut bytes = [0; CACHE_END];
        File::open(path).unwrap().read_exact(&mut bytes).unwrap();
        assert_eq!(&bytes[..8], b"NSCIR\0\0\x01");
        assert_eq!(&bytes[12..20], b"NSFPREP1");
        bytes
    };
    let intended = prefix(original);
    let altered = prefix(changed);
    assert_ne!(&intended[CACHE_START..CACHE_END], &altered[CACHE_START..CACHE_END]);
    assert_eq!(
        &intended[CACHE_END - 8..],
        &altered[CACHE_END - 8..],
        "raw preimage lengths"
    );
    let mut file = OpenOptions::new().write(true).open(changed).unwrap();
    file.seek(SeekFrom::Start(CACHE_START as u64)).unwrap();
    file.write_all(&intended[CACHE_START..CACHE_END]).unwrap();
}

#[cfg(feature = "metal")]
#[test]
#[ignore = "Production Metal proof with changed private constraint and copied identity claims; run separately under the 300-second cap."]
fn metal_verifier_rejects_changed_constraint_with_original_cached_identity_and_same_public_state() {
    use neo_reductions::superneo_eval::SuperneoCachedRelationError;
    let started = std::time::Instant::now();
    let directory = TestDirectory::new();
    let reference = reference();
    let original = Circuit::compile(&reference, private_zero_application(F::ONE)).unwrap();
    let changed = Circuit::compile(&reference, private_zero_application(F::ZERO)).unwrap();
    drop(reference);
    assert_ne!(original.identity(), changed.identity());
    assert_eq!(
        original.compiled.package.application(),
        changed.compiled.package.application()
    );
    assert_eq!(
        original.compiled.package.row_count(),
        changed.compiled.package.row_count()
    );
    assert_eq!(
        original.compiled.package.logical_column_count(),
        changed.compiled.package.logical_column_count()
    );
    let initial = [2, 3, 5, 7].map(F::from_u64);
    assert!(matches!(
        original.compiled.application.execute(initial, &[F::ONE]),
        Err(ApplicationError::UnsatisfiedRow(0))
    ));
    assert_eq!(
        changed
            .compiled
            .application
            .execute(initial, &[F::ONE])
            .unwrap()
            .output_state(),
        initial
    );

    let original_path = directory.path("original.package");
    let changed_path = directory.path("changed.package");
    original.write(&original_path).unwrap();
    changed.write(&changed_path).unwrap();
    drop(changed);
    copy_cached_identity_components(&original_path, &changed_path);

    let prover = Prover::load(&changed_path, Engine::Metal, 114).unwrap();
    assert_eq!(prover.engine(), Engine::Metal);
    assert_eq!(
        prover.compiled.binding, original.compiled.binding,
        "all claimed bindings match"
    );
    let expected = Stage1State::new(1, initial, initial);
    let proof = prover.prove(initial, &[F::ONE]).unwrap();
    assert_eq!(proof.state(), &expected);
    drop(prover);
    eprintln!("changed-package Metal proof built elapsed={:?}", started.elapsed());

    let verifier = Verifier::from_package(&original, Engine::Metal, 114).unwrap();
    assert_eq!(verifier.engine(), Engine::Metal);
    match verifier.verify(&expected, &proof) {
        Err(Error::Verify(VerifyError::FreshRelation(SuperneoCachedRelationError::UnsatisfiedRow { row }))) => {
            eprintln!(
                "original-package Metal verifier rejected row={row} elapsed={:?}",
                started.elapsed()
            );
        }
        other => panic!("expected a false-constraint row after matching state and bindings, got {other:?}"),
    }
}
