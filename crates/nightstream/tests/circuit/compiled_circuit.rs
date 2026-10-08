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
        .as_ref()
        .unwrap()
        .execute(input, &private)
        .unwrap();
    let actual_witness = actual
        .compiled
        .application
        .as_ref()
        .unwrap()
        .execute(input, &private)
        .unwrap();
    assert_eq!(expected_witness.output_state(), [169, 4, 5, 17].map(F::from_u64));
    assert_eq!(actual_witness.values(), expected_witness.values());
    assert_eq!(actual_witness.output_state(), expected_witness.output_state());
    assert_eq!(
        actual
            .compiled
            .application
            .as_ref()
            .unwrap()
            .prepared_output_forms()
            .unwrap(),
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
fn compiled_load_rejects_identity_unbound_output_form_substitution() {
    let original = compiled_fixture();
    let directory = TestDirectory::new();
    let path = directory.path("changed-output-form.package");
    original.write(&path).unwrap();

    // Lane zero is a variable output and therefore uses C=2. A=0 is also a
    // recognized tag, but it reads the wrong side of this saved equality row.
    let mut file = OpenOptions::new().write(true).open(&path).unwrap();
    file.seek(SeekFrom::Start(8)).unwrap();
    file.write_all(&[0]).unwrap();
    drop(file);

    let loaded = Circuit::load(&path);
    if let Ok(changed) = &loaded {
        assert_eq!(changed.identity(), original.identity());
        assert_eq!(changed.compiled.binding, original.compiled.binding);
        assert_eq!(
            changed
                .compiled
                .application
                .as_ref()
                .unwrap()
                .prepared_output_forms()
                .unwrap(),
            [0, 0, 2, 0]
        );

        let input = [2, 3, 5, 7].map(F::from_u64);
        let private = [11, 13].map(F::from_u64);
        assert!(original
            .compiled
            .application
            .as_ref()
            .unwrap()
            .execute(input, &private)
            .is_ok());
        assert!(changed
            .compiled
            .application
            .as_ref()
            .unwrap()
            .execute(input, &private)
            .is_err());
    }
    assert!(
        loaded.is_err(),
        "an output-form change outside the circuit identity must not load"
    );
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
        original
            .compiled
            .application
            .as_ref()
            .unwrap()
            .execute(initial, &[F::ONE]),
        Err(ApplicationError::UnsatisfiedRow(0))
    ));
    assert_eq!(
        changed
            .compiled
            .application
            .as_ref()
            .unwrap()
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

/// A well-shaped active proof with signed-unit witnesses. It does not verify.
fn shaped_proof(prover: &Prover) -> Stage1Envelope {
    use crate::folding::{CcsClaim, CcsInstance, CcsWitness, CeClaim, RunningInstance};
    use neo_math::{D, K};

    let structure = prover.lifecycle.structure();
    let public_width = prover.compiled.package.logical_public_input_count();
    let columns = structure.m.div_ceil(D);
    let extension = |value: u64| neo_math::from_complex(F::from_u64(value), F::from_u64(value + 1));
    let evaluations = |seed: u64| {
        let mut values: Vec<K> = (0..D as u64).map(|lane| extension(seed + lane)).collect();
        values.resize(D.next_power_of_two(), K::ZERO);
        values
    };
    let commitment = |seed: u64| neo_ajtai::Commitment {
        d: D,
        kappa: 22,
        data: (0..D as u64 * 22)
            .map(|word| F::from_u64(seed + word))
            .collect(),
    };
    let claim = |seed: u64| CeClaim {
        c: commitment(seed),
        X: neo_ccs::Mat::from_row_major(
            D,
            public_width / D,
            (0..public_width as u64)
                .map(|word| F::from_u64(seed + word))
                .collect(),
        ),
        r: (0..28).map(|round| extension(seed + round)).collect(),
        eval_k: evaluations(seed),
        eval_a: (0..structure.t() as u64)
            .map(|matrix| evaluations(seed + matrix))
            .collect(),
        m_in: public_width,
        fold_digest: [seed as u8; 32],
        adv: None,
    };
    let witness = |seed: usize| {
        let mut positive = vec![0; columns];
        let mut negative = vec![0; columns];
        positive[seed % columns] = 0b101;
        negative[(seed * 7) % columns] |= 0b10;
        neo_ccs::Mat::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap()
    };
    let running = RunningInstance::new(
        (0..16).map(claim).collect(),
        (0..16).map(witness).collect(),
        Some(claim(99)),
    );
    let fresh = CcsInstance {
        claim: CcsClaim {
            c: commitment(17),
            x: (0..public_width as u64).map(F::from_u64).collect(),
            m_in: public_width,
            adv: None,
        },
        witness: CcsWitness {
            w: Vec::new(),
            Z: witness(17),
        },
    };
    Stage1Envelope::from_parts(Stage1State::new(3, [F::ONE; 4], [F::TWO; 4]), running, fresh)
}

/// Offsets computed independently of the codec: magic and kind, the state,
/// then one running claim and one witness column.
fn proof_offsets(prover: &Prover) -> (usize, usize, usize) {
    let structure = prover.lifecycle.structure();
    let public_width = prover.compiled.package.logical_public_input_count();
    let d = neo_math::D;
    let header = 16 + 8 + 9 * 8;
    let claim = (d * 22 + public_width + 2 * 28 + (structure.t() + 1) * 2 * d) * 8 + 32;
    let witness = structure.m.div_ceil(d) * 16;
    (header, claim, witness)
}

#[test]
fn proof_bytes_round_trip_exactly() {
    let prover = compiled_fixture().prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(compiled_fixture(), Engine::Optimized, 114).unwrap();
    let public_width = prover.compiled.package.logical_public_input_count();
    let (header, claim, witness) = proof_offsets(&prover);

    let proof = shaped_proof(&prover);
    let bytes = prover.encode_proof(&proof).unwrap();
    assert_eq!(
        bytes.len(),
        header + 17 * claim + 16 * witness + (neo_math::D * 22 + public_width) * 8 + witness
    );
    let decoded = verifier.decode_proof(&bytes).unwrap();
    assert_eq!(decoded.state(), proof.state());
    assert_eq!(decoded.running(), proof.running());
    let (left, right) = (decoded.fresh().unwrap(), proof.fresh().unwrap());
    assert_eq!(left.claim.c, right.claim.c);
    assert_eq!(left.claim.x, right.claim.x);
    assert_eq!(left.witness.Z, right.witness.Z);
    assert_eq!(prover.encode_proof(&decoded).unwrap(), bytes);

    let initial = Stage1Envelope::initial([F::ONE, F::TWO, F::ZERO, F::ONE]);
    let bytes = prover.encode_proof(&initial).unwrap();
    assert_eq!(bytes.len(), 16 + 8 + 4 * 8);
    let decoded = verifier.decode_proof(&bytes).unwrap();
    assert!(decoded.is_initial());
    assert_eq!(decoded.state(), initial.state());
}

#[test]
fn proof_decoder_rejects_resized_and_noncanonical_bytes() {
    let prover = compiled_fixture().prover(Engine::Optimized, 114).unwrap();
    let verifier = Verifier::from_package(compiled_fixture(), Engine::Optimized, 114).unwrap();
    let (header, claim, _) = proof_offsets(&prover);
    let mut bytes = prover.encode_proof(&shaped_proof(&prover)).unwrap();
    let rejects = |bytes: &[u8]| matches!(verifier.decode_proof(bytes), Err(Error::ProofBytes(_)));

    assert!(rejects(&bytes[..bytes.len() - 1]), "truncated proof");
    let mut extended = bytes.clone();
    extended.push(0);
    assert!(rejects(&extended), "trailing data");
    drop(extended);

    let mut mutate = |offset: usize, value: u64, reason: &str| {
        let original: [u8; 8] = bytes[offset..offset + 8].try_into().unwrap();
        bytes[offset..offset + 8].copy_from_slice(&value.to_le_bytes());
        assert!(rejects(&bytes), "{reason}");
        bytes[offset..offset + 8].copy_from_slice(&original);
    };
    mutate(0, u64::from_le_bytes(*b"NS-OTHER"), "format tag");
    mutate(16, 2, "unknown proof kind");
    mutate(header, 0xffff_ffff_0000_0001, "noncanonical field word");
    let first_column = header + 17 * claim;
    mutate(first_column, 1 << 60, "mask bit outside the 54 lanes");
    mutate(first_column + 8, 0b101, "overlapping masks");
    assert!(verifier.decode_proof(&bytes).is_ok(), "restored bytes");
}

#[test]
fn proof_encoder_rejects_values_outside_the_selected_shape() {
    let prover = compiled_fixture().prover(Engine::Optimized, 114).unwrap();
    let parts = |proof: Stage1Envelope| {
        let running = proof.running().unwrap().clone();
        let fresh = proof.fresh().unwrap().clone();
        (*proof.state(), running, fresh)
    };
    let rejects = |state, running, fresh| {
        matches!(
            prover.encode_proof(&Stage1Envelope::from_parts(state, running, fresh)),
            Err(Error::ProofBytes(_))
        )
    };

    let (state, mut running, fresh) = parts(shaped_proof(&prover));
    let lane = running.claims[0].c.clone();
    running.claims[0].adv = Some(neo_ccs::LaneCommitments {
        ops: lane.clone(),
        is: lane.clone(),
        fs: lane,
    });
    assert!(rejects(state, running, fresh), "auxiliary lane commitments");

    let (state, mut running, fresh) = parts(shaped_proof(&prover));
    running.claims[0].eval_k[neo_math::D] = neo_math::K::ONE;
    assert!(rejects(state, running, fresh), "nonzero evaluation surplus");

    let (state, running, mut fresh) = parts(shaped_proof(&prover));
    let columns = fresh.witness.Z.cols();
    fresh.witness.Z = neo_ccs::Mat::virtual_constant(neo_math::D, columns, F::TWO);
    assert!(rejects(state, running, fresh), "witness value outside the signed units");

    let (state, running, mut fresh) = parts(shaped_proof(&prover));
    fresh.witness.w = vec![F::ONE];
    assert!(rejects(state, running, fresh), "redundant private witness copy");
}
