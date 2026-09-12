//! Save and replay the actual selected C/R parent before D openings.
//! Original package and fixture inputs own replay. Saved witness values are
//! checked by recomputing their fixed-key commitment; file metadata is not authority.

#[path = "stage1_assemble.rs"]
pub mod assemble;

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::Path,
    time::Instant,
};

use neo_ajtai::{nightstream_fprime_setup::commit_production_signed_unit_matrix, Commitment};
use neo_ccs::{CeClaim, Mat, V1_1Evaluations};
use neo_fold_clean::{
    engine::transcript::{Poseidon2TranscriptSnapshot, Transcript},
    paper::{
        construction2::RunningInstance,
        params::Params,
        pi_ccs, pi_dec, pi_rlc,
        relations::{ajtai_dec_mixer, ajtai_rlc_mixer, CcsClaim, Structure},
    },
};
use neo_math::{D, F, K};
use neo_reductions::{
    common::{split_b_matrix_k_with_nonzero_flags, validate_superneo_witness_mat},
    superneo_eval::SuperneoZBlocks,
};
use p3_field::PrimeCharacteristicRing;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::stage1_actual::{self, ActualBase};

type Claim = CeClaim<Commitment, F, K>;

#[derive(Serialize, Deserialize)]
struct SavedParent {
    schema: u64,
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    verification_key_digest: [u64; 4],
    sumcheck: pi_ccs::SumcheckProof,
    ccs_outputs: Vec<Claim>,
    rlc_parent: Claim,
    transcript_state: [F; 8],
    transcript_absorbed: usize,
}

struct CommittedSplit {
    digits: Vec<Mat<F>>,
    nonzero: Vec<bool>,
    commitments: Vec<Commitment>,
}

#[derive(Serialize, Deserialize)]
struct SavedSplit {
    schema: u64,
    parent: Claim,
    nonzero: Vec<bool>,
    commitments: Vec<Commitment>,
}

#[derive(Serialize, Deserialize)]
struct SavedChildOpening {
    schema: u64,
    child: usize,
    parent: Claim,
    commitment: Commitment,
    opening: V1_1Evaluations<K>,
    transcript_state: [F; 8],
    transcript_absorbed: usize,
}

fn save<T: Serialize>(path: &Path, value: &T) {
    let file = OpenOptions::new()
        .write(true)
        .create_new(true)
        .open(path)
        .expect("fresh checkpoint file");
    let mut writer = BufWriter::new(file);
    serde_json::to_writer(&mut writer, value).expect("serialize existing checkpoint members");
    writer.write_all(b"\n").expect("checkpoint newline");
    writer.flush().expect("complete checkpoint file");
}

fn load<T: DeserializeOwned>(path: &Path) -> T {
    serde_json::from_reader(BufReader::new(File::open(path).expect("checkpoint file")))
        .expect("complete typed checkpoint members")
}

fn replay(
    params: &Params,
    structure: &Structure,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &pi_ccs::Proof,
    parent: &Claim,
) -> Poseidon2TranscriptSnapshot {
    let prior_parent = running
        .parent_authority
        .as_ref()
        .expect("original selected running parent");
    pi_dec::verify(
        params,
        structure,
        ajtai_dec_mixer,
        prior_parent,
        &pi_dec::Proof {
            children: running.claims.clone(),
        },
    )
    .expect("original running-parent authority");
    let mut transcript = Transcript::session();
    let outputs = pi_ccs::verify(
        &mut transcript,
        params,
        structure,
        std::slice::from_ref(fresh),
        running,
        proof,
    )
    .expect("saved C proof on original sources");
    let verified = pi_rlc::verify(
        &mut transcript,
        params,
        structure,
        ajtai_rlc_mixer,
        &outputs,
        &pi_rlc::Proof {
            combined: parent.clone(),
        },
    )
    .expect("saved R proof on actual C outputs");
    assert_eq!(&verified, parent);
    transcript.snapshot()
}

fn commit_parent(params: &Params, witness: &Mat<F>, expected: &Commitment) -> CommittedSplit {
    let (digits, nonzero) = split_b_matrix_k_with_nonzero_flags(witness, params.k_rho() as usize, params.b())
        .expect("actual bounded R witness decomposition");
    let commitments = digits
        .iter()
        .map(commit_production_signed_unit_matrix)
        .collect::<Result<Vec<_>, _>>()
        .expect("fixed-key commitment of every actual digit");
    assert_eq!(
        ajtai_dec_mixer(&commitments, params.b()),
        *expected,
        "saved R witness opens the transcript-verified R commitment"
    );
    CommittedSplit {
        digits,
        nonzero,
        commitments,
    }
}

/// Produce C/R from the original selected witness. No expected proof or
/// opening data is supplied. The parent commitment check is a separate stage.
pub fn prove(package_path: &Path, fixture_path: &Path, output: &Path) {
    assert!(!output.exists(), "fresh parent checkpoint directory");
    let started = Instant::now();
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load(package_path, &fs::read(fixture_path).expect("original base fixture"));
    let fresh_claim = fresh.claim.clone();
    let prior = running.claims_only();
    let (proof, parent) = package
        .prove_parent(vec![fresh], running)
        .expect("shared normal C/R prover");
    let position = replay(
        &params,
        package.structure(),
        &fresh_claim,
        &prior,
        &proof,
        &parent.claim,
    );
    let record = SavedParent {
        schema: 1,
        structural_identifier: package.structural_identifier(),
        package_identity: package.package_identity(),
        verification_key_digest: package.verification_key_digest(),
        sumcheck: proof.sumcheck,
        ccs_outputs: proof.outputs,
        rlc_parent: parent.claim,
        transcript_state: position.state(),
        transcript_absorbed: position.absorbed(),
    };
    fs::create_dir(output).expect("fresh parent checkpoint directory");
    save(&output.join("parent.json"), &record);
    save(&output.join("parent-witness.json"), &parent.witness);
    println!(
        "actual_selected_parent=saved commitment_check=pending output={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
}

fn load_verified_parent(package_path: &Path, fixture_path: &Path, input: &Path) -> (ActualBase, SavedParent) {
    let saved: SavedParent = load(&input.join("parent.json"));
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load(package_path, &fs::read(fixture_path).expect("original base fixture"));
    assert_eq!(saved.schema, 1);
    assert_eq!(saved.structural_identifier, package.structural_identifier());
    assert_eq!(saved.package_identity, package.package_identity());
    assert_eq!(saved.verification_key_digest, package.verification_key_digest());
    let proof = pi_ccs::Proof {
        sumcheck: saved.sumcheck.clone(),
        outputs: saved.ccs_outputs.clone(),
    };
    let position = replay(
        &params,
        package.structure(),
        &fresh.claim,
        &running,
        &proof,
        &saved.rlc_parent,
    );
    assert_eq!(position.state(), saved.transcript_state, "replayed C/R state");
    assert_eq!(
        position.absorbed(),
        saved.transcript_absorbed,
        "replayed C/R sponge position"
    );
    (
        ActualBase {
            package,
            params,
            fresh,
            running,
        },
        saved,
    )
}

/// Recreate the original sources, replay the saved C/R proofs and recompute
/// the saved witness commitment. Persist the computed split and commitments
/// for the later D owner; no child claims or openings are produced here.
pub fn check(package_path: &Path, fixture_path: &Path, input: &Path, material: &Path) {
    assert!(!material.exists(), "fresh checked parent material directory");
    let started = Instant::now();
    let (actual, saved) = load_verified_parent(package_path, fixture_path, input);
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = actual;
    drop((fresh, running));
    let witness: Mat<F> = load(&input.join("parent-witness.json"));
    validate_superneo_witness_mat(&witness, package.structure().m).expect("selected R witness carrier");
    let split = commit_parent(&params, &witness, &saved.rlc_parent.c);
    drop(witness);
    fs::create_dir(material).expect("fresh checked parent material directory");
    for (index, digit) in split.digits.iter().enumerate() {
        save(&material.join(format!("digit-{index}.json")), digit);
    }
    save(
        &material.join("split.json"),
        &SavedSplit {
            schema: 1,
            parent: saved.rlc_parent.clone(),
            nonzero: split.nonzero.clone(),
            commitments: split.commitments.clone(),
        },
    );
    println!(
        "actual_selected_parent=checked transcript=replayed witness_commitment=recomputed material={} elapsed={:?}",
        material.display(),
        started.elapsed()
    );
}

fn check_digit(params: &Params, parent: &Mat<F>, loaded: &Mat<F>, child: usize) -> bool {
    let (expected, flags) = split_b_matrix_k_with_nonzero_flags(parent, params.k_rho() as usize, params.b())
        .expect("canonical split of the saved R witness");
    assert!(child < expected.len(), "selected D child index");
    assert!(
        loaded == &expected[child],
        "saved digit equals the canonical parent split"
    );
    flags[child]
}

/// Compute one actual D opening with the normal selected evaluator. The
/// saved digit is checked against a freshly computed canonical split, and
/// its commitment is recomputed before the evaluator sees it. The output
/// is proof data; later assembly must run the normal D and NIFS verifiers.
pub fn open_child(
    package_path: &Path,
    fixture_path: &Path,
    input: &Path,
    material: &Path,
    child: usize,
    output: &Path,
) {
    assert!(!output.exists(), "fresh child opening file");
    let started = Instant::now();
    let (actual, saved) = load_verified_parent(package_path, fixture_path, input);
    let ActualBase {
        package,
        params,
        fresh,
        running,
    } = actual;
    drop((fresh, running));
    println!(
        "actual_selected_child={child} original_sources_replayed_elapsed={:?}",
        started.elapsed()
    );
    let split: SavedSplit = load(&material.join("split.json"));
    assert_eq!(split.schema, 1);
    assert_eq!(split.parent, saved.rlc_parent, "same replayed R parent");
    assert_eq!(split.commitments.len(), params.k_rho() as usize);
    assert_eq!(split.nonzero.len(), params.k_rho() as usize);
    assert!(child < params.k_rho() as usize, "selected D child index");
    assert_eq!(
        ajtai_dec_mixer(&split.commitments, params.b()),
        saved.rlc_parent.c,
        "ordered saved commitments recompose to the replayed R parent"
    );
    let parent: Mat<F> = load(&input.join("parent-witness.json"));
    let digit: Mat<F> = load(&material.join(format!("digit-{child}.json")));
    validate_superneo_witness_mat(&parent, package.structure().m).expect("selected R witness carrier");
    validate_superneo_witness_mat(&digit, package.structure().m).expect("selected D digit carrier");
    let nonzero = check_digit(&params, &parent, &digit, child);
    assert_eq!(split.nonzero[child], nonzero, "recomputed child activity");
    drop(parent);
    println!(
        "actual_selected_child={child} parent_loaded_split_compared_elapsed={:?}",
        started.elapsed()
    );
    let commitment = commit_production_signed_unit_matrix(&digit).expect("actual saved child commitment");
    assert_eq!(commitment, split.commitments[child], "same ordered child commitment");
    println!(
        "actual_selected_child={child} commitment_recomputed_elapsed={:?}",
        started.elapsed()
    );
    let opening = if nonzero {
        let cache = package
            .build_superneo_cache()
            .expect("owner-built selected matrix rows");
        println!(
            "actual_selected_child={child} cache_built_elapsed={:?}",
            started.elapsed()
        );
        let blocks = SuperneoZBlocks::from_witness_mat(&digit, package.structure().m).expect("actual child block view");
        let mut openings = cache
            .eval_real_v1_1_openings(&saved.rlc_parent.r, std::slice::from_ref(&blocks))
            .expect("normal selected D child opening evaluator");
        assert_eq!(openings.len(), 1);
        let opening = openings.pop().unwrap();
        println!(
            "actual_selected_child={child} opening_evaluated_elapsed={:?}",
            started.elapsed()
        );
        opening
    } else {
        // Same inactive-digit result as optimized_engine::common::witness_openings.
        // The exact split above establishes inactivity; the saved flag is not authority.
        println!(
            "actual_selected_child={child} zero_opening_from_checked_split_elapsed={:?}",
            started.elapsed()
        );
        V1_1Evaluations {
            eval_k: vec![K::ZERO; D],
            eval_a: vec![vec![K::ZERO; D]; package.structure().t()],
        }
    };
    drop(digit);
    save(
        output,
        &SavedChildOpening {
            schema: 1,
            child,
            parent: saved.rlc_parent,
            commitment,
            opening,
            transcript_state: saved.transcript_state,
            transcript_absorbed: saved.transcript_absorbed,
        },
    );
    println!(
        "actual_selected_child={child} opening=computed output={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
}

#[test]
fn saved_digit_must_equal_canonical_parent_split() {
    let params = Params::production();
    let mut values = vec![F::ZERO; D];
    values[0] = F::from_u64(5);
    let parent = Mat::from_row_major(D, 1, values);
    let (digits, _) = split_b_matrix_k_with_nonzero_flags(&parent, params.k_rho() as usize, params.b()).unwrap();
    assert!(check_digit(&params, &parent, &digits[0], 0));
    assert!(!check_digit(&params, &parent, &digits[1], 1));
    assert!(check_digit(&params, &parent, &digits[2], 2));
    assert!(std::panic::catch_unwind(|| check_digit(&params, &parent, &digits[1], 0)).is_err());
    assert!(std::panic::catch_unwind(|| check_digit(&params, &parent, &digits[0], 1)).is_err());
}

#[test]
fn saved_witness_must_open_parent_commitment() {
    use neo_ajtai::nightstream_fprime_setup::PRODUCTION_MESSAGE_COLUMNS;

    let columns = PRODUCTION_MESSAGE_COLUMNS as usize;
    let mut positive = vec![0u64; columns];
    let negative = vec![0u64; columns];
    positive[0] = 1;
    let original = Mat::<F>::compact_signed_unit_from_column_masks(D, columns, &positive, &negative).unwrap();
    let expected = commit_production_signed_unit_matrix(&original).unwrap();
    drop((original, positive, negative));
    let altered = Mat::virtual_constant(D, columns, F::ZERO);
    let params = Params::production();
    let zero = Commitment::zeros(D, params.inner().kappa as usize);
    assert_ne!(expected, zero, "the original saved witness has a nonzero commitment");
    let checked = commit_parent(&params, &altered, &zero);
    assert!(checked.nonzero.iter().all(|flag| !flag));
    assert_eq!(checked.digits.len(), params.k_rho() as usize);
    assert!(std::panic::catch_unwind(|| commit_parent(&params, &altered, &expected)).is_err());
}
