//! Save and replay the actual selected C/R parent before D openings.
//! Original package and source inputs own replay. Saved witness values are
//! checked by recomputing their fixed-key commitment; file metadata is not authority.

#[path = "stage1_assemble.rs"]
pub mod assemble;

use std::{
    fs::{self, File, OpenOptions},
    io::{BufReader, BufWriter, Write},
    path::{Path, PathBuf},
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
    Poseidon2HashChainV1Package,
};
use neo_math::{D, F, K};
use neo_reductions::{
    common::{split_b_matrix_k_with_nonzero_flags, validate_superneo_witness_mat},
    optimized_engine::optimized_verify_with_trace,
    superneo_eval::SuperneoZBlocks,
};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use serde::{de::DeserializeOwned, Deserialize, Serialize};

use super::stage1_actual::{self, ActualSources};
use super::stage1_values::{ccs_input, ccs_phase};

type Claim = CeClaim<Commitment, F, K>;

#[derive(Serialize, Deserialize)]
pub(super) struct SavedCcs {
    pub(super) schema: u64,
    pub(super) structural_identifier: [u64; 4],
    pub(super) package_identity: [u64; 4],
    pub(super) verification_key_digest: [u64; 4],
    pub(super) sumcheck: pi_ccs::SumcheckProof,
    pub(super) outputs: Vec<Claim>,
}

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

pub(super) fn replay_ccs(
    params: &Params,
    structure: &Structure,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &pi_ccs::Proof,
) -> (Transcript, Vec<Claim>) {
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
    (transcript, outputs)
}

fn replay(
    params: &Params,
    structure: &Structure,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &pi_ccs::Proof,
    parent: &Claim,
) -> Poseidon2TranscriptSnapshot {
    let (mut transcript, outputs) = replay_ccs(params, structure, fresh, running, proof);
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

fn save_parent(
    output: &Path,
    package: &Poseidon2HashChainV1Package,
    proof: pi_ccs::Proof,
    parent: pi_rlc::Output,
    position: Poseidon2TranscriptSnapshot,
) {
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
pub fn prove(package_path: &Path, source_path: &Path, output: &Path) {
    assert!(!output.exists(), "fresh parent checkpoint directory");
    let started = Instant::now();
    let ActualSources {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load_path(package_path, source_path);
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
    save_parent(output, &package, proof, parent, position);
    println!(
        "actual_selected_parent=saved commitment_check=pending output={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
}

/// Execute only C on the original source witnesses. Complete source openings
/// are a separate required stage; native C acceptance checks the proof against
/// the original public claims. The same proof is replayed before R.
pub fn prove_ccs(package_path: &Path, source_path: &Path, output: &Path) {
    let input_output = output.with_extension("input.json");
    let phase_output = output.with_extension("phase.json");
    assert!(
        [output, &input_output, &phase_output]
            .into_iter()
            .all(|path| !path.exists()),
        "fresh C proof and conformance files"
    );
    let started = Instant::now();
    let ActualSources {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load_path(package_path, source_path);
    println!("actual_selected_ccs_sources_loaded_elapsed={:?}", started.elapsed());
    let proof = package
        .prove_pi_ccs(
            std::slice::from_ref(&fresh.claim),
            std::slice::from_ref(&fresh.witness),
            &running.claims,
            &running.witnesses,
        )
        .expect("selected C prover on original complete witnesses");
    println!("actual_selected_ccs_proved_elapsed={:?}", started.elapsed());
    let (verified_transcript, _) = replay_ccs(&params, package.structure(), &fresh.claim, &running, &proof);
    let mut transcript = Poseidon2Transcript::from_state_and_absorbed([F::ZERO; 8], 0);
    let (valid, trace) = optimized_verify_with_trace(
        &mut transcript,
        params.inner(),
        package.structure(),
        std::slice::from_ref(&fresh.claim),
        &running.claims,
        &proof.outputs,
        &proof.sumcheck,
    )
    .expect("C trace on the original claims");
    assert!(valid);
    assert_eq!(verified_transcript.snapshot().state(), trace.outgoing_state);
    assert!(proof
        .outputs
        .iter()
        .all(|claim| claim.r == trace.round_challenges));
    save(&input_output, &ccs_input(&fresh.claim, &running.claims, &proof));
    save(&phase_output, &ccs_phase(&proof, &trace, valid));
    println!("actual_selected_ccs_verified_elapsed={:?}", started.elapsed());
    let record = SavedCcs {
        schema: 1,
        structural_identifier: package.structural_identifier(),
        package_identity: package.package_identity(),
        verification_key_digest: package.verification_key_digest(),
        sumcheck: proof.sumcheck,
        outputs: proof.outputs,
    };
    save(output, &record);
    println!(
        "actual_selected_ccs=saved output={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
}

/// Replay C on the same original sources, then compute R from their
/// original complete matrices. The public wrapper borrows these moved Mats
/// and calls the normal prove_refs implementation without cloning them.
pub fn prove_after_ccs(package_path: &Path, source_path: &Path, ccs_file: &Path, output: &Path) {
    assert!(!output.exists(), "fresh parent checkpoint directory");
    let started = Instant::now();
    let saved: SavedCcs = load(ccs_file);
    let ActualSources {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load_path(package_path, source_path);
    println!("actual_selected_rlc_sources_loaded_elapsed={:?}", started.elapsed());
    assert_eq!(saved.schema, 1);
    assert_eq!(saved.structural_identifier, package.structural_identifier());
    assert_eq!(saved.package_identity, package.package_identity());
    assert_eq!(saved.verification_key_digest, package.verification_key_digest());
    let proof = pi_ccs::Proof {
        sumcheck: saved.sumcheck,
        outputs: saved.outputs,
    };
    let (mut transcript, outputs) = replay_ccs(&params, package.structure(), &fresh.claim, &running, &proof);
    println!("actual_selected_rlc_ccs_replayed_elapsed={:?}", started.elapsed());
    let prior = running.claims_only();
    let fresh_claim = fresh.claim;
    let mut witnesses = Vec::with_capacity(1 + running.witnesses.len());
    witnesses.push(fresh.witness.Z);
    witnesses.extend(running.witnesses);
    let (parent, _) = pi_rlc::prove(
        &mut transcript,
        &params,
        package.structure(),
        ajtai_rlc_mixer,
        &outputs,
        &witnesses,
    )
    .expect("normal R prover on original C source matrices");
    drop((witnesses, outputs));
    println!("actual_selected_rlc_proved_elapsed={:?}", started.elapsed());
    let position = replay(
        &params,
        package.structure(),
        &fresh_claim,
        &prior,
        &proof,
        &parent.claim,
    );
    assert_eq!(
        position,
        transcript.snapshot(),
        "C/R producer and verifier positions agree"
    );
    println!("actual_selected_rlc_verified_elapsed={:?}", started.elapsed());
    save_parent(output, &package, proof, parent, position);
    println!(
        "actual_selected_parent=saved from_saved_ccs=true commitment_check=pending output={} elapsed={:?}",
        output.display(),
        started.elapsed()
    );
}

fn load_verified_parent(package_path: &Path, source_path: &Path, input: &Path) -> (ActualSources, SavedParent) {
    let saved: SavedParent = load(&input.join("parent.json"));
    let ActualSources {
        package,
        params,
        fresh,
        running,
    } = stage1_actual::load_path(package_path, source_path);
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
        ActualSources {
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
pub fn check(package_path: &Path, source_path: &Path, input: &Path, material: &Path) {
    assert!(!material.exists(), "fresh checked parent material directory");
    let started = Instant::now();
    let (actual, saved) = load_verified_parent(package_path, source_path, input);
    let ActualSources {
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

fn check_digit(expected: &[Mat<F>], flags: &[bool], loaded: &Mat<F>, child: usize) -> bool {
    assert!(child < expected.len(), "selected D child index");
    assert!(
        loaded == &expected[child],
        "saved digit equals the canonical parent split"
    );
    flags[child]
}

fn validate_children(children: &[usize], count: usize) {
    assert!(!children.is_empty(), "nonempty D child batch");
    for (position, &child) in children.iter().enumerate() {
        assert!(child < count, "selected D child index");
        assert!(!children[..position].contains(&child), "distinct D child indices");
    }
}

/// Compute one actual D opening through the same checked batch core.
pub fn open_child(package_path: &Path, source_path: &Path, input: &Path, material: &Path, child: usize, output: &Path) {
    open_child_batch(
        package_path,
        source_path,
        input,
        material,
        &[(child, output.to_path_buf())],
    );
}

/// Replay the authoritative sources once and share the canonical split and
/// matrix cache across the requested children. Saved digits and commitments
/// are checked before the normal evaluator receives any witness blocks.
pub fn open_children(
    package_path: &Path,
    source_path: &Path,
    input: &Path,
    material: &Path,
    children: &[usize],
    output: &Path,
) {
    assert!(!output.exists(), "fresh child opening directory");
    let outputs = children
        .iter()
        .map(|&child| (child, output.join(format!("child-{child}.json"))))
        .collect::<Vec<_>>();
    fs::create_dir(output).expect("fresh child opening directory");
    open_child_batch(package_path, source_path, input, material, &outputs);
}

fn open_child_batch(
    package_path: &Path,
    source_path: &Path,
    input: &Path,
    material: &Path,
    outputs: &[(usize, PathBuf)],
) {
    assert!(
        outputs.iter().all(|(_, output)| !output.exists()),
        "fresh child opening files"
    );
    let started = Instant::now();
    let (actual, saved) = load_verified_parent(package_path, source_path, input);
    let ActualSources {
        package,
        params,
        fresh,
        running,
    } = actual;
    drop((fresh, running));
    let children = outputs.iter().map(|(child, _)| *child).collect::<Vec<_>>();
    validate_children(&children, params.k_rho() as usize);
    println!(
        "actual_selected_children={children:?} original_sources_replayed_elapsed={:?}",
        started.elapsed()
    );
    let split: SavedSplit = load(&material.join("split.json"));
    assert_eq!(split.schema, 1);
    assert_eq!(split.parent, saved.rlc_parent, "same replayed R parent");
    assert_eq!(split.commitments.len(), params.k_rho() as usize);
    assert_eq!(split.nonzero.len(), params.k_rho() as usize);
    assert_eq!(
        ajtai_dec_mixer(&split.commitments, params.b()),
        saved.rlc_parent.c,
        "ordered saved commitments recompose to the replayed R parent"
    );
    let parent: Mat<F> = load(&input.join("parent-witness.json"));
    validate_superneo_witness_mat(&parent, package.structure().m).expect("selected R witness carrier");
    let (expected, flags) = split_b_matrix_k_with_nonzero_flags(&parent, params.k_rho() as usize, params.b())
        .expect("canonical split of the saved R witness");
    drop(parent);
    println!(
        "actual_selected_children={children:?} parent_split_elapsed={:?}",
        started.elapsed()
    );
    let mut blocks = Vec::with_capacity(children.len());
    let mut commitments = Vec::with_capacity(children.len());
    for &child in &children {
        let digit: Mat<F> = load(&material.join(format!("digit-{child}.json")));
        validate_superneo_witness_mat(&digit, package.structure().m).expect("selected D digit carrier");
        let nonzero = check_digit(&expected, &flags, &digit, child);
        assert_eq!(split.nonzero[child], nonzero, "recomputed child activity");
        let commitment = commit_production_signed_unit_matrix(&digit).expect("actual saved child commitment");
        assert_eq!(commitment, split.commitments[child], "same ordered child commitment");
        blocks.push(SuperneoZBlocks::from_witness_mat(&digit, package.structure().m).expect("actual child block view"));
        commitments.push(commitment);
        println!(
            "actual_selected_child={child} digit_checked_commitment_recomputed_elapsed={:?}",
            started.elapsed()
        );
    }
    drop(expected);
    let openings = if children.iter().any(|&child| flags[child]) {
        let cache = package
            .build_superneo_cache()
            .expect("owner-built selected matrix rows");
        println!(
            "actual_selected_children={children:?} cache_built_elapsed={:?}",
            started.elapsed()
        );
        let openings = cache
            .eval_real_v1_1_openings(&saved.rlc_parent.r, &blocks)
            .expect("normal selected D child opening evaluator");
        println!(
            "actual_selected_children={children:?} openings_evaluated_elapsed={:?}",
            started.elapsed()
        );
        openings
    } else {
        // Inactivity follows from the recomputed split and exact digit equality.
        // This is the same inactive-digit result as the normal D evaluator.
        children
            .iter()
            .map(|_| V1_1Evaluations {
                eval_k: vec![K::ZERO; D],
                eval_a: vec![vec![K::ZERO; D]; package.structure().t()],
            })
            .collect()
    };
    assert_eq!(openings.len(), children.len());
    drop(blocks);
    for (((child, output), commitment), opening) in outputs.iter().zip(commitments).zip(openings) {
        save(
            output,
            &SavedChildOpening {
                schema: 1,
                child: *child,
                parent: saved.rlc_parent.clone(),
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
}

#[cfg(test)]
#[path = "stage1_parent_tests.rs"]
mod tests;
