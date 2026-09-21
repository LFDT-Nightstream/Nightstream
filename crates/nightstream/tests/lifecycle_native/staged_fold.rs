//! Native C/R/D phases with replayed public inputs and recomputed split material.
use super::*;
use neo_reductions::{
    common::{split_b_matrix_k_with_nonzero_flags, validate_superneo_witness_mat},
    superneo_eval::SuperneoZBlocks,
};
use std::io::{self, BufRead};

#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedCcs {
    schema: u64,
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    verification_key_digest: [u64; 4],
    sumcheck: pi_ccs::SumcheckProof,
    outputs: Vec<CeClaim>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedParent {
    schema: u64,
    structural_identifier: [u64; 4],
    package_identity: [u64; 4],
    verification_key_digest: [u64; 4],
    sumcheck: pi_ccs::SumcheckProof,
    ccs_outputs: Vec<CeClaim>,
    pub(super) rlc_parent: CeClaim,
    transcript_state: [F; 8],
    transcript_absorbed: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedSplit {
    schema: u64,
    parent: CeClaim,
    nonzero: Vec<bool>,
    commitments: Vec<Commitment>,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
struct SavedChildOpening {
    schema: u64,
    child: usize,
    parent: CeClaim,
    commitment: Commitment,
    opening: V1_1Evaluations<K>,
    transcript_state: [F; 8],
    transcript_absorbed: usize,
}
#[derive(Serialize, Deserialize)]
#[serde(deny_unknown_fields)]
pub(super) struct SavedNifs {
    parent: SavedParent,
    children: Vec<CeClaim>,
}
impl SavedNifs {
    pub(super) fn proof(&self) -> NifsProof {
        NifsProof {
            pi_ccs: pi_ccs::Proof {
                sumcheck: self.parent.sumcheck.clone(),
                outputs: self.parent.ccs_outputs.clone(),
            },
            pi_rlc: pi_rlc::Proof {
                combined: self.parent.rlc_parent.clone(),
            },
            pi_dec: pi_dec::Proof {
                children: self.children.clone(),
            },
        }
    }
    pub(super) fn verify(
        &self,
        package: &PreparedLifecycle,
        source: &Path,
        step: u64,
    ) -> (Stage1State, CcsClaim, RunningInstance, NifsProof) {
        check_identity(
            package,
            self.parent.schema,
            self.parent.structural_identifier,
            self.parent.package_identity,
            self.parent.verification_key_digest,
        );
        let (state, fresh, running) = load_claims(package, source, step);
        let proof = self.proof();
        let mut transcript = Transcript::session();
        folding::verify(
            &mut transcript,
            &params(package),
            &package.structure,
            ajtai_rlc_mixer,
            ajtai_dec_mixer,
            std::slice::from_ref(&fresh),
            &running,
            &proof,
        )
        .unwrap();
        assert_eq!(transcript.snapshot().state(), self.parent.transcript_state);
        assert_eq!(transcript.snapshot().absorbed(), self.parent.transcript_absorbed);
        (state, fresh, running, proof)
    }
}
fn check_identity(package: &PreparedLifecycle, schema: u64, structural: [u64; 4], identity: [u64; 4], key: [u64; 4]) {
    assert_eq!(schema, 1);
    assert_eq!(structural, package.package.structural_identifier());
    assert_eq!(identity, package.package_identity());
    assert_eq!(key, package.binding.verification_key_digest());
}
fn replay_ccs(
    package: &PreparedLifecycle,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &pi_ccs::Proof,
) -> (Transcript, Vec<CeClaim>) {
    let params = params(package);
    folding::validate_running_parent_authority(&params, &package.structure, ajtai_dec_mixer, running).unwrap();
    let mut transcript = Transcript::session();
    let outputs = pi_ccs::verify(
        &mut transcript,
        &params,
        &package.structure,
        std::slice::from_ref(fresh),
        running,
        proof,
    )
    .unwrap();
    (transcript, outputs)
}
fn read_parent(package: &PreparedLifecycle, root: &Path, step: u64) -> SavedParent {
    let parent: SavedParent = load(&fold_dir(root, step).join("parent.json"));
    check_identity(
        package,
        parent.schema,
        parent.structural_identifier,
        parent.package_identity,
        parent.verification_key_digest,
    );
    let (_, fresh, running) = load_claims(package, &step_dir(root, step), step);
    let ccs = pi_ccs::Proof {
        sumcheck: parent.sumcheck.clone(),
        outputs: parent.ccs_outputs.clone(),
    };
    let (mut transcript, outputs) = replay_ccs(package, &fresh, &running, &ccs);
    pi_rlc::verify(
        &mut transcript,
        &params(package),
        &package.structure,
        ajtai_rlc_mixer,
        &outputs,
        &pi_rlc::Proof {
            combined: parent.rlc_parent.clone(),
        },
    )
    .unwrap();
    assert_eq!(
        transcript.snapshot().state(),
        parent.transcript_state,
        "replayed C/R state"
    );
    assert_eq!(
        transcript.snapshot().absorbed(),
        parent.transcript_absorbed,
        "replayed C/R cursor"
    );
    parent
}
fn committed_split(
    package: &PreparedLifecycle,
    witness: &Mat<F>,
    parent: &CeClaim,
) -> (Vec<Mat<F>>, Vec<bool>, Vec<Commitment>) {
    validate_superneo_witness_mat(witness, package.structure.m).unwrap();
    let params = params(package);
    let (digits, flags) = split_b_matrix_k_with_nonzero_flags(witness, params.k_rho() as usize, params.b()).unwrap();
    assert_eq!(digits.len(), 16);
    let commitments = commit_production_signed_unit_prefix_matrices(&digits).unwrap();
    assert_eq!(
        ajtai_dec_mixer(&commitments, params.b()),
        parent.c,
        "actual R witness commitment"
    );
    (digits, flags, commitments)
}
fn zero_opening(package: &PreparedLifecycle) -> V1_1Evaluations<K> {
    V1_1Evaluations {
        eval_k: vec![K::ZERO; D],
        eval_a: vec![vec![K::ZERO; D]; package.structure.t()],
    }
}

fn equal_contents(mut left: impl BufRead, mut right: impl BufRead) -> io::Result<bool> {
    loop {
        let a = left.fill_buf()?;
        let b = right.fill_buf()?;
        let count = a.len().min(b.len());
        if count == 0 {
            return Ok(a.is_empty() && b.is_empty());
        }
        if a[..count] != b[..count] {
            return Ok(false);
        }
        left.consume(count);
        right.consume(count);
    }
}

#[path = "staged_compare.rs"]
mod compare_tests;

pub(super) fn ccs(root: &Path, step: u64, engine: EvaluationEngine, cpu_reference: Option<&Path>) {
    #[cfg(feature = "metal")]
    if matches!(engine, EvaluationEngine::Metal) {
        assert!(
            cpu_reference.is_some(),
            "Metal production acceptance requires a CPU reference"
        );
    }
    let cpu_proof: Option<SavedCcs> = cpu_reference.map(|reference| {
        assert_ne!(fs::canonicalize(root).unwrap(), fs::canonicalize(reference).unwrap());
        let files = ["envelope.json", "fresh-claim.json", "fresh-witness.json"]
            .into_iter()
            .map(str::to_owned)
            .chain((0..16).map(|child| format!("digit-{child}.json")));
        for name in files {
            let actual = BufReader::new(File::open(step_dir(root, step).join(&name)).unwrap());
            let expected = BufReader::new(File::open(step_dir(reference, step).join(&name)).unwrap());
            assert!(
                equal_contents(actual, expected).unwrap(),
                "CPU/Metal source differs: {name}"
            );
        }
        load(&fold_dir(reference, step).join("ccs.json"))
    });
    let directory = fold_dir(root, step);
    fs::create_dir_all(&directory).unwrap();
    let package = prepare();
    let source = load_sources(&package, &step_dir(root, step), step);
    let started = Instant::now();
    let cache = package.build_superneo_cache().unwrap();
    eprintln!("C cache elapsed={:?}", started.elapsed());
    let mut transcript = Transcript::session();
    let proving = Instant::now();
    let proof = match engine {
        EvaluationEngine::Optimized => pi_ccs::prove_from_parts_with_rows(
            &mut transcript,
            &params(&package),
            &package.structure,
            cache,
            std::slice::from_ref(&source.fresh.claim),
            std::slice::from_ref(&source.fresh.witness),
            &source.running,
        )
        .unwrap(),
        #[cfg(feature = "metal")]
        EvaluationEngine::Metal => {
            let mut device = neo_prover_metal::MetalRowProver::new().unwrap();
            let (outputs, sumcheck, _, _) =
                neo_reductions::optimized_engine::optimized_prove_with_row_cache_and_backend(
                    transcript.inner_mut(),
                    params(&package).inner(),
                    &package.structure,
                    std::slice::from_ref(&source.fresh.claim),
                    std::slice::from_ref(&source.fresh.witness),
                    &source.running.claims,
                    &source.running.witnesses,
                    cache,
                    &mut device,
                )
                .unwrap();
            assert!(device.activity().dispatches > 0);
            eprintln!("C Metal activity={:?}", device.activity());
            pi_ccs::Proof { outputs, sumcheck }
        }
    };
    eprintln!("C proving engine={engine:?} elapsed={:?}", proving.elapsed());
    let (verified, _) = replay_ccs(&package, &source.fresh.claim, &source.running, &proof);
    assert_eq!(verified.snapshot(), transcript.snapshot());
    if let Some(reference) = cpu_proof {
        check_identity(
            &package,
            reference.schema,
            reference.structural_identifier,
            reference.package_identity,
            reference.verification_key_digest,
        );
        let reference = pi_ccs::Proof {
            sumcheck: reference.sumcheck,
            outputs: reference.outputs,
        };
        assert!(
            proof.canonical_bytes() == reference.canonical_bytes(),
            "CPU/Metal PiCCS proof bytes differ"
        );
        eprintln!("C CPU proof byte equality passed; source files also match");
    }
    save(
        &directory.join("ccs.json"),
        &SavedCcs {
            schema: 1,
            structural_identifier: package.package.structural_identifier(),
            package_identity: package.package_identity(),
            verification_key_digest: package.binding.verification_key_digest(),
            sumcheck: proof.sumcheck,
            outputs: proof.outputs,
        },
    );
    eprintln!("C completed elapsed={:?}", started.elapsed());
}
pub(super) fn rlc(root: &Path, step: u64) {
    let directory = fold_dir(root, step);
    let package = prepare();
    let source = load_sources(&package, &step_dir(root, step), step);
    let ccs: SavedCcs = load(&directory.join("ccs.json"));
    check_identity(
        &package,
        ccs.schema,
        ccs.structural_identifier,
        ccs.package_identity,
        ccs.verification_key_digest,
    );
    let proof = pi_ccs::Proof {
        sumcheck: ccs.sumcheck,
        outputs: ccs.outputs,
    };
    let (mut transcript, outputs) = replay_ccs(&package, &source.fresh.claim, &source.running, &proof);
    let witnesses: Vec<_> = std::iter::once(&source.fresh.witness.Z)
        .chain(source.running.witnesses.iter())
        .collect();
    let (parent, rproof) = pi_rlc::prove_refs(
        &mut transcript,
        &params(&package),
        &package.structure,
        ajtai_rlc_mixer,
        &outputs,
        &witnesses,
    )
    .unwrap();
    drop(witnesses);
    let (mut verifier, outputs) = replay_ccs(&package, &source.fresh.claim, &source.running, &proof);
    pi_rlc::verify(
        &mut verifier,
        &params(&package),
        &package.structure,
        ajtai_rlc_mixer,
        &outputs,
        &rproof,
    )
    .unwrap();
    assert_eq!(transcript.snapshot(), verifier.snapshot());
    let record = SavedParent {
        schema: 1,
        structural_identifier: package.package.structural_identifier(),
        package_identity: package.package_identity(),
        verification_key_digest: package.binding.verification_key_digest(),
        sumcheck: proof.sumcheck,
        ccs_outputs: proof.outputs,
        rlc_parent: parent.claim,
        transcript_state: transcript.snapshot().state(),
        transcript_absorbed: transcript.snapshot().absorbed(),
    };
    drop(source);
    save(&directory.join("parent-witness.json"), &parent.witness);
    save(&directory.join("parent.json"), &record);
}
pub(super) fn split(root: &Path, step: u64) {
    let directory = fold_dir(root, step);
    let package = prepare();
    let parent = read_parent(&package, root, step);
    let witness: Mat<F> = load(&directory.join("parent-witness.json"));
    let (digits, nonzero, commitments) = committed_split(&package, &witness, &parent.rlc_parent);
    drop(witness);
    for (child, digit) in digits.iter().enumerate() {
        save(&directory.join(format!("digit-{child}.json")), digit);
    }
    save(
        &directory.join("split.json"),
        &SavedSplit {
            schema: 1,
            parent: parent.rlc_parent,
            nonzero,
            commitments,
        },
    );
}
pub(super) fn child(root: &Path, step: u64, child: usize, engine: EvaluationEngine) {
    assert!(child < 16, "selected child index");
    let directory = fold_dir(root, step);
    let package = prepare();
    let parent = read_parent(&package, root, step);
    let split: SavedSplit = load(&directory.join("split.json"));
    assert_eq!(split.schema, 1);
    assert_eq!(split.parent, parent.rlc_parent);
    assert_eq!(split.nonzero.len(), 16);
    assert_eq!(split.commitments.len(), 16);
    assert_eq!(
        ajtai_dec_mixer(&split.commitments, params(&package).b()),
        parent.rlc_parent.c
    );
    let witness: Mat<F> = load(&directory.join("parent-witness.json"));
    validate_superneo_witness_mat(&witness, package.structure.m).unwrap();
    let digit: Mat<F> = load(&directory.join(format!("digit-{child}.json")));
    let params = params(&package);
    let (expected, flags) = split_b_matrix_k_with_nonzero_flags(&witness, params.k_rho() as usize, params.b()).unwrap();
    assert_eq!(digit, expected[child], "exact canonical digit");
    let active = flags[child];
    assert_eq!(split.nonzero[child], active);
    drop((expected, flags, witness));
    let commitment = commit_production_signed_unit_prefix_matrix(&digit).unwrap();
    assert_eq!(commitment, split.commitments[child]);
    let opening = if active {
        let cache = package.build_superneo_cache().unwrap();
        let started = Instant::now();
        let mut openings = match engine {
            EvaluationEngine::Optimized => {
                let blocks = SuperneoZBlocks::from_witness_mat(&digit, package.structure.m).unwrap();
                cache
                    .eval_real_v1_1_openings(&parent.rlc_parent.r, std::slice::from_ref(&blocks))
                    .unwrap()
            }
            #[cfg(feature = "metal")]
            EvaluationEngine::Metal => {
                let mut device = neo_prover_metal::MetalRowProver::new().unwrap();
                let openings = device
                    .child_openings(
                        std::sync::Arc::clone(cache),
                        std::slice::from_ref(&digit),
                        &parent.rlc_parent.r,
                        package.structure.m,
                    )
                    .unwrap();
                assert!(device.activity().dispatches > 0);
                eprintln!("D Metal activity={:?}", device.activity());
                openings
            }
        };
        eprintln!(
            "D opening engine={engine:?} child={child} elapsed={:?}",
            started.elapsed()
        );
        assert_eq!(openings.len(), 1);
        openings.pop().unwrap()
    } else {
        zero_opening(&package)
    };
    save(
        &directory.join(format!("child-{child}.json")),
        &SavedChildOpening {
            schema: 1,
            child,
            parent: parent.rlc_parent,
            commitment,
            opening,
            transcript_state: parent.transcript_state,
            transcript_absorbed: parent.transcript_absorbed,
        },
    );
}
pub(super) fn nifs(root: &Path, step: u64) {
    let directory = fold_dir(root, step);
    let package = prepare();
    let parent = read_parent(&package, root, step);
    let split: SavedSplit = load(&directory.join("split.json"));
    let witness: Mat<F> = load(&directory.join("parent-witness.json"));
    let (expected, flags, commitments) = committed_split(&package, &witness, &parent.rlc_parent);
    drop(witness);
    assert_eq!(split.schema, 1);
    assert_eq!(split.parent, parent.rlc_parent);
    assert_eq!(split.nonzero, flags, "activity derived from actual split");
    assert_eq!(split.commitments, commitments, "all commitments recomputed");
    let mut digits = Vec::with_capacity(16);
    let mut openings = Vec::with_capacity(16);
    for (child, expected_digit) in expected.into_iter().enumerate() {
        let digit: Mat<F> = load(&directory.join(format!("digit-{child}.json")));
        assert_eq!(digit, expected_digit, "exact saved digit {child}");
        let opening = if flags[child] {
            let saved: SavedChildOpening = load(&directory.join(format!("child-{child}.json")));
            assert_eq!(saved.schema, 1);
            assert_eq!(saved.child, child);
            assert_eq!(saved.parent, parent.rlc_parent);
            assert_eq!(saved.commitment, commitments[child]);
            assert_eq!(saved.transcript_state, parent.transcript_state);
            assert_eq!(saved.transcript_absorbed, parent.transcript_absorbed);
            assert_eq!(saved.opening.eval_k.len(), D);
            assert_eq!(saved.opening.eval_a.len(), package.structure.t());
            assert!(saved.opening.eval_a.iter().all(|family| family.len() == D));
            saved.opening
        } else {
            zero_opening(&package)
        };
        digits.push(digit);
        openings.push(opening);
    }
    // The complete split was recomputed and every saved digit compared above.
    let (children, ok_y, ok_x, ok_c) =
        neo_reductions::api::dec_children_with_commit_superneo_cached_from_trusted_split_digits(
            neo_reductions::api::FoldingMode::Optimized,
            &package.structure,
            params(&package).inner(),
            &parent.rlc_parent,
            &digits,
            &flags,
            D.next_power_of_two().trailing_zeros() as usize,
            &commitments,
            ajtai_dec_mixer,
            None,
            None,
            Some(&openings),
        );
    assert!(ok_y && ok_x && ok_c);
    let record = SavedNifs { parent, children };
    let (_, _, _, proof) = record.verify(&package, &step_dir(root, step), step);
    let wire = proof.canonical_bytes();
    if step == 1 {
        assert_eq!(
            wire,
            fs::read(PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/stage1_actual_nifs/proof.native"))
                .unwrap()
        );
        let expected = read(artifact("nightstream-fprime-stage1-base-nifs-result-v1.json"));
        let state = record
            .parent
            .transcript_state
            .map(|value| value.as_canonical_u64());
        assert_eq!(json!(state), expected[7][9]);
        assert_eq!(json!(state), expected[9][14]);
        assert_eq!(record.parent.transcript_absorbed, 0);
    }
    save_bytes(&directory.join("proof.native"), &wire);
    save(&directory.join("nifs.json"), &record);
}
