//! Original selected sources for the staged conformance driver. Base files
//! execute the base witness; saved directories retain their actual matrices.

use neo_ajtai::{
    nightstream_fprime_setup::{commit_production_signed_unit_matrix, commit_production_signed_units},
    Commitment,
};
use neo_ccs::{CcsWitness, Mat};
use neo_fold_clean::{
    paper::{
        construction2::{LaneCommitmentMode, RunningInstance},
        params::Params,
        pi_dec,
        relations::{ajtai_dec_mixer, CcsClaim, CcsInstance, CeClaim},
    },
    stage1::{encode_pi_ccs_v1_1_public_input, pi_ccs_v1_1_state_hash, serialize_pi_ccs_v1_1_state_preimage},
    Poseidon2HashChainV1Package,
};
use neo_math::{D, F};
use neo_reductions::common::{project_x_from_witness_mat, validate_fresh_witness_tail_zero};
use nightstream_fprime::{
    load_poseidon2_hash_chain_v1_package, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS, PI_CCS_V1_1_STATE_PREIMAGE_WORDS,
    PI_DEC_V1_1_CHILD_COUNT,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::{de::DeserializeOwned, Deserialize};
use serde_json::Value;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize)]
struct Fixture(u64, [u64; 4], Vec<u64>, Vec<u64>, Value);

pub struct ActualSources {
    pub package: Poseidon2HashChainV1Package,
    pub params: Params,
    pub fresh: CcsInstance,
    pub running: RunningInstance,
}

pub fn load(package_path: &Path, fixture_bytes: &[u8]) -> ActualSources {
    let started = Instant::now();
    let bytes = fs::read(package_path).unwrap();
    let producer = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let Fixture(schema, context, private, public, fixture_result) = serde_json::from_slice(fixture_bytes).unwrap();
    drop(fixture_result);
    assert_eq!(schema, 1);
    // serializePreimage places iteration after the 23-word domain and
    // length-prefixed four-word context. A recursive caller is not a base source.
    const ITERATION_WORD: usize = 23 + 1 + 4;
    assert_eq!(
        private.get(ITERATION_WORD),
        Some(&0),
        "the file must contain a base caller"
    );
    assert_eq!(
        private.get(PI_CCS_V1_1_STATE_PREIMAGE_WORDS + ITERATION_WORD),
        Some(&1),
        "the base output must have iteration one"
    );
    assert_eq!(
        context,
        producer
            .production_verifier_binding()
            .unwrap()
            .verifier_context()
            .digest()
    );
    let physical = producer.execute_witness(&private, &public).unwrap();
    let logical = producer.execute_logical_assignment(&physical).unwrap();
    let logical_width = logical.len();
    let blocks = logical_width.div_ceil(D);
    let mut carrier = logical.balanced_values().to_vec();
    carrier.resize(blocks * D, 0);
    let commitment = commit_production_signed_units(&carrier).unwrap();
    let mut positive = vec![0u64; blocks];
    let mut negative = vec![0u64; blocks];
    for (index, &value) in carrier.iter().enumerate() {
        let mask = 1u64 << (index % D);
        match value {
            0 => {}
            1 => positive[index / D] |= mask,
            -1 => negative[index / D] |= mask,
            _ => panic!("non-unit actual source"),
        }
    }
    let public_width = PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS;
    let x = (0..public_width)
        .map(|index| F::from_u64(logical.value(index).unwrap()))
        .collect::<Vec<_>>();
    let mut prior_digest = [0u8; 32];
    for lane in 0..4 {
        let word = (0..64).fold(0u64, |word, bit| {
            let digit = x[1 + lane * 64 + bit].as_canonical_u64();
            assert!(digit <= 1);
            word | (digit << bit)
        });
        prior_digest[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    let fresh = CcsClaim {
        c: commitment,
        x,
        m_in: public_width,
        adv: None,
    };
    let witness = CcsWitness {
        w: Vec::new(),
        Z: Mat::<F>::compact_signed_unit_from_column_masks(D, blocks, &positive, &negative).unwrap(),
    };
    drop((
        producer, physical, logical, carrier, positive, negative, private, public,
    ));
    println!("actual_sources_and_commitment_elapsed={:?}", started.elapsed());

    let package = Poseidon2HashChainV1Package::load(&bytes).unwrap();
    drop(bytes);
    assert_eq!(package.structure().m, logical_width);
    let params = Params::for_ccs_shape(
        package.structure().n,
        package.structure().m,
        package.structure().t(),
        package.structure().max_degree(),
    )
    .unwrap();
    let mut running =
        RunningInstance::canonical_zero(&params, package.structure(), public_width, LaneCommitmentMode::Plain).unwrap();
    for claim in &mut running.claims {
        claim.fold_digest = prior_digest;
    }
    running.parent_authority.as_mut().unwrap().fold_digest = prior_digest;
    ActualSources {
        package,
        params,
        fresh: CcsInstance { claim: fresh, witness },
        running,
    }
}

#[derive(Deserialize)]
struct SavedEnvelope {
    schema: u64,
    iteration: u64,
    z0: [u64; 4],
    current: [u64; 4],
    child_witness_count: usize,
    running_claims: Vec<CeClaim>,
    running_parent: Option<CeClaim>,
}

fn read<T: DeserializeOwned>(path: &Path) -> T {
    let bytes = fs::read(path).expect("original source file");
    serde_json::from_slice(&bytes).expect("complete typed source data")
}

fn state_fields(words: [u64; 4]) -> [F; 4] {
    words.map(|word| {
        assert!(word < F::ORDER_U64, "canonical saved state word");
        F::from_u64(word)
    })
}

/// Load an original base fixture or the exact saved envelope files. Directory
/// paths inside envelope metadata are not followed; all source files are
/// read from the directory selected by the caller. Saved witnesses are prover
/// inputs: `check_path` separately checks their complete commitment openings.
pub fn load_path(package_path: &Path, source_path: &Path) -> ActualSources {
    if !source_path.is_dir() {
        return load(package_path, &fs::read(source_path).expect("original base fixture"));
    }
    let started = Instant::now();
    let bytes = fs::read(package_path).expect("selected package");
    let package = Poseidon2HashChainV1Package::load(&bytes).expect("allowlisted selected package identity");
    let binding = load_poseidon2_hash_chain_v1_package(&bytes)
        .expect("same selected package")
        .production_verifier_binding()
        .expect("package-owned verifier context");
    assert_eq!(package.package_identity(), binding.package_identity());
    assert_eq!(package.verification_key_digest(), binding.verification_key_digest());
    drop(bytes);
    let record: SavedEnvelope = read(&source_path.join("envelope.json"));
    assert_eq!(record.schema, 1, "saved envelope schema");
    assert!(
        record.iteration > 0 && record.iteration < F::ORDER_U64,
        "canonical active source counter"
    );
    assert_eq!(record.child_witness_count, PI_DEC_V1_1_CHILD_COUNT);
    assert!(
        record
            .running_claims
            .iter()
            .all(|claim| claim.adv.is_none()),
        "plain source claims"
    );
    let preimage = serialize_pi_ccs_v1_1_state_preimage(
        binding.verifier_context().digest().map(F::from_u64),
        record.iteration,
        state_fields(record.z0),
        state_fields(record.current),
        &record.running_claims,
        1,
    )
    .expect("selected source state and semantic running shapes");
    let digest = pi_ccs_v1_1_state_hash(&preimage).expect("recomputed source state hash");
    let public = encode_pi_ccs_v1_1_public_input(digest).expect("complete encoded source hash");
    let mut frame = [0u8; 32];
    for (lane, word) in digest.into_iter().enumerate() {
        frame[lane * 8..lane * 8 + 8].copy_from_slice(&word.to_le_bytes());
    }
    assert!(
        record
            .running_claims
            .iter()
            .chain(record.running_parent.iter())
            .all(|claim| claim.fold_digest == frame),
        "original source frames match the recomputed state; the loader does not reframe proof inputs"
    );
    let params = Params::for_ccs_shape(
        package.structure().n,
        package.structure().m,
        package.structure().t(),
        package.structure().max_degree(),
    )
    .expect("selected source parameters");
    pi_dec::verify(
        &params,
        package.structure(),
        ajtai_dec_mixer,
        record
            .running_parent
            .as_ref()
            .expect("saved original running parent"),
        &pi_dec::Proof {
            children: record.running_claims.clone(),
        },
    )
    .expect("saved original running-parent public authority");
    let claim: CcsClaim = read(&source_path.join("fresh-claim.json"));
    assert!(claim.adv.is_none(), "plain fresh source");
    assert_eq!(claim.m_in, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);
    assert_eq!(
        claim.x,
        public.into_iter().map(F::from_u64).collect::<Vec<_>>(),
        "source state/public link"
    );
    let fresh = CcsInstance {
        claim,
        witness: CcsWitness {
            w: Vec::new(),
            Z: read(&source_path.join("fresh-witness.json")),
        },
    };
    let witnesses = (0..PI_DEC_V1_1_CHILD_COUNT)
        .map(|child| read(&source_path.join(format!("digit-{child}.json"))))
        .collect();
    let running = RunningInstance::new(record.running_claims, witnesses, record.running_parent);
    println!(
        "saved_sources_loaded iteration={} elapsed={:?}",
        record.iteration,
        started.elapsed()
    );
    let actual = ActualSources {
        package,
        params,
        fresh,
        running,
    };
    let width = actual.package.structure().m;
    validate_fresh_witness_tail_zero(&actual.fresh.witness.Z, width, "saved fresh source")
        .expect("complete fresh shape and zero tail");
    check_saved_public(&actual.fresh.witness.Z, &fresh_public(&actual.fresh), width)
        .expect("original fresh source public projection");
    assert_eq!(actual.running.claims.len(), actual.running.witnesses.len());
    for (claim, witness) in actual.running.claims.iter().zip(&actual.running.witnesses) {
        check_saved_public(witness, &claim.X, width).expect("original running source public projection");
    }
    actual
}

fn fresh_public(fresh: &CcsInstance) -> Mat<F> {
    let mut public = Mat::zero(D, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS / D, F::ZERO);
    assert_eq!(fresh.claim.x.len(), PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS);
    for (column, value) in fresh.claim.x.iter().copied().enumerate() {
        public[(column % D, column / D)] = value;
    }
    public
}

fn check_saved_public(witness: &Mat<F>, public: &Mat<F>, logical_width: usize) -> Result<(), &'static str> {
    let projected = project_x_from_witness_mat(witness, logical_width, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS)
        .map_err(|_| "source complete witness shape")?;
    if projected != *public {
        return Err("source public projection differs");
    }
    Ok(())
}

fn check_saved_opening(
    witness: &Mat<F>,
    commitment: &Commitment,
    public: &Mat<F>,
    logical_width: usize,
) -> Result<(), &'static str> {
    check_saved_public(witness, public, logical_width)?;
    let actual =
        commit_production_signed_unit_matrix(witness).map_err(|_| "source fixed-key shape or strict unit norm")?;
    if actual != *commitment {
        return Err("source commitment differs from the complete witness");
    }
    Ok(())
}

/// C/R replay checks claims and proofs. Each original Z must also open its
/// own claimed commitment; a recomposed R commitment cannot replace this.
fn check_saved_source_openings(actual: &ActualSources, started: Instant) {
    let width = actual.package.structure().m;
    validate_fresh_witness_tail_zero(&actual.fresh.witness.Z, width, "saved fresh source")
        .expect("complete fresh shape and zero tail");
    check_saved_opening(
        &actual.fresh.witness.Z,
        &actual.fresh.claim.c,
        &fresh_public(&actual.fresh),
        width,
    )
    .expect("original fresh source opening");
    println!("saved_fresh_source_opening=checked elapsed={:?}", started.elapsed());
    assert_eq!(actual.running.claims.len(), actual.running.witnesses.len());
    for (child, (claim, witness)) in actual
        .running
        .claims
        .iter()
        .zip(&actual.running.witnesses)
        .enumerate()
    {
        check_saved_opening(witness, &claim.c, &claim.X, width)
            .unwrap_or_else(|reason| panic!("original running source {child}: {reason}"));
        println!(
            "saved_running_source={child} opening=checked elapsed={:?}",
            started.elapsed()
        );
    }
}

/// Required source-opening stage before a saved-source conformance sequence.
/// A private signed-unit substitution keeps the public input, shape and claim.
pub fn check_path(package_path: &Path, source_path: &Path) {
    assert!(
        source_path.is_dir(),
        "source substitution uses a saved envelope directory"
    );
    let started = Instant::now();
    let actual = load_path(package_path, source_path);
    check_saved_source_openings(&actual, started);
    let witness = &actual.fresh.witness.Z;
    let (positive, negative) = witness
        .packed_signed_unit_column_masks()
        .expect("retained source column masks");
    let mut positive = positive.to_vec();
    let mut negative = negative.to_vec();
    let coordinate = actual.fresh.claim.m_in;
    assert!(
        coordinate < actual.package.structure().m,
        "the source has a private coordinate"
    );
    let block = coordinate / D;
    let mask = 1u64 << (coordinate % D);
    if (positive[block] | negative[block]) & mask == 0 {
        positive[block] |= mask;
    } else {
        positive[block] &= !mask;
        negative[block] &= !mask;
    }
    let changed = Mat::compact_signed_unit_from_column_masks(D, witness.cols(), &positive, &negative)
        .expect("same-shape signed-unit substitution");
    assert_eq!(
        check_saved_opening(
            &changed,
            &actual.fresh.claim.c,
            &fresh_public(&actual.fresh),
            actual.package.structure().m
        ),
        Err("source commitment differs from the complete witness"),
        "a changed private coordinate cannot retain the original source commitment"
    );
    println!(
        "saved_source_private_substitution=rejected elapsed={:?}",
        started.elapsed()
    );
}
