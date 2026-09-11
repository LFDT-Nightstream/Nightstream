//! Actual selected base witness and initial zero sources, shared by conformance tests and the fixture CLI.

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_units;
use neo_ccs::{CcsClaim, CcsWitness, Mat};
use neo_fold_clean::{
    paper::{
        construction2::{LaneCommitmentMode, RunningInstance},
        params::Params,
        relations::CcsInstance,
    },
    Poseidon2HashChainV1Package,
};
use neo_math::{D, F};
use nightstream_fprime::{load_poseidon2_hash_chain_v1_package, PI_CCS_V1_1_PRIOR_PUBLIC_INPUT_WORDS};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use serde::Deserialize;
use serde_json::Value;
#[cfg(test)]
use std::path::PathBuf;
use std::{fs, path::Path, time::Instant};

#[derive(Deserialize)]
struct Fixture(u64, [u64; 4], Vec<u64>, Vec<u64>, Value);

pub struct ActualBase {
    pub package: Poseidon2HashChainV1Package,
    pub params: Params,
    pub fresh: CcsInstance,
    pub running: RunningInstance,
}

#[cfg(test)]
pub fn artifact(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("../../formal/nightstream-fprime/artifacts")
        .join(name)
}

pub fn load(package_path: &Path, fixture_bytes: &[u8]) -> ActualBase {
    let started = Instant::now();
    let bytes = fs::read(package_path).unwrap();
    let producer = load_poseidon2_hash_chain_v1_package(&bytes).unwrap();
    let Fixture(schema, context, private, public, fixture_result) = serde_json::from_slice(fixture_bytes).unwrap();
    drop(fixture_result);
    assert_eq!(schema, 1);
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
    ActualBase {
        package,
        params,
        fresh: CcsInstance { claim: fresh, witness },
        running,
    }
}
