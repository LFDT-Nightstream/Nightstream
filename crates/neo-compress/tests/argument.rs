//! The argument end to end on small claims whose values come from the
//! definitions (`support`): an honest proof verifies, and a changed proof, a
//! false claim, a witness outside its claim and another key do not.

mod support;

use std::sync::OnceLock;

use neo_ajtai::nightstream_fprime_setup::commit_production_signed_unit_prefix_matrix;
use neo_ccs::Mat;
use neo_compress::{commit, verify, Error, Layout, Norm, Proof, VerificationKey};
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use rand::SeedableRng;
use rand_chacha::ChaCha8Rng;
use support::{claim_of, instances, key, layout, statements, Instance};

/// Prove `instances` with the witnesses `witnesses`.
fn prove_with(instances: &[Instance], witnesses: &[&Mat<F>]) -> Result<Proof, Error> {
    let committed = commit(key(), layout(instances), witnesses)?;
    let mut transcript = Poseidon2Transcript::new_v1_2();
    Ok(committed.prove(&mut transcript, &statements(instances))?.0)
}

fn prove(instances: &[Instance]) -> Result<Proof, Error> {
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    prove_with(instances, &witnesses)
}

fn accepts_under(key: &VerificationKey, instances: &[Instance], proof: &Proof) -> Result<(), Error> {
    let mut transcript = Poseidon2Transcript::new_v1_2();
    verify(&mut transcript, key, &statements(instances), proof).map(|_| ())
}

fn accepts(instances: &[Instance], proof: &Proof) -> Result<(), Error> {
    accepts_under(key().verification_key(), instances, proof)
}

/// The honest instances and their proof.
fn honest() -> &'static (Vec<Instance>, Proof) {
    static HONEST: OnceLock<(Vec<Instance>, Proof)> = OnceLock::new();
    HONEST.get_or_init(|| {
        let instances = instances();
        let proof = prove(&instances).expect("proof");
        (instances, proof)
    })
}

#[test]
fn an_honest_proof_verifies() {
    let (instances, proof) = honest();
    let layout = layout(instances);
    assert_eq!((layout.witness_vars(), layout.inverse_vars()), (16, 16));
    accepts(instances, proof).expect("the honest proof verifies");
}

#[test]
fn a_proof_for_another_transcript_does_not_verify() {
    let (instances, proof) = honest();
    let mut transcript = Poseidon2Transcript::new_v1_2();
    transcript.absorb_v1_2(&[F::ONE]);
    assert!(verify(&mut transcript, key().verification_key(), &statements(instances), proof).is_err());
}

#[test]
fn a_changed_proof_does_not_verify() {
    let (instances, proof) = honest();
    let flip = |bytes: &mut Vec<u8>, at: usize| bytes[at] ^= 1;
    let rounds = proof.rounds.len();
    let mut changes: Vec<(String, Box<dyn Fn(&mut Proof)>)> = vec![
        ("witness value".into(), Box::new(|proof| proof.witness_value += K::ONE)),
        ("fewer rounds".into(), Box::new(|proof| proof.rounds.truncate(1))),
        (
            "the key's part of the weights".into(),
            Box::new(|proof| proof.key_weight += K::ONE),
        ),
        (
            "the value of the key polynomial".into(),
            Box::new(|proof| proof.key_value += K::ONE),
        ),
        (
            "fewer key rounds".into(),
            Box::new(|proof| proof.key_rounds.truncate(14)),
        ),
        (
            "one more key round".into(),
            Box::new(|proof| proof.key_rounds.push([K::ZERO; 3])),
        ),
    ];
    for round in [0, rounds / 2, rounds - 1] {
        for coefficient in 0..4 {
            changes.push((
                format!("round {round} coefficient {coefficient}"),
                Box::new(move |proof| proof.rounds[round][coefficient] += K::ONE),
            ));
        }
    }
    for round in 0..proof.key_rounds.len() {
        for coefficient in 0..3 {
            changes.push((
                format!("key round {round} coefficient {coefficient}"),
                Box::new(move |proof| proof.key_rounds[round][coefficient] += K::ONE),
            ));
        }
    }
    for column in 0..proof.inverse_values.len() {
        changes.push((
            format!("inverse value {column}"),
            Box::new(move |proof| proof.inverse_values[column] += K::ONE),
        ));
    }
    for (name, length) in [
        ("witness commitment", proof.witness_commitment.len()),
        ("inverse commitment", proof.inverse_commitment.len()),
        ("opening", proof.opening.len()),
    ] {
        for at in [0, length / 2, length - 1] {
            changes.push((
                format!("{name} byte {at}"),
                Box::new(move |proof| {
                    flip(
                        match name {
                            "witness commitment" => &mut proof.witness_commitment,
                            "inverse commitment" => &mut proof.inverse_commitment,
                            _ => &mut proof.opening,
                        },
                        at,
                    )
                }),
            ));
        }
    }
    // Akita reads each of its byte strings to the end.
    changes.push((
        "witness commitment with one more byte".into(),
        Box::new(|proof| proof.witness_commitment.push(0)),
    ));
    changes.push((
        "inverse commitment with one more byte".into(),
        Box::new(|proof| proof.inverse_commitment.push(0)),
    ));
    changes.push((
        "opening with one more byte".into(),
        Box::new(|proof| proof.opening.push(0)),
    ));
    changes.push((
        "the two commitments in the other order".into(),
        Box::new(|proof| std::mem::swap(&mut proof.witness_commitment, &mut proof.inverse_commitment)),
    ));
    for (name, change) in changes {
        let mut changed = proof.clone();
        change(&mut changed);
        assert!(accepts(instances, &changed).is_err(), "{name}");
    }
}

/// The key sum-check has one round for each lane variable and each block
/// variable of the key polynomial of 512 blocks. A false value of the key's
/// part of the weights with a first round that agrees with it does not
/// verify: the second round does not agree with the first.
#[test]
fn a_false_key_weight_with_a_consistent_first_round_does_not_verify() {
    let (instances, proof) = honest();
    assert_eq!(proof.key_rounds.len(), 6 + 9);
    let mut changed = proof.clone();
    changed.key_weight += K::ONE;
    // p(0) + p(1) = 2·c0 + c1 + c2 moves by one with c1.
    changed.key_rounds[0][1] += K::ONE;
    assert!(matches!(accepts(instances, &changed), Err(Error::Rejected(_))));
}

/// A verification key is authority. A key with one changed byte is another
/// key or no key, and the honest proof does not verify under it. The bytes
/// of a key decode to that key only.
#[test]
fn another_verification_key_does_not_verify() {
    let (instances, proof) = honest();
    let honest_key = key().verification_key();
    let bytes = honest_key.to_bytes();
    assert_eq!(&VerificationKey::from_bytes(&bytes).expect("key"), honest_key);
    assert_eq!(honest_key.blocks(), 512);
    for at in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        match VerificationKey::from_bytes(&changed) {
            Ok(other) => {
                assert_ne!(&other, honest_key, "byte {at}");
                assert!(accepts_under(&other, instances, proof).is_err(), "byte {at}");
            }
            Err(error) => assert!(
                matches!(error, Error::Key(_) | Error::Rejected(_)),
                "byte {at}: {error}"
            ),
        }
    }
    assert!(matches!(
        VerificationKey::from_bytes(&bytes[..bytes.len() - 1]),
        Err(Error::Key(_))
    ));
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(matches!(VerificationKey::from_bytes(&longer), Err(Error::Key(_))));
    // The block count is a field of the key: 1024 blocks have no schedule
    // row, and 2^19 blocks have another commitment profile.
    for blocks in [0u64, 511, 1024, 1 << 19, u64::MAX] {
        let mut changed = bytes.clone();
        changed[16..24].copy_from_slice(&blocks.to_le_bytes());
        assert!(
            matches!(VerificationKey::from_bytes(&changed), Err(Error::Key(_))),
            "{blocks} blocks"
        );
    }
    // A length above the bytes that follow it decodes nothing.
    for huge in [u64::MAX, 1 << 40, bytes.len() as u64] {
        let mut changed = bytes.clone();
        changed[24..32].copy_from_slice(&huge.to_le_bytes());
        assert!(
            matches!(VerificationKey::from_bytes(&changed), Err(Error::Key(_))),
            "length {huge}"
        );
    }
}

/// Anyone derives the verification key from the public seeds, without the
/// prover's key.
#[test]
fn the_derived_verification_key_is_the_key_of_the_prover() {
    assert_eq!(&VerificationKey::derive(512).expect("key"), key().verification_key());
    assert_eq!(&VerificationKey::derive(1).expect("key"), key().verification_key());
    assert!(matches!(VerificationKey::derive((1 << 19) + 1), Err(Error::Layout(_))));
}

/// A claim of 513 blocks reads a key element that the key polynomial of 512
/// blocks does not hold: it has no layout, so no proof and no acceptance.
/// A key size without schedule rows has no layout.
#[test]
fn a_claim_with_more_blocks_than_the_key_has_no_layout() {
    let blocks = key().verification_key().blocks();
    let table = Norm::Table { bound: 1 };
    assert!(Layout::new(&[(blocks / 2, table)], blocks).is_ok());
    assert!(matches!(
        Layout::new(&[(blocks + 1, table)], blocks),
        Err(Error::Layout(_))
    ));
    assert!(matches!(
        Layout::new(&[(61, table), (blocks + 1, Norm::Caller)], blocks),
        Err(Error::Layout(_))
    ));
    assert!(matches!(Layout::new(&[(61, table)], 2 * blocks), Err(Error::Layout(_))));
    let mut rng = ChaCha8Rng::seed_from_u64(7);
    let large = vec![Instance::random(&mut rng, blocks + 1, table)];
    let (_, proof) = honest();
    assert!(accepts(&large, proof).is_err());
}

/// `V` holds the inverses of the claims with a table, so claims without a
/// table have no layout, and no proof for them verifies.
#[test]
fn claims_without_a_table_have_no_layout() {
    assert!(matches!(Layout::new(&[], 512), Err(Error::Layout(_))));
    assert!(matches!(
        Layout::new(&[(90, Norm::Caller), (61, Norm::Caller)], 512),
        Err(Error::Layout(_))
    ));
    assert!(Layout::new(&[(90, Norm::Caller), (61, Norm::Table { bound: 1 })], 512).is_ok());
    let (instances, proof) = honest();
    assert!(matches!(accepts(&instances[2..], proof), Err(Error::Layout(_))));
}

/// The commitment that the native path makes for a witness is the one that
/// the argument opens: both read the columns `0..blocks` of the production
/// key. `claim_of` computes the commitment of the claim from those columns.
#[test]
fn the_native_commitment_is_the_commitment_of_the_claim() {
    let (instances, _) = honest();
    let instance = &instances[2];
    assert!(matches!(instance.norm, Norm::Caller));
    let native = commit_production_signed_unit_prefix_matrix(&instance.witness).expect("native commitment");
    assert_eq!(native, instance.claim.c);
}

/// Each change makes one part of a claim false: a commitment row, `Eval_K`,
/// one `Eval_A`, the public input, the point. The verifier rejects the honest
/// proof for the changed claim, and the proof of a prover that proves the
/// changed claim.
#[test]
fn a_false_claim_does_not_verify() {
    let changes: [(&str, fn(&mut Instance)); 6] = [
        ("first commitment row", |instance| {
            let mut data = instance.claim.c.data().to_vec();
            data[3] += F::ONE;
            instance.claim.c = neo_ajtai::Commitment::new(support::KAPPA, data).expect("commitment");
        }),
        ("last commitment row", |instance| {
            let mut data = instance.claim.c.data().to_vec();
            *data.last_mut().expect("data") += F::ONE;
            instance.claim.c = neo_ajtai::Commitment::new(support::KAPPA, data).expect("commitment");
        }),
        ("Eval_K", |instance| instance.claim.eval_k[5] += K::ONE),
        ("Eval_A", |instance| instance.claim.eval_a[2][63] += K::ONE),
        ("public input", |instance| {
            let mut cells = instance.claim.X.to_dense_vec();
            cells[70] += F::ONE;
            instance.claim.X = Mat::from_row_major(D, support::PUBLIC_BLOCKS, cells);
        }),
        ("point", |instance| instance.claim.r[9] += K::ONE),
    ];
    let (_, proof) = honest();
    for claim in 0..3 {
        for (name, change) in changes {
            let mut instances = instances();
            change(&mut instances[claim]);
            assert!(
                accepts(&instances, proof).is_err(),
                "claim {claim}, {name}: honest proof"
            );
            let forged = prove(&instances).expect("the prover does not check its witness");
            assert!(
                accepts(&instances, &forged).is_err(),
                "claim {claim}, {name}: forged proof"
            );
        }
    }
}

/// A witness with one coefficient at `bound + 1` or `-(bound + 1)`, with the
/// claim that it satisfies: only the table check is false.
#[test]
fn a_witness_outside_its_table_does_not_verify() {
    for (claim, sign) in [(0, F::ONE), (1, F::NEG_ONE)] {
        let mut instances = instances();
        let instance = &mut instances[claim];
        let Norm::Table { bound } = instance.norm else {
            panic!("claim {claim} has a table");
        };
        let mut cells = instance.witness.to_dense_vec();
        cells[1234] = sign * F::from_u64(bound + 1);
        instance.witness = Mat::from_row_major(D, instance.witness.cols(), cells);
        instance.claim = claim_of(&instance.rows, &instance.witness, &instance.claim.r.clone());
        let proof = prove(&instances).expect("the prover does not check its witness");
        assert!(accepts(&instances, &proof).is_err(), "claim {claim}");

        // The same witness at the bound verifies.
        let instance = &mut instances[claim];
        let mut cells = instance.witness.to_dense_vec();
        cells[1234] = sign * F::from_u64(bound);
        instance.witness = Mat::from_row_major(D, instance.witness.cols(), cells);
        instance.claim = claim_of(&instance.rows, &instance.witness, &instance.claim.r.clone());
        let proof = prove(&instances).expect("proof");
        accepts(&instances, &proof).unwrap_or_else(|error| panic!("claim {claim} at the bound: {error}"));
    }
}

/// The prover commits a witness with one other coefficient: it opens no
/// commitment row of the claim.
#[test]
fn another_witness_does_not_verify() {
    let (instances, _) = honest();
    for claim in 0..3 {
        let mut cells = instances[claim].witness.to_dense_vec();
        cells[777] = if cells[777] == F::ZERO { F::ONE } else { F::ZERO };
        let other = Mat::from_row_major(D, instances[claim].witness.cols(), cells);
        let witnesses: Vec<_> = instances
            .iter()
            .enumerate()
            .map(|(index, instance)| if index == claim { &other } else { &instance.witness })
            .collect();
        let proof = prove_with(instances, &witnesses).expect("the prover does not check its witness");
        assert!(accepts(instances, &proof).is_err(), "claim {claim}");
    }
}
