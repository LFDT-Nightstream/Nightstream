//! The argument end to end on small claims whose values come from the
//! definitions (`support`): an honest proof verifies, and a changed proof, a
//! false claim and a witness outside its claim do not.

mod support;

use std::sync::OnceLock;

use neo_ccs::Mat;
use neo_compress::{commit, verify, Error, Layout, Norm, Proof, Statement};
use neo_math::{D, F, K};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;
use support::{claim_of, instances, Instance};

fn statements(instances: &[Instance]) -> Vec<Statement<'_>> {
    instances.iter().map(Instance::statement).collect()
}

fn layout(instances: &[Instance]) -> Layout {
    let claims: Vec<_> = instances
        .iter()
        .map(|instance| (instance.witness.cols(), instance.norm))
        .collect();
    Layout::new(&claims).expect("layout")
}

/// Prove `instances` with the witnesses `witnesses`.
fn prove_with(instances: &[Instance], witnesses: &[&Mat<F>]) -> Result<Proof, Error> {
    let committed = commit(layout(instances), witnesses)?;
    let mut transcript = Poseidon2Transcript::new_v1_2();
    Ok(committed.prove(&mut transcript, &statements(instances))?.0)
}

fn prove(instances: &[Instance]) -> Result<Proof, Error> {
    let witnesses: Vec<_> = instances.iter().map(|instance| &instance.witness).collect();
    prove_with(instances, &witnesses)
}

fn accepts(instances: &[Instance], proof: &Proof) -> Result<(), Error> {
    let mut transcript = Poseidon2Transcript::new_v1_2();
    verify(&mut transcript, &statements(instances), proof).map(|_| ())
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
    assert!(verify(&mut transcript, &statements(instances), proof).is_err());
}

#[test]
fn a_changed_proof_does_not_verify() {
    let (instances, proof) = honest();
    let flip = |bytes: &mut Vec<u8>, at: usize| bytes[at] ^= 1;
    let rounds = proof.rounds.len();
    let mut changes: Vec<(String, Box<dyn Fn(&mut Proof)>)> = vec![
        ("witness value".into(), Box::new(|proof| proof.witness_value += K::ONE)),
        ("fewer rounds".into(), Box::new(|proof| proof.rounds.truncate(1))),
    ];
    for round in [0, rounds / 2, rounds - 1] {
        for coefficient in 0..4 {
            changes.push((
                format!("round {round} coefficient {coefficient}"),
                Box::new(move |proof| proof.rounds[round][coefficient] += K::ONE),
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
        ("witness opening", proof.witness_opening.len()),
        ("inverse opening", proof.inverse_opening.len()),
    ] {
        for at in [0, length / 2, length - 1] {
            changes.push((
                format!("{name} byte {at}"),
                Box::new(move |proof| {
                    flip(
                        match name {
                            "witness commitment" => &mut proof.witness_commitment,
                            "inverse commitment" => &mut proof.inverse_commitment,
                            "witness opening" => &mut proof.witness_opening,
                            _ => &mut proof.inverse_opening,
                        },
                        at,
                    )
                }),
            ));
        }
    }
    for (name, change) in changes {
        let mut changed = proof.clone();
        change(&mut changed);
        assert!(accepts(instances, &changed).is_err(), "{name}");
    }
}

/// `V` holds the inverses of the claims with a table, so claims without a
/// table have no layout, and no proof for them verifies.
#[test]
fn claims_without_a_table_have_no_layout() {
    assert!(matches!(Layout::new(&[]), Err(Error::Layout(_))));
    assert!(matches!(
        Layout::new(&[(90, Norm::Caller), (61, Norm::Caller)]),
        Err(Error::Layout(_))
    ));
    assert!(Layout::new(&[(90, Norm::Caller), (61, Norm::Table { bound: 1 })]).is_ok());
    let (instances, proof) = honest();
    assert!(matches!(accepts(&instances[2..], proof), Err(Error::Layout(_))));
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
