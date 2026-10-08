//! The layer-1 verifier read by `Recorder`: satisfied rows for a true claim,
//! a noted failure for every tampered input, and rows that do not depend on
//! the proof.

use p3_field_v08::PrimeCharacteristicRing;

use super::layer1::{claim_mutations, proof_mutations, relation_of, setup_toy, transcript};
use crate::circuit::record::{Recorder, Trace};
use crate::circuit::{poseidon2, Backend};
use crate::field::Gl;
use crate::hash::seed;
use crate::verifier::{verify, ProofView};
use crate::{prove, Claim, Proof, Relation, Statement};

/// Record the verifier with the seed and the statement as statement words.
fn record(
    relation: &Relation<'_>,
    seed: [Gl; 4],
    claim: &Claim,
    proof: Option<&Proof>,
) -> (Trace, Option<&'static str>) {
    let statement = Statement::new(&relation.shape, claim).unwrap();
    let mut recorder = Recorder::new(Trace::default());
    let b = &mut recorder;
    let seed = seed.map(|word| b.public(word));
    let mut words = |values: &[[Gl; 2]]| -> Vec<[<Recorder<Trace> as Backend>::F; 2]> {
        values
            .iter()
            .map(|pair| pair.map(|word| b.public(word)))
            .collect()
    };
    let point = words(&statement.point);
    let eval_k = words(&statement.eval_k).try_into().unwrap();
    let eval_a = statement
        .eval_a
        .iter()
        .map(|values| words(values).try_into().unwrap())
        .collect();
    let mut rings = |values: &[[Gl; neo_math::D]]| -> Vec<[<Recorder<Trace> as Backend>::F; neo_math::D]> {
        values
            .iter()
            .map(|ring| ring.map(|word| b.public(word)))
            .collect()
    };
    let statement = Statement {
        commitment: rings(&statement.commitment),
        public: rings(&statement.public),
        point,
        eval_k,
        eval_a,
    };
    let view = ProofView::read(b, relation, proof).unwrap();
    verify(b, relation, seed, &statement, &view).unwrap();
    recorder.finish()
}

#[test]
fn recorded_layer1_verifier_holds_rejects_and_keeps_its_shape() {
    let (toy, setup, _scratch) = setup_toy(8, 4, 6);
    let relation = relation_of(&toy, setup.key());
    let proof = prove(&relation, &setup, transcript(1), &toy.claim, &toy.witness).unwrap();
    let honest_seed = seed(transcript(1));

    let (honest, failure) = record(&relation, honest_seed, &toy.claim, Some(&proof));
    assert_eq!(failure, None);
    for r in 0..honest.rows.len() {
        assert_eq!(honest.row_value(r), Gl::ZERO, "row {r}");
    }
    for block in 0..honest.blocks.len() / poseidon2::CELLS {
        assert_eq!(honest.block_failure(block), None, "block {block}");
    }
    eprintln!(
        "layer-1 verifier: {} rows, {} glue cells, {} permutations",
        honest.rows.len(),
        honest.glue.len(),
        honest.blocks.len() / poseidon2::CELLS
    );

    let (shape, _) = record(&relation, honest_seed, &toy.claim, None);
    assert_eq!(honest.rows, shape.rows);
    assert_eq!(honest.entries, shape.entries);
    assert_eq!(honest.glue.len(), shape.glue.len());
    assert_eq!(honest.blocks.len(), shape.blocks.len());
    assert_eq!(honest.public.len(), shape.public.len());

    let (_, failure) = record(&relation, seed(transcript(2)), &toy.claim, Some(&proof));
    assert!(failure.is_some(), "fold transcript");
    for (name, claim) in claim_mutations(&toy.claim) {
        assert!(
            record(&relation, honest_seed, &claim, Some(&proof))
                .1
                .is_some(),
            "{name}"
        );
    }
    for (name, changed) in proof_mutations(&proof) {
        assert!(
            record(&relation, honest_seed, &toy.claim, Some(&changed))
                .1
                .is_some(),
            "{name}"
        );
    }
}
