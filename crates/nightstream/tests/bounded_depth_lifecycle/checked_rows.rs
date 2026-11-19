use std::sync::Arc;

use neo_reductions::superneo_eval::MatrixRows;

use super::super::relation::{FreshMatrixRows, Relation};
use super::{fold_gate_bits, hash_chain_step, prover, words, BoundedDepth, Envelope, Mat, D, F};

fn first(prover: &BoundedDepth, seed: u64) -> Envelope {
    let initial = words([seed, seed + 1, seed + 2, seed + 3]);
    let message = words([1, 3, 5, 7]);
    let envelope = prover.initial(initial).unwrap();
    prover
        .extend(&envelope, &message, hash_chain_step(initial, message), None)
        .unwrap()
}

fn rows<'a>(prover: &'a BoundedDepth, envelope: &'a Envelope) -> FreshMatrixRows<'a> {
    FreshMatrixRows {
        relation: &prover.relations()[0],
        prepared: envelope.checked_rows.as_deref(),
        witness: &envelope.fresh.as_ref().unwrap().witness.Z,
    }
}

#[test]
fn checked_source_cache_is_bounded_by_relation_and_proof_lifetime() {
    let prover = prover(2);
    let (a, b) = rayon::join(|| first(&prover, 10), || first(&prover, 20));
    assert!(a.checked_rows.is_some() || b.checked_rows.is_some());
    for proof in [&a, &b] {
        prover.verify(1, proof.z0, proof.current, proof).unwrap();
    }
    let c = first(&prover, 30);
    assert!(a
        .checked_rows
        .iter()
        .chain(&b.checked_rows)
        .all(|rows| !rows.retained()));
    assert!(rows(&prover, &a).take_checked_fresh_rows().is_none());
    assert!(rows(&prover, &b).take_checked_fresh_rows().is_none());

    let live = rows(&prover, &c);
    let other_relation = Relation::new(prover.relations()[0].step(), fold_gate_bits()).unwrap();
    let foreign = FreshMatrixRows {
        relation: &other_relation,
        ..rows(&prover, &c)
    };
    assert!(foreign.take_checked_fresh_rows().is_none());
    let (p, n) = live.witness.packed_signed_unit_column_masks().unwrap();
    let distinct = Mat::<F>::compact_signed_unit_from_column_masks(D, p.len(), p, n).unwrap();
    let other = FreshMatrixRows {
        witness: &distinct,
        ..live
    };
    assert!(other.take_checked_fresh_rows().is_none());
    assert!(rows(&prover, &c).take_checked_fresh_rows().is_some());
    assert!(rows(&prover, &c).take_checked_fresh_rows().is_none());

    let imported = prover
        .decode_proof(&prover.encode_proof(&a).unwrap())
        .unwrap();
    let message = words([2, 4, 6, 8]);
    let next = hash_chain_step(a.current, message);
    let actual = prover.extend(&a, &message, next, None).unwrap();
    let expected = prover.extend(&imported, &message, next, None).unwrap();
    prover.verify(2, a.z0, next, &actual).unwrap();
    assert_eq!(
        prover.encode_proof(&actual).unwrap(),
        prover.encode_proof(&expected).unwrap()
    );

    let idle = first(&prover, 40);
    let handle = Arc::downgrade(idle.checked_rows.as_ref().unwrap());
    drop(idle);
    assert!(handle.upgrade().is_none());
    assert!(rows(&prover, &expected).take_checked_fresh_rows().is_none());
}
