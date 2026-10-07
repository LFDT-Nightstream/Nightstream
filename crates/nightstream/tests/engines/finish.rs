//! Compression on the parity fixtures: PiCCS + PiRLC without PiDEC, then the
//! layer-1 argument for the parent claim, and the verifier's replay of both.

use super::Fixture;
use crate::folding::{
    self, ajtai_dec_mixer, ajtai_rlc_mixer, pi_ccs, pi_rlc, transcript::Transcript, CcsClaim, CeClaim, RunningInstance,
};
use neo_math::{D, F, K};
use neo_reductions::superneo_eval::CachedMatrixRows;
use p3_field::PrimeCharacteristicRing;

#[derive(Clone)]
struct Finished {
    pi_ccs: pi_ccs::Proof,
    pi_rlc: pi_rlc::Proof,
    layer1: neo_spartan::Proof,
}

fn relation<'r>(fixture: &Fixture, rows: &'r CachedMatrixRows<'_>, parent: &CeClaim) -> neo_spartan::Relation<'r> {
    let params = &fixture.params;
    let bound = 17 * params.T() * (params.b() - 1);
    let bits = params.compression_security_bits().unwrap();
    neo_spartan::Relation::new(rows, parent.m_in / D, parent.r.len(), bound, bits).unwrap()
}

/// Layer 0, then layer 1. `extra` is absorbed between them to model a layer-1
/// proof made for a different layer-0 transcript.
fn finish(fixture: &Fixture, extra: Option<F>) -> Finished {
    let rows = CachedMatrixRows::new(&fixture.cache).unwrap();
    let mut transcript = Transcript::session();
    let (pi_ccs, parent, pi_rlc) = folding::prove_parent_with_rows(
        &mut transcript,
        &fixture.params,
        &fixture.structure,
        &rows,
        fixture.workspace_bytes(&rows),
        vec![fixture.fresh.clone()],
        fixture.running.clone(),
    )
    .unwrap();
    if let Some(word) = extra {
        transcript.inner_mut().absorb_v1_1(&[word]);
    }
    let relation = relation(fixture, &rows, &parent.claim);
    let layer1 = neo_spartan::prove(&relation, transcript.into_inner(), &parent.claim, &parent.witness).unwrap();
    Finished { pi_ccs, pi_rlc, layer1 }
}

fn accepts(fixture: &Fixture, fresh: &CcsClaim, running: &RunningInstance, proof: &Finished) -> bool {
    let rows = CachedMatrixRows::new(&fixture.cache).unwrap();
    let mut transcript = Transcript::session();
    let Ok(parent) = folding::verify_parent(
        &mut transcript,
        &fixture.params,
        &fixture.structure,
        ajtai_rlc_mixer,
        ajtai_dec_mixer,
        std::slice::from_ref(fresh),
        running,
        &proof.pi_ccs,
        &proof.pi_rlc,
    ) else {
        return false;
    };
    let relation = relation(fixture, &rows, &parent);
    neo_spartan::verify(&relation, transcript.into_inner(), &parent, &proof.layer1).is_ok()
}

fn check(fixture: Fixture) {
    let proof = finish(&fixture, None);
    let fresh = &fixture.fresh.claim;
    let running = fixture.running.claims_only();
    assert!(accepts(&fixture, fresh, &running, &proof));
    let decoded = Finished {
        layer1: neo_spartan::Proof::from_bytes(&proof.layer1.to_bytes()).unwrap(),
        ..proof.clone()
    };
    assert!(accepts(&fixture, fresh, &running, &decoded), "layer-1 bytes round trip");

    let mut changed = fresh.clone();
    changed.x[0] += F::ONE;
    assert!(!accepts(&fixture, &changed, &running, &proof), "fresh public input");
    let mut changed = running.clone();
    changed.claims[0].eval_k[0] += K::ONE;
    assert!(!accepts(&fixture, fresh, &changed, &proof), "running evaluation");
    let mut changed = proof.clone();
    changed.pi_ccs.outputs[1].eval_a[0][3] += K::ONE;
    assert!(!accepts(&fixture, fresh, &running, &changed), "PiCCS output");
    let mut changed = proof.clone();
    changed.pi_rlc.combined.eval_k[5] += K::ONE;
    assert!(!accepts(&fixture, fresh, &running, &changed), "PiRLC parent");
    let other = Finished {
        layer1: finish(&fixture, Some(F::ONE)).layer1,
        ..proof.clone()
    };
    assert!(
        !accepts(&fixture, fresh, &running, &other),
        "layer-1 proof of another transcript"
    );
}

#[test]
fn bit_fixture_finishes_and_verifies() {
    check(Fixture::bit());
}

#[test]
fn selected_polynomial_finishes_and_verifies() {
    check(Fixture::selected_polynomial());
}
