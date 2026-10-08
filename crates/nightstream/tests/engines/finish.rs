//! Compression on the parity fixtures: PiCCS + PiRLC without PiDEC, then the
//! layer-1 argument for the parent claim, and the verifier's replay of both
//! with only the compression key.

use std::path::PathBuf;

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

/// A fixture's setup files in a scratch directory removed on drop.
struct Compression {
    setup: neo_spartan::Setup,
    dir: PathBuf,
}

impl Compression {
    fn new(fixture: &Fixture, name: &str) -> Self {
        let dir = std::env::temp_dir().join(format!("nightstream-finish-{name}-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let rows = CachedMatrixRows::new(&fixture.cache).unwrap();
        let setup = neo_spartan::Setup::build(&rows, &dir).unwrap();
        Self { setup, dir }
    }
}

impl Drop for Compression {
    fn drop(&mut self) {
        let _ = std::fs::remove_dir_all(&self.dir);
    }
}

fn relation<'k>(
    fixture: &Fixture,
    key: &'k neo_spartan::Key,
    parent: &CeClaim,
) -> Result<neo_spartan::Relation<'k>, neo_spartan::Error> {
    let params = &fixture.params;
    let bound = 17 * params.T() * (params.b() - 1);
    let bits = params.compression_security_bits().unwrap();
    neo_spartan::Relation::new(key, parent.m_in / D, parent.r.len(), bound, bits)
}

/// Layer 0, then layer 1. `extra` is absorbed between them to model a layer-1
/// proof made for a different layer-0 transcript.
fn finish(fixture: &Fixture, setup: &neo_spartan::Setup, extra: Option<F>) -> Finished {
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
    let relation = relation(fixture, setup.key(), &parent.claim).unwrap();
    let layer1 = neo_spartan::prove(
        &relation,
        setup,
        transcript.into_inner(),
        &parent.claim,
        &parent.witness,
    )
    .unwrap();
    Finished { pi_ccs, pi_rlc, layer1 }
}

fn accepts(
    fixture: &Fixture,
    key: &neo_spartan::Key,
    fresh: &CcsClaim,
    running: &RunningInstance,
    proof: &Finished,
) -> bool {
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
    relation(fixture, key, &parent)
        .is_ok_and(|relation| neo_spartan::verify(&relation, transcript.into_inner(), &parent, &proof.layer1).is_ok())
}

fn check(fixture: Fixture, other: Fixture, name: &str) {
    let compression = Compression::new(&fixture, name);
    let key = compression.setup.key();
    let proof = finish(&fixture, &compression.setup, None);
    let fresh = &fixture.fresh.claim;
    let running = fixture.running.claims_only();
    assert!(accepts(&fixture, key, fresh, &running, &proof));
    let decoded = Finished {
        layer1: neo_spartan::Proof::from_bytes(&proof.layer1.to_bytes()).unwrap(),
        ..proof.clone()
    };
    assert!(
        accepts(&fixture, key, fresh, &running, &decoded),
        "layer-1 bytes round trip"
    );

    let mut changed = fresh.clone();
    changed.x[0] += F::ONE;
    assert!(
        !accepts(&fixture, key, &changed, &running, &proof),
        "fresh public input"
    );
    let mut changed = running.clone();
    changed.claims[0].eval_k[0] += K::ONE;
    assert!(!accepts(&fixture, key, fresh, &changed, &proof), "running evaluation");
    let mut changed = proof.clone();
    changed.pi_ccs.outputs[1].eval_a[0][3] += K::ONE;
    assert!(!accepts(&fixture, key, fresh, &running, &changed), "PiCCS output");
    let mut changed = proof.clone();
    changed.pi_rlc.combined.eval_k[5] += K::ONE;
    assert!(!accepts(&fixture, key, fresh, &running, &changed), "PiRLC parent");
    let another = Finished {
        layer1: finish(&fixture, &compression.setup, Some(F::ONE)).layer1,
        ..proof.clone()
    };
    assert!(
        !accepts(&fixture, key, fresh, &running, &another),
        "layer-1 proof of another transcript"
    );
    let other_key = neo_spartan::Key::derive(&CachedMatrixRows::new(&other.cache).unwrap()).unwrap();
    assert!(
        !accepts(&fixture, &other_key, fresh, &running, &proof),
        "key of the other fixture"
    );
}

#[test]
fn bit_fixture_finishes_and_verifies() {
    check(Fixture::bit(), Fixture::selected_polynomial(), "bit");
}

#[test]
fn selected_polynomial_finishes_and_verifies() {
    check(Fixture::selected_polynomial(), Fixture::bit(), "selected");
}
