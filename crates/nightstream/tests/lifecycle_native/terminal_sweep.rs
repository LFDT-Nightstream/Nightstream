//! Terminal tamper sweep: every field of an active envelope either changes the
//! terminal decision or is named below as non-authoritative.
use super::*;
use crate::folding::{CcsInstance, CcsWitness};
use crate::lifecycle::{LatestInstance, ProofState, Stage1Envelope};
use crate::{Circuit, Engine, Verifier};
use std::time::Instant;

#[derive(Clone)]
struct Parts {
    state: Stage1State,
    running: RunningInstance,
    instances: Vec<CcsInstance>,
}

impl Parts {
    fn envelope(self) -> Stage1Envelope {
        Stage1Envelope::from_state_and_proof(
            self.state,
            ProofState::active(self.running, LatestInstance::from_instances(self.instances)),
        )
    }

    fn last_claim(&mut self) -> &mut CeClaim {
        self.running.claims.last_mut().unwrap()
    }

    fn fresh(&mut self) -> &mut CcsInstance {
        &mut self.instances[0]
    }
}

/// Change one signed-unit coordinate of a witness to another unit value.
fn flip(matrix: &mut Mat<F>, position: usize) {
    let (row, column) = (position % D, position / D);
    let value = if matrix[(row, column)] == F::ZERO {
        F::ONE
    } else {
        F::ZERO
    };
    matrix.set(row, column, value);
}

fn lanes(commitment: &Commitment) -> LaneCommitments<Commitment> {
    LaneCommitments {
        ops: commitment.clone(),
        is: commitment.clone(),
        fs: commitment.clone(),
    }
}

type Change = Box<dyn Fn(&mut Parts)>;

#[test]
#[ignore = "Production base proof and terminal tamper sweep; run separately under the 300-second cap."]
fn terminal_rejects_every_authoritative_envelope_change() {
    let started = Instant::now();
    let fixture = read(artifact("nightstream-fprime-stage1-base-step-fixture-v1.json"));
    let private: Vec<u64> = serde_json::from_value(fixture[2].clone()).unwrap();
    let initial: [F; 4] = std::array::from_fn(|lane| field(private[30 + lane]));
    let message: [F; 4] = std::array::from_fn(|lane| field(private[private.len() - 4 + lane]));
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let circuit = Circuit::compile(&bytes, crate::application::poseidon2_hash_chain_v1().unwrap()).unwrap();
    drop(bytes);
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let proof = circuit
        .prover(Engine::Optimized, 114)
        .unwrap()
        .prove(initial, &message)
        .unwrap();
    let expected = Stage1State::new(1, initial, output(initial, message));
    verifier.verify(&expected, &proof).unwrap();
    eprintln!("honest base proof accepted elapsed={:?}", started.elapsed());

    let (state, proof) = proof.into_parts();
    let ProofState::Active { running, latest } = proof else {
        panic!("the base proof is active")
    };
    let LatestInstance { instances } = latest;
    let honest = Parts {
        state,
        running,
        instances,
    };
    // A new envelope field breaks one of these patterns and needs a sweep case.
    let RunningInstance {
        claims: _,
        witnesses: _,
        parent_authority: _,
    } = &honest.running;
    let CeClaim {
        c: _,
        X: _,
        r: _,
        eval_k: _,
        eval_a: _,
        m_in: _,
        fold_digest: _,
        adv: _,
    } = &honest.running.claims[0];
    let CcsInstance { claim: _, witness: _ } = &honest.instances[0];
    let CcsClaim {
        c: _,
        x: _,
        m_in: _,
        adv: _,
    } = &honest.instances[0].claim;
    let CcsWitness { w: _, Z: _ } = &honest.instances[0].witness;
    let Commitment {
        d: _,
        kappa: _,
        data: _,
    } = &honest.instances[0].claim.c;

    // The first private coordinate follows the public input.
    let private_position = honest.running.claims[0].m_in;
    let changes: Vec<(&str, Change)> = vec![
        (
            "state iteration",
            Box::new(|parts: &mut Parts| parts.state = Stage1State::new(2, parts.state.z0(), parts.state.current())),
        ),
        (
            "state z0",
            Box::new(|parts: &mut Parts| {
                let mut z0 = parts.state.z0();
                z0[0] += F::ONE;
                parts.state = Stage1State::new(1, z0, parts.state.current())
            }),
        ),
        (
            "state current",
            Box::new(|parts: &mut Parts| {
                let mut current = parts.state.current();
                current[0] += F::ONE;
                parts.state = Stage1State::new(1, parts.state.z0(), current)
            }),
        ),
        (
            "running claim count",
            Box::new(|parts: &mut Parts| {
                parts.running.claims.pop();
            }),
        ),
        (
            "running witness count",
            Box::new(|parts: &mut Parts| {
                parts.running.witnesses.pop();
            }),
        ),
        (
            "running commitment d",
            Box::new(|parts: &mut Parts| parts.last_claim().c.d += 1),
        ),
        (
            "running commitment kappa",
            Box::new(|parts: &mut Parts| parts.last_claim().c.kappa += 1),
        ),
        (
            "running commitment data",
            Box::new(|parts: &mut Parts| parts.last_claim().c.data[0] += F::ONE),
        ),
        (
            "running public input X",
            Box::new(|parts: &mut Parts| parts.last_claim().X[(0, 0)] += F::ONE),
        ),
        (
            "running point r of one claim",
            Box::new(|parts: &mut Parts| parts.last_claim().r[0] += K::ONE),
        ),
        (
            "running point r of every claim",
            Box::new(|parts: &mut Parts| {
                for claim in &mut parts.running.claims {
                    claim.r[0] += K::ONE;
                }
            }),
        ),
        (
            "running Eval_K",
            Box::new(|parts: &mut Parts| parts.last_claim().eval_k[0] += K::ONE),
        ),
        (
            "running Eval_K surplus",
            Box::new(|parts: &mut Parts| parts.last_claim().eval_k.push(K::ONE)),
        ),
        (
            "running Eval_A",
            Box::new(|parts: &mut Parts| parts.last_claim().eval_a[0][0] += K::ONE),
        ),
        (
            "running Eval_A count",
            Box::new(|parts: &mut Parts| {
                parts.last_claim().eval_a.pop();
            }),
        ),
        (
            "running m_in",
            Box::new(|parts: &mut Parts| parts.last_claim().m_in += D),
        ),
        (
            "running auxiliary commitments",
            Box::new(|parts: &mut Parts| {
                let claim = parts.last_claim();
                claim.adv = Some(lanes(&claim.c));
            }),
        ),
        (
            "running witness public coordinate",
            Box::new(|parts: &mut Parts| flip(parts.running.witnesses.last_mut().unwrap(), 0)),
        ),
        (
            "running witness private coordinate",
            Box::new(move |parts: &mut Parts| flip(parts.running.witnesses.last_mut().unwrap(), private_position)),
        ),
        (
            "fresh instance missing",
            Box::new(|parts: &mut Parts| parts.instances.clear()),
        ),
        (
            "extra fresh instance",
            Box::new(|parts: &mut Parts| parts.instances.push(parts.instances[0].clone())),
        ),
        (
            "fresh commitment d",
            Box::new(|parts: &mut Parts| parts.fresh().claim.c.d += 1),
        ),
        (
            "fresh commitment kappa",
            Box::new(|parts: &mut Parts| parts.fresh().claim.c.kappa += 1),
        ),
        (
            "fresh commitment data",
            Box::new(|parts: &mut Parts| parts.fresh().claim.c.data[0] += F::ONE),
        ),
        (
            "fresh public input x",
            Box::new(|parts: &mut Parts| parts.fresh().claim.x[1] += F::ONE),
        ),
        (
            "fresh m_in",
            Box::new(|parts: &mut Parts| parts.fresh().claim.m_in += D),
        ),
        (
            "fresh auxiliary commitments",
            Box::new(|parts: &mut Parts| {
                let claim = &mut parts.fresh().claim;
                claim.adv = Some(lanes(&claim.c));
            }),
        ),
        (
            "fresh witness public coordinate",
            Box::new(|parts: &mut Parts| flip(&mut parts.fresh().witness.Z, 1)),
        ),
        (
            "fresh witness private coordinate",
            Box::new(move |parts: &mut Parts| flip(&mut parts.fresh().witness.Z, private_position)),
        ),
    ];
    for (name, change) in &changes {
        let mut parts = honest.clone();
        change(&mut parts);
        let case = Instant::now();
        let result = verifier.verify(&expected, &parts.envelope());
        eprintln!("{name}: {result:?} elapsed={:?}", case.elapsed());
        assert!(result.is_err(), "the terminal verifier accepted a changed {name}");
    }
    assert!(verifier
        .verify(&expected, &Stage1Envelope::initial(initial))
        .is_err());

    // These fields are non-authoritative: the terminal decision must ignore them.
    let mut ignored = honest.clone();
    for claim in &mut ignored.running.claims {
        claim.fold_digest[0] ^= 1;
    }
    ignored.running.parent_authority = Some(ignored.running.claims[0].clone());
    ignored.fresh().witness.w = vec![F::ONE];
    verifier.verify(&expected, &ignored.envelope()).unwrap();
    eprintln!("terminal tamper sweep elapsed={:?}", started.elapsed());
}
