//! Terminal tamper sweep on a folded proof: every field of an active envelope
//! either produces its expected rejection or is named below as
//! non-authoritative. Exhaustive patterns make a new proof variant or a new
//! claim, instance, witness, or commitment field a compile error until it has
//! a case. `Stage1State` fields are private; the three state cases cover them.
use super::*;
use crate::folding::{CcsInstance, CcsWitness};
use crate::lifecycle::{ProofState, Stage1Envelope};
use crate::{Circuit, Engine, Verifier};
use nightstream_fprime::PI_DEC_V1_1_CHILD_COUNT;
use std::time::Instant;

#[derive(Clone)]
struct Parts {
    state: Stage1State,
    running: RunningInstance,
    fresh: CcsInstance,
}

impl Parts {
    fn envelope(self) -> Stage1Envelope {
        Stage1Envelope::from_state_and_proof(self.state, ProofState::active(self.running, self.fresh))
    }

    fn with_state(&mut self, iteration: u64, z0: [F; 4], current: [F; 4]) {
        self.state = Stage1State::new(iteration, z0, current);
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

const STATEMENT: &str = "selected terminal statement";
const FRESH: &str = "selected terminal fresh opening";
const HASH: &str = "selected terminal fresh opening: public input differs from the recomputed terminal state hash";

/// One change: a name, the change, and the exact expected rejection.
type Case = (&'static str, fn(&mut Parts), String);

fn running(index: usize, reason: &str) -> String {
    format!("selected terminal running child {index}: {reason}")
}

fn cases() -> Vec<Case> {
    let last = PI_DEC_V1_1_CHILD_COUNT - 1;
    vec![
        (
            "state iteration",
            |parts| parts.with_state(parts.state.iteration() + 1, parts.state.z0(), parts.state.current()),
            HASH.into(),
        ),
        (
            "state z0",
            |parts| {
                let mut z0 = parts.state.z0();
                z0[0] += F::ONE;
                parts.with_state(parts.state.iteration(), z0, parts.state.current())
            },
            HASH.into(),
        ),
        (
            "state current",
            |parts| {
                let mut current = parts.state.current();
                current[0] += F::ONE;
                parts.with_state(parts.state.iteration(), parts.state.z0(), current)
            },
            HASH.into(),
        ),
        (
            "running claim count",
            |parts| drop(parts.running.claims.pop()),
            format!("{STATEMENT}: running claim or witness count differs from the selected profile"),
        ),
        (
            "running witness count",
            |parts| drop(parts.running.witnesses.pop()),
            format!("{STATEMENT}: running claim or witness count differs from the selected profile"),
        ),
        (
            "last running commitment d",
            |parts| parts.running.claims.last_mut().unwrap().c.d += 1,
            running(last, "commitment or public-input shape"),
        ),
        (
            "last running commitment kappa",
            |parts| parts.running.claims.last_mut().unwrap().c.kappa += 1,
            running(last, "commitment or public-input shape"),
        ),
        (
            "first running commitment data",
            |parts| parts.running.claims[0].c.data[0] += F::ONE,
            HASH.into(),
        ),
        (
            "last running commitment data",
            |parts| parts.running.claims.last_mut().unwrap().c.data[0] += F::ONE,
            HASH.into(),
        ),
        (
            "first running public input X",
            |parts| parts.running.claims[0].X[(0, 0)] += F::ONE,
            running(0, "witness public projection differs from X"),
        ),
        (
            "last running public input X",
            |parts| parts.running.claims.last_mut().unwrap().X[(0, 0)] += F::ONE,
            running(last, "witness public projection differs from X"),
        ),
        (
            "last running point r",
            |parts| parts.running.claims.last_mut().unwrap().r[0] += K::ONE,
            running(last, "running claims must share the selected evaluation point"),
        ),
        (
            "every running point r",
            |parts| {
                for claim in &mut parts.running.claims {
                    claim.r[0] += K::ONE;
                }
            },
            HASH.into(),
        ),
        (
            "first running Eval_K",
            |parts| parts.running.claims[0].eval_k[0] += K::ONE,
            HASH.into(),
        ),
        (
            "last running Eval_K",
            |parts| parts.running.claims.last_mut().unwrap().eval_k[0] += K::ONE,
            HASH.into(),
        ),
        (
            "last running Eval_K surplus",
            |parts| parts.running.claims.last_mut().unwrap().eval_k.push(K::ONE),
            running(last, "evaluation shape or nonzero surplus coefficients"),
        ),
        (
            "last running Eval_A",
            |parts| parts.running.claims.last_mut().unwrap().eval_a[0][0] += K::ONE,
            HASH.into(),
        ),
        (
            "last running Eval_A count",
            |parts| drop(parts.running.claims.last_mut().unwrap().eval_a.pop()),
            running(last, "evaluation shape or nonzero surplus coefficients"),
        ),
        (
            "last running m_in",
            |parts| parts.running.claims.last_mut().unwrap().m_in += D,
            running(last, "commitment or public-input shape"),
        ),
        (
            "last running auxiliary commitments",
            |parts| {
                let claim = parts.running.claims.last_mut().unwrap();
                claim.adv = Some(lanes(&claim.c));
            },
            running(last, "plain claims cannot carry auxiliary commitments"),
        ),
        (
            "first running witness public coordinate",
            |parts| flip(&mut parts.running.witnesses[0], 0),
            running(0, "witness public projection differs from X"),
        ),
        (
            "last running witness public coordinate",
            |parts| flip(parts.running.witnesses.last_mut().unwrap(), 0),
            running(last, "witness public projection differs from X"),
        ),
        (
            "first running witness private coordinate",
            |parts| {
                let position = parts.running.claims[0].m_in;
                flip(&mut parts.running.witnesses[0], position)
            },
            running(0, "fixed-key commitment differs from the witness"),
        ),
        (
            "last running witness private coordinate",
            |parts| {
                let position = parts.running.claims[0].m_in;
                flip(parts.running.witnesses.last_mut().unwrap(), position)
            },
            running(last, "fixed-key commitment differs from the witness"),
        ),
        (
            "fresh commitment d",
            |parts| parts.fresh.claim.c.d += 1,
            format!("{FRESH}: commitment or public-input shape"),
        ),
        (
            "fresh commitment kappa",
            |parts| parts.fresh.claim.c.kappa += 1,
            format!("{FRESH}: commitment or public-input shape"),
        ),
        (
            "fresh commitment data",
            |parts| parts.fresh.claim.c.data[0] += F::ONE,
            format!("{FRESH}: fixed-key commitment differs from the witness"),
        ),
        (
            "fresh public input x",
            |parts| parts.fresh.claim.x[1] += F::ONE,
            HASH.into(),
        ),
        (
            "fresh m_in",
            |parts| parts.fresh.claim.m_in += D,
            format!("{FRESH}: commitment or public-input shape"),
        ),
        (
            "fresh auxiliary commitments",
            |parts| parts.fresh.claim.adv = Some(lanes(&parts.fresh.claim.c)),
            format!("{FRESH}: plain claims cannot carry auxiliary commitments"),
        ),
        (
            "fresh witness public coordinate",
            |parts| flip(&mut parts.fresh.witness.Z, 1),
            format!("{FRESH}: witness public projection differs from x"),
        ),
        (
            "fresh witness private coordinate",
            |parts| {
                let position = parts.fresh.claim.m_in;
                flip(&mut parts.fresh.witness.Z, position)
            },
            format!("{FRESH}: fixed-key commitment differs from the witness"),
        ),
    ]
}

#[test]
#[ignore = "Production two-step proof and terminal tamper sweep; run separately under the 300-second cap."]
fn terminal_rejects_every_authoritative_envelope_change() {
    let started = Instant::now();
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8].map(F::from_u64);
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let circuit = Circuit::compile(&bytes, crate::application::poseidon2_hash_chain_v1().unwrap()).unwrap();
    drop(bytes);
    let verifier = Verifier::from_package(&circuit, Engine::Optimized, 114).unwrap();
    let prover = circuit.prover(Engine::Optimized, 114).unwrap();
    // The second step folds a real running instance; the base step's running
    // claims are all the zero default.
    let proof = prover
        .extend(&prover.prove(initial, &message).unwrap(), &message)
        .unwrap();
    let expected = Stage1State::new(2, initial, output(output(initial, message), message));
    verifier.verify(&expected, &proof).unwrap();
    eprintln!("honest folded proof accepted elapsed={:?}", started.elapsed());

    let (state, proof) = proof.into_parts();
    let (running, fresh) = match proof {
        ProofState::Initial => panic!("the folded proof is active"),
        ProofState::Active { running, fresh } => (running, fresh),
    };
    let honest = Parts { state, running, fresh };
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
    let CcsInstance { claim: _, witness: _ } = &honest.fresh;
    let CcsClaim {
        c: _,
        x: _,
        m_in: _,
        adv: _,
    } = &honest.fresh.claim;
    let CcsWitness { w: _, Z: _ } = &honest.fresh.witness;
    let Commitment {
        d: _,
        kappa: _,
        data: _,
    } = &honest.fresh.claim.c;
    let LaneCommitments { ops: _, is: _, fs: _ } = lanes(&honest.fresh.claim.c);
    assert_ne!(
        honest.running.claims[0].eval_k, honest.running.claims[1].eval_k,
        "the sweep needs distinct running claims"
    );

    // Each change is verified against the envelope's own state, so the state
    // cases must fail at the state-hash binding, not at the state comparison.
    for (name, change, expected_error) in cases() {
        let mut parts = honest.clone();
        change(&mut parts);
        let state = parts.state.clone();
        let case = Instant::now();
        let error = verifier
            .verify(&state, &parts.envelope())
            .expect_err(&format!("the terminal verifier accepted a changed {name}"));
        eprintln!("{name}: {error} elapsed={:?}", case.elapsed());
        assert_eq!(error.to_string(), expected_error, "{name}");
    }
    let bottom = Stage1Envelope::from_state_and_proof(expected.clone(), ProofState::Initial);
    assert_eq!(
        verifier.verify(&expected, &bottom).unwrap_err().to_string(),
        format!("{STATEMENT}: bottom requires zero iterations and equal endpoints")
    );

    // These fields are non-authoritative: the terminal decision must ignore
    // them. Zero coefficients beyond `D` are padding of the same evaluation.
    let mut ignored = honest.clone();
    for claim in &mut ignored.running.claims {
        claim.fold_digest[0] ^= 1;
        claim.eval_k.push(K::ZERO);
        claim.eval_a[0].push(K::ZERO);
    }
    ignored.running.parent_authority = match ignored.running.parent_authority {
        Some(_) => None,
        None => Some(ignored.running.claims[0].clone()),
    };
    ignored.fresh.witness.w = vec![F::ONE];
    verifier.verify(&expected, &ignored.envelope()).unwrap();
    eprintln!("terminal tamper sweep elapsed={:?}", started.elapsed());
}
