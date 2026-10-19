//! Terminal tamper sweep on a folded proof. Each case changes one envelope
//! field and checks the exact decision: the first check that fails, or
//! acceptance for a documented non-authoritative field. The rejected cases stop
//! at the statement, shape, projection, state-hash and commitment checks. The
//! accepted cases run the complete verifier. Two balanced `Π_DEC` opening
//! changes, completed through `prove_active` and `complete_fold` with a rebuilt
//! fresh witness, pass every earlier check and reach the `Eval_K` and `Eval_A`
//! opening checks. A relation failure is covered by the recursive and staged
//! terminal tests. The patterns
//! below name every proof variant and every claim, instance, witness and
//! commitment field, so a new field must at least be named here. The second
//! fold starts from decoded proof bytes. The
//! `Stage1State` and `Stage1Envelope` fields are private; `Parts` and the
//! state cases cover them.
use super::staged::opening_tests::{change_openings, labels, Opening};
use super::*;
use crate::engine::Backend;
use crate::folding::{CcsInstance, CcsWitness};
use crate::lifecycle::{ProofState, Stage1Envelope, VerifyError};
use nightstream_fprime::PI_DEC_V1_2_CHILD_COUNT;
use std::time::Instant;

#[derive(Clone)]
struct Parts {
    state: Stage1State,
    running: RunningInstance,
    fresh: CcsInstance,
}

impl Parts {
    fn envelope(self) -> Stage1Envelope {
        Stage1Envelope::from_parts(self.state, self.running, self.fresh)
    }

    fn with_state(&mut self, iteration: u64, z0: [F; 4], current: [F; 4]) {
        self.state = Stage1State::new(iteration, z0, current);
    }

    fn last_claim(&mut self) -> &mut CeClaim {
        self.running.claims.last_mut().unwrap()
    }
}

/// The terminal checks a case can reach, by error kind and reason.
#[derive(Debug, PartialEq)]
enum Rejection {
    Statement(&'static str),
    Running(usize, &'static str),
    Fresh(&'static str),
}

fn rejection(error: VerifyError) -> Rejection {
    match error {
        VerifyError::Statement(reason) => Rejection::Statement(reason),
        VerifyError::Running { index, reason } => Rejection::Running(index, reason),
        VerifyError::Fresh(reason) => Rejection::Fresh(reason),
        other => panic!("unexpected terminal error: {other:?}"),
    }
}

/// One envelope change and the terminal decision it must produce.
struct Case {
    name: &'static str,
    change: fn(&mut Parts),
    expected: Result<(), Rejection>,
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

const LAST: usize = PI_DEC_V1_2_CHILD_COUNT - 1;
const HASH: Rejection = Rejection::Fresh("public input differs from the recomputed terminal state hash");

fn rejected(name: &'static str, change: fn(&mut Parts), rejection: Rejection) -> Case {
    Case {
        name,
        change,
        expected: Err(rejection),
    }
}

fn ignored(name: &'static str, change: fn(&mut Parts)) -> Case {
    Case {
        name,
        change,
        expected: Ok(()),
    }
}

fn cases() -> Vec<Case> {
    use Rejection::{Fresh, Running, Statement};
    vec![
        rejected(
            "state iteration",
            |parts| parts.with_state(parts.state.iteration() + 1, parts.state.z0(), parts.state.current()),
            HASH,
        ),
        rejected(
            "zero state iteration",
            |parts| parts.with_state(0, parts.state.z0(), parts.state.current()),
            Statement("an active proof requires a positive iteration"),
        ),
        rejected(
            "non-canonical state iteration",
            |parts| parts.with_state(F::ORDER_U64, parts.state.z0(), parts.state.current()),
            Statement("iteration is not a canonical Goldilocks counter"),
        ),
        rejected(
            "state z0",
            |parts| {
                let mut z0 = parts.state.z0();
                z0[0] += F::ONE;
                parts.with_state(parts.state.iteration(), z0, parts.state.current())
            },
            HASH,
        ),
        rejected(
            "state current",
            |parts| {
                let mut current = parts.state.current();
                current[0] += F::ONE;
                parts.with_state(parts.state.iteration(), parts.state.z0(), current)
            },
            HASH,
        ),
        rejected(
            "running claim count",
            |parts| drop(parts.running.claims.pop()),
            Statement("running claim or witness count differs from the selected profile"),
        ),
        rejected(
            "running witness count",
            |parts| drop(parts.running.witnesses.pop()),
            Statement("running claim or witness count differs from the selected profile"),
        ),
        rejected(
            "last running commitment d",
            |parts| parts.last_claim().c.d += 1,
            Running(LAST, "commitment or public-input shape"),
        ),
        rejected(
            "last running commitment kappa",
            |parts| parts.last_claim().c.kappa += 1,
            Running(LAST, "commitment or public-input shape"),
        ),
        rejected(
            "first running commitment data",
            |parts| parts.running.claims[0].c.data[0] += F::ONE,
            HASH,
        ),
        rejected(
            "last running commitment data",
            |parts| parts.last_claim().c.data[0] += F::ONE,
            HASH,
        ),
        rejected(
            "first running public input X",
            |parts| parts.running.claims[0].X[(0, 0)] += F::ONE,
            Running(0, "witness public projection differs from X"),
        ),
        rejected(
            "last running public input X",
            |parts| parts.last_claim().X[(0, 0)] += F::ONE,
            Running(LAST, "witness public projection differs from X"),
        ),
        rejected(
            "last running point r length",
            |parts| parts.last_claim().r.push(K::ZERO),
            Running(LAST, "running claims must share the selected evaluation point"),
        ),
        rejected(
            "last running public input X shape",
            |parts| parts.last_claim().X.append_zero_rows(1, F::ZERO),
            Running(LAST, "witness public projection differs from X"),
        ),
        rejected(
            "last running point r",
            |parts| parts.last_claim().r[0] += K::ONE,
            Running(LAST, "running claims must share the selected evaluation point"),
        ),
        rejected(
            "every running point r",
            |parts| {
                for claim in &mut parts.running.claims {
                    claim.r[0] += K::ONE;
                }
            },
            HASH,
        ),
        rejected(
            "first running Eval_K",
            |parts| parts.running.claims[0].eval_k[0] += K::ONE,
            HASH,
        ),
        rejected(
            "last running Eval_K",
            |parts| parts.last_claim().eval_k[0] += K::ONE,
            HASH,
        ),
        rejected(
            "last running Eval_K surplus",
            |parts| parts.last_claim().eval_k[D] = K::ONE,
            Running(LAST, "evaluation shape or nonzero surplus coefficients"),
        ),
        rejected(
            "last running Eval_K width",
            |parts| parts.last_claim().eval_k.push(K::ZERO),
            Running(LAST, "evaluation shape or nonzero surplus coefficients"),
        ),
        rejected(
            "last running Eval_A width",
            |parts| parts.last_claim().eval_a[0].push(K::ZERO),
            Running(LAST, "evaluation shape or nonzero surplus coefficients"),
        ),
        rejected(
            "last running Eval_A",
            |parts| parts.last_claim().eval_a[0][0] += K::ONE,
            HASH,
        ),
        rejected(
            "last running Eval_A surplus",
            |parts| parts.last_claim().eval_a[0][D] = K::ONE,
            Running(LAST, "evaluation shape or nonzero surplus coefficients"),
        ),
        rejected(
            "last running Eval_A count",
            |parts| drop(parts.last_claim().eval_a.pop()),
            Running(LAST, "evaluation shape or nonzero surplus coefficients"),
        ),
        rejected(
            "last running m_in",
            |parts| parts.last_claim().m_in += D,
            Running(LAST, "commitment or public-input shape"),
        ),
        rejected(
            "last running auxiliary commitments",
            |parts| {
                let claim = parts.last_claim();
                claim.adv = Some(lanes(&claim.c));
            },
            Running(LAST, "plain claims cannot carry auxiliary commitments"),
        ),
        rejected(
            "first running witness public coordinate",
            |parts| flip(&mut parts.running.witnesses[0], 0),
            Running(0, "witness public projection differs from X"),
        ),
        rejected(
            "last running witness public coordinate",
            |parts| flip(parts.running.witnesses.last_mut().unwrap(), 0),
            Running(LAST, "witness public projection differs from X"),
        ),
        rejected(
            "last running witness shape",
            |parts| {
                parts
                    .running
                    .witnesses
                    .last_mut()
                    .unwrap()
                    .append_zero_rows(1, F::ZERO)
            },
            Running(LAST, "complete witness shape"),
        ),
        rejected(
            "first running witness private coordinate",
            |parts| {
                let position = parts.running.claims[0].m_in;
                flip(&mut parts.running.witnesses[0], position)
            },
            Running(0, "fixed-key commitment differs from the witness"),
        ),
        rejected(
            "last running witness private coordinate",
            |parts| {
                let position = parts.last_claim().m_in;
                flip(parts.running.witnesses.last_mut().unwrap(), position)
            },
            Running(LAST, "fixed-key commitment differs from the witness"),
        ),
        rejected(
            "fresh commitment d",
            |parts| parts.fresh.claim.c.d += 1,
            Fresh("commitment or public-input shape"),
        ),
        rejected(
            "fresh commitment kappa",
            |parts| parts.fresh.claim.c.kappa += 1,
            Fresh("commitment or public-input shape"),
        ),
        rejected(
            "fresh commitment data",
            |parts| parts.fresh.claim.c.data[0] += F::ONE,
            Fresh("fixed-key commitment differs from the witness"),
        ),
        rejected("fresh public input x", |parts| parts.fresh.claim.x[1] += F::ONE, HASH),
        rejected(
            "fresh public input x length",
            |parts| parts.fresh.claim.x.push(F::ZERO),
            Fresh("commitment or public-input shape"),
        ),
        rejected(
            "fresh m_in",
            |parts| parts.fresh.claim.m_in += D,
            Fresh("commitment or public-input shape"),
        ),
        rejected(
            "fresh auxiliary commitments",
            |parts| parts.fresh.claim.adv = Some(lanes(&parts.fresh.claim.c)),
            Fresh("plain claims cannot carry auxiliary commitments"),
        ),
        rejected(
            "fresh witness public coordinate",
            |parts| flip(&mut parts.fresh.witness.Z, 1),
            Fresh("witness public projection differs from x"),
        ),
        rejected(
            "fresh witness private coordinate",
            |parts| {
                let position = parts.fresh.claim.m_in;
                flip(&mut parts.fresh.witness.Z, position)
            },
            Fresh("fixed-key commitment differs from the witness"),
        ),
        rejected(
            "fresh witness shape",
            |parts| parts.fresh.witness.Z.append_zero_rows(1, F::ZERO),
            Fresh("complete witness shape or nonzero fresh completion tail"),
        ),
        rejected(
            "fresh witness completion tail",
            |parts| {
                let position = parts.fresh.witness.Z.cols() * D - 1;
                flip(&mut parts.fresh.witness.Z, position)
            },
            Fresh("complete witness shape or nonzero fresh completion tail"),
        ),
        ignored("running fold digests", |parts| {
            for claim in &mut parts.running.claims {
                claim.fold_digest[0] ^= 1;
            }
        }),
        ignored("fresh private witness cache w", |parts| {
            parts.fresh.witness.w = vec![F::ONE]
        }),
    ]
}

#[test]
#[ignore = "Production two-step proof and terminal tamper sweep; run separately under the 300-second cap."]
fn terminal_rejects_every_authoritative_envelope_change() {
    let started = Instant::now();
    let initial = [1, 2, 3, 4].map(F::from_u64);
    let message = [5, 6, 7, 8].map(F::from_u64);
    let words = message.map(|word| word.as_canonical_u64());
    let application = crate::application::poseidon2_hash_chain_v1().unwrap();
    let bytes = fs::read(artifact("nightstream-fprime-stage1-poseidon2-hash-chain-v1.json")).unwrap();
    let (source, binding) = crate::assembly::prepare(&bytes, &application).unwrap();
    drop(bytes);
    let package = PreparedLifecycle::from_package(source.into(), binding, Backend::Optimized, 114).unwrap();
    assert_ne!(package.structure.m % D, 0, "the selected carrier has a completion tail");
    let first = output(initial, message);
    let base = package
        .extend_with_output(Stage1Envelope::initial(initial), &words, first, None)
        .unwrap();
    // Fold from the decoded bytes, as a separate prover would. In the base
    // envelope only the state and the fresh instance are nonzero;
    // `proof_bytes_round_trip_exactly` covers nonzero running claims.
    let base = package
        .decode_proof(&package.encode_proof(&base).unwrap())
        .unwrap();
    let second = output(first, message);
    // Prove the second fold once; the honest envelope and the balanced
    // opening changes complete from the same proof.
    let (state, base) = base.into_parts();
    let ProofState::Active { running, fresh } = base else {
        panic!("the base proof is active");
    };
    let fold = package.prove_active(state, running, fresh).unwrap();
    let proof = package
        .complete_fold(fold.clone(), &words, second, None)
        .unwrap();
    let expected = Stage1State::new(2, initial, second);
    package.verify(&expected, &proof).unwrap();
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
        "the folded proof carries a non-default running instance"
    );

    // Each change is verified against the envelope's own state, so the state
    // cases must fail at the state-hash binding, not at the state comparison.
    for case in cases() {
        let mut parts = honest.clone();
        (case.change)(&mut parts);
        let state = parts.state;
        let timer = Instant::now();
        let actual = package.verify(&state, &parts.envelope()).map_err(rejection);
        eprintln!("{}: {actual:?} elapsed={:?}", case.name, timer.elapsed());
        assert_eq!(actual, case.expected, "{}", case.name);
    }
    // A balanced change to two `Π_DEC` child openings keeps the weighted
    // recomposition, so the NIFS verifier accepts it and the fresh witness is
    // valid; only the terminal opening checks can reject the result.
    let radix = K::from(F::from_u64(package.params.b() as u64));
    for opening in [Opening::K, Opening::A] {
        let (name, reason) = labels(opening);
        let mut changed = fold.clone();
        change_openings(&mut changed.proof.pi_dec.children, opening, radix);
        let envelope = package
            .complete_fold(changed, &words, second, None)
            .unwrap();
        let timer = Instant::now();
        let actual = package.verify(&expected, &envelope).map_err(rejection);
        eprintln!("balanced {name}: {actual:?} elapsed={:?}", timer.elapsed());
        assert_eq!(actual, Err(Rejection::Running(0, reason)), "balanced {name}");
    }
    let bottom = Stage1Envelope::from_state_and_proof(expected, ProofState::Initial);
    assert_eq!(
        package.verify(&expected, &bottom).map_err(rejection),
        Err(Rejection::Statement(
            "bottom requires zero iterations and equal endpoints"
        ))
    );
    eprintln!("terminal tamper sweep elapsed={:?}", started.elapsed());
}
