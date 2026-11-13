//! The compressed proof (SPEC.md section 6) on the Poseidon2 hash chain with
//! cap 2: the proof of each circuit verifies from its bytes under the
//! compression key, and a changed statement, a changed proof, a proof from a
//! bad witness and another key do not.
//!
//! The relations have one size, so each test that compresses a step holds
//! the compression setup of 10 GB and needs up to 30 GB. Those tests are
//! ignored. `neo-compress` tests the argument on small claims.

use std::sync::OnceLock;
use std::time::{Duration, Instant};

use super::super::compression::{FinalProof, Fresh, Steps};
use super::super::{BoundedDepth, BoundedDepthError, Envelope};
use super::support::{hash_chain_step, packages, words};
use super::{addition, prover, MINIMUM_SECURITY_BITS};
use crate::folding::transcript::Transcript;
use crate::folding::{pi_ccs, CcsClaim, RunningInstance};
use neo_ajtai::nightstream_fprime_setup::{
    coefficient_block, commit_production_signed_unit_prefix_matrix, PRODUCTION_SEED,
};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_compress::{ProverKey, VerificationKey};
use neo_math::{D, F, K};
use nightstream_fprime::{
    PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS,
};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const STEPS: u64 = 5;

fn z0() -> [F; 4] {
    words([10, 11, 12, 13])
}

/// `STEPS` hash-chain steps from `z0`, and the envelope after each. The
/// circuits are light, light, light, refresh, light.
fn envelopes(prover: &BoundedDepth) -> Vec<Envelope> {
    let mut envelope = prover.initial(z0()).expect("initial proof");
    (0..STEPS)
        .map(|step| {
            let message = words([20 + step, 21 + step, 22 + step, 23 + step]);
            let next = hash_chain_step(envelope.current, message);
            envelope = prover
                .extend(&envelope, &message, next, None)
                .expect("step");
            envelope.clone()
        })
        .collect()
}

/// The compression setup of the hash chain and the time to make it, once
/// for the tests of this binary: it holds about 10 GB. Each prover of these
/// tests has the same relations, so each makes the same setup.
fn setup_and_time(prover: &BoundedDepth) -> &'static (ProverKey, Duration) {
    static SETUP: OnceLock<(ProverKey, Duration)> = OnceLock::new();
    SETUP.get_or_init(|| {
        let start = Instant::now();
        let setup = prover.compression_setup().expect("compression setup");
        (setup, start.elapsed())
    })
}

fn setup(prover: &BoundedDepth) -> &'static ProverKey {
    &setup_and_time(prover).0
}

fn key(prover: &BoundedDepth) -> &'static VerificationKey {
    setup(prover).verification_key()
}

/// `proof` after its bytes, when they decode.
fn through_bytes(proof: &FinalProof) -> Option<FinalProof> {
    FinalProof::from_bytes(&proof.to_bytes().expect("bytes")).ok()
}

fn accepts(prover: &BoundedDepth, envelope: &Envelope, proof: &FinalProof) -> bool {
    prover
        .verify_final(key(prover), envelope.iteration, envelope.z0, envelope.current, proof)
        .is_ok()
}

fn milliseconds(duration: Duration) -> f64 {
    duration.as_secs_f64() * 1e3
}

fn increment(commitment: &mut Commitment, word: usize) {
    let mut data = commitment.data().to_vec();
    data[word] += F::ONE;
    *commitment = Commitment::new(commitment.kappa(), data).expect("commitment");
}

/// The proofs of the refresh circuit and of the light circuit verify from
/// their bytes under the key from its bytes. The printed lines give the
/// times of one run and the sizes.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_compressed_proof_verifies_and_a_changed_one_does_not() {
    let prover = prover(2);
    // `neo-compress` tests its key reduction on this number of blocks.
    assert_eq!(prover.relations[0].blocks(), 74_530);
    let envelopes = envelopes(&prover);
    let (setup, setup_time) = setup_and_time(&prover);
    let key_bytes = setup.verification_key().to_bytes();
    let key = VerificationKey::from_bytes(&key_bytes).expect("decoded key");
    assert_eq!(&key, setup.verification_key());
    println!(
        "compression setup {:.0} ms on {} threads: the key polynomial holds {} blocks of each of the 18 key rows, \
         verification key {} bytes",
        milliseconds(*setup_time),
        rayon::current_num_threads(),
        key.blocks(),
        key_bytes.len(),
    );
    let single = rayon::ThreadPoolBuilder::new()
        .num_threads(1)
        .build()
        .expect("thread pool");
    let mut last = None;
    for envelope in &envelopes[STEPS as usize - 2..] {
        let layout = prover
            .compression_layout(
                key.blocks(),
                envelope.pc,
                envelope.slots.each_ref().map(|slot| slot.depth),
            )
            .expect("layout");
        let start = Instant::now();
        let (proof, times) = prover.compress(setup, envelope).expect("compressed proof");
        let compress = start.elapsed();
        let bytes = proof.to_bytes().expect("bytes");
        let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
        let verify = || {
            let start = Instant::now();
            let times = prover
                .verify_final(&key, envelope.iteration, envelope.z0, envelope.current, &decoded)
                .unwrap_or_else(|error| panic!("the proof of circuit {} verifies: {error}", envelope.pc));
            (start.elapsed(), times)
        };
        let (verify_time, verify_times) = verify();
        let (single_time, single_times) = single.install(verify);
        let argument = &decoded.steps.as_ref().expect("steps").argument;
        let error_bits = neo_compress::error_bits(&layout, PUBLIC_WORDS, PI_CCS_V1_2_MATRIX_COUNT);
        assert!(error_bits > 119.0, "the argument's own error is 2^-{error_bits:.2}");
        let commitments = argument.witness_commitment.len() + argument.inverse_commitment.len();
        let sumchecks = 16 * (4 * argument.rounds.len() + 3 * argument.key_rounds.len());
        let values = 16 * (1 + argument.inverse_values.len() + 2);
        let lengths = 8 * 5;
        let akita = commitments + argument.opening.len();
        println!(
            "compressed proof, circuit {}: {} witness cells, W has {} variables, V has {} variables, {} bytes: \
             Akita opening {}, Akita commitments {}, sum-check rounds {} ({} and {} rounds), values {}, \
             claims and PiCCS {}",
            envelope.pc,
            layout.witness_cells(),
            layout.witness_vars(),
            layout.inverse_vars(),
            bytes.len(),
            argument.opening.len(),
            commitments,
            sumchecks,
            argument.rounds.len(),
            argument.key_rounds.len(),
            values,
            bytes.len() - akita - sumchecks - values - lengths,
        );
        println!("  argument error without PiCCS and the Akita opening: 2^-{error_bits:.2}");
        println!("  compress {:.0} ms: {times}", milliseconds(compress));
        println!(
            "  verify_final {:.1} ms on {} threads: {verify_times}",
            milliseconds(verify_time),
            rayon::current_num_threads(),
        );
        println!(
            "  verify_final {:.1} ms on 1 thread: {single_times}",
            milliseconds(single_time)
        );
        last = Some((envelope, proof));
    }
    let (envelope, proof) = last.expect("the last proof");
    prover
        .verify(envelope.iteration, envelope.z0, envelope.current, envelope)
        .expect("the proof with its witnesses verifies");

    let mut other = envelope.current;
    other[0] += F::ONE;
    assert!(prover
        .verify_final(&key, envelope.iteration, envelope.z0, other, &proof)
        .is_err());
    assert!(prover
        .verify_final(&key, envelope.iteration + 1, envelope.z0, envelope.current, &proof)
        .is_err());

    let changes: Vec<(&str, Box<dyn Fn(&mut FinalProof)>)> = vec![
        ("iteration", Box::new(|proof| proof.iteration += 1)),
        ("initial state", Box::new(|proof| proof.z0[3] += F::ONE)),
        ("circuit", Box::new(|proof| proof.steps.as_mut().unwrap().pc = 2)),
        (
            "slot 0 depth",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots[0].1 += 1),
        ),
        (
            "slot 1 depth",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots[1].1 += 1),
        ),
        (
            "slot 0 commitment word",
            Box::new(|proof| increment(&mut proof.steps.as_mut().unwrap().slots[0].0.c, 70)),
        ),
        (
            "slot 1 public input word",
            Box::new(|proof| {
                let claim = &mut proof.steps.as_mut().unwrap().slots[1].0;
                let mut cells = claim.X.to_dense_vec();
                cells[9] += F::ONE;
                claim.X = Mat::from_row_major(D, cells.len() / D, cells);
            }),
        ),
        (
            "slot 0 point",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots[0].0.r[27] += K::ONE),
        ),
        (
            "slot 0 Eval_K entry",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots[0].0.eval_k[63] += K::ONE),
        ),
        (
            "slot 1 Eval_A entry",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots[1].0.eval_a[3][0] += K::ONE),
        ),
        (
            "fresh commitment word",
            Box::new(|proof| increment(&mut proof.steps.as_mut().unwrap().fresh.c, 1151)),
        ),
        (
            "fresh point",
            Box::new(|proof| proof.steps.as_mut().unwrap().fresh.r[0] += K::ONE),
        ),
        (
            "fresh Eval_K entry",
            Box::new(|proof| proof.steps.as_mut().unwrap().fresh.eval_k[1] += K::ONE),
        ),
        (
            "fresh Eval_A entry",
            Box::new(|proof| proof.steps.as_mut().unwrap().fresh.eval_a[0][17] += K::ONE),
        ),
        (
            "fresh digest",
            Box::new(|proof| proof.steps.as_mut().unwrap().fresh.fold_digest[8] ^= 1),
        ),
        (
            "PiCCS round coefficient",
            Box::new(|proof| proof.steps.as_mut().unwrap().pi_ccs.sumcheck_rounds[27][8] += K::ONE),
        ),
        (
            "sum-check round coefficient",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.rounds[13][2] += K::ONE),
        ),
        (
            "last sum-check round",
            Box::new(|proof| {
                let rounds = &mut proof.steps.as_mut().unwrap().argument.rounds;
                rounds.last_mut().unwrap()[3] += K::ONE;
            }),
        ),
        (
            "witness value",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.witness_value += K::ONE),
        ),
        (
            "inverse value",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.inverse_values[2] += K::ONE),
        ),
        (
            "witness commitment byte",
            Box::new(|proof| {
                *proof
                    .steps
                    .as_mut()
                    .unwrap()
                    .argument
                    .witness_commitment
                    .last_mut()
                    .unwrap() ^= 1
            }),
        ),
        (
            "inverse commitment byte",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.inverse_commitment[40] ^= 1),
        ),
        (
            "the key's part of the weights",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.key_weight += K::ONE),
        ),
        (
            "key sum-check round coefficient",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.key_rounds[20][1] += K::ONE),
        ),
        (
            "last key sum-check round",
            Box::new(|proof| {
                let rounds = &mut proof.steps.as_mut().unwrap().argument.key_rounds;
                rounds.last_mut().unwrap()[2] += K::ONE;
            }),
        ),
        (
            "the value of the key polynomial",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.key_value += K::ONE),
        ),
        (
            "opening byte",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.opening[30_000] ^= 1),
        ),
        (
            "last opening byte",
            Box::new(|proof| {
                *proof
                    .steps
                    .as_mut()
                    .unwrap()
                    .argument
                    .opening
                    .last_mut()
                    .unwrap() ^= 1
            }),
        ),
        (
            "opening with one more byte",
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.opening.push(0)),
        ),
        (
            "the slot claims in the other order",
            Box::new(|proof| proof.steps.as_mut().unwrap().slots.swap(0, 1)),
        ),
    ];
    for (name, change) in changes {
        let mut changed = proof.clone();
        change(&mut changed);
        let decoded = through_bytes(&changed).unwrap_or_else(|| panic!("{name}: the changed proof decodes"));
        assert!(!accepts(&prover, envelope, &decoded), "{name}");
    }
}

/// A final proof with the slot claims of `envelope`, the claim of slot 1 as
/// its fresh claim, and a PiCCS proof and an argument that prove nothing.
/// The bytes of a proof, and the checks of `verify_final` before the
/// argument, read no more.
fn unproven(envelope: &Envelope) -> FinalProof {
    let slots = envelope.slots.each_ref();
    let claim = &slots[1].accumulator.claim;
    FinalProof {
        iteration: envelope.iteration,
        z0: envelope.z0,
        current: envelope.current,
        steps: Some(Steps {
            pc: envelope.pc,
            slots: slots.map(|slot| (slot.accumulator.claim.clone(), slot.depth)),
            fresh: Fresh {
                c: claim.c.clone(),
                r: claim.r.clone(),
                eval_k: claim.eval_k.clone(),
                eval_a: claim.eval_a.clone(),
                fold_digest: [0; 32],
            },
            pi_ccs: pi_ccs::SumcheckProof::new(vec![
                vec![K::ONE; PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT];
                PI_CCS_V1_2_ROUND_COUNT
            ]),
            argument: neo_compress::Proof {
                witness_commitment: vec![1; 471],
                inverse_commitment: vec![2; 471],
                rounds: vec![[K::ONE; 4]; 26],
                witness_value: K::ONE,
                inverse_values: [K::ONE; 4],
                key_weight: K::ONE,
                key_rounds: vec![[K::ONE; 3]; 25],
                key_value: K::ONE,
                opening: vec![3; 1000],
            },
        }),
    }
}

/// A final proof has one encoding, and `verify_final` checks the circuit
/// and each slot depth before it reads the argument. The proof of this test
/// proves nothing, so the test needs no compression setup of the hash chain.
#[test]
fn a_final_proof_has_one_encoding_and_a_circuit_and_depths_in_range() {
    let prover = prover(2);
    let setup = small_setup();
    let key = setup.verification_key();
    let initial = prover.initial(z0()).expect("initial proof");
    let message = words([20, 21, 22, 23]);
    let envelope = &prover
        .extend(&initial, &message, hash_chain_step(initial.current, message), None)
        .expect("step");
    let proof = unproven(envelope);
    let bytes = proof.to_bytes().expect("bytes");
    let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
    assert_eq!(decoded.to_bytes().expect("bytes"), bytes);
    // The key of 512 blocks holds no relation of the hash chain.
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &decoded),
        Err(BoundedDepthError::Compression(neo_compress::Error::Layout(_)))
    ));

    // A depth above the light fold cap of 64 has no accumulator bound.
    let mut deep = proof.clone();
    deep.steps.as_mut().unwrap().slots[0].1 = 65;
    let deep = through_bytes(&deep).expect("depth 65 decodes");
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &deep),
        Err(BoundedDepthError::Slot { slot: 0, .. })
    ));
    // The circuits are 1 and 2.
    let mut third = proof.clone();
    third.steps.as_mut().unwrap().pc = 3;
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &third),
        Err(BoundedDepthError::Statement(_))
    ));
    assert!(through_bytes(&third).is_none(), "circuit 3 decodes");

    // A length above the bytes that follow it decodes nothing.
    let argument = &proof.steps.as_ref().unwrap().argument;
    let commitment = &argument.witness_commitment[..];
    let length_at = bytes
        .windows(commitment.len())
        .position(|window| window == commitment)
        .expect("the witness commitment")
        - 8;
    let rounds_at = length_at + 16 + commitment.len() + argument.inverse_commitment.len();
    let key_rounds_at = rounds_at + 8 + 64 * argument.rounds.len() + 16 * 6;
    let opening_at = key_rounds_at + 8 + 48 * argument.key_rounds.len() + 16;
    assert_eq!(opening_at + 8 + argument.opening.len(), bytes.len());
    for (name, at, honest) in [
        ("witness commitment length", length_at, commitment.len()),
        ("sum-check round count", rounds_at, argument.rounds.len()),
        ("key sum-check round count", key_rounds_at, argument.key_rounds.len()),
        ("opening length", opening_at, argument.opening.len()),
    ] {
        assert_eq!(bytes[at..at + 8], (honest as u64).to_le_bytes(), "{name}");
        for huge in [u64::MAX, 1 << 40, bytes.len() as u64] {
            let mut changed = bytes.clone();
            changed[at..at + 8].copy_from_slice(&huge.to_le_bytes());
            assert!(FinalProof::from_bytes(&changed).is_err(), "{name} {huge}");
        }
    }

    // A proof has one encoding.
    assert!(FinalProof::from_bytes(&bytes[..bytes.len() - 1]).is_err(), "truncated");
    let mut longer = bytes.clone();
    longer.push(0);
    assert!(FinalProof::from_bytes(&longer).is_err(), "a byte after the proof");
    for (name, at) in [("iteration", 16 + 8), ("slot 0 depth", 16 + 80)] {
        let word = u64::from_le_bytes(bytes[at..at + 8].try_into().unwrap());
        let mut aliased = bytes.clone();
        aliased[at..at + 8].copy_from_slice(&(word + F::ORDER_U64).to_le_bytes());
        assert!(FinalProof::from_bytes(&aliased).is_err(), "{name} word {word} + q");
    }
}

/// A prover whose envelope does not verify gets no proof that `verify_final`
/// accepts. The argument is made for a bad slot witness, and the verifier
/// rejects it. PiCCS makes no proof for a bad fresh witness.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_bad_witness_gives_no_accepted_proof() {
    let prover = prover(2);
    let envelope = envelopes(&prover).pop().expect("the last proof");
    let invalid = |name: &str, bad: &Envelope| {
        assert!(
            prover
                .verify(bad.iteration, bad.z0, bad.current, bad)
                .is_err(),
            "{name}: the envelope verifies"
        );
    };

    // One coefficient of slot 0 goes one above the bound of its depth.
    let mut above = envelope.clone();
    let slot = &mut above.slots[0];
    let bound = prover.accumulator_bound(slot.depth);
    assert!(prover.slot_within_depth_bound(slot).is_ok());
    slot.accumulator.witness.as_mut_slice()[12_345] = F::from_u64(bound + 1);
    assert!(prover.slot_within_depth_bound(slot).is_err());
    // One coefficient of slot 1 changes inside the bound: no key row opens
    // the slot's commitment.
    let mut moved = envelope.clone();
    let cell = &mut moved.slots[1].accumulator.witness.as_mut_slice()[777];
    *cell = if *cell == F::ZERO { F::ONE } else { F::ZERO };
    for (name, bad) in [
        ("slot 0 coefficient above its bound", &above),
        ("slot 1 coefficient", &moved),
    ] {
        invalid(name, bad);
        let (proof, _) = prover
            .compress(setup(&prover), bad)
            .unwrap_or_else(|error| panic!("{name}: the prover does not check a slot witness: {error}"));
        assert!(!accepts(&prover, bad, &proof), "{name}");
    }

    // One entry of the fresh witness is 2.
    let mut two = envelope.clone();
    let fresh = two.fresh.as_mut().expect("fresh instance");
    let blocks = fresh.witness.Z.cols();
    let mut cells = fresh.witness.Z.to_dense_vec();
    cells[3 * blocks + PUBLIC_WORDS / D + 100] = F::TWO;
    fresh.witness.Z = Mat::from_row_major(D, blocks, cells);
    invalid("fresh entry 2", &two);
    assert!(prover.compress(setup(&prover), &two).is_err(), "fresh entry 2");
}

/// The proof of a prover that commits `committed` as the fresh witness in
/// `W`, and makes PiCCS for the fresh witness of `envelope` under the
/// commitment `c`. With the witness and the commitment of `envelope` it is
/// the proof of `compress`.
fn compress_committing(prover: &BoundedDepth, envelope: &Envelope, committed: &Mat<F>, c: &Commitment) -> FinalProof {
    let fresh = envelope.fresh.as_ref().expect("fresh instance");
    let (pc, slots) = (envelope.pc, envelope.slots.each_ref());
    let layout = prover
        .compression_layout(key(prover).blocks(), pc, slots.map(|slot| slot.depth))
        .expect("layout");
    let witnesses = [&slots[0].accumulator.witness, &slots[1].accumulator.witness, committed];
    let committed = neo_compress::commit(setup(prover), layout, &witnesses).expect("commitment of W");
    let commitment = neo_compress::byte_words(committed.commitment());
    let mut transcript = Transcript::session();
    let claim = CcsClaim {
        c: c.clone(),
        ..fresh.claim.clone()
    };
    let mut ccs = pi_ccs::prove_from_parts_with_rows(
        &mut transcript,
        &prover.backend,
        prover.relations[(pc - 1) as usize].context(),
        &BoundedDepth::compression_profile(&commitment),
        &claim,
        &fresh.witness,
        &RunningInstance::default(),
    )
    .expect("PiCCS proof");
    let output = ccs.outputs.swap_remove(0);
    let statements =
        prover.compression_statements(pc, slots.map(|slot| (&slot.accumulator.claim, slot.depth)), &output);
    let (argument, _) = committed
        .prove(transcript.inner_mut(), &statements)
        .expect("the prover does not check its witness");
    FinalProof {
        iteration: envelope.iteration,
        z0: envelope.z0,
        current: envelope.current,
        steps: Some(Steps {
            pc,
            slots: slots.map(|slot| (slot.accumulator.claim.clone(), slot.depth)),
            fresh: Fresh {
                c: output.c,
                r: output.r,
                eval_k: output.eval_k,
                eval_a: output.eval_a,
                fold_digest: output.fold_digest,
            },
            pi_ccs: ccs.sumcheck,
            argument,
        }),
    }
}

/// The commitment of the witness of `c` with `delta` added in block `block`,
/// lane `lane`: each row gets `delta · X^lane` times its key element.
fn moved_commitment(c: &Commitment, block: usize, lane: usize, delta: F) -> Commitment {
    let mut data = c.data().to_vec();
    for (row, coefficients) in data.chunks_exact_mut(D).enumerate() {
        let key = coefficient_block(&PRODUCTION_SEED, row as u32, block as u64);
        for (index, key) in key.into_iter().enumerate() {
            let term = delta * F::new(key);
            if index + lane < D {
                coefficients[index + lane] += term;
            } else {
                coefficients[index + lane - D] -= term;
            }
        }
    }
    Commitment::new(c.kappa(), data).expect("commitment")
}

/// PiCCS proves the rows and the entries {-1, 0, 1} of a witness, and the
/// argument proves that `W` holds that witness: the evaluations of the PiCCS
/// claim are at a point that follows the commitment of `W`. So a prover that
/// commits another fresh witness gets no accepted proof. It makes PiCCS for
/// the honest witness, under the honest Ajtai commitment or under the Ajtai
/// commitment of the witness in `W`, and the argument rejects each.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_committed_fresh_witness_that_piccs_did_not_judge_is_rejected() {
    let prover = prover(2);
    let envelope = envelopes(&prover).pop().expect("the last proof");
    let fresh = envelope.fresh.as_ref().expect("fresh instance");
    let (honest, c) = (&fresh.witness.Z, &fresh.claim.c);
    let proof = compress_committing(&prover, &envelope, honest, c);
    prover
        .verify_final(key(&prover), envelope.iteration, envelope.z0, envelope.current, &proof)
        .expect("the prover of this test makes an accepted proof");

    let blocks = honest.cols();
    let (lane, block) = (3, PUBLIC_WORDS / D + 100);
    let cells = honest.to_dense_vec();
    let old = cells[lane * blocks + block];
    let with = |value: F| {
        let mut cells = cells.clone();
        cells[lane * blocks + block] = value;
        Mat::from_row_major(D, blocks, cells)
    };
    let unit = if old == F::ZERO { F::ONE } else { F::ZERO };
    assert_eq!(
        moved_commitment(c, block, lane, unit - old),
        commit_production_signed_unit_prefix_matrix(&with(unit)).expect("native commitment")
    );
    let cases = [
        ("another unit under the honest commitment", unit, c.clone()),
        (
            "another unit under its own commitment",
            unit,
            moved_commitment(c, block, lane, unit - old),
        ),
        ("the entry 2 under the honest commitment", F::TWO, c.clone()),
        (
            "the entry 2 under its own commitment",
            F::TWO,
            moved_commitment(c, block, lane, F::TWO - old),
        ),
    ];
    for (name, value, c) in cases {
        let proof = compress_committing(&prover, &envelope, &with(value), &c);
        let proof = through_bytes(&proof).unwrap_or_else(|| panic!("{name}: the proof decodes"));
        let result = prover.verify_final(key(&prover), envelope.iteration, envelope.z0, envelope.current, &proof);
        assert!(
            matches!(
                result,
                Err(BoundedDepthError::Compression(neo_compress::Error::Rejected(_)))
            ),
            "{name}: {result:?}"
        );
    }
}

/// The state hash holds the key of each circuit, so a verifier of other keys
/// rejects the proof: with the same relations and one other key word, and
/// with the addition application in place of the hash chain.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_verifier_of_other_keys_rejects_the_proof() {
    let prover = prover(2);
    let envelope = envelopes(&prover).pop().expect("the last proof");
    let (proof, _) = prover
        .compress(setup(&prover), &envelope)
        .expect("compressed proof");
    assert!(accepts(&prover, &envelope, &proof));

    let mut other_key = super::prover(2);
    other_key.keys[0][0] += F::ONE;
    assert!(matches!(
        other_key.verify_final(key(&prover), envelope.iteration, envelope.z0, envelope.current, &proof),
        Err(BoundedDepthError::Fresh(_))
    ));

    let application = addition();
    let packages = packages()
        .with_application(
            application.records().clone(),
            application.private_input_count(),
            application.shape(),
        )
        .expect("packages with the application");
    let other_circuit = BoundedDepth::new(packages, MINIMUM_SECURITY_BITS, crate::engine::Backend::Optimized)
        .expect("bounded-depth verifier");
    assert_ne!(other_circuit.keys, prover.keys);
    assert!(!accepts(&other_circuit, &envelope, &proof));
}

/// The compression key is authority, and anyone computes it from the public
/// seeds. The key that a verifier derives is the key of the setup. A proof
/// does not verify under the key of a smaller key polynomial, which does not
/// hold the blocks of the relations, or under a key with one changed byte.
#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_proof_does_not_verify_under_another_compression_key() {
    let prover = prover(2);
    let envelope = envelopes(&prover).pop().expect("the last proof");
    let (proof, _) = prover
        .compress(setup(&prover), &envelope)
        .expect("compressed proof");
    let start = Instant::now();
    let derived = prover.compression_key().expect("derived key");
    println!(
        "compression_key {:.0} ms on {} threads",
        milliseconds(start.elapsed()),
        rayon::current_num_threads()
    );
    assert_eq!(&derived, key(&prover));
    prover
        .verify_final(&derived, envelope.iteration, envelope.z0, envelope.current, &proof)
        .expect("the proof verifies under the derived key");

    let small = VerificationKey::derive(512).expect("the key of 512 blocks");
    assert!(small.blocks() < prover.relations[0].blocks());
    assert!(matches!(
        prover.verify_final(&small, envelope.iteration, envelope.z0, envelope.current, &proof),
        Err(BoundedDepthError::Compression(neo_compress::Error::Layout(_)))
    ));

    let bytes = key(&prover).to_bytes();
    for at in [16, 24, 40, 300, bytes.len() - 600, bytes.len() - 300, bytes.len() - 1] {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        if let Ok(other) = VerificationKey::from_bytes(&changed) {
            assert_ne!(&other, key(&prover), "byte {at}");
            let result = prover.verify_final(&other, envelope.iteration, envelope.z0, envelope.current, &proof);
            assert!(result.is_err(), "byte {at}");
        }
    }
}

/// The key polynomial of 512 blocks, which holds no relation of the hash
/// chain.
fn small_setup() -> ProverKey {
    ProverKey::new(512).expect("the key of 512 blocks")
}

/// Zero steps: the proof has no argument and reads no key, and it verifies
/// exactly for iteration 0 with the current state `z0`.
#[test]
fn the_proof_of_zero_steps_compresses_to_its_statement() {
    let prover = prover(2);
    let setup = small_setup();
    let key = setup.verification_key();
    let envelope = prover.initial(z0()).expect("initial proof");
    let (proof, _) = prover
        .compress(&setup, &envelope)
        .expect("compressed proof");
    let bytes = proof.to_bytes().expect("bytes");
    assert_eq!(bytes.len(), 16 + 8 * 10);
    let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
    prover
        .verify_final(key, 0, z0(), z0(), &decoded)
        .expect("zero steps verify");
    assert!(prover.verify_final(key, 1, z0(), z0(), &decoded).is_err());
    let other = words([10, 11, 12, 14]);
    assert!(prover.verify_final(key, 0, z0(), other, &decoded).is_err());
    assert!(prover.verify_final(key, 0, other, other, &decoded).is_err());
    // An initial proof for another current state does not decode.
    let mut moved = bytes.clone();
    moved[bytes.len() - 8] ^= 1;
    assert!(FinalProof::from_bytes(&moved).is_err());
}

/// The compression setup holds the blocks of the larger relation up to a
/// power of two. A setup of fewer blocks than a relation gives no proof of
/// a step.
#[test]
fn a_setup_that_does_not_hold_the_relations_gives_no_proof() {
    let prover = prover(2);
    assert!(prover.relations[0].blocks() <= prover.relations[1].blocks());
    assert_eq!(prover.key_blocks(), prover.relations[1].blocks());
    let setup = small_setup();
    assert!(setup.verification_key().blocks() < prover.relations[0].blocks());
    let initial = prover.initial(z0()).expect("initial proof");
    let message = words([20, 21, 22, 23]);
    let envelope = prover
        .extend(&initial, &message, hash_chain_step(initial.current, message), None)
        .expect("step");
    assert!(matches!(
        prover.compress(&setup, &envelope),
        Err(BoundedDepthError::Compression(neo_compress::Error::Layout(_)))
    ));
}
