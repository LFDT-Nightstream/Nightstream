//! SPEC.md section 6.

use std::sync::{Arc, OnceLock};
use std::time::{Duration, Instant};

use super::super::compression::{FinalProof, Fresh, Steps};
use super::super::{BoundedDepth, BoundedDepthError, CompiledSteps, Compression, Envelope};
use super::support::{fold_gate_bits, hash_chain_step, packages, words};
use super::{addition, prover};
use crate::folding::transcript::Transcript;
use crate::folding::{pi_ccs, CcsClaim, RunningInstance};
use neo_ajtai::nightstream_fprime_setup::{
    coefficient_block, commit_production_signed_unit_prefix_matrix, PRODUCTION_SEED,
};
use neo_ajtai::Commitment;
use neo_ccs::Mat;
use neo_math::{D, F, K};
use nightstream_fprime::{
    PI_CCS_V1_2_MATRIX_COUNT, PI_CCS_V1_2_ROUND_COEFFICIENT_COUNT, PI_CCS_V1_2_ROUND_COUNT, PUBLIC_WORDS,
};
use ns_compress::{ProverKey, VerificationKey};
use p3_field::{PrimeCharacteristicRing, PrimeField64};

const STEPS: u64 = 5;

fn z0() -> [F; 4] {
    words([10, 11, 12, 13])
}

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

fn setup_and_time(prover: &BoundedDepth) -> &'static (Arc<ProverKey>, Duration) {
    static SETUP: OnceLock<(Arc<ProverKey>, Duration)> = OnceLock::new();
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

#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_compressed_proof_verifies_and_a_changed_one_does_not() {
    let prover = prover(2);
    assert_eq!(prover.relations[0].blocks(), 74_146);
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
                &envelope.slots.each_ref().map(|slot| slot.depth),
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
        let error_bits = ns_compress::error_bits(&layout, PUBLIC_WORDS, PI_CCS_V1_2_MATRIX_COUNT);
        assert!(error_bits >= 119, "the argument's own error is at most 2^-{error_bits}");
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
        println!("  argument error without PiCCS and the Akita opening: at most 2^-{error_bits}");
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
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.key_part += K::ONE),
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
            Box::new(|proof| proof.steps.as_mut().unwrap().argument.key_polynomial_value += K::ONE),
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

fn unproven(envelope: &Envelope) -> FinalProof {
    let slots = envelope.slots.each_ref();
    let claim = &slots[1].accumulator.claim;
    FinalProof {
        iteration: envelope.iteration,
        z0: envelope.z0,
        current: envelope.current,
        steps: Some(Steps {
            pc: envelope.pc,
            slots: slots
                .iter()
                .map(|slot| (slot.accumulator.claim.clone(), slot.depth))
                .collect(),
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
            argument: ns_compress::Proof {
                witness_commitment: vec![1; 471],
                inverse_commitment: vec![2; 471],
                rounds: vec![[K::ONE; 4]; 26],
                witness_value: K::ONE,
                inverse_values: [K::ONE; 4],
                key_part: K::ONE,
                key_rounds: vec![[K::ONE; 3]; 25],
                key_polynomial_value: K::ONE,
                opening: vec![3; 1000],
            },
        }),
    }
}

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
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &decoded),
        Err(BoundedDepthError::Compression(ns_compress::Error::Key(_)))
    ));
    assert!(matches!(
        prover.decode_compression_key(&key.to_bytes()),
        Err(BoundedDepthError::Compression(ns_compress::Error::Key(_)))
    ));

    let mut deep = proof.clone();
    deep.steps.as_mut().unwrap().slots[0].1 = 65;
    let deep = through_bytes(&deep).expect("depth 65 decodes");
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &deep),
        Err(BoundedDepthError::Slot { slot: 0, .. })
    ));
    let mut third = proof.clone();
    third.steps.as_mut().unwrap().pc = 3;
    assert!(matches!(
        prover.verify_final(key, envelope.iteration, envelope.z0, envelope.current, &third),
        Err(BoundedDepthError::Statement(_))
    ));
    assert!(third.to_bytes().is_err(), "circuit 3 encodes");
    let mut tagged = bytes.clone();
    tagged[16..24].copy_from_slice(&3u64.to_le_bytes());
    assert!(FinalProof::from_bytes(&tagged).is_err(), "circuit 3 decodes");
    let mut unknown = proof.clone();
    unknown.steps.as_mut().unwrap().pc = 0;
    assert!(unknown.to_bytes().is_err(), "circuit 0 with steps encodes");
    let mut far = proof.clone();
    far.steps.as_mut().unwrap().slots[1].1 = F::ORDER_U64;
    assert!(far.to_bytes().is_err(), "a depth of q encodes");
    let mut late = proof.clone();
    late.iteration = F::ORDER_U64;
    assert!(late.to_bytes().is_err(), "an iteration of q encodes");
    let moved = FinalProof {
        iteration: 1,
        z0: envelope.z0,
        current: envelope.z0,
        steps: None,
    };
    assert!(moved.to_bytes().is_err(), "zero steps at iteration 1 encode");

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

    let mut above = envelope.clone();
    let slot = &mut above.slots[0];
    let bound = prover.accumulator_bound(slot.depth);
    assert!(prover.slot_within_depth_bound(slot).is_ok());
    slot.accumulator.witness.as_mut_slice()[12_345] = F::from_u64(bound + 1);
    assert!(prover.slot_within_depth_bound(slot).is_err());
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

    let mut two = envelope.clone();
    let fresh = two.fresh.as_mut().expect("fresh instance");
    let blocks = fresh.witness.Z.cols();
    let mut cells = fresh.witness.Z.to_dense_vec();
    cells[3 * blocks + PUBLIC_WORDS / D + 100] = F::TWO;
    fresh.witness.Z = Mat::from_row_major(D, blocks, cells);
    invalid("fresh entry 2", &two);
    assert!(prover.compress(setup(&prover), &two).is_err(), "fresh entry 2");
}

fn compress_committing(prover: &BoundedDepth, envelope: &Envelope, committed: &Mat<F>, c: &Commitment) -> FinalProof {
    let fresh = envelope.fresh.as_ref().expect("fresh instance");
    let (pc, slots) = (envelope.pc, envelope.slots.each_ref());
    let layout = prover
        .compression_layout(key(prover).blocks(), pc, &slots.map(|slot| slot.depth))
        .expect("layout");
    let witnesses = [&slots[0].accumulator.witness, &slots[1].accumulator.witness, committed];
    let committed = ns_compress::commit(setup(prover), layout, &witnesses).expect("commitment of W");
    let commitment = ns_compress::byte_words(committed.commitment_to_bind());
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
        prover.compression_statements(pc, &slots.map(|slot| (&slot.accumulator.claim, slot.depth)), &output);
    let (argument, _) = committed
        .prove(transcript.inner_mut(), &statements)
        .expect("the prover does not check its witness");
    FinalProof {
        iteration: envelope.iteration,
        z0: envelope.z0,
        current: envelope.current,
        steps: Some(Steps {
            pc,
            slots: slots
                .iter()
                .map(|slot| (slot.accumulator.claim.clone(), slot.depth))
                .collect(),
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
                Err(BoundedDepthError::Compression(ns_compress::Error::Rejected(_)))
            ),
            "{name}: {result:?}"
        );
    }
}

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
    let other_circuit = BoundedDepth::new(
        &CompiledSteps::new(packages).expect("compiled steps"),
        fold_gate_bits(),
        crate::engine::Backend::Optimized,
    )
    .expect("bounded-depth verifier");
    assert_ne!(other_circuit.keys, prover.keys);
    assert!(!accepts(&other_circuit, &envelope, &proof));
}

#[test]
#[ignore = "Production-size compression, 30 GB; run separately under the 300-second cap."]
fn a_proof_does_not_verify_under_another_compression_key() {
    let prover = prover(2);
    let envelope = envelopes(&prover).pop().expect("the last proof");
    let (proof, _) = prover
        .compress(setup(&prover), &envelope)
        .expect("compressed proof");
    let pinned = prover.compression_key().expect("pinned key");
    assert_eq!(&pinned, key(&prover));
    prover
        .verify_final(&pinned, envelope.iteration, envelope.z0, envelope.current, &proof)
        .expect("the proof verifies under the pinned key");
    assert_eq!(
        &prover
            .decode_compression_key(&pinned.to_bytes())
            .expect("decoded key"),
        &pinned
    );

    let small = VerificationKey::new(512).expect("the key of 512 blocks");
    assert!(small.blocks() < prover.relations[0].blocks());
    assert!(matches!(
        prover.verify_final(&small, envelope.iteration, envelope.z0, envelope.current, &proof),
        Err(BoundedDepthError::Compression(ns_compress::Error::Key(_)))
    ));

    let bytes = pinned.to_bytes();
    for at in 0..bytes.len() {
        let mut changed = bytes.clone();
        changed[at] ^= 1;
        assert!(prover.decode_compression_key(&changed).is_err(), "byte {at}");
    }
}

fn small_setup() -> ProverKey {
    ProverKey::new(512).expect("the key of 512 blocks")
}

/// SPEC.md section 4.5, Algorithms 9 and 10; Lean `FinalVerifier.ModeChecks`.
#[test]
fn a_light_final_proof_carries_slot_zero_alone_and_needs_its_mode() {
    let both = prover(2);
    let light = prover(2).with_compression(Compression::Light);
    assert_eq!(light.key_blocks(), light.relations[0].blocks());
    assert_eq!(
        both.key_blocks(),
        both.relations[0].blocks().max(both.relations[1].blocks())
    );
    let setup = small_setup();
    let key = setup.verification_key();
    let initial = light.initial(z0()).expect("initial proof");
    let message = words([20, 21, 22, 23]);
    let envelope = &light
        .extend(&initial, &message, hash_chain_step(initial.current, message), None)
        .expect("step");
    let full = unproven(envelope);
    let mut proof = full.clone();
    proof.steps.as_mut().unwrap().slots.truncate(1);
    let bytes = proof.to_bytes().expect("bytes");
    assert_eq!(&bytes[..16], b"NS-OPTIONB-FIN-Z");
    let decoded = FinalProof::from_bytes(&bytes).expect("decoded proof");
    assert_eq!(decoded.steps.as_ref().unwrap().slots.len(), 1);
    assert_eq!(decoded.to_bytes().expect("bytes"), bytes);
    let verify = |verifier: &BoundedDepth, proof: &FinalProof| {
        verifier.verify_final(key, envelope.iteration, envelope.z0, envelope.current, proof)
    };
    assert!(matches!(
        verify(&light, &decoded),
        Err(BoundedDepthError::Compression(ns_compress::Error::Key(_)))
    ));
    assert!(matches!(verify(&both, &decoded), Err(BoundedDepthError::Statement(_))));
    assert!(matches!(verify(&light, &full), Err(BoundedDepthError::Statement(_))));
    let mut refresh = decoded.clone();
    refresh.steps.as_mut().unwrap().pc = 2;
    assert!(matches!(verify(&light, &refresh), Err(BoundedDepthError::Statement(_))));
    assert!(refresh.to_bytes().is_err(), "slot 0 alone with pc = 2 encodes");

    let mut moved = envelope.clone();
    moved.slots[1].depth = 1;
    assert!(matches!(
        light.compress(&setup, &moved),
        Err(BoundedDepthError::Slot { slot: 1, .. })
    ));
    let mut refreshed = envelope.clone();
    refreshed.pc = 2;
    assert!(matches!(
        light.compress(&setup, &refreshed),
        Err(BoundedDepthError::Statement(_))
    ));
    let mut tagged = bytes.clone();
    tagged[16..24].copy_from_slice(&2u64.to_le_bytes());
    assert!(
        FinalProof::from_bytes(&tagged).is_err(),
        "the light tag with pc = 2 decodes"
    );
    let mut retagged = bytes.clone();
    retagged[..16].copy_from_slice(b"NS-OPTIONB-FIN-2");
    assert!(
        FinalProof::from_bytes(&retagged).is_err(),
        "one slot under the tag of both decodes"
    );
}

fn light_wire_fixture() -> FinalProof {
    let initial = prover(2).initial(z0()).expect("initial envelope");
    let mut proof = unproven(&initial);
    let steps = proof.steps.as_mut().unwrap();
    steps.pc = 1;
    steps.slots.truncate(1);
    proof.iteration = 1;
    proof
}

fn assert_public_round_trip(bytes: &[u8]) {
    let proof = crate::FinalProof::from_bytes(bytes).expect("decode public proof");
    assert_eq!(proof.to_bytes().expect("encode public proof"), bytes);
}

#[test]
fn compact_light_public_words_choose_one_encoding() {
    let mut proof = light_wire_fixture();
    proof.steps.as_mut().unwrap().slots[0].1 = 1;
    let start = 96 + 8 * (1 + nightstream_fprime::COMMITMENT_WORDS);
    let zero_public = proof.to_bytes().unwrap();
    assert_eq!(&zero_public[..16], b"NS-OPTIONB-FIN-S");
    let mut default_as_signed = zero_public;
    default_as_signed[96..104].fill(0);
    assert!(crate::FinalProof::from_bytes(&default_as_signed).is_err());

    let public = &mut proof.steps.as_mut().unwrap().slots[0].0.X;
    public.set(0, 0, F::from_i64(-128));
    public.set(0, 1, -F::ONE);
    public.set(0, 2, F::from_u64(127));
    let signed = proof.to_bytes().unwrap();
    assert_eq!(&signed[..16], b"NS-OPTIONB-FIN-S");
    assert_eq!(&signed[start..start + 3], &[128, 255, 127]);
    assert_public_round_trip(&signed);
    let decoded = FinalProof::from_bytes(&signed).unwrap();
    assert_eq!(
        decoded.steps.unwrap().slots[0].0.X,
        proof.steps.as_ref().unwrap().slots[0].0.X
    );

    for wide in [F::from_i64(-129), F::from_u64(128)] {
        proof.steps.as_mut().unwrap().slots[0]
            .0
            .X
            .set(D - 1, PUBLIC_WORDS / D - 1, wide);
        let full = proof.to_bytes().unwrap();
        assert_eq!(&full[..16], b"NS-OPTIONB-FIN-L");
        assert_eq!(full.len() - signed.len(), PUBLIC_WORDS * 7);
        assert_public_round_trip(&full);
        let last = start + 8 * (PUBLIC_WORDS - 1);
        assert_eq!(&full[last..last + 8], &wide.as_canonical_u64().to_le_bytes());
        let mut signed_as_full = full;
        signed_as_full[last..last + 8].fill(0);
        assert!(crate::FinalProof::from_bytes(&signed_as_full).is_err());
    }
}

#[test]
fn compact_light_round_mask_restores_the_full_polynomials() {
    let mut proof = light_wire_fixture();
    let argument = &mut proof.steps.as_mut().unwrap().argument;
    argument.rounds = vec![[K::ONE; 4]; 30];
    for i in [0, 8, 29] {
        argument.rounds[i] = [K::ONE, K::ZERO, K::ONE, K::ZERO];
    }
    let bytes = proof.to_bytes().unwrap();
    let argument = &proof.steps.as_ref().unwrap().argument;
    let inverse = &argument.inverse_commitment;
    let count_at = bytes
        .windows(inverse.len())
        .position(|w| w == inverse)
        .unwrap()
        + inverse.len();
    assert_eq!(&bytes[count_at..count_at + 8], &30u64.to_le_bytes());
    let mask_at = count_at + 8;
    let mask = 1u32 | 1 << 8 | 1 << 29;
    assert_eq!(&bytes[mask_at..mask_at + 4], &mask.to_le_bytes());
    assert_public_round_trip(&bytes);
    let decoded = FinalProof::from_bytes(&bytes).unwrap().steps.unwrap();
    assert_eq!(decoded.argument.rounds, argument.rounds);

    let mut full_quadratic = bytes.clone();
    full_quadratic[mask_at..mask_at + 4].copy_from_slice(&(mask ^ 1).to_le_bytes());
    let high_at = mask_at + 4 + 3 * 16;
    full_quadratic.splice(high_at..high_at, [0u8; 16]);
    assert!(crate::FinalProof::from_bytes(&full_quadratic).is_err());

    let mut surplus = bytes.clone();
    surplus[mask_at..mask_at + 4].copy_from_slice(&(mask | 1 << 30).to_le_bytes());
    assert!(crate::FinalProof::from_bytes(&surplus).is_err());
    assert!(crate::FinalProof::from_bytes(&bytes[..mask_at + 3]).is_err());
    let mut oversized = bytes;
    oversized[count_at..mask_at].copy_from_slice(&31u64.to_le_bytes());
    assert!(crate::FinalProof::from_bytes(&oversized).is_err());
}

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
    let mut moved = bytes.clone();
    moved[bytes.len() - 8] ^= 1;
    assert!(FinalProof::from_bytes(&moved).is_err());
}

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
        Err(BoundedDepthError::Compression(ns_compress::Error::Layout(_)))
    ));
}
