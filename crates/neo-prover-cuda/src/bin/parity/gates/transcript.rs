//! Device Poseidon2 transcript gate: raw permutation parity plus a long
//! mixed absorb/challenge op stream against the host sponge.

use super::*;

use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_prover_cuda::kernels::poseidon2::{launch_hash_fields, launch_permute_states, load_poseidon2_kernels, WIDTH};
use neo_prover_cuda::transcript::{upload_round_constants, DeviceTranscript, TranscriptIoOp, TranscriptOp};
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeField64;

pub fn transcript() {
    const STATES: usize = 4096;
    const ROUNDS: usize = 200;
    let mut rng = StdRng::seed_from_u64(0x7472_616e_7363_7231);

    let device = Device::open().expect("open CUDA device");
    let module = load_poseidon2_kernels(device.ctx()).expect("load poseidon2 kernels");
    let rc = upload_round_constants(&device).expect("upload round constants");

    // 1. Raw permutation parity, one thread per state.
    let states: Vec<[F; WIDTH]> = (0..STATES)
        .map(|_| core::array::from_fn(|_| rand_f(&mut rng)))
        .collect();
    let (expect, cpu_ms) = timed(|| {
        states
            .iter()
            .map(|&s| p2::permute_state(s))
            .collect::<Vec<_>>()
    });
    let flat: Vec<u64> = states
        .iter()
        .flatten()
        .map(|f| f.as_canonical_u64())
        .collect();
    let mut dev_states = DeviceBuffer::from_host(device.stream(), &flat).expect("upload states");
    let (_, gpu_ms) = timed(|| {
        launch_permute_states(&module, device.stream(), STATES, &mut dev_states, &rc).expect("launch permute");
        device.sync().expect("sync after permute");
    });
    let got = dev_states
        .to_host_vec(device.stream())
        .expect("download states");
    device.sync().expect("sync after download");
    for (i, exp) in expect.iter().enumerate() {
        for j in 0..WIDTH {
            assert_eq!(
                got[WIDTH * i + j],
                exp[j].as_canonical_u64(),
                "permutation mismatch at state {i} lane {j}"
            );
        }
    }

    // 2. Stateless hash parity: one thread per variable-length preimage.
    let hash_inputs: Vec<Vec<F>> = (0..512)
        .map(|case| {
            let len = match case % 16 {
                0 => 0,
                1 => 1,
                2 => 3,
                3 => 4,
                4 => 5,
                5 => 7,
                6 => 8,
                7 => 9,
                _ => 10 + (case % 37),
            };
            (0..len).map(|_| rand_f(&mut rng)).collect()
        })
        .collect();
    let mut hash_offsets = Vec::with_capacity(hash_inputs.len());
    let mut hash_lengths = Vec::with_capacity(hash_inputs.len());
    let mut hash_fields = Vec::new();
    for input in &hash_inputs {
        hash_offsets.push(hash_fields.len() as u64);
        hash_lengths.push(input.len() as u64);
        hash_fields.extend(input.iter().map(|field| field.as_canonical_u64()));
    }
    if hash_fields.is_empty() {
        hash_fields.push(0);
    }
    let hash_expect: Vec<[F; p2::DIGEST_LEN]> = hash_inputs
        .iter()
        .map(|input| p2::poseidon2_hash(input))
        .collect();
    let hash_fields_dev = DeviceBuffer::from_host(device.stream(), &hash_fields).expect("upload hash fields");
    let hash_offsets_dev = DeviceBuffer::from_host(device.stream(), &hash_offsets).expect("upload hash offsets");
    let hash_lengths_dev = DeviceBuffer::from_host(device.stream(), &hash_lengths).expect("upload hash lengths");
    let mut hash_out_dev =
        DeviceBuffer::zeroed(device.stream(), hash_inputs.len() * p2::DIGEST_LEN).expect("alloc hash out");
    launch_hash_fields(
        &module,
        device.stream(),
        hash_inputs.len(),
        &hash_fields_dev,
        &hash_offsets_dev,
        &hash_lengths_dev,
        &mut hash_out_dev,
        &rc,
    )
    .expect("launch hash fields");
    let hash_got = hash_out_dev
        .to_host_vec(device.stream())
        .expect("download hash outputs");
    device.sync().expect("sync after hash outputs");
    for (i, exp) in hash_expect.iter().enumerate() {
        for lane in 0..p2::DIGEST_LEN {
            assert_eq!(
                hash_got[i * p2::DIGEST_LEN + lane],
                exp[lane].as_canonical_u64(),
                "hash mismatch at input {i} lane {lane}"
            );
        }
    }

    // 3. Sponge parity: identical raw op stream on host transcript and device,
    // lengths chosen to cross every RATE/DIGEST_LEN boundary.
    let seed_fields: Vec<F> = (0..7).map(|_| rand_f(&mut rng)).collect();
    let mut host_tr = Poseidon2Transcript::new_raw_fields(&seed_fields);
    let mut dev_tr = DeviceTranscript::from_state_and_absorbed(&device, host_tr.state(), host_tr.absorbed())
        .expect("seed device transcript");

    let mut ops = Vec::new();
    let mut host_challenges: Vec<F> = Vec::new();
    for round in 0..ROUNDS {
        let fs: Vec<F> = (0..round % 9).map(|_| rand_f(&mut rng)).collect();
        // append_fields_raw absorbs a length prefix, then the fields.
        let mut absorb = vec![F::from_u64(fs.len() as u64)];
        absorb.extend(fs.iter().copied());
        host_tr.append_fields_raw(&fs);
        ops.push(TranscriptOp::AbsorbFields(absorb));

        let n = 1 + round % 9;
        host_challenges.extend(host_tr.challenge_fields_raw(n));
        ops.push(TranscriptOp::Challenge(n));
    }
    let dev_challenges = dev_tr
        .run(&device, &module, &rc, &ops)
        .expect("run device op stream");
    assert_eq!(dev_challenges.len(), host_challenges.len(), "challenge count");
    for (i, (d, h)) in dev_challenges.iter().zip(&host_challenges).enumerate() {
        assert_eq!(d.as_canonical_u64(), h.as_canonical_u64(), "challenge mismatch at {i}");
    }
    let (dev_state, dev_absorbed) = dev_tr.state_and_absorbed(&device).expect("download sponge");
    let host_state = host_tr.state();
    for j in 0..WIDTH {
        assert_eq!(
            dev_state[j].as_canonical_u64(),
            host_state[j].as_canonical_u64(),
            "final state lane {j}"
        );
    }
    assert_eq!(dev_absorbed, host_tr.absorbed(), "absorb cursor");

    // 4. IO parity: coeff-like payloads may live in device memory, and
    // challenges may be written to device memory for the next kernel.
    let io_seed_fields: Vec<F> = (0..5).map(|_| rand_f(&mut rng)).collect();
    let mut io_host_tr = Poseidon2Transcript::new_raw_fields(&io_seed_fields);
    let mut io_dev_tr = DeviceTranscript::from_state_and_absorbed(&device, io_host_tr.state(), io_host_tr.absorbed())
        .expect("seed IO device transcript");
    let mut io_ops = Vec::new();
    let mut device_payload_words = Vec::new();
    let mut expected_host_challenges = Vec::new();
    let mut expected_device_challenges = Vec::new();
    let mut device_out_len = 0usize;

    for round in 0..ROUNDS {
        let fs: Vec<F> = (0..1 + round % 7).map(|_| rand_f(&mut rng)).collect();
        let mut absorb = vec![F::from_u64(fs.len() as u64)];
        absorb.extend(fs.iter().copied());
        io_host_tr.append_fields_raw(&fs);

        if round % 3 == 0 {
            io_ops.push(TranscriptIoOp::AbsorbHost(absorb));
        } else {
            let offset = device_payload_words.len();
            device_payload_words.extend(absorb.iter().map(|f| f.as_canonical_u64()));
            io_ops.push(TranscriptIoOp::AbsorbDevice {
                offset,
                len: absorb.len(),
            });
        }

        let n = 1 + round % 5;
        let expected = io_host_tr.challenge_fields_raw(n);
        if round % 2 == 0 {
            let offset = device_out_len;
            device_out_len += n;
            expected_device_challenges.extend(expected.iter().map(|f| f.as_canonical_u64()));
            io_ops.push(TranscriptIoOp::ChallengeDevice { offset, len: n });
        } else {
            expected_host_challenges.extend(expected);
            io_ops.push(TranscriptIoOp::ChallengeHost(n));
        }
    }

    let device_payload = DeviceBuffer::from_host(device.stream(), &device_payload_words).expect("upload IO payload");
    let mut device_out = DeviceBuffer::zeroed(device.stream(), device_out_len.max(1)).expect("alloc IO out");
    let got_host_challenges = io_dev_tr
        .run_io(&device, &module, &rc, &io_ops, &device_payload, &mut device_out)
        .expect("run device IO op stream");
    assert_eq!(
        got_host_challenges.len(),
        expected_host_challenges.len(),
        "IO host challenge count"
    );
    for (i, (got, exp)) in got_host_challenges
        .iter()
        .zip(&expected_host_challenges)
        .enumerate()
    {
        assert_eq!(
            got.as_canonical_u64(),
            exp.as_canonical_u64(),
            "IO host challenge mismatch at {i}"
        );
    }
    let got_device_challenges = device_out
        .to_host_vec(device.stream())
        .expect("download IO out");
    device.sync().expect("sync after IO download");
    assert_eq!(
        &got_device_challenges[..device_out_len],
        expected_device_challenges.as_slice(),
        "IO device challenges"
    );
    let (io_dev_state, io_dev_absorbed) = io_dev_tr
        .state_and_absorbed(&device)
        .expect("download IO sponge");
    let io_host_state = io_host_tr.state();
    for j in 0..WIDTH {
        assert_eq!(
            io_dev_state[j].as_canonical_u64(),
            io_host_state[j].as_canonical_u64(),
            "IO final state lane {j}"
        );
    }
    assert_eq!(io_dev_absorbed, io_host_tr.absorbed(), "IO absorb cursor");

    println!(
        "[parity transcript] OK: {STATES} permutations identical (cpu {cpu_ms:.2}ms gpu {gpu_ms:.2}ms), \
         {} stateless hashes identical, {ROUNDS}-round sponge op stream identical ({} challenges), \
         IO stream identical ({} host + {} device challenges)",
        hash_inputs.len(),
        host_challenges.len(),
        expected_host_challenges.len(),
        expected_device_challenges.len()
    );
}
