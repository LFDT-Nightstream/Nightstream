//! The device arithmetic of `kernels::field` against `neo_math` and the CPU
//! Poseidon2 permutation, word for word.

use cuda_core::{CudaContext, DeviceBuffer, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread};
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::{from_complex, KExtensions, F, K};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rand::{Rng, SeedableRng};

use crate::kernels::field::{GOLDILOCKS_MODULUS, POSEIDON_WIDTH};

#[cuda_module]
mod probe {
    use super::*;
    use crate::kernels::field::*;

    /// Per pair: from_word(a), a + b, a - b, a·b, a^7, a·7 on canonical
    /// pairs; the wide reduction and the product of the raw words.
    #[kernel]
    pub fn gl_ops(a: *const u64, b: *const u64, out: *mut u64, count: u64) {
        let i = thread::index_1d().get();
        if i as u64 >= count {
            return;
        }
        // SAFETY: the host gives `count` words per input and 8 per output.
        unsafe {
            let (x, y) = (*a.add(i), *b.add(i));
            let (u, v) = (gl_from_word(x), gl_from_word(y));
            let words = [
                u,
                gl_add(u, v),
                gl_sub(u, v),
                gl_mul(u, v),
                gl_sbox(u),
                gl_mul_by_7(u),
                gl_reduce_wide(x, y),
                gl_mul(x, y),
            ];
            for (lane, word) in words.into_iter().enumerate() {
                *out.add(8 * i + lane) = word;
            }
        }
    }

    /// Per pair of interleaved `K` values: a + b, a - b, a·b.
    #[kernel]
    pub fn kx_ops(a: *const u64, b: *const u64, out: *mut u64, count: u64) {
        let i = thread::index_1d().get() as u64;
        if i >= count {
            return;
        }
        // SAFETY: the host gives `count` values per input and 3 per output.
        unsafe {
            let (x, y) = (load_k(a, i), load_k(b, i));
            store_k(out, 3 * i, kx_add(x, y));
            store_k(out, 3 * i + 1, kx_sub(x, y));
            store_k(out, 3 * i + 2, kx_mul(x, y));
        }
    }

    /// `out[i] = shape[i % 6] + i`: a shape passed by value.
    #[kernel]
    pub fn by_value(out: *mut u64, shape: [u64; 6], count: u64) {
        let i = thread::index_1d().get();
        if i as u64 >= count {
            return;
        }
        // SAFETY: the host gives `count` output words.
        unsafe { *out.add(i) = shape[i % 6] + i as u64 };
    }

    /// The permutation of each state of `POSEIDON_WIDTH` words, in place.
    #[kernel]
    pub fn permute(states: *mut u64, round_constants: *const u64, count: u64) {
        let i = thread::index_1d().get();
        if i as u64 >= count {
            return;
        }
        let mut state = [0u64; POSEIDON_WIDTH];
        // SAFETY: the host gives `count` states and every round constant.
        unsafe {
            for (lane, word) in state.iter_mut().enumerate() {
                *word = gl_from_word(*states.add(POSEIDON_WIDTH * i + lane));
            }
            poseidon_permute(&mut state, round_constants);
            for (lane, &word) in state.iter().enumerate() {
                *states.add(POSEIDON_WIDTH * i + lane) = word;
            }
        }
    }
}

fn pointer(buffer: &DeviceBuffer<u64>) -> *mut u64 {
    buffer.cu_deviceptr() as *mut u64
}

fn words(rng: &mut rand::rngs::StdRng, count: usize) -> Vec<u64> {
    const EDGES: [u64; 10] = [
        0,
        1,
        2,
        0xffff_ffff,
        1 << 32,
        GOLDILOCKS_MODULUS - 1,
        GOLDILOCKS_MODULUS,
        GOLDILOCKS_MODULUS + 1,
        u64::MAX - 1,
        u64::MAX,
    ];
    let mut words = Vec::with_capacity(count);
    for &a in &EDGES {
        words.extend(EDGES.iter().map(|_| a));
    }
    words.extend(std::iter::repeat_with(|| EDGES).take(EDGES.len()).flatten());
    words.truncate(count);
    while words.len() < count {
        words.push(rng.random());
    }
    words
}

#[test]
fn goldilocks_and_k_match_the_cpu() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let module = probe::load(&context).expect("probe kernels");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x4355_4441);
    let count = 1 << 16;
    let a = words(&mut rng, count);
    let mut b = words(&mut rng, 2 * count);
    b.drain(..count);
    // The first block pairs every edge word with every edge word.
    b[..100].copy_from_slice(&std::array::from_fn::<u64, 100, _>(|i| a[10 * (i % 10)]));
    let (device_a, device_b) = (
        DeviceBuffer::from_host(&stream, &a).unwrap(),
        DeviceBuffer::from_host(&stream, &b).unwrap(),
    );
    let out = DeviceBuffer::<u64>::zeroed(&stream, 8 * count).unwrap();
    module
        .gl_ops(
            &stream,
            LaunchConfig::for_num_elems(count as u32),
            pointer(&device_a),
            pointer(&device_b),
            pointer(&out),
            count as u64,
        )
        .expect("launch");
    let out = out.to_host_vec(&stream).unwrap();
    let canonical = |word: u64| F::from_u64(word);
    for (i, (&x, &y)) in a.iter().zip(&b).enumerate() {
        let (u, v) = (canonical(x), canonical(y));
        let wide = ((u128::from(y) << 64) | u128::from(x)) % u128::from(GOLDILOCKS_MODULUS);
        let expected = [
            u,
            u + v,
            u - v,
            u * v,
            u.exp_u64(7),
            u * F::from_u64(7),
            F::from_u64(wide as u64),
            u * v,
        ]
        .map(|value| value.as_canonical_u64());
        assert_eq!(out[8 * i..8 * i + 8], expected, "pair {i}: {x:#x}, {y:#x}");
    }

    let k = |words: &[u64]| -> Vec<K> {
        words
            .chunks_exact(2)
            .map(|pair| from_complex(F::from_u64(pair[0]), F::from_u64(pair[1])))
            .collect()
    };
    let (ka, kb) = (k(&a), k(&b));
    let out = DeviceBuffer::<u64>::zeroed(&stream, 3 * count).unwrap();
    module
        .kx_ops(
            &stream,
            LaunchConfig::for_num_elems(ka.len() as u32),
            pointer(&device_a),
            pointer(&device_b),
            pointer(&out),
            ka.len() as u64,
        )
        .expect("launch");
    let out = out.to_host_vec(&stream).unwrap();
    for (i, (&x, &y)) in ka.iter().zip(&kb).enumerate() {
        let expected = [x + y, x - y, x * y];
        for (j, value) in expected.iter().enumerate() {
            let (c0, c1) = value.to_limbs_u64();
            assert_eq!(
                out[6 * i + 2 * j..6 * i + 2 * j + 2],
                [c0, c1],
                "K pair {i} operation {j}"
            );
        }
    }
}

#[test]
fn poseidon2_matches_the_cpu_permutation() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let module = probe::load(&context).expect("probe kernels");
    let mut rng = rand::rngs::StdRng::seed_from_u64(0x5032);
    let count = 4096;
    let mut states = words(&mut rng, count * POSEIDON_WIDTH);
    states[..POSEIDON_WIDTH].fill(0);
    let device = DeviceBuffer::from_host(&stream, &states).unwrap();
    let constants = DeviceBuffer::from_host(&stream, &crate::poseidon2::round_constant_words()).unwrap();
    module
        .permute(
            &stream,
            LaunchConfig::for_num_elems(count as u32),
            pointer(&device),
            pointer(&constants),
            count as u64,
        )
        .expect("launch");
    let out = device.to_host_vec(&stream).unwrap();
    for (i, (state, got)) in states
        .chunks_exact(POSEIDON_WIDTH)
        .zip(out.chunks_exact(POSEIDON_WIDTH))
        .enumerate()
    {
        let expected = p2::permute_state(std::array::from_fn(|lane| F::from_u64(state[lane])));
        assert_eq!(got, expected.map(|value| value.as_canonical_u64()), "state {i}");
    }
}

#[test]
fn shapes_pass_by_value() {
    let context = CudaContext::new(0).expect("CUDA context");
    let stream = context.default_stream();
    let module = probe::load(&context).expect("probe kernels");
    let out = DeviceBuffer::<u64>::zeroed(&stream, 1000).unwrap();
    let shape = [7, 1 << 40, 3, u64::MAX - 2000, 0, 9];
    module
        .by_value(&stream, LaunchConfig::for_num_elems(1000), pointer(&out), shape, 1000)
        .expect("launch");
    let out = out.to_host_vec(&stream).unwrap();
    assert!(out
        .iter()
        .enumerate()
        .all(|(i, &word)| word == shape[i % 6] + i as u64));
}
