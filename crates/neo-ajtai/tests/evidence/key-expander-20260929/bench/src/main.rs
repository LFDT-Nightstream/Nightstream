//! Time one complete indexed Ajtai key pass for four expanders on the CPU and
//! on Metal. Every expander gives 54 coefficients per key element
//! (row, column); each coefficient is 32 little-endian bytes reduced modulo
//! the Goldilocks prime. Only the byte source differs.
//!
//! Usage: `key-expander-bench [cpu|metal|both]`. The pass covers the
//! production key prefix: 22 rows by 3,221,095 columns.

use std::time::{Duration, Instant};

mod chacha_v1;

use chacha_v1::{
    block_words, coefficient, coefficient_block, PRODUCTION_MESSAGE_COLUMNS, PRODUCTION_SEED, PRODUCTION_VERIFIER_ROWS,
};
use objc2::rc::Retained;
use objc2::runtime::ProtocolObject;
use objc2_foundation::NSString;
use objc2_metal::{
    MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLCommandQueue, MTLComputeCommandEncoder, MTLComputePipelineState,
    MTLCreateSystemDefaultDevice, MTLDevice, MTLLibrary, MTLResourceOptions, MTLSize,
};
use rayon::prelude::*;
use sha3::digest::{ExtendableOutput, Update, XofReader};
use sha3::{Shake128, Shake256};

use chacha_v1::D;
const P: u64 = 0xFFFF_FFFF_0000_0001;
const TAG: [u8; 32] = *b"nightstream-ajtai-shake-bench-v0";
const SIGMA: [u32; 4] = [0x6170_7865, 0x3320_646e, 0x7962_2d32, 0x6b20_6574];
const METAL_SOURCE: &str = include_str!("expander.metal");
const METAL_THREADS: u64 = 1 << 17;

#[derive(Clone, Copy, Debug)]
enum Expander {
    /// The current setup: one ChaCha20 block per coefficient.
    ChaCha20,
    /// Both halves of each ChaCha20 block: 27 blocks per key element.
    ChaCha20Contiguous,
    Shake128,
    Shake256,
}

const EXPANDERS: [Expander; 4] = [
    Expander::ChaCha20,
    Expander::ChaCha20Contiguous,
    Expander::Shake128,
    Expander::Shake256,
];

impl Expander {
    fn kernel(self) -> &'static str {
        match self {
            Self::ChaCha20 => "expand_chacha20",
            Self::ChaCha20Contiguous => "expand_chacha20_contiguous",
            Self::Shake128 => "expand_shake128",
            Self::Shake256 => "expand_shake256",
        }
    }

    fn element(self, row: u32, column: u64) -> [u64; D] {
        match self {
            Self::ChaCha20 => coefficient_block(&PRODUCTION_SEED, row, column),
            Self::ChaCha20Contiguous => chacha_contiguous(row, column),
            Self::Shake128 => shake::<Shake128>(row, column),
            Self::Shake256 => shake::<Shake256>(row, column),
        }
    }
}

/// Add two canonical values; a carry adds 2^64 = 2^32 - 1 (mod P).
fn add(x: u64, y: u64) -> u64 {
    let (mut s, carry) = x.overflowing_add(y);
    if carry {
        s += 0xFFFF_FFFF;
    }
    if s >= P {
        s -= P;
    }
    s
}

/// Reduce eight little-endian words modulo P, as the Metal kernels do. With
/// x = 2^32, x^2 = x - 1 and x^6 = 1, so the value is a + b*x.
#[inline(always)]
fn reduce(w: [u32; 8]) -> u64 {
    let w = w.map(i64::from);
    let a = w[0] - w[2] - w[3] + w[5] + w[6];
    let b = w[1] + w[2] - w[4] - w[5] + w[7];
    let canonical = |v: i64| if v < 0 { P - v.unsigned_abs() } else { v as u64 };
    let (a, b) = (canonical(a), canonical(b));
    // b * 2^32 = high * 2^64 + low * 2^32 = high * (2^32 - 1) + low * 2^32.
    let (high, low) = (b >> 32, b & 0xFFFF_FFFF);
    add(a, add(low << 32, (high << 32) - high))
}

/// The reference reduction: the 256-bit integer, divided step by step.
fn reduce_by_division(w: [u32; 8]) -> u64 {
    w.iter()
        .rev()
        .fold(0u128, |value, &word| ((value << 32) + u128::from(word)) % u128::from(P)) as u64
}

fn le_words(bytes: &[u8]) -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(bytes[4 * i..4 * i + 4].try_into().unwrap()))
}

fn seed_words() -> [u32; 8] {
    std::array::from_fn(|i| u32::from_le_bytes(PRODUCTION_SEED[4 * i..4 * i + 4].try_into().unwrap()))
}

#[inline(always)]
fn quarter_round<const L: usize>(s: &mut [[u32; L]; 16], a: usize, b: usize, c: usize, d: usize) {
    for l in 0..L {
        s[a][l] = s[a][l].wrapping_add(s[b][l]);
        s[d][l] = (s[d][l] ^ s[a][l]).rotate_left(16);
        s[c][l] = s[c][l].wrapping_add(s[d][l]);
        s[b][l] = (s[b][l] ^ s[c][l]).rotate_left(12);
        s[a][l] = s[a][l].wrapping_add(s[b][l]);
        s[d][l] = (s[d][l] ^ s[a][l]).rotate_left(8);
        s[c][l] = s[c][l].wrapping_add(s[d][l]);
        s[b][l] = (s[b][l] ^ s[c][l]).rotate_left(7);
    }
}

/// 27 ChaCha20 blocks, evaluated lane-parallel like the production code.
/// Block k gives lanes 2k (words 0..8) and 2k+1 (words 8..16).
fn chacha_contiguous(row: u32, column: u64) -> [u64; D] {
    const L: usize = D / 2;
    let key = seed_words();
    let mut initial = [[0u32; L]; 16];
    for (word, constant) in initial[..4].iter_mut().zip(SIGMA) {
        *word = [constant; L];
    }
    for (word, key_word) in initial[4..12].iter_mut().zip(key) {
        *word = [key_word; L];
    }
    initial[12] = std::array::from_fn(|lane| lane as u32);
    initial[13] = [row; L];
    initial[14] = [column as u32; L];
    initial[15] = [(column >> 32) as u32; L];
    let mut s = initial;
    for _ in 0..10 {
        quarter_round(&mut s, 0, 4, 8, 12);
        quarter_round(&mut s, 1, 5, 9, 13);
        quarter_round(&mut s, 2, 6, 10, 14);
        quarter_round(&mut s, 3, 7, 11, 15);
        quarter_round(&mut s, 0, 5, 10, 15);
        quarter_round(&mut s, 1, 6, 11, 12);
        quarter_round(&mut s, 2, 7, 8, 13);
        quarter_round(&mut s, 3, 4, 9, 14);
    }
    let mut out = [0u64; D];
    for lane in 0..L {
        let word = |w: usize| s[w][lane].wrapping_add(initial[w][lane]);
        out[2 * lane] = reduce(std::array::from_fn(word));
        out[2 * lane + 1] = reduce(std::array::from_fn(|w| word(8 + w)));
    }
    out
}

fn shake_input(row: u32, column: u64) -> [u8; 76] {
    let mut input = [0u8; 76];
    input[..32].copy_from_slice(&TAG);
    input[32..64].copy_from_slice(&PRODUCTION_SEED);
    input[64..68].copy_from_slice(&row.to_le_bytes());
    input[68..76].copy_from_slice(&column.to_le_bytes());
    input
}

fn shake<X: Default + Update + ExtendableOutput>(row: u32, column: u64) -> [u64; D] {
    let mut xof = X::default();
    xof.update(&shake_input(row, column));
    let mut reader = xof.finalize_xof();
    let mut bytes = [0u8; 32 * D];
    reader.read(&mut bytes);
    std::array::from_fn(|lane| reduce(le_words(&bytes[32 * lane..32 * lane + 32])))
}

fn element_checksum(index: u64, coefficients: &[u64; D]) -> u64 {
    coefficients
        .iter()
        .enumerate()
        .fold(index.wrapping_mul(0x9E37_79B9_7F4A_7C15), |sum, (lane, &c)| {
            sum.wrapping_add(c.wrapping_mul(2 * lane as u64 + 1))
        })
}

/// Independent references on scattered elements: the scalar production
/// coefficient, the scalar RFC 8439 block, and division-based reduction.
fn check_references() {
    for (row, column) in [(0u32, 0u64), (7, 1_234_567), (21, PRODUCTION_MESSAGE_COLUMNS - 1)] {
        let current = Expander::ChaCha20.element(row, column);
        let contiguous = Expander::ChaCha20Contiguous.element(row, column);
        for block in 0..D as u32 / 2 {
            let words = block_words(&PRODUCTION_SEED, row, column, block);
            assert_eq!(
                contiguous[2 * block as usize],
                coefficient(&PRODUCTION_SEED, row, column, block)
            );
            let second: [u32; 8] = words[8..].try_into().unwrap();
            assert_eq!(contiguous[2 * block as usize + 1], reduce_by_division(second));
        }
        let bytes = {
            let mut xof = Shake128::default();
            xof.update(&shake_input(row, column));
            let mut bytes = [0u8; 32 * D];
            xof.finalize_xof().read(&mut bytes);
            bytes
        };
        let shake = Expander::Shake128.element(row, column);
        for lane in 0..D {
            assert_eq!(
                shake[lane],
                reduce_by_division(le_words(&bytes[32 * lane..32 * lane + 32]))
            );
        }
        for lane in 0..D as u32 {
            assert_eq!(current[lane as usize], coefficient(&PRODUCTION_SEED, row, column, lane));
        }
    }
    println!("reference checks: ok");
}

fn cpu_pass(expander: Expander, rows: u64, columns: u64) -> (u64, Duration) {
    let started = Instant::now();
    let checksum = (0..(rows * columns) as usize)
        .into_par_iter()
        .with_min_len(1 << 12)
        .map(|index| {
            let index = index as u64;
            element_checksum(index, &expander.element((index / columns) as u32, index % columns))
        })
        .reduce(|| 0, |left, right| left ^ right);
    (checksum, started.elapsed())
}

type Buffer = Retained<ProtocolObject<dyn MTLBuffer>>;

fn shared_buffer<T: Copy>(device: &ProtocolObject<dyn MTLDevice>, values: &[T]) -> Buffer {
    let bytes = size_of_val(values);
    let buffer = device
        .newBufferWithLength_options(bytes, MTLResourceOptions::StorageModeShared)
        .expect("buffer");
    unsafe {
        std::ptr::copy_nonoverlapping(
            values.as_ptr().cast::<u8>(),
            buffer.contents().as_ptr().cast::<u8>(),
            bytes,
        );
    }
    buffer
}

/// One command buffer per key row; all rows are committed, then awaited.
fn metal_pass(expander: Expander, rows: u64, columns: u64) -> (u64, Duration) {
    let device = MTLCreateSystemDefaultDevice().expect("Metal device");
    let queue = device.newCommandQueue().expect("command queue");
    let library = device
        .newLibraryWithSource_options_error(&NSString::from_str(METAL_SOURCE), None)
        .expect("Metal source compiles");
    let function = library
        .newFunctionWithName(&NSString::from_str(expander.kernel()))
        .expect("kernel");
    let pipeline = device
        .newComputePipelineStateWithFunction_error(&function)
        .expect("pipeline");
    let seed = shared_buffer(&device, &seed_words());
    let prefix: Vec<u64> = TAG
        .chunks_exact(8)
        .chain(PRODUCTION_SEED.chunks_exact(8))
        .map(|bytes| u64::from_le_bytes(bytes.try_into().unwrap()))
        .collect();
    let prefix = shared_buffer(&device, &prefix);
    let out = shared_buffer(&device, &vec![0u64; METAL_THREADS as usize]);
    let shapes: Vec<Buffer> = (0..rows)
        .map(|row| shared_buffer(&device, &[row, columns, METAL_THREADS]))
        .collect();
    let width = pipeline.maxTotalThreadsPerThreadgroup().clamp(1, 256);
    let started = Instant::now();
    let mut commands = Vec::with_capacity(rows as usize);
    for shape in &shapes {
        let command = queue.commandBuffer().expect("command buffer");
        let encoder = command.computeCommandEncoder().expect("encoder");
        encoder.setComputePipelineState(&pipeline);
        unsafe {
            encoder.setBuffer_offset_atIndex(Some(&seed), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(&prefix), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(shape), 0, 2);
            encoder.setBuffer_offset_atIndex(Some(&out), 0, 3);
        }
        encoder.dispatchThreads_threadsPerThreadgroup(
            MTLSize {
                width: METAL_THREADS as usize,
                height: 1,
                depth: 1,
            },
            MTLSize {
                width,
                height: 1,
                depth: 1,
            },
        );
        encoder.endEncoding();
        command.commit();
        commands.push(command);
    }
    for command in &commands {
        command.waitUntilCompleted();
        assert!(
            command.error().is_none(),
            "Metal execution error: {:?}",
            command.error()
        );
    }
    let elapsed = started.elapsed();
    let values = unsafe { std::slice::from_raw_parts(out.contents().as_ptr().cast::<u64>(), METAL_THREADS as usize) };
    (values.iter().fold(0, |left, right| left ^ right), elapsed)
}

fn main() {
    let mode = std::env::args().nth(1).unwrap_or_else(|| "both".into());
    let (rows, columns) = (PRODUCTION_VERIFIER_ROWS, PRODUCTION_MESSAGE_COLUMNS);
    println!(
        "key pass: {rows} rows x {columns} columns = {} key elements, {} coefficients; rayon threads {}; sha3 asm {}",
        rows * columns,
        rows * columns * D as u64,
        rayon::current_num_threads(),
        cfg!(feature = "asm")
    );
    check_references();
    for expander in EXPANDERS {
        let cpu = (mode != "metal").then(|| cpu_pass(expander, rows, columns));
        let metal = (mode != "cpu").then(|| metal_pass(expander, rows, columns));
        if let Some((checksum, elapsed)) = cpu {
            println!(
                "cpu   {expander:?}: {:.3} s checksum {checksum:016x}",
                elapsed.as_secs_f64()
            );
        }
        if let Some((checksum, elapsed)) = metal {
            println!(
                "metal {expander:?}: {:.3} s checksum {checksum:016x}",
                elapsed.as_secs_f64()
            );
        }
        if let (Some(cpu), Some(metal)) = (cpu, metal) {
            assert_eq!(cpu.0, metal.0, "{expander:?}: CPU and Metal checksums differ");
        }
    }
}
