//! Time one complete indexed Ajtai key pass for two expanders on the CPU and
//! on Metal. Every expander gives 54 coefficients per key element
//! (row, column); each coefficient is 32 little-endian bytes reduced modulo
//! the Goldilocks prime. Only the byte source differs.
//!
//! Usage: `key-expander-bench [cpu|metal|both]`. The pass covers the
//! production key prefix: 22 rows by 3,221,095 columns.

use std::time::{Duration, Instant};

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

const D: usize = 54;
const P: u64 = 0xFFFF_FFFF_0000_0001;
const PRODUCTION_VERIFIER_ROWS: u64 = 22;
const PRODUCTION_MESSAGE_COLUMNS: u64 = 3_221_095;
const PRODUCTION_SEED: [u8; 32] = [
    252, 64, 73, 132, 212, 76, 27, 135, 141, 104, 166, 168, 0, 146, 215, 215, 171, 68, 216, 26, 193, 123, 69, 168, 231,
    189, 76, 31, 30, 55, 23, 2,
];
const TAG: [u8; 32] = *b"nightstream-ajtai-shake-bench-v0";
const METAL_SOURCE: &str = include_str!("expander.metal");
const METAL_THREADS: u64 = 1 << 17;

#[derive(Clone, Copy, Debug)]
enum Expander {
    Shake128,
    Shake256,
}

const EXPANDERS: [Expander; 2] = [Expander::Shake128, Expander::Shake256];

impl Expander {
    fn kernel(self) -> &'static str {
        match self {
            Self::Shake128 => "expand_shake128",
            Self::Shake256 => "expand_shake256",
        }
    }

    fn element(self, row: u32, column: u64) -> [u64; D] {
        match self {
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

/// Independent reference on scattered elements: division-based reduction.
fn check_references() {
    for (row, column) in [(0u32, 0u64), (7, 1_234_567), (21, PRODUCTION_MESSAGE_COLUMNS - 1)] {
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
            encoder.setBuffer_offset_atIndex(Some(&prefix), 0, 0);
            encoder.setBuffer_offset_atIndex(Some(shape), 0, 1);
            encoder.setBuffer_offset_atIndex(Some(&out), 0, 2);
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
