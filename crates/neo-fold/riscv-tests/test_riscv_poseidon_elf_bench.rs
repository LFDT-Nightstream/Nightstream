#![cfg(feature = "poseidon-precompile")]

use neo_fold::pi_ccs::FoldingMode;
use neo_fold::riscv_trace_shard::Rv32TraceWiring;
use std::fs;
use std::path::{Path, PathBuf};
use std::time::{Duration, Instant};

const INPUT_ADDR: u64 = 0x104;
const INPUT_WORDS_U64: [u64; 4] = [5, 7, 11, 13];
const CHUNK_ROWS: usize = 2_048;

#[derive(Debug)]
struct GuestBench {
    name: &'static str,
    elf_size_bytes: usize,
    rom_size_bytes: usize,
    rom_words: usize,
    trace_rows: usize,
    fold_steps: usize,
    requires_poseidon_stage: bool,
    prove_wall: Duration,
    prove_internal: Duration,
    setup_ms: f64,
    chunk_build_commit_ms: f64,
    fold_and_prove_ms: f64,
    verify_wall: Duration,
    verify_internal: Option<Duration>,
}

fn read_u16_le(data: &[u8], off: usize) -> Result<u16, String> {
    let bytes = data
        .get(off..off + 2)
        .ok_or_else(|| format!("u16 out of bounds at offset {off}"))?;
    let arr: [u8; 2] = bytes
        .try_into()
        .map_err(|_| format!("u16 conversion failed at offset {off}"))?;
    Ok(u16::from_le_bytes(arr))
}

fn read_u32_le(data: &[u8], off: usize) -> Result<u32, String> {
    let bytes = data
        .get(off..off + 4)
        .ok_or_else(|| format!("u32 out of bounds at offset {off}"))?;
    let arr: [u8; 4] = bytes
        .try_into()
        .map_err(|_| format!("u32 conversion failed at offset {off}"))?;
    Ok(u32::from_le_bytes(arr))
}

fn read_cstr(buf: &[u8], off: usize) -> Result<&str, String> {
    let tail = buf
        .get(off..)
        .ok_or_else(|| format!("string offset out of bounds: {off}"))?;
    let end_rel = tail
        .iter()
        .position(|&b| b == 0)
        .ok_or_else(|| format!("unterminated C string at offset {off}"))?;
    std::str::from_utf8(&tail[..end_rel]).map_err(|e| format!("invalid UTF-8 in section name: {e}"))
}

fn extract_elf32_section(data: &[u8], section_name: &str) -> Result<(u32, Vec<u8>), String> {
    if data.get(..4) != Some(b"\x7FELF") {
        return Err("not an ELF file".into());
    }
    let ei_class = *data.get(4).ok_or_else(|| "missing EI_CLASS".to_string())?;
    let ei_data = *data.get(5).ok_or_else(|| "missing EI_DATA".to_string())?;
    if ei_class != 1 {
        return Err(format!("expected ELF32 class=1, got {ei_class}"));
    }
    if ei_data != 1 {
        return Err(format!("expected little-endian data=1, got {ei_data}"));
    }

    let e_shoff = read_u32_le(data, 0x20)? as usize;
    let e_shentsize = read_u16_le(data, 0x2E)? as usize;
    let e_shnum = read_u16_le(data, 0x30)? as usize;
    let e_shstrndx = read_u16_le(data, 0x32)? as usize;
    if e_shoff == 0 || e_shentsize == 0 || e_shnum == 0 {
        return Err("ELF missing section headers".into());
    }
    if e_shstrndx >= e_shnum {
        return Err(format!("invalid e_shstrndx={e_shstrndx} for e_shnum={e_shnum}"));
    }

    let read_shdr = |idx: usize| -> Result<(u32, u32, u32, u32), String> {
        let off = e_shoff
            .checked_add(
                idx.checked_mul(e_shentsize)
                    .ok_or_else(|| "section index overflow".to_string())?,
            )
            .ok_or_else(|| "section header offset overflow".to_string())?;
        let sh_name = read_u32_le(data, off)?;
        let sh_addr = read_u32_le(data, off + 0x0C)?;
        let sh_offset = read_u32_le(data, off + 0x10)?;
        let sh_size = read_u32_le(data, off + 0x14)?;
        Ok((sh_name, sh_addr, sh_offset, sh_size))
    };

    let (_shstr_name, _shstr_addr, shstr_off, shstr_size) = read_shdr(e_shstrndx)?;
    let shstr_start = shstr_off as usize;
    let shstr_end = shstr_start
        .checked_add(shstr_size as usize)
        .ok_or_else(|| "shstrtab overflow".to_string())?;
    let shstr = data
        .get(shstr_start..shstr_end)
        .ok_or_else(|| "shstrtab out of bounds".to_string())?;

    for i in 0..e_shnum {
        let (sh_name, sh_addr, sh_offset, sh_size) = read_shdr(i)?;
        let name = read_cstr(shstr, sh_name as usize)?;
        if name != section_name {
            continue;
        }
        let start = sh_offset as usize;
        let end = start
            .checked_add(sh_size as usize)
            .ok_or_else(|| format!("section {section_name} size overflow"))?;
        let bytes = data
            .get(start..end)
            .ok_or_else(|| format!("section {section_name} out of bounds"))?;
        return Ok((sh_addr, bytes.to_vec()));
    }

    Err(format!("missing section {section_name}"))
}

fn guest_dir(guest_name: &str) -> PathBuf {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .join("riscv-tests")
        .join("guests")
        .join(guest_name)
}

fn with_poseidon_input(mut wiring: Rv32TraceWiring) -> Rv32TraceWiring {
    for (i, &value) in INPUT_WORDS_U64.iter().enumerate() {
        let base = INPUT_ADDR + (i as u64) * 8;
        let lo = (value & 0xFFFF_FFFF) as u32;
        let hi = (value >> 32) as u32;
        wiring = wiring.ram_init_u32(base, lo).ram_init_u32(base + 4, hi);
    }
    wiring
}

fn bench_guest(guest_name: &'static str, bin_name: &'static str) -> GuestBench {
    let guest_dir = guest_dir(guest_name);
    let elf_path = guest_dir
        .join("target")
        .join("riscv32im-unknown-none-elf")
        .join("release")
        .join(bin_name);
    if !elf_path.exists() {
        panic!(
            "missing ELF {}.\nBuild it first with:\n  cd {}\n  cargo build --release",
            elf_path.display(),
            guest_dir.display()
        );
    }
    let elf = fs::read(&elf_path).unwrap_or_else(|e| panic!("read ELF {} failed: {e}", elf_path.display()));
    let elf_size_bytes = elf.len();
    let (program_base, rom_bytes) = extract_elf32_section(&elf, ".neo_start")
        .unwrap_or_else(|e| panic!("extract .neo_start from {} failed: {e}", elf_path.display()));
    assert_eq!(
        program_base,
        0,
        "expected .neo_start base to be zero for {}",
        elf_path.display()
    );

    let wiring = with_poseidon_input(
        Rv32TraceWiring::from_rom(program_base as u64, &rom_bytes)
            .xlen(32)
            .mode(FoldingMode::Optimized)
            .min_trace_len(8)
            .chunk_rows(CHUNK_ROWS)
            .shout_auto_minimal(),
    );

    let prove_wall_start = Instant::now();
    let mut run = wiring
        .prove()
        .unwrap_or_else(|e| panic!("prove failed for {guest_name}: {e}"));
    let prove_wall = prove_wall_start.elapsed();
    let prove_internal = run.prove_duration();

    let verify_wall_start = Instant::now();
    run.verify()
        .unwrap_or_else(|e| panic!("verify failed for {guest_name}: {e}"));
    let verify_wall = verify_wall_start.elapsed();
    let verify_internal = run.verify_duration();

    let phases = run.prove_phase_durations();
    GuestBench {
        name: guest_name,
        elf_size_bytes,
        rom_size_bytes: rom_bytes.len(),
        rom_words: rom_bytes.len() / 4,
        trace_rows: run.trace_len(),
        fold_steps: run.fold_count(),
        requires_poseidon_stage: run.requires_poseidon_stage(),
        prove_wall,
        prove_internal,
        setup_ms: phases.setup.as_secs_f64() * 1000.0,
        chunk_build_commit_ms: phases.chunk_build_commit.as_secs_f64() * 1000.0,
        fold_and_prove_ms: phases.fold_and_prove.as_secs_f64() * 1000.0,
        verify_wall,
        verify_internal,
    }
}

fn ratio(a: Duration, b: Duration) -> f64 {
    if b.is_zero() {
        return f64::INFINITY;
    }
    a.as_secs_f64() / b.as_secs_f64()
}

#[test]
#[ignore = "benchmark: precompile-only prove/verify timing"]
fn bench_poseidon_precompile_from_elf() {
    let pre = bench_guest("rv32-poseidon2", "rv32_poseidon2");
    println!("==== poseidon_elf_bench (precompile only) ====");
    println!("input_u64={:?}", INPUT_WORDS_U64);
    println!(
        "trace_rows={} fold_steps={} poseidon_stage={}",
        pre.trace_rows, pre.fold_steps, pre.requires_poseidon_stage
    );
    println!(
        "prove_wall_ms={:.1} setup_ms={:.1} chunk_build_commit_ms={:.1} fold_and_prove_ms={:.1}",
        pre.prove_wall.as_secs_f64() * 1000.0,
        pre.setup_ms,
        pre.chunk_build_commit_ms,
        pre.fold_and_prove_ms
    );
    println!("verify_wall_ms={:.1}", pre.verify_wall.as_secs_f64() * 1000.0);
    println!("============================");
}

#[test]
#[ignore = "benchmark: precompile vs soft comparison"]
fn compare_poseidon_precompile_vs_soft_from_elf() {
    let pre = bench_guest("rv32-poseidon2", "rv32_poseidon2");
    let soft = bench_guest("rv32-poseidon2-soft", "rv32_poseidon2_soft");

    println!("==== poseidon_elf_bench ====");
    println!("input_u64={:?}", INPUT_WORDS_U64);
    for row in [&pre, &soft] {
        println!(
            "name={} elf_bytes={} neo_start_bytes={} neo_start_words={} trace_rows={} fold_steps={} requires_poseidon_stage={} prove_wall_ms={:.3} prove_internal_ms={:.3} verify_wall_ms={:.3} verify_internal_ms={:.3}",
            row.name,
            row.elf_size_bytes,
            row.rom_size_bytes,
            row.rom_words,
            row.trace_rows,
            row.fold_steps,
            row.requires_poseidon_stage,
            row.prove_wall.as_secs_f64() * 1000.0,
            row.prove_internal.as_secs_f64() * 1000.0,
            row.verify_wall.as_secs_f64() * 1000.0,
            row.verify_internal.unwrap_or(row.verify_wall).as_secs_f64() * 1000.0
        );
    }
    println!(
        "delta soft/precompile prove_wall_x={:.3} verify_wall_x={:.3} trace_rows_x={:.3}",
        ratio(soft.prove_wall, pre.prove_wall),
        ratio(soft.verify_wall, pre.verify_wall),
        (soft.trace_rows as f64) / (pre.trace_rows as f64)
    );
    println!("============================");
}
