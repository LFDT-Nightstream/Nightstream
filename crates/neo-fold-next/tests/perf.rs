//! Performance/debugging reports for the current CHIP-8 proof path.

#[path = "support/chip8.rs"]
mod chip8_support;

use std::time::Instant;

use serde::Serialize;

use neo_fold_next::chip8::decider::{prove_chip8_spartan2_decider, setup_chip8_spartan2_decider};
use neo_fold_next::chip8::proof::prove_recursive as prove_chip8_recursive;
use neo_fold_next::nightstream::chip8::{
    build_chip8_nightstream_from_recursive_proof, verify_chip8_nightstream_from_recursive_proof,
};

#[derive(Clone, Copy)]
struct SerializedSizeRow<'a> {
    label: &'a str,
    bytes: usize,
}

fn millis_since(started: Instant) -> f64 {
    started.elapsed().as_secs_f64() * 1_000.0
}

fn bytes_to_kib(bytes: usize) -> f64 {
    bytes as f64 / 1024.0
}

fn serialized_size_bytes<T: Serialize>(value: &T) -> usize {
    bincode::serialized_size(value).expect("measure serialized size") as usize
}

fn print_section(title: &str) {
    println!();
    println!("{title}");
    println!("{}", "=".repeat(title.len()));
}

fn print_kv(label: &str, value: impl std::fmt::Display) {
    println!("  {:32} {}", label, value);
}

fn print_serialized_size_table(title: &str, rows: &[SerializedSizeRow<'_>], total_bytes: usize) {
    print_section(title);
    println!("  {:32} {:>10} {:>10}", "component", "bytes", "KiB");
    for row in rows {
        println!("  {:32} {:>10} {:>10.3}", row.label, row.bytes, bytes_to_kib(row.bytes));
    }
    println!(
        "  {:32} {:>10} {:>10.3}",
        "total",
        total_bytes,
        bytes_to_kib(total_bytes)
    );
}

#[test]
#[ignore = "performance/debugging snapshot; run with --release -- --ignored --nocapture"]
fn chip8_nightstream_perf_snapshot() {
    let input = chip8_support::build_jump_kernel_input(4);

    let recursive_started = Instant::now();
    let (recursive_statement, final_proof) = prove_chip8_recursive(&input).expect("prove chip8 recursive");
    let recursive_ms = millis_since(recursive_started);

    let decider_setup_started = Instant::now();
    let (decider_pk, _decider_vk) =
        setup_chip8_spartan2_decider(&recursive_statement, &final_proof).expect("setup chip8 spartan2 decider");
    let decider_setup_ms = millis_since(decider_setup_started);

    let decider_prove_started = Instant::now();
    let decider_proof = prove_chip8_spartan2_decider(&decider_pk, &recursive_statement, &final_proof)
        .expect("prove chip8 spartan2 decider");
    let decider_prove_ms = millis_since(decider_prove_started);

    let nightstream_build_started = Instant::now();
    let (nightstream_statement, nightstream_proof) =
        build_chip8_nightstream_from_recursive_proof(&recursive_statement, &final_proof)
            .expect("build chip8 nightstream proof");
    let nightstream_build_ms = millis_since(nightstream_build_started);

    let nightstream_verify_started = Instant::now();
    verify_chip8_nightstream_from_recursive_proof(
        &recursive_statement,
        &final_proof,
        &nightstream_statement,
        &nightstream_proof,
    )
    .expect("verify chip8 nightstream proof");
    let nightstream_verify_ms = millis_since(nightstream_verify_started);

    let decider_proof_bytes = serialized_size_bytes(&decider_proof);
    let nightstream_serialized_sizes = [
        SerializedSizeRow {
            label: "nightstream.total",
            bytes: serialized_size_bytes(&(nightstream_statement.clone(), nightstream_proof.clone())),
        },
        SerializedSizeRow {
            label: "nightstream.statement",
            bytes: serialized_size_bytes(&nightstream_statement),
        },
        SerializedSizeRow {
            label: "nightstream.proof",
            bytes: serialized_size_bytes(&nightstream_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_decider_proof",
            bytes: serialized_size_bytes(&nightstream_proof.main_decider_proof),
        },
        SerializedSizeRow {
            label: "nightstream.main_residual_proof",
            bytes: serialized_size_bytes(&nightstream_proof.main_residual_proof),
        },
    ];
    let nightstream_total_bytes = nightstream_serialized_sizes[0].bytes;

    print_section("CHIP-8 Nightstream Perf Snapshot");
    print_kv("semantic_rows", input.witness.semantic_trace_rows.len());
    print_kv("chunk_count", recursive_statement.folded.chunk_count);
    print_kv("semantic_step_count", recursive_statement.folded.semantic_step_count);
    print_kv(
        "fold_schedule",
        format!("{:?}", recursive_statement.folded.fold_schedule),
    );
    print_kv("final_pc_word", recursive_statement.final_state.pc_word);

    print_section("Raw Timing");
    print_kv("prove_chip8_recursive", format!("{recursive_ms:.3} ms"));
    print_kv("setup_chip8_spartan2_decider", format!("{decider_setup_ms:.3} ms"));
    print_kv("prove_chip8_spartan2_decider", format!("{decider_prove_ms:.3} ms"));
    print_kv("build_chip8_nightstream", format!("{nightstream_build_ms:.3} ms"));
    print_kv("verify_chip8_nightstream", format!("{nightstream_verify_ms:.3} ms"));

    print_section("Nightstream Published Boundary");
    print_kv(
        "spartan_decider_proof_size",
        format!(
            "{decider_proof_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(decider_proof_bytes)
        ),
    );
    print_serialized_size_table(
        "Serialized Sizes (Nightstream)",
        &nightstream_serialized_sizes,
        nightstream_total_bytes,
    );

    print_section("Final Summary");
    print_kv(
        "nightstream published size",
        format!(
            "{nightstream_total_bytes} bytes ({:.3} KiB)",
            bytes_to_kib(nightstream_total_bytes)
        ),
    );
}
