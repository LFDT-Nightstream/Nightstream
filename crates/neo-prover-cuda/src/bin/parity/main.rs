//! CPU-vs-CUDA parity gates, one per accelerated protocol piece.
//!
//! Run via `cargo +nightly-2026-04-03 oxide run --features cuda --bin parity`.
//! With no argument every gate runs in order (cargo-oxide does not forward
//! program args); `argv[1]` selects a single gate when the built binary is
//! invoked directly. Every gate asserts field-identical results against the
//! canonical CPU implementation and panics on any mismatch.

mod fixtures;
mod gates;
mod sha256_workload;

const GATES: [(&str, fn()); 36] = [
    ("smoke", gates::smoke),
    ("transcript", gates::transcript),
    ("ajtai", gates::ajtai),
    ("fresh", gates::fresh),
    ("fresh_bench", gates::fresh_bench),
    ("dec", gates::dec),
    ("dec_bench", gates::dec_bench),
    ("ccs_fe", gates::ccs_fe),
    ("ccs_nc", gates::ccs_nc),
    ("ccs_prove", gates::ccs_prove),
    ("ccs_output_digest", gates::ccs_output_digest),
    ("ccs_graph_replay", gates::ccs_graph_replay),
    ("ccs_phase_summary", gates::ccs_phase_summary),
    ("ccs_graph_replay_bench", gates::ccs_graph_replay_bench),
    ("ccs_phase_bench", gates::ccs_phase_bench),
    ("ccs_bench", gates::ccs_bench),
    ("rlc", gates::rlc),
    ("rlc_bench", gates::rlc_bench),
    ("nifs", gates::nifs),
    ("nifs_nebula", gates::nifs_nebula),
    ("nebula_lifecycle", gates::nebula_lifecycle),
    ("nifs_whole_phase", gates::nifs_whole_phase),
    ("nifs_bench", gates::nifs_bench),
    ("e2e_bench", gates::e2e_bench),
    ("e2e_gpu_fast_bench", gates::e2e_gpu_fast_bench),
    ("e2e_multichain_bench", gates::e2e_multichain_bench),
    ("e2e_multichain8_bench", gates::e2e_multichain8_bench),
    ("e2e_multichain8_fast_bench", gates::e2e_multichain8_fast_bench),
    ("e2e_multichain16_fast_bench", gates::e2e_multichain16_fast_bench),
    ("e2e_whole_fe_bench", gates::e2e_whole_fe_bench),
    ("e2e_whole_fe_fast_bench", gates::e2e_whole_fe_fast_bench),
    ("e2e_graph_bench", gates::e2e_graph_bench),
    ("e2e_graph_once_bench", gates::e2e_graph_once_bench),
    ("e2e_graph_two_bench", gates::e2e_graph_two_bench),
    ("e2e_graph_three_bench", gates::e2e_graph_three_bench),
    (
        "e2e_graph_three_recapture_bench",
        gates::e2e_graph_three_recapture_bench,
    ),
];

fn main() {
    match std::env::args().nth(1).as_deref() {
        None => {
            for (name, gate) in GATES {
                eprintln!("[parity] running `{name}`");
                gate();
            }
        }
        // Parity-only sweep: every gate except the real-scale benches.
        Some("quick") => {
            for (name, gate) in GATES.iter().filter(|(name, _)| !name.ends_with("_bench")) {
                eprintln!("[parity] running `{name}`");
                gate();
            }
        }
        Some(name) => match GATES.iter().find(|(gate_name, _)| *gate_name == name) {
            Some((_, gate)) => gate(),
            None => {
                let names: Vec<&str> = GATES.iter().map(|(gate_name, _)| *gate_name).collect();
                eprintln!("unknown parity gate `{name}`; available: {}", names.join(", "));
                std::process::exit(2);
            }
        },
    }
}
