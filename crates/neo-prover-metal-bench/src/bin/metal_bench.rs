use neo_prover_metal_bench::{run_benchmark_json, BenchmarkConfig};

fn main() {
    let mut config = BenchmarkConfig::default();
    for argument in std::env::args().skip(1) {
        match argument.as_str() {
            "--smoke" => config = BenchmarkConfig::smoke(),
            "--m6" => config = BenchmarkConfig::m6(),
            "--primitives-only" => {
                config.run_sha256_lifecycle = false;
                config.run_nebula_lifecycle = false;
            }
            "-h" | "--help" => {
                println!("Usage: metal_bench [--smoke|--m6] [--primitives-only]");
                return;
            }
            _ => {
                eprintln!("unknown argument: {argument}");
                std::process::exit(2);
            }
        }
    }
    match run_benchmark_json(config) {
        Ok(report) => println!("{report}"),
        Err(error) => {
            eprintln!("{error}");
            std::process::exit(1);
        }
    }
}
