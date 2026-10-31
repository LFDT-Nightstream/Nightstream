//! Writes a self-contained HTML audit of the WASM application relation.

use std::{fs, path::Path};

use neo_application::render_column_audit_html;
use neo_wasm::{build_wasm_relation, build_wasm_relation_layout, RANGE_BITS_REGION};

fn main() {
    if let Err(error) = run() {
        eprintln!("error: {error}");
        eprintln!("usage: wasm_column_audit");
        std::process::exit(2);
    }
}

fn run() -> Result<(), String> {
    if std::env::args().nth(1).is_some() {
        return Err("this command takes no arguments".to_owned());
    }

    let relation = build_wasm_relation()?;
    let layout = build_wasm_relation_layout();
    let document = render_column_audit_html(
        "WASM column audit",
        &relation,
        &layout.auxiliary.memory,
        &layout.auxiliary.continuity,
        &[RANGE_BITS_REGION],
    );

    let workspace = Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neo-wasm belongs to the workspace crates directory");
    let output_dir = workspace.join("target/wasm-column-audit");
    fs::create_dir_all(&output_dir)
        .map_err(|error| format!("create audit output directory {}: {error}", output_dir.display()))?;
    let output = output_dir.join("index.html");
    fs::write(&output, document).map_err(|error| format!("write audit report {}: {error}", output.display()))?;
    println!("wrote {}", output.display());
    Ok(())
}
