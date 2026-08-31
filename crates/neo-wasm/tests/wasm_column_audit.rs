use std::{fs, path::Path, process::Command};

fn workspace_root() -> &'static Path {
    Path::new(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("neo-wasm belongs to the workspace crates directory")
}

#[test]
fn html_report_embeds_unique_rows_and_per_column_references() {
    let output = Command::new(env!("CARGO_BIN_EXE_wasm_column_audit"))
        .output()
        .expect("generate wasm column audit HTML");
    assert!(
        output.status.success(),
        "HTML generation failed: {}",
        String::from_utf8_lossy(&output.stderr)
    );

    let html = fs::read_to_string(workspace_root().join("target/wasm-column-audit/index.html"))
        .expect("read generated audit HTML");
    let marker = "<script id=\"report-data\" type=\"application/json\">";
    let json_start = html.find(marker).expect("embedded report data") + marker.len();
    let json_end = html[json_start..]
        .find("</script>")
        .expect("end of embedded report data")
        + json_start;
    let report: serde_json::Value = serde_json::from_str(&html[json_start..json_end]).expect("valid embedded JSON");
    let columns = report["columns"].as_array().expect("column records");
    let rows = report["rows"].as_array().expect("unique row records");

    assert_eq!(report["title"], "WASM column audit");
    assert_eq!(columns.len(), neo_wasm::RANGE_CHECKED_WITNESS_WIDTH);
    assert_eq!(
        rows.len(),
        neo_wasm::build_wasm_relation()
            .unwrap()
            .r1cs()
            .catalog()
            .len()
    );
    let select_condition = columns
        .iter()
        .find(|column| column["name"] == "COL_SELECT_COND_IS_ZERO")
        .expect("select condition column");
    let row_indices = select_condition["rowIndices"]
        .as_array()
        .expect("row references");
    assert!(row_indices.windows(2).all(|pair| pair[0] != pair[1]));
    let range_bit = columns
        .iter()
        .find(|column| column["region"] == neo_wasm::RANGE_BITS_REGION)
        .expect("range-bit column");
    assert_eq!(range_bit["generated"], true);
    assert!(html.contains("function searchColumns(query)"));
    assert!(html.contains("function approximateDamerauLevenshtein(pattern, text)"));
    assert!(html.contains("term.column === columnIndex ? ' match'"));
    assert!(html.contains("R1CS rows show pre-coalescing builder terms"));
    assert!(!html.contains("regionName !== 'range_bits'"));
}
