use super::*;

const RECURSIVE_FAMILIES: &[&str] = &[
    "fprime.recursive.prelude",
    "fprime.recursive.transcript",
    "fprime.recursive.nifs",
    "fprime.recursive.prior_link",
    "fprime.recursive.nebula",
    "fprime.recursive.accumulator",
    "fprime.recursive.counter",
    "fprime.recursive.output",
];

const TERMINAL_FAMILIES: &[&str] = &[
    "terminal.nifs",
    "terminal.running_link",
    "terminal.parent_link",
    "terminal.latest_link",
    "terminal.accumulator",
];

const RECURSIVE_NIFS_FAMILIES: &[&str] = &["nifs.pi_ccs", "nifs.pi_rlc", "nifs.pi_dec", "nifs.point_binding"];

const TERMINAL_NIFS_FAMILIES: &[&str] = &[
    "terminal.transcript",
    "nifs.pi_ccs",
    "nifs.pi_rlc",
    "nifs.pi_dec",
    "nifs.point_binding",
];

const PI_CCS_FAMILIES: &[&str] = &[
    "nifs.pi_ccs.allocation",
    "nifs.pi_ccs.authority",
    "nifs.pi_ccs.fresh_digests",
    "nifs.pi_ccs.running_authority",
    "nifs.pi_ccs.transcript",
    "nifs.pi_ccs.fe_initial",
    "nifs.pi_ccs.fe_sumcheck",
    "nifs.pi_ccs.nc_sumcheck",
    "nifs.pi_ccs.output_binding",
    "nifs.pi_ccs.fe_terminal",
    "nifs.pi_ccs.nc_terminal",
    "nifs.pi_ccs.catchup",
];

const PI_RLC_FAMILIES: &[&str] = &[
    "nifs.pi_rlc.transcript_rhos",
    "nifs.pi_rlc.shape",
    "nifs.pi_rlc.linear_folds",
    "nifs.pi_rlc.projection_binding",
    "nifs.pi_rlc.projection_shared",
    "nifs.pi_rlc.projection_identities",
];

fn unique_range_inside<'a>(builder: &'a R1csBuilder, parent: &RowFamilyRange, name: &str) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name && parent.row_start <= range.row_start && range.row_end <= parent.row_end)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one {name} range inside {}", parent.name);
    matches[0]
}

fn named_partition<'a>(builder: &'a R1csBuilder, parent: &RowFamilyRange, names: &[&str]) -> Vec<&'a RowFamilyRange> {
    let mut ranges = names
        .iter()
        .map(|name| unique_range_inside(builder, parent, name))
        .collect::<Vec<_>>();
    ranges.sort_by_key(|range| range.row_start);
    let mut cursor = parent.row_start;
    for range in &ranges {
        assert_eq!(
            range.row_start, cursor,
            "gap or overlap before {} inside {}",
            range.name, parent.name
        );
        cursor = range.row_end;
    }
    assert_eq!(cursor, parent.row_end, "subowners do not cover {}", parent.name);
    ranges
}

fn parent_range<'a>(builder: &'a R1csBuilder, name: &str) -> &'a RowFamilyRange {
    let matches = builder
        .row_family_ranges()
        .iter()
        .filter(|range| range.name == name)
        .collect::<Vec<_>>();
    assert_eq!(matches.len(), 1, "expected one parent owner {name}");
    matches[0]
}

pub fn nested_full_history_manifest(builder: &R1csBuilder) -> Value {
    let recursive = parent_range(builder, "decider.step.recursive");
    let recursive_families = named_partition(builder, recursive, RECURSIVE_FAMILIES);
    let recursive_nifs = unique_range_inside(builder, recursive, "fprime.recursive.nifs");
    let recursive_nifs_families = named_partition(builder, recursive_nifs, RECURSIVE_NIFS_FAMILIES);
    let recursive_pi_ccs = unique_range_inside(builder, recursive_nifs, "nifs.pi_ccs");
    let recursive_pi_ccs_families = named_partition(builder, recursive_pi_ccs, PI_CCS_FAMILIES);
    let recursive_pi_rlc = unique_range_inside(builder, recursive_nifs, "nifs.pi_rlc");
    let recursive_pi_rlc_families = named_partition(builder, recursive_pi_rlc, PI_RLC_FAMILIES);

    let terminal = parent_range(builder, "decider.terminal_fold");
    let terminal_families = named_partition(builder, terminal, TERMINAL_FAMILIES);
    let terminal_nifs = unique_range_inside(builder, terminal, "terminal.nifs");
    let terminal_nifs_families = named_partition(builder, terminal_nifs, TERMINAL_NIFS_FAMILIES);
    let terminal_pi_ccs = unique_range_inside(builder, terminal_nifs, "nifs.pi_ccs");
    let terminal_pi_ccs_families = named_partition(builder, terminal_pi_ccs, PI_CCS_FAMILIES);
    let terminal_pi_rlc = unique_range_inside(builder, terminal_nifs, "nifs.pi_rlc");
    let terminal_pi_rlc_families = named_partition(builder, terminal_pi_rlc, PI_RLC_FAMILIES);

    json!({
        "recursive_families": recursive_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "recursive_nifs_families": recursive_nifs_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "recursive_pi_ccs_families": recursive_pi_ccs_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "recursive_pi_rlc_families": recursive_pi_rlc_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "terminal_families": terminal_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "terminal_nifs_families": terminal_nifs_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "terminal_pi_ccs_families": terminal_pi_ccs_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
        "terminal_pi_rlc_families": terminal_pi_rlc_families
            .iter()
            .map(|range| full_history_range_json(builder, range))
            .collect::<Vec<_>>(),
    })
}

fn render_range_list(name: &str, ranges: &[Value]) -> String {
    let mut rendered = format!("def {name} : List FPrimeRecursiveManifest.RowRange :=\n");
    for (index, range) in ranges.iter().enumerate() {
        let prefix = if index == 0 { "  [" } else { "  ," };
        writeln!(
            rendered,
            "{prefix} {{ name := {}, rowStart := {}, rowEnd := {}, nonzeroEntries := {}, sha256 := {} }}",
            full_history_lean_string(range["name"].as_str().expect("range name")),
            range["row_start"],
            range["row_end"],
            range["nonzero_entries"],
            full_history_lean_string(range["sha256"].as_str().expect("range hash")),
        )
        .expect("render nested range");
    }
    rendered.push_str("  ]\n");
    rendered
}

pub fn render_nested_lean_definitions(manifest: &Value) -> String {
    let mut rendered = String::new();
    for (json_name, lean_name) in [
        ("recursive_families", "recursiveFamilies"),
        ("recursive_nifs_families", "recursiveNifsFamilies"),
        ("recursive_pi_ccs_families", "recursivePiCcsFamilies"),
        ("recursive_pi_rlc_families", "recursivePiRlcFamilies"),
        ("terminal_families", "terminalFamilies"),
        ("terminal_nifs_families", "terminalNifsFamilies"),
        ("terminal_pi_ccs_families", "terminalPiCcsFamilies"),
        ("terminal_pi_rlc_families", "terminalPiRlcFamilies"),
    ] {
        rendered.push_str(&render_range_list(
            lean_name,
            manifest[json_name].as_array().expect("nested ranges"),
        ));
    }
    rendered
}
