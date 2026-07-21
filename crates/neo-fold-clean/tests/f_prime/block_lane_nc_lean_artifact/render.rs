//! Exact fixture, boundary, ownership-range, and round-map rendering.

use std::fmt::Write as _;
use std::ops::Range;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcBlockLaneNcSelectiveRowsAudit;

use super::{
    generated_header, lean_nat_list, GeneratedLeanFile, TinyFixtureScope, GENERATED_ROOT, IMPORT_ROOT, NAMESPACE_ROOT,
};

const STEADY_ARM: usize = 2;
const MAX_NESTED_RECORDS: usize = 256;

fn lean_range(range: Range<usize>) -> String {
    format!("{{ start := {}, stop := {} }}", range.start, range.end)
}

fn lean_pair(pair: [usize; 2]) -> String {
    format!("{{ c0 := {}, c1 := {} }}", pair[0], pair[1])
}

fn lean_pairs(values: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .copied()
            .map(lean_pair)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_pair_matrix(values: &[Vec<[usize; 2]>]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(|row| lean_pairs(row))
            .collect::<Vec<_>>()
            .join(",\n       ")
    )
}

fn lean_ranges(values: &[Range<usize>]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .cloned()
            .map(lean_range)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

pub(super) fn source_shape(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> (usize, usize) {
    let final_round = audit
        .fixed_point()
        .rounds()
        .last()
        .expect("fixed-point audit contains its stabilizing round");
    let arm = final_round.arms[STEADY_ARM];
    (arm.rows, arm.columns)
}

fn render_boundary(contents: &mut String, audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) {
    let boundary = audit.boundary();
    let (source_rows, source_columns) = source_shape(audit);
    let pending_old_block = boundary
        .pending_old_block_cols
        .as_deref()
        .expect("recursive boundary exports pending old-block columns");
    let pending_parent = boundary
        .pending_parent_y_zcol_cols
        .as_deref()
        .expect("recursive boundary exports pending parent y_zcol columns");
    assert!(audit.output_padding_source_ranges().len() <= MAX_NESTED_RECORDS);
    assert!(boundary.beta_lane_cols.len() <= MAX_NESTED_RECORDS);
    assert!(boundary.beta_block_cols.len() <= MAX_NESTED_RECORDS);
    assert!(pending_old_block.len() <= MAX_NESTED_RECORDS);
    assert!(pending_parent.len() <= MAX_NESTED_RECORDS);
    assert!(boundary.output_y_zcol_cols.len() <= MAX_NESTED_RECORDS);
    assert!(boundary
        .output_y_zcol_cols
        .iter()
        .all(|output| output.len() <= MAX_NESTED_RECORDS));
    assert!(boundary.block_point_cols.len() <= MAX_NESTED_RECORDS);
    assert!(boundary.lane_point_cols.len() <= MAX_NESTED_RECORDS);
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def boundary : RawBoundaryMap :=").expect("render boundary header");
    writeln!(contents, "  {{ schemaVersion := 1").expect("render boundary schema");
    writeln!(contents, "    sourceRows := {source_rows}").expect("render source rows");
    writeln!(contents, "    sourceColumns := {source_columns}").expect("render source columns");
    contents.push_str("    constantOneColumn := 0\n");
    writeln!(
        contents,
        "    claimedInitialRows := {}",
        lean_range(boundary.claimed_initial_rows.clone())
    )
    .expect("render claimed-initial rows");
    writeln!(
        contents,
        "    terminalIdentityRows := {}",
        lean_range(boundary.terminal_identity_rows.clone())
    )
    .expect("render terminal rows");
    writeln!(
        contents,
        "    terminalFinalEqualityRows := {}",
        lean_range(boundary.terminal_final_equality_rows.clone())
    )
    .expect("render terminal equality rows");
    writeln!(
        contents,
        "    outputYZcolPaddingRows := {}",
        lean_ranges(audit.output_padding_source_ranges())
    )
    .expect("render output padding ranges");
    writeln!(contents, "    gammaColumns := {}", lean_pair(boundary.gamma_cols)).expect("render gamma");
    writeln!(
        contents,
        "    betaLaneColumns := {}",
        lean_pairs(&boundary.beta_lane_cols)
    )
    .expect("render lane beta");
    writeln!(
        contents,
        "    betaBlockColumns := {}",
        lean_pairs(&boundary.beta_block_cols)
    )
    .expect("render block beta");
    writeln!(
        contents,
        "    producerBetaColumns := {}",
        lean_pair(boundary.producer_beta_cols)
    )
    .expect("render producer beta");
    writeln!(
        contents,
        "    batchWeightColumns := {}",
        lean_pair(boundary.batch_weight_cols)
    )
    .expect("render batch weight");
    writeln!(
        contents,
        "    pendingOldBlockColumns := {}",
        lean_pairs(pending_old_block)
    )
    .expect("render pending block");
    writeln!(
        contents,
        "    pendingParentYZcolColumns := {}",
        lean_pairs(pending_parent)
    )
    .expect("render pending parent");
    writeln!(
        contents,
        "    outputYZcolColumns := {}",
        lean_pair_matrix(&boundary.output_y_zcol_cols)
    )
    .expect("render output columns");
    writeln!(
        contents,
        "    blockPointColumns := {}",
        lean_pairs(&boundary.block_point_cols)
    )
    .expect("render block point");
    writeln!(
        contents,
        "    lanePointColumns := {}",
        lean_pairs(&boundary.lane_point_cols)
    )
    .expect("render lane point");
    writeln!(
        contents,
        "    claimedInitialColumns := {}",
        lean_pair(boundary.claimed_initial_cols)
    )
    .expect("render initial columns");
    writeln!(
        contents,
        "    finalSumColumns := {}",
        lean_pair(boundary.final_sum_cols)
    )
    .expect("render final sum");
    writeln!(
        contents,
        "    terminalRhsColumns := {} }}",
        lean_pair(boundary.terminal_rhs_cols)
    )
    .expect("render terminal rhs");
}

pub(super) fn metadata(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit, fixture: TinyFixtureScope) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Metadata");
    let (source_rows, source_columns) = source_shape(audit);
    let projected = audit.projected_rows();
    let provenance = projected
        .source_provenance()
        .expect("combined-NC audit contains source provenance");
    let decoder = projected
        .decoder_provenance()
        .expect("combined-NC audit contains decoder provenance");
    assert_eq!(provenance.arm(), STEADY_ARM, "source provenance arm");
    assert_eq!(decoder.arm(), STEADY_ARM, "decoder provenance arm");
    assert_eq!(audit.source_rows().len(), audit.source_row_artifacts().len());
    assert!(
        audit.source_row_ranges().len() <= MAX_NESTED_RECORDS,
        "source ownership ranges require another proof-free shard"
    );
    assert!(
        audit
            .source_rows()
            .iter()
            .copied()
            .eq(audit.source_row_artifacts().iter().map(|row| row.index())),
        "source row indices and coefficient artifacts agree"
    );

    let mut contents =
        generated_header("the exact bounded fixture scope, row ownership ranges, and delayed-NC boundary");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render metadata import");
    writeln!(contents, "namespace {namespace}\n").expect("render metadata namespace");
    writeln!(
        contents,
        "def parameterConstraintCount : Nat := {}",
        fixture.parameter_constraint_count
    )
    .expect("render parameter rows");
    writeln!(contents, "def commitmentWidth : Nat := {}", fixture.commitment_width).expect("render commitment width");
    writeln!(contents, "def securityBits : Nat := {}", fixture.security_bits).expect("render security bits");
    writeln!(
        contents,
        "def applicationRows : Nat := {}",
        fixture.application_row_count
    )
    .expect("render app rows");
    writeln!(
        contents,
        "def applicationColumns : Nat := {}",
        fixture.application_column_count
    )
    .expect("render app columns");
    writeln!(
        contents,
        "def applicationPublicInputs : Nat := {}",
        fixture.application_public_input_count
    )
    .expect("render app public inputs");
    writeln!(contents, "def sourceRelationRows : Nat := {source_rows}").expect("render source rows");
    writeln!(contents, "def sourceRelationColumns : Nat := {source_columns}").expect("render source columns");
    writeln!(contents, "def finalRelationRows : Nat := {}", projected.rows()).expect("render final rows");
    writeln!(contents, "def finalRelationColumns : Nat := {}", projected.columns()).expect("render final columns");
    writeln!(
        contents,
        "def steadySelectorColumn : Nat := {}",
        projected.selector_columns()[STEADY_ARM]
    )
    .expect("render steady selector");
    writeln!(contents, "def sourceRowCount : Nat := {}", audit.source_rows().len()).expect("render source row count");
    writeln!(
        contents,
        "def emittedRowCount : Nat := {}",
        projected.row_artifacts().len()
    )
    .expect("render emitted row count");
    writeln!(
        contents,
        "def sourceColumnCount : Nat := {}",
        provenance.source_columns().len()
    )
    .expect("render source column count");
    writeln!(contents, "def decoderCount : Nat := {}", decoder.decoders().len()).expect("render decoder count");
    writeln!(
        contents,
        "def sourceRowRanges : List RawRowRange := {}",
        lean_ranges(audit.source_row_ranges())
    )
    .expect("render source row ranges");
    contents.push('\n');
    render_boundary(&mut contents, audit);
    writeln!(contents, "\nend {namespace}").expect("render metadata namespace end");
    assert!(contents.lines().count() < 1_500, "metadata file line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Metadata.lean"),
        contents,
    }
}

pub(super) fn round_maps(audit: &R1csIvcBlockLaneNcSelectiveRowsAudit) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.RoundMaps");
    let (source_rows, source_columns) = source_shape(audit);
    assert_eq!(audit.rounds().len(), 25, "combined-NC round count");
    assert_eq!(audit.round_column_maps().len(), 25, "combined-NC round-map count");

    let mut contents = generated_header("25 exact proof-free local-to-source five-coefficient SumCheck round maps");
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render round import");
    writeln!(contents, "namespace {namespace}\n").expect("render round namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def values : List RawRoundMap := [\n");
    for (index, (round, map)) in audit
        .rounds()
        .iter()
        .zip(audit.round_column_maps())
        .enumerate()
    {
        assert!(round.allocated_cols.len() <= MAX_NESTED_RECORDS);
        assert!(round.coefficient_cols.len() <= MAX_NESTED_RECORDS);
        assert!(map.len() <= MAX_NESTED_RECORDS);
        if index != 0 {
            contents.push_str(",\n");
        }
        writeln!(contents, "  {{ schemaVersion := 1").expect("render round schema");
        writeln!(contents, "    sourceRows := {source_rows}").expect("render round source rows");
        writeln!(contents, "    sourceColumns := {source_columns}").expect("render round source columns");
        writeln!(contents, "    roundIndex := {index}").expect("render round index");
        writeln!(
            contents,
            "    rowRange := {}",
            lean_range(round.row_start..round.row_end)
        )
        .expect("render round rows");
        writeln!(contents, "    firstAllocatedColumn := {}", round.first_allocated_column)
            .expect("render allocation anchor");
        writeln!(
            contents,
            "    allocatedColumns := {}",
            lean_nat_list(round.allocated_cols.iter().copied())
        )
        .expect("render allocated columns");
        writeln!(
            contents,
            "    coefficientColumns := {}",
            lean_pairs(&round.coefficient_cols)
        )
        .expect("render coefficient columns");
        writeln!(contents, "    challengeColumns := {}", lean_pair(round.challenge_cols))
            .expect("render challenge columns");
        writeln!(contents, "    claimInColumns := {}", lean_pair(round.claim_in_cols))
            .expect("render claim-in columns");
        writeln!(contents, "    claimOutColumns := {}", lean_pair(round.claim_out_cols))
            .expect("render claim-out columns");
        writeln!(contents, "    columnMap := {} }}", lean_nat_list(map.iter().copied())).expect("render column map");
    }
    writeln!(contents, "]\n\nend {namespace}").expect("render round namespace end");
    assert!(contents.lines().count() < 1_500, "round-map file line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/RoundMaps.lean"),
        contents,
    }
}
