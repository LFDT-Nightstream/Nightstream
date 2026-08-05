//! Drift gate for the bounded active strict-PiDEC source-R1CS artifact.
//!
//! Owns: the exact tiny-fixture (`kappa = 4`) outer steady-recursive PiDEC
//! layout and every sparse A/B/C row in its contiguous strict source range.
//!
//! Does not own: selective-CCS row materialization, decoder provenance,
//! witness values, production-security parameters, or
//! permission to remove constraints.

#[path = "../support/mod.rs"]
mod support;

use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};

use neo_fold_clean::engine::r1cs_circuit::builder::{PiDecClaimAudit, PiDecCommitmentAudit, PiDecStrictAudit};
use neo_fold_clean::frontends::r1cs_f_prime::ivc::{R1csIvcPiDecSourceRowAudit, R1csIvcRelation};
use neo_math::{D, F};
use p3_field::PrimeField64;
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

const GENERATED_DIRECTORY: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/Nifs/PiDec/Generated";
const SHARD_SIZE: usize = 250;
const ACTIVE_ROW_COUNT: usize = 11_845;
const ACTIVE_CHILD_COUNT: usize = 14;
const ACTIVE_LOGICAL_X: usize = 270;
const ACTIVE_MATRIX_COUNT: usize = 13;
const ACTIVE_POINT_DIMENSION: usize = 24;
const ACTIVE_COMMITMENT_ROWS: usize = 4;

struct GeneratedLeanFile {
    relative_path: String,
    contents: String,
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn compare_or_write_expected(root: &Path, file: GeneratedLeanFile, drifted: &mut Vec<String>) {
    let path = root.join(&file.relative_path);
    if fs::read_to_string(&path).unwrap_or_default() == file.contents {
        return;
    }
    let expected = path.with_extension("lean.expected");
    fs::create_dir_all(expected.parent().expect("generated PiDEC parent")).expect("create generated PiDEC directory");
    fs::write(&expected, file.contents).expect("write generated PiDEC candidate");
    drifted.push(
        expected
            .strip_prefix(root)
            .unwrap_or(&expected)
            .display()
            .to_string(),
    );
}

fn committed_lean_files(directory: &Path, root: &Path, files: &mut BTreeSet<String>) {
    if !directory.exists() {
        return;
    }
    for entry in fs::read_dir(directory).expect("read generated PiDEC directory") {
        let path = entry.expect("read generated PiDEC entry").path();
        if path.is_dir() {
            committed_lean_files(&path, root, files);
        } else if path
            .extension()
            .is_some_and(|extension| extension == "lean")
        {
            files.insert(
                path.strip_prefix(root)
                    .unwrap_or(&path)
                    .display()
                    .to_string(),
            );
        }
    }
}

fn lean_nat_list(values: impl IntoIterator<Item = usize>) -> String {
    format!(
        "[{}]",
        values
            .into_iter()
            .map(|value| value.to_string())
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_compact_nat_sequence(values: &[usize]) -> String {
    if values.is_empty() {
        return "[]".to_owned();
    }
    let mut pieces = Vec::new();
    let mut literals = Vec::new();
    let flush_literals = |literals: &mut Vec<usize>, pieces: &mut Vec<String>| {
        if !literals.is_empty() {
            pieces.push(lean_nat_list(literals.drain(..)));
        }
    };
    let mut index = 0;
    while index < values.len() {
        let mut repeated_end = index + 1;
        while repeated_end < values.len() && values[repeated_end] == values[index] {
            repeated_end += 1;
        }
        if repeated_end - index >= 4 {
            flush_literals(&mut literals, &mut pieces);
            pieces.push(format!("List.replicate {} {}", repeated_end - index, values[index]));
            index = repeated_end;
            continue;
        }
        if index + 3 < values.len() && values[index + 1] > values[index] {
            let step = values[index + 1] - values[index];
            let mut end = index + 2;
            while end < values.len() && values[end] > values[end - 1] && values[end] - values[end - 1] == step {
                end += 1;
            }
            if end - index >= 4 {
                flush_literals(&mut literals, &mut pieces);
                if values[index] == 0 && step == 1 {
                    pieces.push(format!("List.range {}", end - index));
                } else {
                    pieces.push(format!(
                        "((List.range {}).map (fun index => {} + {} * index))",
                        end - index,
                        values[index],
                        step
                    ));
                }
                index = end;
                continue;
            }
        }
        literals.push(values[index]);
        index += 1;
    }
    flush_literals(&mut literals, &mut pieces);
    pieces.join(" ++\n        ")
}

fn render_pairs(pairs: &[[usize; 2]]) -> String {
    format!(
        "[{}]",
        pairs
            .iter()
            .map(|pair| format!("({}, {})", pair[0], pair[1]))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn render_commitment(commitment: &PiDecCommitmentAudit) -> String {
    format!(
        "{{ dCol := {}, kappaCol := {}, dataCols := {} }}",
        commitment.d_col,
        commitment.kappa_col,
        lean_compact_nat_sequence(&commitment.data_cols)
    )
}

fn render_claim(claim: &PiDecClaimAudit) -> String {
    let active_width = claim.m_in.div_ceil(D);
    let active_x = (0..claim.x_rows)
        .flat_map(|row| (0..active_width).map(move |column| claim.x_cols[row * claim.x_width + column]))
        .collect::<Vec<_>>();
    let inactive_x = (0..claim.x_rows)
        .flat_map(|row| (active_width..claim.x_width).map(move |column| claim.x_cols[row * claim.x_width + column]))
        .collect::<Vec<_>>();
    let inactive_column = *inactive_x
        .first()
        .expect("active PiDEC claim has inactive public-X storage");
    assert!(
        inactive_x.iter().all(|&column| column == inactive_column),
        "active PiDEC inactive public-X coordinates share one zero wire"
    );
    assert!(claim.adv.is_none(), "active PiDEC carries no advice");
    let y_rows = claim
        .y_ring_cols
        .iter()
        .map(|row| lean_compact_nat_sequence(row))
        .collect::<Vec<_>>()
        .join(",\n        ");
    format!(
        "{{\n      commitment := {}\n      xActiveCols := {}\n      xInactiveCol := {}\n      xRows := {}\n      xWidth := {}\n      xRowsCol := {}\n      xWidthCol := {}\n      mIn := {}\n      mInCol := {}\n      yRingCols :=\n        [{}]\n      ctCols := {}\n      rCols := {}\n      foldDigestCols := {} }}",
        render_commitment(&claim.commitment),
        lean_compact_nat_sequence(&active_x),
        inactive_column,
        claim.x_rows,
        claim.x_width,
        claim.x_rows_col,
        claim.x_width_col,
        claim.m_in,
        claim.m_in_col,
        y_rows,
        render_pairs(&claim.ct_cols),
        render_pairs(&claim.r_cols),
        lean_compact_nat_sequence(&claim.fold_digest_cols),
    )
}

fn render_layout(strict: &PiDecStrictAudit) -> String {
    let children = strict
        .children
        .iter()
        .map(|child| format!("    {}", render_claim(child)))
        .collect::<Vec<_>>()
        .join(",\n");
    let traces = strict
        .x_sign_traces
        .iter()
        .map(|pair| format!("({}, {})", pair[0], pair[1]))
        .collect::<Vec<_>>()
        .join(", ");
    format!(
        "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Schema\n\n\
         /-! Generated bounded active strict-PiDEC source layout. Do not hand-edit.\n\n\
         Owns: the proof-free Rust-exported layout record.\n\n\
         Does not own: layout validity, compiler semantics, acceptance, or row removal.\n\n\
         Emits constraints: no.\n\n\
         | Payload | Meaning | Authority |\n\
         |---|---|---|\n\
         | `value` | exact active source columns and trace pairs | untrusted until checked |\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout\n\n\
         open Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec\n\n\
         set_option maxRecDepth 100000 in\n\
         def value : RawLayout := {{\n\
           schemaVersion := 1\n\
           radix := {}\n\
           ringDimension := {}\n\
           extensionLimbs := 2\n\
           firstAllocatedColumn := {}\n\
           parent := {}\n\
           children :=\n\
         [{children}]\n\
           xSignTraces := [{traces}] }}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Layout\n",
        strict.radix,
        D,
        strict.first_allocated_column,
        render_claim(&strict.parent),
    )
}

fn lean_terms(terms: &[(usize, F)]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|&(column, coefficient)| { format!("({column}, {})", coefficient.as_canonical_u64()) })
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_row(row: &R1csIvcPiDecSourceRowAudit) -> String {
    format!(
        "⟨{}, {}, {}⟩",
        lean_terms(row.a()),
        lean_terms(row.b()),
        lean_terms(row.c())
    )
}

fn render_row_chunk(index: usize, rows: &[&R1csIvcPiDecSourceRowAudit]) -> String {
    let values = rows
        .iter()
        .map(|row| lean_row(row))
        .collect::<Vec<_>>()
        .join(",\n  ");
    format!(
        "import Nightstream.Implementation.R1CS.Core.Semantics\n\n\
         /-! Generated exact active strict-PiDEC source rows. Do not hand-edit.\n\n\
         Owns: at most 250 ordered sparse A/B/C rows.\n\n\
         Does not own: coefficient interpretation, satisfaction, acceptance, or row removal.\n\n\
         Emits constraints: no.\n\n\
         | Payload | Meaning | Authority |\n\
         |---|---|---|\n\
         | `values` | one bounded source-row shard | untrusted until checked |\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows.Chunk{index}\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         set_option maxRecDepth 100000 in\n\
         def values : List Row := [\n  {values}\n]\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows.Chunk{index}\n"
    )
}

fn render_metadata(strict: &PiDecStrictAudit, leaf_counts: &[usize]) -> String {
    format!(
        "/-! Generated bounded active strict-PiDEC source metadata. Do not hand-edit.\n\n\
         Owns: exact tiny-fixture scope and source-row census.\n\n\
         Does not own: sparse coefficients, semantic acceptance, or row removal.\n\n\
         Emits constraints: no.\n\n\
         | Payload | Meaning | Authority |\n\
         |---|---|---|\n\
         | source range | one contiguous outer strict-PiDEC program | untrusted until checked |\n\
         | profile | `kappa = 4`, `t = 13`, `r = 24` | bounded fixture only |\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata\n\n\
         def schemaVersion : Nat := 1\n\
         def commitmentRows : Nat := {ACTIVE_COMMITMENT_ROWS}\n\
         def childCount : Nat := {ACTIVE_CHILD_COUNT}\n\
         def matrixCount : Nat := {ACTIVE_MATRIX_COUNT}\n\
         def pointDimension : Nat := {ACTIVE_POINT_DIMENSION}\n\
         def logicalPublicWidth : Nat := {ACTIVE_LOGICAL_X}\n\
         def sourceRowStart : Nat := {}\n\
         def sourceRowEnd : Nat := {}\n\
         def sourceRowCount : Nat := {ACTIVE_ROW_COUNT}\n\
         def shardSize : Nat := {SHARD_SIZE}\n\
         def shardCount : Nat := {}\n\
         def leafRowCounts : List Nat := {}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Metadata\n",
        strict.row_start,
        strict.row_end,
        ACTIVE_ROW_COUNT.div_ceil(SHARD_SIZE),
        lean_nat_list(leaf_counts.iter().copied()),
    )
}

fn render_rows_aggregate(shard_count: usize) -> String {
    let imports = (0..shard_count)
        .map(|index| {
            format!(
                "import Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated.Rows.Chunk{index}"
            )
        })
        .collect::<Vec<_>>()
        .join("\n");
    let chunks = (0..shard_count)
        .map(|index| format!("Rows.Chunk{index}.values"))
        .collect::<Vec<_>>()
        .join(" ++\n    ");
    format!(
        "{imports}\n\n\
         /-! Generated exact strict-PiDEC source-row aggregation. Do not hand-edit.\n\n\
         Owns: ordered concatenation of every bounded row shard.\n\n\
         Does not own: coefficient checking, satisfaction, acceptance, or row removal.\n\n\
         Emits constraints: no.\n\n\
         | Payload | Meaning | Authority |\n\
         |---|---|---|\n\
         | `sourceRows` | all 11,845 rows in source order | untrusted until checked |\n\
         -/\n\n\
         namespace Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated\n\n\
         open Nightstream.Implementation.R1CS\n\n\
         def sourceRows : List Row :=\n    {chunks}\n\n\
         end Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.Nifs.PiDec.Generated\n"
    )
}

fn generated_files() -> Vec<GeneratedLeanFile> {
    let params = tiny_params();
    assert_eq!(params.kappa() as usize, ACTIVE_COMMITMENT_ROWS);
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let audit = R1csIvcRelation::audit_fixed_point_pi_dec_source_rows(&params, &app.into(), &plan)
        .expect("audit bounded active strict-PiDEC source rows");
    let strict = audit.strict();
    assert_eq!(strict.children.len(), ACTIVE_CHILD_COUNT);
    assert_eq!(strict.x_sign_traces.len(), ACTIVE_LOGICAL_X);
    assert_eq!(strict.row_end - strict.row_start, ACTIVE_ROW_COUNT);

    let strict_rows = audit
        .source_row_artifacts()
        .iter()
        .filter(|row| (strict.row_start..strict.row_end).contains(&row.index()))
        .collect::<Vec<_>>();
    assert_eq!(strict_rows.len(), ACTIVE_ROW_COUNT);
    assert!(
        strict_rows
            .iter()
            .map(|row| row.index())
            .eq(strict.row_start..strict.row_end),
        "generated rows are the exact ordered contiguous strict source interval"
    );
    let leaf_counts = audit
        .leaf_source_ranges()
        .iter()
        .map(|range| range.row_end - range.row_start)
        .collect::<Vec<_>>();
    assert_eq!(
        leaf_counts,
        vec![216, 0, 270, 1_404, 70, 672, 532, 15, 4_320, 390, 3_900, 56]
    );

    let mut files = vec![
        GeneratedLeanFile {
            relative_path: format!("{GENERATED_DIRECTORY}/Metadata.lean"),
            contents: render_metadata(strict, &leaf_counts),
        },
        GeneratedLeanFile {
            relative_path: format!("{GENERATED_DIRECTORY}/Layout.lean"),
            contents: render_layout(strict),
        },
    ];
    files.extend(
        strict_rows
            .chunks(SHARD_SIZE)
            .enumerate()
            .map(|(index, rows)| GeneratedLeanFile {
                relative_path: format!("{GENERATED_DIRECTORY}/Rows/Chunk{index}.lean"),
                contents: render_row_chunk(index, rows),
            }),
    );
    let shard_count = ACTIVE_ROW_COUNT.div_ceil(SHARD_SIZE);
    files.push(GeneratedLeanFile {
        relative_path: format!("{GENERATED_DIRECTORY}/Rows.lean"),
        contents: render_rows_aggregate(shard_count),
    });
    files
}

#[test]
fn active_strict_pi_dec_source_artifact_matches_rust_rows() {
    let files = generated_files();
    assert_eq!(files.len(), 3 + ACTIVE_ROW_COUNT.div_ceil(SHARD_SIZE));
    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    let mut drifted = Vec::new();
    for file in files {
        compare_or_write_expected(&root, file, &mut drifted);
    }
    let mut committed_paths = BTreeSet::new();
    committed_lean_files(&root.join(GENERATED_DIRECTORY), &root, &mut committed_paths);
    for stale in committed_paths.difference(&expected_paths) {
        drifted.push(format!("stale generated module: {stale}"));
    }
    assert!(
        drifted.is_empty(),
        "active strict-PiDEC source artifact drifted; inspect every generated `.lean.expected` candidate before promotion: {drifted:?}"
    );
}

#[test]
#[ignore = "deliberately promotes reviewed generated Lean candidates"]
fn regenerate_active_strict_pi_dec_source_artifact() {
    let root = repo_root();
    for file in generated_files() {
        let path = root.join(&file.relative_path);
        fs::create_dir_all(path.parent().expect("generated PiDEC parent")).expect("create generated PiDEC directory");
        fs::write(&path, file.contents).expect("write generated PiDEC artifact");
        let expected = path.with_extension("lean.expected");
        if expected.exists() {
            fs::remove_file(expected).expect("remove promoted PiDEC candidate");
        }
    }
}
