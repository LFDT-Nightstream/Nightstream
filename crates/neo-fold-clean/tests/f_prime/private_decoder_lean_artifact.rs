//! Exact compact private-decoder artifact for the active recursive fixed point.
//!
//! Owns: fail-closed extraction of every steady-recursive private source
//! disposition from the same prepared selective layout used by production,
//! plus a bounded template/SIS-batch representation suitable for Lean.
//!
//! Does not own: source values, definition semantics, derived products,
//! sparse A/B/C equality, CCS/CE membership, key alignment, commitment
//! binding, or permission to remove rows.

#[path = "../support/mod.rs"]
mod support;

#[path = "private_decoder_lean_artifact/batch_grammar.rs"]
mod batch_grammar;

#[path = "private_decoder_lean_artifact/artifact.rs"]
mod artifact;

use neo_fold_clean::frontends::r1cs_f_prime::ivc::R1csIvcRelation;
use neo_fold_clean::frontends::r1cs_f_prime::SelectiveProjectedSourceResolutionRun;
use std::collections::BTreeSet;
use std::fs;
use std::path::{Path, PathBuf};
use support::r1cs_compiler_fixtures::{make_tiny_lifecycle_plan, one_product_r1cs, tiny_params};

const SOURCE_START: usize = 257;
const SOURCE_STOP: usize = 10_997_363;
const FINAL_START: usize = 311;
const BRANCH_STOP: usize = 10_340_178;
const FINAL_STOP: usize = 11_437_010;
const FINAL_COLUMNS: usize = 11_437_038;
const ELIMINATED_COLUMNS: usize = 3_963_194;
const UNIT_COLUMNS: usize = 6_863_364;
const BALANCED_COLUMNS: usize = 170_295;
const BINARY_COLUMNS: usize = 253;
const DECOMPOSITION_ALIASES: usize = 3_459_864;
const EQUALITY_ALIASES: usize = 1_760;
const EQUALITY_ALIAS_SAVINGS: usize = 61_920;
const CENTERED_COLUMNS: usize = 3_454_708;
const BRANCH_COORDINATES: usize = 10_339_867;
const DERIVED_PRODUCTS: usize = 26_752;
const DERIVED_COORDINATES: usize = 1_096_832;

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum Resolution {
    Direct {
        start: usize,
        start_stride: usize,
        width: usize,
        centered: bool,
    },
    DecompositionAlias {
        source: usize,
        source_stride: usize,
        digit: usize,
        digit_stride: usize,
        start: usize,
        start_stride: usize,
        centered: bool,
    },
    EqualityAlias {
        source: usize,
        source_stride: usize,
        start: usize,
        start_stride: usize,
        width: usize,
        centered: bool,
    },
    LinearDefinition,
    TraceEliminated,
}

impl From<SelectiveProjectedSourceResolutionRun> for Resolution {
    fn from(value: SelectiveProjectedSourceResolutionRun) -> Self {
        match value {
            SelectiveProjectedSourceResolutionRun::Direct {
                start,
                start_stride,
                width,
                centered,
            } => Self::Direct {
                start,
                start_stride,
                width,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::DecompositionAlias {
                source,
                source_stride,
                digit,
                digit_stride,
                start,
                start_stride,
                centered,
            } => Self::DecompositionAlias {
                source,
                source_stride,
                digit,
                digit_stride,
                start,
                start_stride,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::EqualityAlias {
                source,
                source_stride,
                start,
                start_stride,
                width,
                centered,
            } => Self::EqualityAlias {
                source,
                source_stride,
                start,
                start_stride,
                width,
                centered,
            },
            SelectiveProjectedSourceResolutionRun::LinearDefinition => Self::LinearDefinition,
            SelectiveProjectedSourceResolutionRun::TraceEliminated => Self::TraceEliminated,
        }
    }
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Run {
    source_start: usize,
    length: usize,
    resolution: Resolution,
}

#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
struct Census {
    eliminated: usize,
    unit: usize,
    balanced: usize,
    binary: usize,
    decomposition_aliases: usize,
    equality_aliases: usize,
    equality_alias_savings: usize,
    retained_coordinates_before_aliases: usize,
    centered_columns: usize,
}

#[derive(Clone, Copy, Debug, Hash, PartialEq, Eq)]
enum RunToken {
    Direct {
        length: usize,
        start_stride: usize,
        width: u8,
        centered: bool,
    },
    DecompositionAlias {
        length: usize,
        source_delta: usize,
        source_stride: usize,
        digit: u8,
        digit_stride: usize,
        start_stride: usize,
        centered: bool,
    },
    EqualityAlias {
        length: usize,
        source_delta: usize,
        source_stride: usize,
        start_stride: usize,
        width: u8,
        centered: bool,
    },
    LinearDefinition {
        length: usize,
    },
    TraceEliminated {
        length: usize,
    },
}

#[derive(Clone, Copy, Debug, PartialEq, Eq)]
struct Stage {
    path: &'static str,
    source_start: usize,
    source_end: usize,
}

fn affine(first: usize, stride: usize, offset: usize) -> Result<usize, String> {
    first
        .checked_add(
            stride
                .checked_mul(offset)
                .ok_or_else(|| "private decoder affine stride overflows".to_string())?,
        )
        .ok_or_else(|| "private decoder affine value overflows".to_string())
}

fn affine_step(first: usize, step: usize, length: usize, next: usize) -> Option<usize> {
    if length == 1 {
        next.checked_sub(first)
    } else {
        (first.checked_add(step.checked_mul(length)?)? == next).then_some(step)
    }
}

fn validate(runs: &[Run]) -> Result<Census, String> {
    if runs.is_empty() {
        return Err("private decoder is empty".into());
    }
    let mut starts = vec![0usize; SOURCE_STOP];
    let mut widths = vec![0u8; SOURCE_STOP];
    let mut centered_flags = vec![false; SOURCE_STOP];
    for column in 0..SOURCE_START {
        starts[column] = column;
        widths[column] = 1;
    }

    let mut source_cursor = SOURCE_START;
    let mut final_cursor = FINAL_START;
    let mut census = Census::default();
    for run in runs {
        if run.length == 0 || run.source_start != source_cursor {
            return Err(format!(
                "private decoder run begins at {}, expected {source_cursor}",
                run.source_start
            ));
        }
        let source_end = run
            .source_start
            .checked_add(run.length)
            .filter(|&end| end <= SOURCE_STOP)
            .ok_or_else(|| "private decoder run exceeds the source interval".to_string())?;
        for offset in 0..run.length {
            let column = run.source_start + offset;
            let (start, width, centered) = match run.resolution {
                Resolution::Direct {
                    start,
                    start_stride,
                    width,
                    centered,
                } => {
                    let start = affine(start, start_stride, offset)?;
                    if start != final_cursor || !matches!(width, 1 | 41 | 64) {
                        return Err(format!(
                            "direct source column {column} does not own the next exact slot"
                        ));
                    }
                    final_cursor = final_cursor
                        .checked_add(width)
                        .ok_or_else(|| "private direct slot cursor overflows".to_string())?;
                    (start, width, centered)
                }
                Resolution::DecompositionAlias {
                    source,
                    source_stride,
                    digit,
                    digit_stride,
                    start,
                    start_stride,
                    centered,
                } => {
                    let source = affine(source, source_stride, offset)?;
                    let digit = affine(digit, digit_stride, offset)?;
                    let start = affine(start, start_stride, offset)?;
                    if source >= column || widths[source] == 0 || digit >= usize::from(widths[source]) {
                        return Err(format!("decomposition alias {column} has an invalid earlier source"));
                    }
                    if start != starts[source] + digit {
                        return Err(format!("decomposition alias {column} has the wrong final digit"));
                    }
                    census.decomposition_aliases += 1;
                    (start, 1, centered)
                }
                Resolution::EqualityAlias {
                    source,
                    source_stride,
                    start,
                    start_stride,
                    width,
                    centered,
                } => {
                    let source = affine(source, source_stride, offset)?;
                    let start = affine(start, start_stride, offset)?;
                    if source >= column
                        || widths[source] == 0
                        || usize::from(widths[source]) != width
                        || starts[source] != start
                        || centered_flags[source] != centered
                    {
                        return Err(format!("equality alias {column} differs from its earlier source"));
                    }
                    census.equality_aliases += 1;
                    census.equality_alias_savings += width;
                    (start, width, centered)
                }
                Resolution::LinearDefinition | Resolution::TraceEliminated => {
                    census.eliminated += 1;
                    continue;
                }
            };
            widths[column] = u8::try_from(width).map_err(|_| "private decoder width exceeds u8".to_string())?;
            starts[column] = start;
            centered_flags[column] = centered;
            census.centered_columns += usize::from(centered);
            census.retained_coordinates_before_aliases += width;
            match width {
                1 => census.unit += 1,
                41 => census.balanced += 1,
                64 => census.binary += 1,
                _ => return Err(format!("private decoder column {column} has unsupported width {width}")),
            }
        }
        source_cursor = source_end;
    }

    let source_columns = SOURCE_STOP - SOURCE_START;
    if source_cursor != SOURCE_STOP
        || final_cursor != BRANCH_STOP
        || census.eliminated != ELIMINATED_COLUMNS
        || census.unit != UNIT_COLUMNS
        || census.balanced != BALANCED_COLUMNS
        || census.binary != BINARY_COLUMNS
        || census.decomposition_aliases != DECOMPOSITION_ALIASES
        || census.equality_aliases != EQUALITY_ALIASES
        || census.equality_alias_savings != EQUALITY_ALIAS_SAVINGS
        || census.centered_columns != CENTERED_COLUMNS
        || census.eliminated + census.unit + census.balanced + census.binary != source_columns
        || census.retained_coordinates_before_aliases != UNIT_COLUMNS + 41 * BALANCED_COLUMNS + 64 * BINARY_COLUMNS
        || census.retained_coordinates_before_aliases - census.decomposition_aliases - census.equality_alias_savings
            != BRANCH_COORDINATES
        || BRANCH_STOP - FINAL_START != BRANCH_COORDINATES
        || BRANCH_STOP + DERIVED_COORDINATES != FINAL_STOP
        || FINAL_STOP + 28 != FINAL_COLUMNS
        || DERIVED_PRODUCTS * 41 != DERIVED_COORDINATES
    {
        return Err(format!("private decoder census drift: {census:?}"));
    }
    Ok(census)
}

fn normalized_run_token(run: Run) -> Result<RunToken, String> {
    Ok(match run.resolution {
        Resolution::Direct {
            start_stride,
            width,
            centered,
            ..
        } => RunToken::Direct {
            length: run.length,
            start_stride,
            width: u8::try_from(width).map_err(|_| "run-token direct width exceeds u8")?,
            centered,
        },
        Resolution::DecompositionAlias {
            source,
            source_stride,
            digit,
            digit_stride,
            start_stride,
            centered,
            ..
        } => RunToken::DecompositionAlias {
            length: run.length,
            source_delta: run
                .source_start
                .checked_sub(source)
                .ok_or_else(|| "run-token decomposition source is not earlier".to_string())?,
            source_stride,
            digit: u8::try_from(digit).map_err(|_| "run-token digit exceeds u8")?,
            digit_stride,
            start_stride,
            centered,
        },
        Resolution::EqualityAlias {
            source,
            source_stride,
            start_stride,
            width,
            centered,
            ..
        } => RunToken::EqualityAlias {
            length: run.length,
            source_delta: run
                .source_start
                .checked_sub(source)
                .ok_or_else(|| "run-token equality source is not earlier".to_string())?,
            source_stride,
            start_stride,
            width: u8::try_from(width).map_err(|_| "run-token equality width exceeds u8")?,
            centered,
        },
        Resolution::LinearDefinition => RunToken::LinearDefinition { length: run.length },
        Resolution::TraceEliminated => RunToken::TraceEliminated { length: run.length },
    })
}

fn slice_run(run: Run, source_start: usize, source_end: usize) -> Result<Run, String> {
    let run_end = run
        .source_start
        .checked_add(run.length)
        .ok_or_else(|| "private decoder run endpoint overflows".to_string())?;
    if source_start < run.source_start || source_start >= source_end || source_end > run_end {
        return Err("private decoder slice escapes its run".into());
    }
    let offset = source_start - run.source_start;
    let resolution = match run.resolution {
        Resolution::Direct {
            start,
            start_stride,
            width,
            centered,
        } => Resolution::Direct {
            start: affine(start, start_stride, offset)?,
            start_stride,
            width,
            centered,
        },
        Resolution::DecompositionAlias {
            source,
            source_stride,
            digit,
            digit_stride,
            start,
            start_stride,
            centered,
        } => Resolution::DecompositionAlias {
            source: affine(source, source_stride, offset)?,
            source_stride,
            digit: affine(digit, digit_stride, offset)?,
            digit_stride,
            start: affine(start, start_stride, offset)?,
            start_stride,
            centered,
        },
        Resolution::EqualityAlias {
            source,
            source_stride,
            start,
            start_stride,
            width,
            centered,
        } => Resolution::EqualityAlias {
            source: affine(source, source_stride, offset)?,
            source_stride,
            start: affine(start, start_stride, offset)?,
            start_stride,
            width,
            centered,
        },
        Resolution::LinearDefinition => Resolution::LinearDefinition,
        Resolution::TraceEliminated => Resolution::TraceEliminated,
    };
    Ok(Run {
        source_start,
        length: source_end - source_start,
        resolution,
    })
}

fn materialized_runs() -> (Vec<Run>, Vec<Stage>) {
    let params = tiny_params();
    let app = one_product_r1cs();
    let plan = make_tiny_lifecycle_plan(app.m(), app.m_in);
    let materialized = R1csIvcRelation::audit_fixed_point_y_zcol_rows(&params, &app.into(), &plan)
        .expect("materialize active selective fixed-point rows");
    let decoder = materialized
        .projected_rows()
        .decoder_run_provenance()
        .expect("complete active private decoder");
    assert_eq!(decoder.arm(), 2);
    assert_eq!(decoder.source_range(), SOURCE_START..SOURCE_STOP);
    let runs = decoder
        .runs()
        .iter()
        .copied()
        .map(|run| Run {
            source_start: run.source_start(),
            length: run.length(),
            resolution: run.resolution().into(),
        })
        .collect();
    let families = decoder
        .source_families()
        .iter()
        .map(|family| {
            let source = family.source_range();
            Stage {
                path: family.path(),
                source_start: source.start,
                source_end: source.end,
            }
        })
        .collect();
    (runs, families)
}

fn mutation_must_fail(runs: &[Run], mutate: impl FnOnce(&mut Vec<Run>), label: &str) {
    let mut changed = runs.to_vec();
    mutate(&mut changed);
    assert!(validate(&changed).is_err(), "{label} mutation must fail closed");
}

fn program_mutation_must_fail(
    program: &batch_grammar::Program,
    runs: &[Run],
    mutate: impl FnOnce(&mut batch_grammar::Program),
    label: &str,
) {
    let mut changed = program.clone();
    mutate(&mut changed);
    assert!(
        batch_grammar::validate(&changed, runs).is_err(),
        "{label} compact-program mutation must fail closed"
    );
}

fn repo_root() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .parent()
        .and_then(Path::parent)
        .expect("workspace root")
        .to_path_buf()
}

fn compare_or_write_expected(root: &Path, file: artifact::GeneratedLeanFile, drifted: &mut Vec<String>) {
    let path = root.join(&file.relative_path);
    if fs::read_to_string(&path).unwrap_or_default() == file.contents {
        return;
    }
    let expected = path.with_extension("lean.expected");
    fs::create_dir_all(expected.parent().expect("private decoder artifact parent"))
        .expect("create private decoder artifact directory");
    fs::write(&expected, file.contents).expect("write private decoder Lean candidate");
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
    for entry in fs::read_dir(directory).expect("read private decoder artifact directory") {
        let path = entry.expect("read private decoder artifact entry").path();
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

fn verify_drift(program: &batch_grammar::Program) {
    let files = artifact::generated_files(program);
    assert_eq!(
        files.len(),
        16,
        "bounded templates, calls, SIS groups, alias links, and metadata"
    );
    let root = repo_root();
    let expected_paths = files
        .iter()
        .map(|file| file.relative_path.clone())
        .collect::<BTreeSet<_>>();
    let mut drifted = Vec::new();
    for file in files {
        compare_or_write_expected(&root, file, &mut drifted);
    }
    let mut committed = BTreeSet::new();
    committed_lean_files(&root.join(artifact::GENERATED_DIRECTORY), &root, &mut committed);
    for stale in committed.difference(&expected_paths) {
        drifted.push(format!("stale generated module: {stale}"));
    }
    assert!(
        drifted.is_empty(),
        "private decoder Lean artifact drifted; inspect and deliberately promote every `.lean.expected` candidate: {drifted:?}"
    );
}

#[test]
fn active_private_decoder_program_is_exact_and_fail_closed() {
    let (runs, families) = materialized_runs();
    let census = validate(&runs).expect("exact active private decoder");
    let program = batch_grammar::report(&runs, &families).expect("bounded private decoder program");
    assert_eq!(program.summary.census, census);
    assert!(program
        .templates
        .iter()
        .all(|template| template.atoms.len() <= 32));

    mutation_must_fail(&runs, |changed| changed[0].source_start += 1, "source boundary");
    mutation_must_fail(&runs, |changed| changed.swap(0, 1), "run order");
    mutation_must_fail(
        &runs,
        |changed| {
            let run = changed
                .iter_mut()
                .find(|run| matches!(run.resolution, Resolution::Direct { .. }))
                .expect("direct run");
            if let Resolution::Direct { start, .. } = &mut run.resolution {
                *start += 1;
            }
        },
        "direct start",
    );
    mutation_must_fail(
        &runs,
        |changed| {
            let run = changed
                .iter_mut()
                .find(|run| run.length > 1 && matches!(run.resolution, Resolution::Direct { .. }))
                .expect("strided direct run");
            if let Resolution::Direct { start_stride, .. } = &mut run.resolution {
                *start_stride += 1;
            }
        },
        "direct stride",
    );
    mutation_must_fail(
        &runs,
        |changed| {
            let run = changed
                .iter_mut()
                .find(|run| matches!(run.resolution, Resolution::Direct { centered: true, .. }))
                .expect("centered direct run");
            if let Resolution::Direct { centered, .. } = &mut run.resolution {
                *centered = false;
            }
        },
        "direct centeredness",
    );
    mutation_must_fail(
        &runs,
        |changed| {
            let run = changed
                .iter_mut()
                .find(|run| matches!(run.resolution, Resolution::DecompositionAlias { .. }))
                .expect("decomposition run");
            if let Resolution::DecompositionAlias { source, .. } = &mut run.resolution {
                *source += 1;
            }
        },
        "decomposition source",
    );
    mutation_must_fail(
        &runs,
        |changed| {
            let run = changed
                .iter_mut()
                .find(|run| matches!(run.resolution, Resolution::EqualityAlias { .. }))
                .expect("equality run");
            if let Resolution::EqualityAlias { source, .. } = &mut run.resolution {
                *source += 1;
            }
        },
        "equality source",
    );
    program_mutation_must_fail(
        &program,
        &runs,
        |changed| changed.calls[0].source_start += 1,
        "call source cursor",
    );
    program_mutation_must_fail(
        &program,
        &runs,
        |changed| changed.templates[0].summary.source_columns += 1,
        "template summary",
    );
    program_mutation_must_fail(
        &program,
        &runs,
        |changed| {
            let group = changed
                .batches
                .iter_mut()
                .flat_map(|batch| &mut batch.groups)
                .find(|group| matches!(group.kind, batch_grammar::OpeningGroupKind::Alias { .. }))
                .expect("alias opening group");
            if let batch_grammar::OpeningGroupKind::Alias { source, .. } = &mut group.kind {
                *source += 1;
            }
        },
        "SIS alias source",
    );
    program_mutation_must_fail(
        &program,
        &runs,
        |changed| changed.alias_links[0].target_offset_stride += 1,
        "alias target link",
    );
    program_mutation_must_fail(
        &program,
        &runs,
        |changed| changed.alias_consumers[0].link_stop += 1,
        "alias consumer link interval",
    );
    verify_drift(&program);
}

#[test]
#[ignore = "deliberately rewrites the reviewed compact private-decoder Lean artifact"]
fn regenerate_active_private_decoder_lean_artifact() {
    let (runs, families) = materialized_runs();
    validate(&runs).expect("exact active private decoder");
    let program = batch_grammar::report(&runs, &families).expect("bounded private decoder program");
    let files = artifact::generated_files(&program);
    assert_eq!(files.len(), 19);
    let root = repo_root();
    for file in files {
        let path = root.join(file.relative_path);
        fs::create_dir_all(path.parent().expect("private decoder artifact parent"))
            .expect("create private decoder artifact directory");
        fs::write(&path, file.contents).expect("write private decoder Lean artifact");
        let expected = path.with_extension("lean.expected");
        if expected.exists() {
            fs::remove_file(expected).expect("remove promoted private decoder candidate");
        }
    }
}
