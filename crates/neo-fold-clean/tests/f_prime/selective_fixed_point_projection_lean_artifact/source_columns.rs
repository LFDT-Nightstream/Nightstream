//! Exact source-column decoder and compiler-substitution provenance.
//!
//! Owns: the transitive source-column closure reachable from the focused
//! 5,724 source rows, retained low-norm slots, compiler linear definitions,
//! trace-local eliminated columns, and referenced grouped-product slots.
//!
//! Does not own: source-row semantics, assignment satisfaction, selector
//! truth, protocol authority, or permission to remove rows.
//!
//! Emits constraints: no; this test generator records existing compiler data.
//!
//! | Artifact leaf | Mathematical obligation | Authority class |
//! |---|---|---|
//! | source provenance | exact closure, slots, definitions, and rewrite order | computed |

use std::collections::{BTreeMap, BTreeSet};
use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveProjectedProductFactor, SelectiveProjectedRewriteOutput,
    SelectiveProjectedRowsAudit, SelectiveProjectedSourceProvenance, SelectiveProjectedSourceTerm,
    SelectiveRewriteKind,
};
use p3_field::PrimeField64;

use super::GeneratedLeanFile;

const GENERATED_ROOT: &str =
    "formal/nightstream-lean/Nightstream/Implementation/R1CS/Artifacts/FPrimeSelectiveFixedPoint/PiRlcProjection/YZcol/Generated";
const IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized.SourceSchema";
const SHARD_IMPORT_ROOT: &str =
    "Nightstream.Implementation.R1CS.Artifacts.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns";
const NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Generated.SourceColumns";
const MATERIALIZED_NAMESPACE: &str =
    "Nightstream.Implementation.R1CS.FPrimeSelectiveFixedPoint.PiRlcProjection.YZcol.Selective.Materialized";
const STEADY_ARM: usize = 2;
const SOURCE_COLUMN_SHARD_SIZE: usize = 1_024;
const TRACE_ELIMINATED_COLUMN_SHARD_SIZE: usize = 1_024;
const RETAINED_SLOT_SHARD_SIZE: usize = 1_024;
const LINEAR_DEFINITION_SHARD_SIZE: usize = 96;
const DERIVED_PRODUCT_SUM_SHARD_SIZE: usize = 32;
const REWRITE_STEP_SHARD_SIZE: usize = 32;

const GENERATED_HEADER: &str = r#"/-
Generated file: source-provenance payload; do not hand-edit.

Owns: the exact ordered payload stored in this generated module.

Does not own: decoding, assignment satisfaction, semantic authority, security
events, or permission to remove rows.

Emits constraints: no.

| Artifact leaf | Mathematical obligation | Authority class |
|---|---|---|
| generated payload | exact Rust-rendered list data and order | computed |
-/

"#;

fn lean_terms(terms: &[SelectiveProjectedSourceTerm]) -> String {
    format!(
        "[{}]",
        terms
            .iter()
            .map(|term| format!(
                "{{ column := {}, coefficient := {} }}",
                term.column(),
                term.coefficient().as_canonical_u64()
            ))
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_factor(factor: &SelectiveProjectedProductFactor) -> String {
    format!(
        "{{ left := {{ constant := {}, terms := {} }}, right := {{ constant := {}, terms := {} }}, coefficient := {} }}",
        factor.left_constant().as_canonical_u64(),
        lean_terms(factor.left_terms()),
        factor.right_constant().as_canonical_u64(),
        lean_terms(factor.right_terms()),
        factor.coefficient().as_canonical_u64(),
    )
}

fn lean_lc(constant: u64, terms: &[SelectiveProjectedSourceTerm]) -> String {
    format!("{{ constant := {constant}, terms := {} }}", lean_terms(terms))
}

fn lean_rewrite_kind(kind: SelectiveRewriteKind) -> &'static str {
    match kind {
        SelectiveRewriteKind::PolynomialEvaluation => "polynomialEvaluation",
        SelectiveRewriteKind::ProductSum => "productSum",
        other => panic!("unsupported focused executable rewrite kind {other:?}"),
    }
}

fn validate(provenance: &SelectiveProjectedSourceProvenance, projected: &SelectiveProjectedRowsAudit) {
    assert_eq!(provenance.arm(), STEADY_ARM, "focused steady arm");
    assert_eq!(provenance.source_columns().first(), Some(&0), "constant source column");
    assert!(
        provenance
            .source_columns()
            .windows(2)
            .all(|pair| pair[0] < pair[1]),
        "source-column closure is strictly increasing"
    );

    let source_columns = provenance
        .source_columns()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    let slots = provenance
        .retained_slots()
        .iter()
        .map(|slot| slot.column())
        .collect::<BTreeSet<_>>();
    let definitions = provenance
        .linear_definitions()
        .iter()
        .map(|definition| definition.target())
        .collect::<BTreeSet<_>>();
    let eliminated = provenance
        .trace_eliminated_columns()
        .iter()
        .copied()
        .collect::<BTreeSet<_>>();
    assert_eq!(slots.len(), provenance.retained_slots().len(), "unique source slots");
    assert_eq!(
        definitions.len(),
        provenance.linear_definitions().len(),
        "unique compiler definitions"
    );
    assert_eq!(
        eliminated.len(),
        provenance.trace_eliminated_columns().len(),
        "unique trace-eliminated columns"
    );
    assert!(slots.is_disjoint(&definitions), "slot/definition partition");
    assert!(slots.is_disjoint(&eliminated), "slot/eliminated partition");
    assert!(definitions.is_disjoint(&eliminated), "definition/eliminated partition");
    let mut partition = BTreeSet::from([0]);
    partition.extend(slots);
    partition.extend(definitions);
    partition.extend(eliminated);
    assert_eq!(partition, source_columns, "complete source-column partition");

    for slot in provenance.retained_slots() {
        assert!(matches!(slot.width(), 1 | 41 | 64), "supported exact decoder width");
    }
    for definition in provenance.linear_definitions() {
        assert!(
            definition
                .terms()
                .iter()
                .all(|term| source_columns.contains(&term.column())),
            "definition dependency belongs to transitive closure"
        );
    }
    let derived_by_index = provenance
        .derived_product_sums()
        .iter()
        .map(|derived| (derived.compiler_index(), derived))
        .collect::<BTreeMap<_, _>>();
    assert_eq!(
        derived_by_index.len(),
        provenance.derived_product_sums().len(),
        "unique derived compiler indices"
    );
    let derived_indices = derived_by_index.keys().copied().collect::<BTreeSet<_>>();
    for derived in provenance.derived_product_sums() {
        assert_eq!(derived.width(), 41, "balanced derived-product width");
        assert!(!derived.factors().is_empty(), "derived product has factors");
        if let Some(previous) = derived.previous() {
            assert!(previous < derived.compiler_index(), "derived predecessor order");
            assert!(derived_indices.contains(&previous), "derived predecessor closure");
        }
    }
    assert_eq!(provenance.rewrite_steps().len(), 1_250, "exact rewrite-step count");
    assert!(
        provenance
            .rewrite_steps()
            .windows(2)
            .all(|pair| pair[0].emitted_row() < pair[1].emitted_row()),
        "rewrite steps follow emitted-row order"
    );
    let expected_rewrite_rows = projected
        .row_artifacts()
        .iter()
        .filter(|row| {
            matches!(
                row.family(),
                SelectiveEmittedRowFamily::PolynomialEvaluation | SelectiveEmittedRowFamily::ProductSum
            )
        })
        .map(|row| row.emitted_row())
        .collect::<BTreeSet<_>>();
    let actual_rewrite_rows = provenance
        .rewrite_steps()
        .iter()
        .map(|step| step.emitted_row())
        .collect::<BTreeSet<_>>();
    assert_eq!(actual_rewrite_rows, expected_rewrite_rows, "exact rewrite-row coverage");
    for step in provenance.rewrite_steps() {
        assert!(matches!(
            step.kind(),
            SelectiveRewriteKind::PolynomialEvaluation | SelectiveRewriteKind::ProductSum
        ));
        assert!(!step.source_rows().is_empty(), "rewrite owns source rows");
        assert!(step.factors().len() <= 5, "selective evaluation port capacity");
        if let Some(previous) = step.previous() {
            assert!(derived_indices.contains(&previous), "rewrite predecessor is exported");
        }
        if let SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } = step.output() {
            assert_eq!(
                step.base_constant().as_canonical_u64(),
                0,
                "derived rewrite has zero base constant"
            );
            assert!(step.base_terms().is_empty(), "derived rewrite has no base terms");
            let derived = derived_by_index
                .get(compiler_index)
                .unwrap_or_else(|| panic!("rewrite output {compiler_index} is exported"));
            assert_eq!(
                step.previous(),
                derived.previous(),
                "rewrite predecessor matches witness-derived encoding"
            );
            assert_eq!(
                step.factors(),
                derived.factors(),
                "rewrite factors match witness-derived encoding"
            );
        }
    }
    let derived_rewrite_outputs = provenance
        .rewrite_steps()
        .iter()
        .filter_map(|step| match step.output() {
            SelectiveProjectedRewriteOutput::Source { .. } => None,
            SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } => Some(*compiler_index),
        })
        .collect::<Vec<_>>();
    let exported_derived_entries = provenance
        .derived_product_sums()
        .iter()
        .map(|derived| derived.compiler_index())
        .collect::<Vec<_>>();
    assert_eq!(
        derived_rewrite_outputs, exported_derived_entries,
        "every exported derived witness entry has exactly one ordered rewrite output"
    );
    assert_eq!(provenance.retained_steps().len(), 4, "exact retained-step count");
    assert!(
        provenance
            .retained_steps()
            .windows(2)
            .all(|pair| pair[0].emitted_row() < pair[1].emitted_row()),
        "retained steps follow emitted-row order"
    );
    let expected_retained_rows = projected
        .row_artifacts()
        .iter()
        .filter(|row| row.family() == SelectiveEmittedRowFamily::Retained)
        .map(|row| row.emitted_row())
        .collect::<BTreeSet<_>>();
    let actual_retained_rows = provenance
        .retained_steps()
        .iter()
        .map(|step| step.emitted_row())
        .collect::<BTreeSet<_>>();
    assert_eq!(
        actual_retained_rows, expected_retained_rows,
        "exact retained-row coverage"
    );
}

fn shard_file(name: &str, value_type: &str, values: Vec<String>) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE}.{name}");
    let mut contents = String::new();
    contents.push_str(GENERATED_HEADER);
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render source shard import");
    writeln!(contents, "namespace {namespace}\n").expect("render source shard namespace");
    writeln!(contents, "open {MATERIALIZED_NAMESPACE}\n").expect("render materialized namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def values : List {value_type} :=\n  [").expect("render source shard definition");
    for (index, value) in values.iter().enumerate() {
        if index != 0 {
            contents.push_str(",\n");
        }
        write!(contents, "    {value}").expect("render source shard value");
    }
    contents.push_str("\n  ]\n\n");
    writeln!(contents, "end {namespace}").expect("render source shard namespace end");
    assert!(
        contents.lines().count() < 1_500,
        "generated source-provenance shard exceeds the repository line limit"
    );
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SourceColumns/{name}.lean"),
        contents,
    }
}

fn metadata_file(provenance: &SelectiveProjectedSourceProvenance) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE}.Metadata");
    let mut contents = String::new();
    contents.push_str(GENERATED_HEADER);
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render metadata import");
    writeln!(contents, "namespace {namespace}\n").expect("render metadata namespace");
    writeln!(contents, "open {MATERIALIZED_NAMESPACE}\n").expect("render materialized namespace");
    writeln!(contents, "def sourceArm : Nat := {}", provenance.arm()).expect("render source arm");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    contents.push_str("def retainedSteps : List RawRetainedStep :=\n  [");
    for (index, step) in provenance.retained_steps().iter().enumerate() {
        if index != 0 {
            contents.push_str(",\n    ");
        }
        write!(
            contents,
            "{{ emittedRow := {}, sourceRow := {}, a := {}, b := {}, c := {} }}",
            step.emitted_row(),
            step.source_row(),
            lean_lc(step.a().constant().as_canonical_u64(), step.a().terms()),
            lean_lc(step.b().constant().as_canonical_u64(), step.b().terms()),
            lean_lc(step.c().constant().as_canonical_u64(), step.c().terms()),
        )
        .expect("render retained source step");
    }
    contents.push_str("]\n\n");
    writeln!(contents, "end {namespace}").expect("render metadata namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SourceColumns/Metadata.lean"),
        contents,
    }
}

fn root_file(shard_names: &[String]) -> GeneratedLeanFile {
    let mut contents = String::new();
    contents.push_str(GENERATED_HEADER);
    writeln!(contents, "import {SHARD_IMPORT_ROOT}.Metadata").expect("render metadata root import");
    for name in shard_names {
        writeln!(contents, "import {SHARD_IMPORT_ROOT}.{name}").expect("render shard root import");
    }
    contents.push('\n');
    writeln!(contents, "namespace {NAMESPACE}\n").expect("render source root namespace");
    writeln!(contents, "open {MATERIALIZED_NAMESPACE}\n").expect("render materialized namespace");
    contents.push_str("def sourceArm : Nat := Metadata.sourceArm\n");
    contents.push_str("def retainedSteps : List RawRetainedStep := Metadata.retainedSteps\n\n");

    for (prefix, value_type, public_name) in [
        ("SourceColumnValues", "Nat", "sourceColumns"),
        ("TraceEliminatedColumns", "Nat", "traceEliminatedColumns"),
        ("RetainedSlots", "RawSourceSlot", "retainedSlots"),
        ("LinearDefinitions", "RawSourceDefinition", "linearDefinitions"),
        ("DerivedProductSums", "RawDerivedProductSum", "derivedProductSums"),
        ("RewriteSteps", "RawRewriteStep", "rewriteSteps"),
    ] {
        let names = shard_names
            .iter()
            .filter(|name| name.starts_with(prefix))
            .collect::<Vec<_>>();
        writeln!(contents, "set_option maxRecDepth 100000 in").expect("render root recursion option");
        writeln!(contents, "def {public_name} : List {value_type} :=").expect("render root list definition");
        if names.is_empty() {
            contents.push_str("  []\n\n");
        } else {
            for (index, name) in names.iter().enumerate() {
                let continuation = if index + 1 == names.len() { "" } else { " ++" };
                writeln!(contents, "  {name}.values{continuation}").expect("render root list shard");
            }
            contents.push('\n');
        }
    }
    writeln!(contents, "end {NAMESPACE}").expect("render source root namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/SourceColumns.lean"),
        contents,
    }
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> Vec<GeneratedLeanFile> {
    let provenance = projected
        .source_provenance()
        .expect("focused projection includes exact source provenance");
    validate(provenance, projected);

    let mut files = vec![metadata_file(provenance)];
    let mut shard_names = Vec::new();

    for (index, chunk) in provenance
        .source_columns()
        .chunks(SOURCE_COLUMN_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("SourceColumnValues{index}");
        let values = chunk.iter().map(usize::to_string).collect();
        files.push(shard_file(&name, "Nat", values));
        shard_names.push(name);
    }

    for (index, chunk) in provenance
        .trace_eliminated_columns()
        .chunks(TRACE_ELIMINATED_COLUMN_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("TraceEliminatedColumns{index}");
        let values = chunk.iter().map(usize::to_string).collect();
        files.push(shard_file(&name, "Nat", values));
        shard_names.push(name);
    }

    for (index, chunk) in provenance
        .retained_slots()
        .chunks(RETAINED_SLOT_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("RetainedSlots{index}");
        let values = chunk
            .iter()
            .map(|slot| {
                format!(
                    "{{ column := {}, start := {}, width := {} }}",
                    slot.column(),
                    slot.start(),
                    slot.width()
                )
            })
            .collect();
        files.push(shard_file(&name, "RawSourceSlot", values));
        shard_names.push(name);
    }

    for (index, chunk) in provenance
        .linear_definitions()
        .chunks(LINEAR_DEFINITION_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("LinearDefinitions{index}");
        let values = chunk
            .iter()
            .map(|definition| {
                format!(
                    "{{ target := {}, constant := {}, terms := {} }}",
                    definition.target(),
                    definition.constant().as_canonical_u64(),
                    lean_terms(definition.terms())
                )
            })
            .collect();
        files.push(shard_file(&name, "RawSourceDefinition", values));
        shard_names.push(name);
    }

    for (index, chunk) in provenance
        .derived_product_sums()
        .chunks(DERIVED_PRODUCT_SUM_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("DerivedProductSums{index}");
        let values = chunk
            .iter()
            .map(|derived| {
                let previous = derived
                    .previous()
                    .map_or_else(|| "none".to_owned(), |value| format!("some {value}"));
                let factors = derived
                    .factors()
                    .iter()
                    .map(lean_factor)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{{ compilerIndex := {}, start := {}, width := {}, factors := [{}], previous := {} }}",
                    derived.compiler_index(),
                    derived.start(),
                    derived.width(),
                    factors,
                    previous
                )
            })
            .collect();
        files.push(shard_file(&name, "RawDerivedProductSum", values));
        shard_names.push(name);
    }

    for (index, chunk) in provenance
        .rewrite_steps()
        .chunks(REWRITE_STEP_SHARD_SIZE)
        .enumerate()
    {
        let name = format!("RewriteSteps{index}");
        let values = chunk
            .iter()
            .map(|step| {
                let source_rows = step
                    .source_rows()
                    .iter()
                    .map(|&(start, stop)| format!("{{ start := {start}, stop := {stop} }}"))
                    .collect::<Vec<_>>()
                    .join(", ");
                let output = match step.output() {
                    SelectiveProjectedRewriteOutput::Source { constant, terms } => {
                        format!(".source {}", lean_lc(constant.as_canonical_u64(), terms))
                    }
                    SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } => {
                        format!(".derivedProductSum {compiler_index}")
                    }
                };
                let previous = step
                    .previous()
                    .map_or_else(|| "none".to_owned(), |value| format!("some {value}"));
                let factors = step
                    .factors()
                    .iter()
                    .map(lean_factor)
                    .collect::<Vec<_>>()
                    .join(", ");
                format!(
                    "{{ emittedRow := {}, rewriteId := {}, kind := .{}, sourceRows := [{}], output := {}, base := {}, previous := {}, factors := [{}] }}",
                    step.emitted_row(),
                    step.rewrite_id(),
                    lean_rewrite_kind(step.kind()),
                    source_rows,
                    output,
                    lean_lc(step.base_constant().as_canonical_u64(), step.base_terms()),
                    previous,
                    factors
                )
            })
            .collect();
        files.push(shard_file(&name, "RawRewriteStep", values));
        shard_names.push(name);
    }

    files.push(root_file(&shard_names));
    files
}
