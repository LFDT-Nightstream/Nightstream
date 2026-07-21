//! Exact selective source-program and source-to-final decoder rendering.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveProjectedProductFactor, SelectiveProjectedRewriteOutput, SelectiveProjectedRowsAudit,
    SelectiveProjectedSourceLinearCombination, SelectiveProjectedSourceResolution, SelectiveProjectedSourceTerm,
    SelectiveRewriteKind,
};
use p3_field::PrimeField64;

use super::{generated_header, lean_option, GeneratedLeanFile, GENERATED_ROOT, IMPORT_ROOT, NAMESPACE_ROOT};

const DEFAULT_SHARD_SIZE: usize = 128;
const REWRITE_SHARD_SIZE: usize = 64;
const MAX_NESTED_RECORDS: usize = 256;

fn assert_terms_bounded(terms: &[SelectiveProjectedSourceTerm], owner: &str) {
    assert!(
        terms.len() <= MAX_NESTED_RECORDS,
        "{owner} has {} terms; render a compact certificate before Lean evaluation",
        terms.len()
    );
}

fn assert_factors_bounded(factors: &[SelectiveProjectedProductFactor], owner: &str) {
    assert!(
        factors.len() <= MAX_NESTED_RECORDS,
        "{owner} has too many product factors"
    );
    for factor in factors {
        assert_terms_bounded(factor.left_terms(), owner);
        assert_terms_bounded(factor.right_terms(), owner);
    }
}

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

fn lean_lc(constant: u64, terms: &[SelectiveProjectedSourceTerm]) -> String {
    format!("{{ constant := {constant}, terms := {} }}", lean_terms(terms))
}

fn lean_source_lc(value: &SelectiveProjectedSourceLinearCombination) -> String {
    lean_lc(value.constant().as_canonical_u64(), value.terms())
}

fn lean_factor(factor: &SelectiveProjectedProductFactor) -> String {
    format!(
        "{{ left := {}, right := {}, coefficient := {} }}",
        lean_lc(factor.left_constant().as_canonical_u64(), factor.left_terms()),
        lean_lc(factor.right_constant().as_canonical_u64(), factor.right_terms()),
        factor.coefficient().as_canonical_u64()
    )
}

fn lean_factors(values: &[SelectiveProjectedProductFactor]) -> String {
    format!(
        "[{}]",
        values
            .iter()
            .map(lean_factor)
            .collect::<Vec<_>>()
            .join(", ")
    )
}

fn lean_rewrite_kind(kind: SelectiveRewriteKind) -> &'static str {
    match kind {
        SelectiveRewriteKind::PolynomialEvaluation => "polynomialEvaluation",
        SelectiveRewriteKind::ProductSum => "productSum",
        other => panic!("unsupported combined-NC executable rewrite kind {other:?}"),
    }
}

fn lean_resolution(resolution: SelectiveProjectedSourceResolution) -> String {
    match resolution {
        SelectiveProjectedSourceResolution::ConstantOne => ".constantOne".to_owned(),
        SelectiveProjectedSourceResolution::Direct { start, width, centered } => {
            format!(".direct {start} {width} {centered}")
        }
        SelectiveProjectedSourceResolution::DecompositionAlias {
            source,
            digit,
            start,
            centered,
        } => format!(".decompositionAlias {source} {digit} {start} {centered}"),
        SelectiveProjectedSourceResolution::EqualityAlias {
            source,
            start,
            width,
            centered,
        } => format!(".equalityAlias {source} {start} {width} {centered}"),
        SelectiveProjectedSourceResolution::LinearDefinition => ".linearDefinition".to_owned(),
        SelectiveProjectedSourceResolution::TraceEliminated => ".traceEliminated".to_owned(),
    }
}

fn shard_file(family: &str, index: usize, value_type: &str, values: &[String]) -> GeneratedLeanFile {
    assert!(!values.is_empty() && values.len() <= DEFAULT_SHARD_SIZE);
    let namespace = format!("{NAMESPACE_ROOT}.Provenance.{family}.Chunk{index}");
    let mut contents = generated_header(&format!("at most 128 proof-free {family} records"));
    writeln!(contents, "import {IMPORT_ROOT}\n").expect("render provenance import");
    writeln!(contents, "namespace {namespace}\n").expect("render provenance namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def values : List {value_type} := [").expect("render provenance type");
    for (offset, value) in values.iter().enumerate() {
        if offset != 0 {
            contents.push_str(",\n");
        }
        write!(contents, "  {value}").expect("render provenance value");
    }
    writeln!(contents, "\n]\n\nend {namespace}").expect("render provenance namespace end");
    assert!(contents.lines().count() < 1_500, "provenance shard line limit");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Provenance/{family}/Chunk{index}.lean"),
        contents,
    }
}

fn family_root(family: &str, shard_count: usize, value_type: &str) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Provenance.{family}");
    let mut contents = generated_header(&format!("the ordered concatenation of every {family} shard"));
    writeln!(contents, "import {IMPORT_ROOT}").expect("render provenance schema import");
    for index in 0..shard_count {
        writeln!(contents, "import {namespace}.Chunk{index}").expect("render provenance shard import");
    }
    writeln!(contents, "\nnamespace {namespace}\n").expect("render provenance root namespace");
    contents.push_str("set_option maxRecDepth 100000 in\n");
    writeln!(contents, "def values : List {value_type} :=").expect("render provenance root type");
    if shard_count == 0 {
        contents.push_str("  []\n");
    } else {
        for index in 0..shard_count {
            let continuation = if index + 1 == shard_count { "" } else { " ++" };
            writeln!(contents, "  Chunk{index}.values{continuation}").expect("render provenance root shard");
        }
    }
    writeln!(contents, "\nend {namespace}").expect("render provenance root namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Provenance/{family}.lean"),
        contents,
    }
}

fn render_family(family: &str, value_type: &str, values: Vec<String>, shard_size: usize) -> Vec<GeneratedLeanFile> {
    assert!(shard_size <= DEFAULT_SHARD_SIZE, "provenance certificate shard ceiling");
    let mut files = values
        .chunks(shard_size)
        .enumerate()
        .map(|(index, values)| shard_file(family, index, value_type, values))
        .collect::<Vec<_>>();
    files.push(family_root(family, files.len(), value_type));
    files
}

fn provenance_root(families: &[(&str, &str)], source_arm: usize, decoder_arm: usize) -> GeneratedLeanFile {
    let namespace = format!("{NAMESPACE_ROOT}.Provenance");
    let mut contents = generated_header("the exact source-program and source-decoder family roots");
    for (family, _) in families {
        writeln!(contents, "import {namespace}.{family}").expect("render provenance family import");
    }
    writeln!(contents, "\nnamespace {namespace}\n").expect("render provenance namespace");
    writeln!(contents, "def sourceArm : Nat := {source_arm}").expect("render source arm");
    writeln!(contents, "def decoderArm : Nat := {decoder_arm}\n").expect("render decoder arm");
    for (family, value_type) in families {
        let public_name = match *family {
            "SourceColumns" => "sourceColumns",
            "RetainedSlots" => "retainedSlots",
            "LinearDefinitions" => "linearDefinitions",
            "TraceEliminatedColumns" => "traceEliminatedColumns",
            "DerivedProductSums" => "derivedProductSums",
            "RewriteSteps" => "rewriteSteps",
            "RetainedSteps" => "retainedSteps",
            "Decoders" => "decoders",
            _ => unreachable!("known provenance family"),
        };
        writeln!(contents, "def {public_name} : List {value_type} := {family}.values")
            .expect("render provenance public definition");
    }
    writeln!(contents, "\nend {namespace}").expect("render provenance namespace end");
    GeneratedLeanFile {
        relative_path: format!("{GENERATED_ROOT}/Provenance.lean"),
        contents,
    }
}

pub(super) fn render(projected: &SelectiveProjectedRowsAudit) -> Vec<GeneratedLeanFile> {
    let provenance = projected
        .source_provenance()
        .expect("combined-NC projection includes source provenance");
    let decoder = projected
        .decoder_provenance()
        .expect("combined-NC projection includes decoder provenance");
    let decoded_columns = decoder
        .decoders()
        .iter()
        .map(|entry| entry.column())
        .collect::<Vec<_>>();
    if provenance.source_columns() != decoded_columns {
        let first_mismatch = provenance
            .source_columns()
            .iter()
            .copied()
            .zip(decoded_columns.iter().copied())
            .position(|(source, decoded)| source != decoded);
        panic!(
            "decoder/source closure mismatch: source={}, decoded={}, first mismatch={first_mismatch:?}",
            provenance.source_columns().len(),
            decoded_columns.len()
        );
    }

    let source_columns = provenance
        .source_columns()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>();
    let retained_slots = provenance
        .retained_slots()
        .iter()
        .map(|slot| {
            format!(
                "{{ column := {}, start := {}, width := {} }}",
                slot.column(),
                slot.start(),
                slot.width()
            )
        })
        .collect::<Vec<_>>();
    let linear_definitions = provenance
        .linear_definitions()
        .iter()
        .map(|definition| {
            assert_terms_bounded(definition.terms(), "source linear definition");
            format!(
                "{{ target := {}, value := {} }}",
                definition.target(),
                lean_lc(definition.constant().as_canonical_u64(), definition.terms())
            )
        })
        .collect::<Vec<_>>();
    let trace_eliminated = provenance
        .trace_eliminated_columns()
        .iter()
        .map(usize::to_string)
        .collect::<Vec<_>>();
    let derived = provenance
        .derived_product_sums()
        .iter()
        .map(|entry| {
            assert_factors_bounded(entry.factors(), "derived product sum");
            format!(
                "{{ compilerIndex := {}, start := {}, width := {}, factors := {}, previous := {} }}",
                entry.compiler_index(),
                entry.start(),
                entry.width(),
                lean_factors(entry.factors()),
                lean_option(entry.previous())
            )
        })
        .collect::<Vec<_>>();
    let rewrites = provenance
        .rewrite_steps()
        .iter()
        .map(|step| {
            assert!(step.source_rows().len() <= MAX_NESTED_RECORDS);
            assert_terms_bounded(step.base_terms(), "rewrite base");
            assert_factors_bounded(step.factors(), "rewrite step");
            let ranges = step
                .source_rows()
                .iter()
                .map(|&(start, stop)| format!("{{ start := {start}, stop := {stop} }}"))
                .collect::<Vec<_>>()
                .join(", ");
            let output = match step.output() {
                SelectiveProjectedRewriteOutput::Source { constant, terms } => format!(
                    ".source {}",
                    lean_lc(constant.as_canonical_u64(), terms)
                ),
                SelectiveProjectedRewriteOutput::DerivedProductSum { compiler_index } => {
                    format!(".derivedProductSum {compiler_index}")
                }
            };
            format!(
                "{{ emittedRow := {}, rewriteId := {}, kind := .{}, sourceRows := [{}], output := {}, base := {}, previous := {}, factors := {} }}",
                step.emitted_row(),
                step.rewrite_id(),
                lean_rewrite_kind(step.kind()),
                ranges,
                output,
                lean_lc(step.base_constant().as_canonical_u64(), step.base_terms()),
                lean_option(step.previous()),
                lean_factors(step.factors())
            )
        })
        .collect::<Vec<_>>();
    let retained = provenance
        .retained_steps()
        .iter()
        .map(|step| {
            assert_terms_bounded(step.a().terms(), "retained A form");
            assert_terms_bounded(step.b().terms(), "retained B form");
            assert_terms_bounded(step.c().terms(), "retained C form");
            format!(
                "{{ emittedRow := {}, sourceRow := {}, a := {}, b := {}, c := {} }}",
                step.emitted_row(),
                step.source_row(),
                lean_source_lc(step.a()),
                lean_source_lc(step.b()),
                lean_source_lc(step.c())
            )
        })
        .collect::<Vec<_>>();
    let decoders = decoder
        .decoders()
        .iter()
        .map(|entry| {
            format!(
                "{{ column := {}, resolution := {} }}",
                entry.column(),
                lean_resolution(entry.resolution())
            )
        })
        .collect::<Vec<_>>();

    let families = [
        ("SourceColumns", "Nat"),
        ("RetainedSlots", "RawSourceSlot"),
        ("LinearDefinitions", "RawSourceDefinition"),
        ("TraceEliminatedColumns", "Nat"),
        ("DerivedProductSums", "RawDerivedProductSum"),
        ("RewriteSteps", "RawRewriteStep"),
        ("RetainedSteps", "RawRetainedStep"),
        ("Decoders", "RawSourceDecoder"),
    ];
    let payloads = [
        source_columns,
        retained_slots,
        linear_definitions,
        trace_eliminated,
        derived,
        rewrites,
        retained,
        decoders,
    ];
    let mut files = Vec::new();
    for ((family, value_type), values) in families.iter().zip(payloads) {
        let shard_size = if *family == "RewriteSteps" {
            REWRITE_SHARD_SIZE
        } else {
            DEFAULT_SHARD_SIZE
        };
        files.extend(render_family(family, value_type, values, shard_size));
    }
    files.push(provenance_root(&families, provenance.arm(), decoder.arm()));
    files
}
