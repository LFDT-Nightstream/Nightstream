//! Shared Lean renderer for compact selective decoder runs.

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveProjectedSourceDecoderRun, SelectiveProjectedSourceResolutionRun,
};

pub(super) fn lean_resolution(resolution: SelectiveProjectedSourceResolutionRun) -> String {
    match resolution {
        SelectiveProjectedSourceResolutionRun::Direct {
            start,
            start_stride,
            width,
            centered,
        } => format!(".direct {start} {start_stride} {width} {centered}"),
        SelectiveProjectedSourceResolutionRun::DecompositionAlias {
            source,
            source_stride,
            digit,
            digit_stride,
            start,
            start_stride,
            centered,
        } => format!(
            ".decompositionAlias {source} {source_stride} {digit} {digit_stride} {start} {start_stride} {centered}"
        ),
        SelectiveProjectedSourceResolutionRun::EqualityAlias {
            source,
            source_stride,
            start,
            start_stride,
            width,
            centered,
        } => format!(".equalityAlias {source} {source_stride} {start} {start_stride} {width} {centered}"),
        SelectiveProjectedSourceResolutionRun::LinearDefinition => ".linearDefinition".to_owned(),
        SelectiveProjectedSourceResolutionRun::TraceEliminated => ".traceEliminated".to_owned(),
    }
}

pub(super) fn write_runs(
    rendered: &mut String,
    name: &str,
    raw_run_type: &str,
    runs: &[SelectiveProjectedSourceDecoderRun],
) {
    writeln!(rendered, "def {name} : List {raw_run_type} :=\n  [").expect("render decoder run header");
    for (index, run) in runs.iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ sourceStart := {}, length := {}, resolution := {} }}",
            run.source_start(),
            run.length(),
            lean_resolution(run.resolution()),
        )
        .expect("render decoder run");
    }
    writeln!(rendered, "  ]\n").expect("render decoder run footer");
}
