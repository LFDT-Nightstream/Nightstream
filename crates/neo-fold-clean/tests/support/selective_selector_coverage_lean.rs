//! Shared deterministic Lean encoding for checked selector-coverage runs.

#![allow(dead_code)]

use std::fmt::Write as _;

use neo_fold_clean::frontends::r1cs_f_prime::{
    SelectiveEmittedRowFamily, SelectiveGatePort, SelectiveSelectorGateCoverage,
    SELECTIVE_SELECTOR_GATE_COVERAGE_SCHEMA_VERSION,
};
use p3_field::PrimeField64;

pub fn lean_family(family: SelectiveEmittedRowFamily) -> &'static str {
    match family {
        SelectiveEmittedRowFamily::SelectorDomain => "selectorDomain",
        SelectiveEmittedRowFamily::SharedDomain => "sharedDomain",
        SelectiveEmittedRowFamily::ArmDomain => "armDomain",
        SelectiveEmittedRowFamily::OneHot => "oneHot",
        SelectiveEmittedRowFamily::PublicPadding => "publicPadding",
        SelectiveEmittedRowFamily::PrivatePadding => "privatePadding",
        SelectiveEmittedRowFamily::Retained => "retained",
        SelectiveEmittedRowFamily::Poseidon2 => "poseidon2",
        SelectiveEmittedRowFamily::CenteredUnit => "centeredUnit",
        SelectiveEmittedRowFamily::ShiftedTernaryCanonical => "shiftedTernaryCanonical",
        SelectiveEmittedRowFamily::PolynomialEvaluation => "polynomialEvaluation",
        SelectiveEmittedRowFamily::ProductSum => "productSum",
        SelectiveEmittedRowFamily::RingPadding => "ringPadding",
    }
}

fn lean_gate_port(port: SelectiveGatePort) -> &'static str {
    match port {
        SelectiveGatePort::General => "general",
        SelectiveGatePort::Evaluation => "evaluation",
        SelectiveGatePort::GeneralEvaluation => "generalEvaluation",
    }
}

pub fn write_raw_coverage(
    rendered: &mut String,
    name: &str,
    coverage: &SelectiveSelectorGateCoverage,
) -> std::fmt::Result {
    writeln!(
        rendered,
        "\ndef {name} : RawCoverage where\n  schemaVersion := {}\n  rows := {}\n  columns := {}\n  selectorColumns := {:?}\n  polynomialArity := {}\n  polynomialTerms := [",
        SELECTIVE_SELECTOR_GATE_COVERAGE_SCHEMA_VERSION,
        coverage.rows(),
        coverage.columns(),
        coverage.selector_columns(),
        coverage.polynomial_arity(),
    )?;
    for (index, term) in coverage.polynomial_terms().iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ coefficient := {}, exponents := {:?} }}",
            term.coefficient().as_canonical_u64(),
            term.exponents(),
        )?;
    }
    writeln!(rendered, "  ]\n  ownerRuns := [")?;
    for (index, run) in coverage.owner_runs().iter().enumerate() {
        let rows = run.emitted_rows();
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ start := {}, stop := {}, family := .{}, arm := {} }}",
            rows.start,
            rows.end,
            lean_family(run.family()),
            run.arm()
                .map_or_else(|| "none".to_owned(), |arm| format!("some {arm}")),
        )?;
    }
    writeln!(rendered, "  ]\n  gateRuns := [")?;
    for (index, run) in coverage.gate_runs().iter().enumerate() {
        let rows = run.emitted_rows();
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ start := {}, stop := {}, port := .{}, column := {}, coefficient := {} }}",
            rows.start,
            rows.end,
            lean_gate_port(run.port()),
            run.column(),
            run.coefficient().as_canonical_u64(),
        )?;
    }
    writeln!(rendered, "  ]")
}

pub fn write_coalesced_raw_coverage(
    rendered: &mut String,
    name: &str,
    coverage: &SelectiveSelectorGateCoverage,
) -> std::fmt::Result {
    writeln!(
        rendered,
        "\ndef {name} : RawCoverage where\n  schemaVersion := {}\n  rows := {}\n  columns := {}\n  selectorColumns := {:?}\n  polynomialArity := {}\n  polynomialTerms := [",
        SELECTIVE_SELECTOR_GATE_COVERAGE_SCHEMA_VERSION,
        coverage.rows(),
        coverage.columns(),
        coverage.selector_columns(),
        coverage.polynomial_arity(),
    )?;
    for (index, term) in coverage.polynomial_terms().iter().enumerate() {
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ coefficient := {}, exponents := {:?} }}",
            term.coefficient().as_canonical_u64(),
            term.exponents(),
        )?;
    }

    let runs = coverage.coalesced_owner_gate_runs();
    writeln!(rendered, "  ]\n  ownerRuns := [")?;
    for (index, run) in runs.iter().enumerate() {
        let rows = run.emitted_rows();
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ start := {}, stop := {}, family := .{}, arm := {} }}",
            rows.start,
            rows.end,
            lean_family(run.family()),
            run.arm()
                .map_or_else(|| "none".to_owned(), |arm| format!("some {arm}")),
        )?;
    }
    writeln!(rendered, "  ]\n  gateRuns := [")?;
    for (index, run) in runs.iter().enumerate() {
        let rows = run.emitted_rows();
        let separator = if index == 0 { "    " } else { "  , " };
        writeln!(
            rendered,
            "{separator}{{ start := {}, stop := {}, port := .{}, column := {}, coefficient := {} }}",
            rows.start,
            rows.end,
            lean_gate_port(run.port()),
            run.column(),
            run.coefficient().as_canonical_u64(),
        )?;
    }
    writeln!(rendered, "  ]")
}
