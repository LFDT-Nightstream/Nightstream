//! One-pass indexes for bounded selected-row projection.

use std::collections::{BTreeMap, BTreeSet};

use neo_ccs::CcsMatrix;
use neo_math::{D, F};
use p3_field::PrimeCharacteristicRing;

use super::{
    structure, trace_error, unique_owner, LowNormR1csError, SelectiveCompilerAudit, SelectiveProjectedGeometricRun,
    SelectiveProjectedPort, SelectiveProjectedRowArtifact, SelectiveProjectedSourceLinearCombination,
    SelectiveProjectedSourceTerm, SelectiveProjectedTerm, SparseR1cs, SELECTIVE_ARITY,
};

type SourcePorts = [SelectiveProjectedSourceLinearCombination; 3];

fn finish_source(mut terms: BTreeMap<usize, F>) -> SelectiveProjectedSourceLinearCombination {
    terms.retain(|_, coefficient| *coefficient != F::ZERO);
    let constant = terms.remove(&0).unwrap_or(F::ZERO);
    SelectiveProjectedSourceLinearCombination {
        constant,
        terms: terms
            .into_iter()
            .map(|(column, coefficient)| SelectiveProjectedSourceTerm { column, coefficient })
            .collect(),
    }
}

fn source_port_rows(
    matrix: &CcsMatrix<F>,
    selected: &BTreeSet<usize>,
) -> Result<BTreeMap<usize, SelectiveProjectedSourceLinearCombination>, LowNormR1csError> {
    let mut rows = selected
        .iter()
        .copied()
        .map(|row| (row, BTreeMap::<usize, F>::new()))
        .collect::<BTreeMap<_, _>>();
    let mut add = |row: usize, column: usize, coefficient: F| {
        if coefficient != F::ZERO {
            *rows
                .get_mut(&row)
                .expect("selected source row")
                .entry(column)
                .or_insert(F::ZERO) += coefficient;
        }
    };
    let mut append_csc = |csc: &neo_ccs::CscMat<F>| {
        for column in 0..csc.ncols {
            for entry in csc.column_range(column) {
                let row = csc.row_index(entry);
                if selected.contains(&row) {
                    add(row, column, csc.vals[entry]);
                }
            }
        }
    };
    match matrix {
        CcsMatrix::Identity { n } => {
            if selected.iter().any(|&row| row >= *n) {
                return Err(trace_error("projected retained source row exceeds identity port"));
            }
            for &row in selected {
                add(row, row, F::ONE);
            }
        }
        CcsMatrix::Csc(csc) => append_csc(csc),
        CcsMatrix::CscWithSeededPhi81 {
            csc,
            blocks,
            geometric_runs,
        } => {
            if blocks.iter().any(|block| {
                selected
                    .range(block.row_start()..block.row_end())
                    .next()
                    .is_some()
            }) {
                return Err(trace_error(
                    "projected retained source row intersects a compact seeded row",
                ));
            }
            append_csc(csc);
            for run in geometric_runs {
                if selected.contains(&run.row()) {
                    let mut coefficient = *run.initial();
                    for column in run.column_start()..run.column_start() + run.len() {
                        add(run.row(), column, coefficient);
                        coefficient *= *run.ratio();
                    }
                }
            }
        }
        CcsMatrix::VerifierArtifact { .. } => {
            return Err(trace_error(
                "projected source-row audit requires materialized matrix content",
            ));
        }
    }
    Ok(rows
        .into_iter()
        .map(|(row, terms)| (row, finish_source(terms)))
        .collect())
}

pub(super) fn source_rows(
    source: &SparseR1cs,
    selected: &BTreeSet<usize>,
) -> Result<BTreeMap<usize, SourcePorts>, LowNormR1csError> {
    if selected.iter().any(|&row| row >= source.n) {
        return Err(trace_error("projected retained source row is out of range"));
    }
    let ports = [
        source_port_rows(&source.a, selected)?,
        source_port_rows(&source.b, selected)?,
        source_port_rows(&source.c, selected)?,
    ];
    Ok(selected
        .iter()
        .copied()
        .map(|row| {
            let values = std::array::from_fn(|port| ports[port][&row].clone());
            (row, values)
        })
        .collect())
}

pub(super) fn project_rows(
    emitted: &structure::EmittedStructureTerms,
    audit: &SelectiveCompilerAudit,
    selected_rows: &[usize],
) -> Result<Vec<SelectiveProjectedRowArtifact>, LowNormR1csError> {
    let selected = selected_rows.iter().copied().collect::<BTreeSet<_>>();
    let mut indexed = (0..SELECTIVE_ARITY)
        .map(|_| BTreeMap::<usize, BTreeMap<usize, F>>::new())
        .collect::<Vec<_>>();
    let mut geometric = (0..SELECTIVE_ARITY)
        .map(|_| BTreeMap::<usize, Vec<SelectiveProjectedGeometricRun>>::new())
        .collect::<Vec<_>>();
    for (port, terms) in emitted.matrix_terms.iter().enumerate() {
        for &(row, column, coefficient) in &terms.explicit {
            if selected.contains(&row) {
                if column >= emitted.columns {
                    return Err(trace_error("projected selective term exceeds the final column domain"));
                }
                *indexed[port]
                    .entry(row)
                    .or_default()
                    .entry(column)
                    .or_insert(F::ZERO) += coefficient;
            }
        }
        for run in &terms.geometric_runs {
            if selected.contains(&run.row()) {
                if run.column_start() + run.len() > emitted.columns {
                    return Err(trace_error("projected geometric run exceeds the final column domain"));
                }
                geometric[port]
                    .entry(run.row())
                    .or_default()
                    .push(SelectiveProjectedGeometricRun {
                        column_start: run.column_start(),
                        length: run.len(),
                        initial: *run.initial(),
                        ratio: *run.ratio(),
                    });
            }
        }
        for block in &terms.seeded {
            if selected
                .range(block.row_start()..block.row_start() + D * block.kappa())
                .next()
                .is_some()
            {
                return Err(trace_error(
                    "bounded selective projection intersects a compact seeded row",
                ));
            }
        }
    }
    selected_rows
        .iter()
        .copied()
        .map(|row| {
            let ports = std::array::from_fn(|port| {
                let mut terms = indexed[port].remove(&row).unwrap_or_default();
                terms.retain(|_, coefficient| *coefficient != F::ZERO);
                SelectiveProjectedPort {
                    explicit: terms
                        .into_iter()
                        .map(|(column, coefficient)| SelectiveProjectedTerm { column, coefficient })
                        .collect(),
                    geometric_runs: geometric[port].remove(&row).unwrap_or_default(),
                    seeded_blocks: Vec::new(),
                }
            });
            let (run_index, owner) = unique_owner(audit, row)?;
            Ok(SelectiveProjectedRowArtifact {
                rows: emitted.rows,
                columns: emitted.columns,
                emitted_row: row,
                run_index,
                family: owner.family(),
                arm: owner.arm(),
                ports,
            })
        })
        .collect()
}
