//! Fixed R1CS implementation of the WASM operation relations.
mod builder;
mod compare;
mod counts;
mod shared;

use crate::sparse_r1cs::SparseR1cs;
use crate::tagged_r1cs_builder::WasmConstraintScope;
use neo_application::{range_checked_variable_widths, ApplicationRelation, AuxiliaryLayout};
use neo_math::F;
use p3_field::PrimeCharacteristicRing;
use std::sync::OnceLock;

use crate::isa::WasmOpcode;
use crate::layout::{
    selector_col, COL_STACK_READ_VALUE_HI, COL_STACK_READ_VALUE_LO, COL_STACK_WRITE0_VALUE_HI,
    COL_STACK_WRITE0_VALUE_LO,
};
use crate::range_check::range_checked_bit_columns;
use builder::{selector_lc, CircuitBuilder, ColumnId, Lc};
use compare::enforce_comparisons;
use counts::enforce_popcount;
struct Inputs {
    left32: Vec<ColumnId>,
    right32: Vec<ColumnId>,
    output32: Vec<ColumnId>,
    left64: Vec<ColumnId>,
    right64: Vec<ColumnId>,
    output64: Vec<ColumnId>,
}

struct Section {
    name: &'static str,
    opcodes: fn() -> Vec<WasmOpcode>,
    constrain: fn(&mut CircuitBuilder, &Inputs) -> Result<(), String>,
}

const SECTIONS: &[Section] = {
    use WasmOpcode::*;
    &[
        Section {
            name: "shared_binary",
            opcodes: shared::opcodes,
            constrain: shared::enforce,
        },
        Section {
            name: "compare",
            opcodes: || {
                compare::comparison_operations(false)
                    .iter()
                    .chain(compare::comparison_operations(true).iter())
                    .map(|entry| entry.0)
                    .collect()
            },
            constrain: enforce_comparisons,
        },
        Section {
            name: "popcnt32",
            opcodes: || vec![I32Popcnt],
            constrain: |b, i| enforce_popcount(b, &i.left32, &i.output32, I32Popcnt),
        },
        Section {
            name: "popcnt64",
            opcodes: || vec![I64Popcnt],
            constrain: |b, i| enforce_popcount(b, &i.left64, &i.output64, I64Popcnt),
        },
    ]
};

impl Inputs {
    fn new() -> Result<Self, String> {
        let left32 = column_bits(COL_STACK_READ_VALUE_LO[0])?;
        let right32 = column_bits(COL_STACK_READ_VALUE_LO[1])?;
        let output32 = column_bits(COL_STACK_WRITE0_VALUE_LO)?;
        let left64 = wide_bits(COL_STACK_READ_VALUE_LO[0], COL_STACK_READ_VALUE_HI[0])?;
        let right64 = wide_bits(COL_STACK_READ_VALUE_LO[1], COL_STACK_READ_VALUE_HI[1])?;
        let output64 = wide_bits(COL_STACK_WRITE0_VALUE_LO, COL_STACK_WRITE0_VALUE_HI)?;
        Ok(Self {
            left32,
            right32,
            output32,
            left64,
            right64,
            output64,
        })
    }
}

fn column_bits(column: usize) -> Result<Vec<ColumnId>, String> {
    range_checked_bit_columns(column)
        .map(|columns| columns.map(ColumnId).collect())
        .ok_or_else(|| format!("missing range-check bits for application column {column}"))
}

fn wide_bits(low: usize, high: usize) -> Result<Vec<ColumnId>, String> {
    let mut bits = column_bits(low)?;
    bits.extend(column_bits(high)?);
    Ok(bits)
}

fn opcode_selector(operation: WasmOpcode) -> ColumnId {
    ColumnId(selector_col(operation).expect("operation opcode has a selector"))
}

fn ops_lc(operations: &[WasmOpcode]) -> Lc {
    selector_lc(
        &operations
            .iter()
            .map(|&operation| opcode_selector(operation))
            .collect::<Vec<_>>(),
    )
}

fn opcode_lc(operation: WasmOpcode) -> Lc {
    Lc::var(opcode_selector(operation))
}

type OperationRelation = ApplicationRelation<WasmConstraintScope>;

/// Frozen operation layout and the complete base-plus-operation relation.
pub struct OperationCircuit {
    relation: OperationRelation,
    assignment: AuxiliaryLayout,
    base_rows: usize,
    groups: Vec<OperationGroup>,
}

/// Named construction groups retain exact column and row ranges for export.
#[derive(Clone, Debug)]
pub struct OperationGroup {
    pub name: &'static str,
    pub columns: std::ops::Range<usize>,
    pub rows: std::ops::Range<usize>,
}

impl OperationCircuit {
    pub fn groups(&self) -> &[OperationGroup] {
        &self.groups
    }
    pub fn relation(&self) -> &OperationRelation {
        &self.relation
    }
    pub fn base_columns(&self) -> usize {
        self.assignment.base_columns()
    }
    pub fn operation_rows(&self) -> std::ops::Range<usize> {
        self.base_rows..self.relation.r1cs().catalog().len()
    }

    /// Assignment cannot alter the frozen relation or allocate more columns.
    pub fn assign(&self, witness: &mut [F]) -> Result<(), OperationCircuitError> {
        self.assignment
            .assign(witness)
            .map_err(|e| OperationCircuitError::Assignment(e.to_string()))
    }
}

/// Build the fixed relation used by batching, the prover, and audits.
pub fn operation_circuit() -> Result<&'static OperationCircuit, OperationCircuitError> {
    static CIRCUIT: OnceLock<Result<OperationCircuit, String>> = OnceLock::new();
    CIRCUIT
        .get_or_init(|| {
            let base = crate::build_wasm_relation().map_err(|e| e.to_string())?;
            let base_rows = base.r1cs().catalog().len();
            let inputs = Inputs::new()?;
            let mut builder = CircuitBuilder::new(&base);
            let mut groups = Vec::new();
            for section in SECTIONS {
                let (column_start, row_start) = builder.position();
                builder.section(section.name, (section.opcodes)());
                (section.constrain)(&mut builder, &inputs)?;
                let (column_end, row_end) = builder.position();
                groups.push(OperationGroup {
                    name: section.name,
                    columns: column_start..column_end,
                    rows: row_start..row_end,
                });
            }
            let (relation, assignment) = builder.finish()?;
            Ok(OperationCircuit {
                relation,
                assignment,
                base_rows,
                groups,
            })
        })
        .as_ref()
        .map_err(|e| OperationCircuitError::Construction(e.clone()))
}

pub(crate) struct OperationShape {
    pub(crate) relation: SparseR1cs,
    pub(crate) widths: Vec<usize>,
}

pub(crate) fn single_relation() -> Result<OperationShape, OperationCircuitError> {
    let complete = operation_circuit()?.relation();
    let r1cs = complete.r1cs();
    let structure = r1cs.structure();
    let relation = SparseR1cs::new(
        structure.matrices()[0].clone(),
        structure.matrices()[1].clone(),
        structure.matrices()[2].clone(),
        structure.n(),
        structure.m(),
        r1cs.public_input_count(),
    )
    .map_err(|e| OperationCircuitError::Construction(e.to_string()))?;
    Ok(OperationShape {
        relation,
        widths: range_checked_variable_widths(complete.columns()),
    })
}

pub(crate) fn extend_witness(mut base: Vec<F>) -> Result<Vec<F>, OperationCircuitError> {
    let circuit = operation_circuit()?;
    if base.len() != circuit.base_columns() {
        return Err(OperationCircuitError::BaseWidth {
            expected: circuit.base_columns(),
            actual: base.len(),
        });
    }
    base.resize(circuit.relation.r1cs().column_count(), F::ZERO);
    circuit.assign(&mut base)?;
    Ok(base)
}

#[doc(hidden)]
pub fn audit_operation_witness(base: &[F]) -> Result<usize, OperationCircuitError> {
    let circuit = operation_circuit()?;
    let witness = extend_witness(base.to_vec())?;
    for index in circuit.operation_rows() {
        let row = circuit.relation.r1cs().catalog().rows()[index].row();
        if evaluate(row.a_terms(), &witness) * evaluate(row.b_terms(), &witness) != evaluate(row.c_terms(), &witness) {
            return Err(OperationCircuitError::Unsatisfied { row: index });
        }
    }
    Ok(witness.len() - base.len())
}

#[doc(hidden)]
pub fn audit_operation_auxiliary_load_bearing(base: &[F]) -> Result<usize, OperationCircuitError> {
    let circuit = operation_circuit()?;
    let mut witness = extend_witness(base.to_vec())?;
    for column in circuit.base_columns()..witness.len() {
        let original = witness[column];
        witness[column] += F::ONE;
        let rejected = circuit.operation_rows().any(|index| {
            let row = circuit.relation.r1cs().catalog().rows()[index].row();
            evaluate(row.a_terms(), &witness) * evaluate(row.b_terms(), &witness) != evaluate(row.c_terms(), &witness)
        });
        witness[column] = original;
        if !rejected {
            return Err(OperationCircuitError::UnconstrainedAuxiliary { column });
        }
    }
    Ok(witness.len() - base.len())
}

fn evaluate(terms: &[(usize, F)], z: &[F]) -> F {
    terms
        .iter()
        .fold(F::ZERO, |sum, &(column, coefficient)| sum + z[column] * coefficient)
}

#[derive(Debug, thiserror::Error)]
pub enum OperationCircuitError {
    #[error("operation circuit construction failed: {0}")]
    Construction(String),
    #[error("operation assignment failed: {0}")]
    Assignment(String),
    #[error("operation assignment has {actual} base columns, expected {expected}")]
    BaseWidth { expected: usize, actual: usize },
    #[error("operation relation row {row} is unsatisfied")]
    Unsatisfied { row: usize },
    #[error("operation auxiliary column {column} can change without violating a row")]
    UnconstrainedAuxiliary { column: usize },
}
