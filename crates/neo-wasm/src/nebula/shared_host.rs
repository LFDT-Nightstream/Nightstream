use neo_application::GadgetDescriptor;
use neo_math::F;
use nightstream::nebula::{RowRelation, SharedColumns};
use p3_field::PrimeCharacteristicRing;
use std::num::NonZeroUsize;

use super::WasmApplicationError;
use crate::ccs::host_event_chain::{absorb_commitment, absorb_flag_column};
use crate::layout::COL_ONE;

const GROUP_ROWS: usize = 8;
const INPUT_LANES: usize = 12;

fn input_columns() -> [usize; INPUT_LANES] {
    let commitment = absorb_commitment();
    core::array::from_fn(|lane| {
        if lane < 4 {
            commitment.previous[lane]
        } else {
            commitment.block[lane - 4]
        }
    })
}

fn shared_columns(width: usize) -> SharedColumns {
    let commitment = absorb_commitment();
    SharedColumns {
        columns: (width..width + INPUT_LANES)
            .chain(commitment.auxiliary_range())
            .chain(commitment.output)
            .collect(),
        rows: NonZeroUsize::new(GROUP_ROWS).unwrap(),
    }
}

/// SPEC.md section 7.6. Keep the source relation intact and substitute only
/// the recorded host commitment's input lanes in the Nebula application.
pub(super) fn transform(relation: &mut RowRelation) -> Result<SharedColumns, WasmApplicationError> {
    let source = crate::operations::r1cs::operation_circuit().map_err(crate::batch::BatchError::from)?;
    let commitment = absorb_commitment();
    let matches = source
        .relation()
        .r1cs()
        .catalog()
        .gadget_occurrences()
        .iter()
        .filter(|gadget| {
            matches!(gadget.descriptor(), GadgetDescriptor::EventCommitment {
            previous, block, output, auxiliary_start, auxiliary_len,
        } if *previous == commitment.previous && *block == commitment.block
            && *output == commitment.output && *auxiliary_start == commitment.auxiliary_start
            && *auxiliary_len == commitment.auxiliary_range().len())
        })
        .collect::<Vec<_>>();
    let [gadget] = matches.as_slice() else {
        return Err(WasmApplicationError::SharedAbsorb);
    };
    let width = relation.width;
    let inputs = input_columns();
    for sides in &mut relation.rows[gadget.row_range().clone()] {
        for (column, _) in sides.iter_mut().flatten() {
            if let Some(lane) = inputs.iter().position(|input| input == column) {
                *column = width + lane;
            }
        }
    }
    for (lane, input) in inputs.into_iter().enumerate() {
        relation.rows.push([
            vec![(absorb_flag_column(), F::ONE)],
            vec![(width + lane, F::ONE), (input, -F::ONE)],
            Vec::new(),
        ]);
    }
    relation.width += INPUT_LANES;
    Ok(shared_columns(width))
}

pub(super) fn step_witness(width: usize, batch: &[F]) -> Result<Vec<F>, WasmApplicationError> {
    let inputs = input_columns();
    let rows = batch
        .chunks_exact(width)
        .map(|row| {
            let mut extended = row.to_vec();
            extended.extend(inputs.map(|column| row[column]));
            extended
        })
        .collect::<Vec<_>>();
    let rows = rows.iter().map(Vec::as_slice).collect::<Vec<_>>();
    Ok(
        shared_columns(width).step_witness(width + INPUT_LANES, COL_ONE, &rows, |group| {
            group
                .iter()
                .position(|row| row[absorb_flag_column()] != F::ZERO)
                .unwrap_or(0)
        })?,
    )
}
