use neo_ccs::Mat;
use neo_math::{D, F};
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use crate::session::{host_view, Arg, Buffer, Grid, MetalSession};
use crate::MetalError;

pub(super) struct LightDeltas {
    output: Buffer,
    cells: usize,
}

impl LightDeltas {
    pub(super) fn add_to(&self, running: &Mat<F>) -> Mat<F> {
        let deltas = host_view::<i16>(&self.output, self.cells);
        let values = if let Some(constant) = running.virtual_constant_value() {
            deltas
                .par_iter()
                .map(|&d| *constant + F::from_i16(d))
                .collect()
        } else {
            running
                .as_slice()
                .par_iter()
                .zip(deltas)
                .map(|(&x, &d)| x + F::from_i16(d))
                .collect()
        };
        Mat::from_row_major(D, self.cells / D, values)
    }
}

impl MetalSession {
    pub(super) fn mix_light(&self, rhos: &[Mat<F>], witnesses: &[&Mat<F>]) -> Result<Option<LightDeltas>, MetalError> {
        let ([rho, weight], [fresh, running]) = (rhos, witnesses) else {
            return Ok(None);
        };
        let cols = fresh.cols();
        let Some((positive, negative)) = fresh.packed_signed_unit_column_masks() else {
            return Ok(None);
        };
        if cols == 0
            || positive.len() != cols
            || negative.len() != cols
            || fresh.rows() != D
            || running.rows() != D
            || running.cols() != cols
            || running.is_packed_signed_unit()
            || rho.cols() != D
            || rho.rows() != D
            || weight.rows() != D
            || weight.cols() != D
            || !weight.is_identity()
        {
            return Ok(None);
        }
        let cells = cols
            .checked_mul(D)
            .filter(|&cells| u32::try_from(cells).is_ok())
            .ok_or(MetalError::Shape("device PiRLC light grid exceeds u32"))?;
        let mut weights = [[0u64; 3]; D];
        for (row, parts) in rho.as_slice().chunks_exact(D).zip(&mut weights) {
            for (lane, value) in row.iter().enumerate() {
                let x = value.as_canonical_u64();
                let neg = F::ORDER_U64 - x;
                match x.min(neg) {
                    0 => continue,
                    1 => parts[1] |= 1 << lane,
                    2 => parts[2] |= 1 << lane,
                    _ => return Ok(None),
                }
                if neg < x {
                    parts[0] |= 1 << lane;
                }
            }
        }
        let weights = self.buffer_from_slice(&weights)?;
        let plus = self.buffer_from_slice(positive)?;
        let minus = self.buffer_from_slice(negative)?;
        let output = self.buffer(cells * size_of::<i16>())?;
        let shape = self.buffer_from_slice(&[cols as u64])?;
        let command = self.command_buffer("nightstream.pi_rlc.light")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        self.launch(
            &encoder,
            &self.rlc_light_deltas,
            &[
                Arg::Buffer(0, &weights),
                Arg::Buffer(1, &plus),
                Arg::Buffer(2, &minus),
                Arg::Buffer(3, &output),
                Arg::Buffer(4, &shape),
            ],
            Grid::Threads(cells),
        );
        encoder.endEncoding();
        self.finish(&command)?;
        Ok(Some(LightDeltas { output, cells }))
    }
}
