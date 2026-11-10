//! The leaves of the layer-1 setup oracles that a proof queries
//! (`neo_spartan::Device::setup_leaves`). Each oracle is made, encoded and
//! gathered on the device, and only its queried leaves come back.

use neo_ajtai::nightstream_fprime_setup::PRODUCTION_VERIFIER_ROWS;
use objc2_metal::{MTLCommandBuffer, MTLCommandEncoder};

use super::dft::PASS_THREADS;
use super::{Arg, Grid, MetalSession};
use crate::MetalError;

/// Lanes per block of a key-row table, and values per coset or power thread.
const LANES: usize = 64;
/// Neighboring butterfly groups per threadgroup of a single-column pass
/// (`SETUP_COLUMNS` in `setup.metal`).
const PASS_COLUMNS: usize = 8;
/// Values per leaf of one oracle.
const LEAF: usize = 8;

/// One layer-1 setup oracle (`neo_spartan::SetupOracle`).
#[derive(Clone, Copy)]
pub enum SetupSource<'a> {
    /// Production key row `row` on the cube `x = 64·b + lane`, zero past
    /// `blocks` blocks and past lane `D`.
    KeyRow { row: u32, blocks: usize },
    /// A table of `2^variables` canonical words.
    Table(&'a [u64]),
}

impl MetalSession {
    /// For each oracle, its codeword at positions `leaf + v·2^{variables-2}`
    /// (`v < 8`) for every leaf in `leaves`: the DFT at `ω` = `generator`
    /// (order `2^{variables+1}`) of its monomial coefficients `c_x` scaled
    /// by `shift^x`, with the upper half zero. Values are canonical.
    pub fn setup_leaves(
        &self,
        oracles: &[SetupSource<'_>],
        variables: usize,
        leaves: &[u32],
        generator: u64,
        shift: u64,
    ) -> Result<Vec<Vec<[u64; LEAF]>>, MetalError> {
        if !(LANES.trailing_zeros() as usize..32).contains(&variables) {
            return Err(MetalError::Shape("setup leaves need 6 to 31 variables"));
        }
        let half = 1usize << variables;
        let quarter = half >> 2;
        if leaves.is_empty() || leaves.iter().any(|&leaf| leaf as usize >= quarter) {
            return Err(MetalError::Shape("a setup leaf is outside the codeword"));
        }
        for oracle in oracles {
            let fits = match *oracle {
                SetupSource::KeyRow { row, blocks } => {
                    u64::from(row) < PRODUCTION_VERIFIER_ROWS && blocks.checked_mul(LANES).is_some_and(|x| x <= half)
                }
                SetupSource::Table(table) => table.len() == half,
            };
            if !fits {
                return Err(MetalError::Shape("a setup oracle does not fit its cube"));
            }
        }
        let rows = 2 * half;
        let data = self.buffer(rows * size_of::<u64>())?;
        let twiddles = self.buffer(half * size_of::<u64>())?;
        let out = self.buffer(leaves.len() * LEAF * size_of::<u64>())?;
        let prefix = self.production_key_prefix()?;
        let leaf_buffer = self.buffer_from_slice(leaves)?;
        let twiddle_shape = self.buffer_from_slice(&[generator])?;
        let coset_shape = self.buffer_from_slice(&[shift])?;
        let zero_shape = self.buffer_from_slice(&[half as u64])?;
        let gather_shape = self.buffer_from_slice(&[u64::from(variables as u32 + 1), quarter as u64])?;
        let mobius_passes = self.column_passes(variables as u32)?;
        let dft_passes = self.column_passes(variables as u32 + 1)?;
        let key_shapes = oracles
            .iter()
            .map(|oracle| match *oracle {
                SetupSource::KeyRow { row, blocks } => self.buffer_from_slice(&[u64::from(row), blocks as u64]),
                SetupSource::Table(_) => self.buffer_from_slice(&[0u64, 0]),
            })
            .collect::<Result<Vec<_>, _>>()?;

        let command = self.command_buffer("nightstream.setup_twiddles")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        self.launch(
            &encoder,
            &self.setup_powers,
            &[Arg::Buffer(0, &twiddles), Arg::Buffer(1, &twiddle_shape)],
            Grid::Threads(half / LANES),
        );
        encoder.endEncoding();
        self.finish(&command)?;

        let mut values = Vec::with_capacity(oracles.len());
        for (oracle, key_shape) in oracles.iter().zip(&key_shapes) {
            // Every earlier command has finished, so the host may write.
            if let SetupSource::Table(table) = *oracle {
                self.write_shared(&data, table)?;
            }
            let command = self.command_buffer("nightstream.setup_leaves")?;
            let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
            match oracle {
                SetupSource::KeyRow { .. } => self.launch(
                    &encoder,
                    &self.setup_key_table,
                    &[
                        Arg::Buffer(0, &prefix),
                        Arg::Buffer(1, key_shape),
                        Arg::Buffer(2, &data),
                    ],
                    Grid::Threads(rows / LANES),
                ),
                SetupSource::Table(_) => self.launch(
                    &encoder,
                    &self.setup_zero,
                    &[Arg::Buffer(0, &data), Arg::Buffer(1, &zero_shape)],
                    Grid::Threads(half),
                ),
            }
            // Serial dispatches: each pass reads what the one before wrote.
            for (shape, groups) in &mobius_passes {
                self.launch(
                    &encoder,
                    &self.setup_mobius_pass,
                    &[Arg::Buffer(0, &data), Arg::Buffer(1, shape)],
                    Grid::Groups {
                        groups: *groups,
                        threads: PASS_THREADS,
                    },
                );
            }
            self.launch(
                &encoder,
                &self.setup_coset,
                &[Arg::Buffer(0, &data), Arg::Buffer(1, &coset_shape)],
                Grid::Threads(half / LANES),
            );
            for (shape, groups) in &dft_passes {
                self.launch(
                    &encoder,
                    &self.setup_dif_pass,
                    &[Arg::Buffer(0, &data), Arg::Buffer(1, &twiddles), Arg::Buffer(2, shape)],
                    Grid::Groups {
                        groups: *groups,
                        threads: PASS_THREADS,
                    },
                );
            }
            self.launch(
                &encoder,
                &self.setup_gather,
                &[
                    Arg::Buffer(0, &data),
                    Arg::Buffer(1, &leaf_buffer),
                    Arg::Buffer(2, &gather_shape),
                    Arg::Buffer(3, &out),
                ],
                Grid::Threads(leaves.len() * LEAF),
            );
            encoder.endEncoding();
            self.finish(&command)?;
            values.push(self.read_buffer::<[u64; LEAF]>(&out, leaves.len()));
        }
        Ok(values)
    }

    /// The shapes of the single-column passes over `2^log_rows` values (the
    /// passes of `dif_passes`), with their threadgroup counts: each group
    /// owns up to `PASS_COLUMNS` neighboring butterfly groups.
    fn column_passes(&self, log_rows: u32) -> Result<Vec<(super::Buffer, usize)>, MetalError> {
        let mut first = 0;
        self.dif_passes(log_rows, 1)?
            .into_iter()
            .map(|(shape, stages)| {
                let stride = 1usize << (log_rows - first - stages);
                first += stages;
                Ok((shape, (1usize << (log_rows - stages)) / stride.min(PASS_COLUMNS)))
            })
            .collect()
    }
}

#[cfg(test)]
#[path = "../../tests/unit/setup_leaves.rs"]
mod tests;
