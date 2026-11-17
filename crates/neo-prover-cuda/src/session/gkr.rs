//! The rounds of one layer of a batched fraction-sum GKR prover on the
//! device.
//!
//! Owns: the round message and the fold over a layer's tables. The tables
//! are copied to the device when the layer starts and copied back to the
//! caller's memory at the last fold. Does not own the protocol or the
//! tables: the caller draws every challenge, builds every layer and reads
//! the children.

use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};

use super::{Buffer, CudaSession, Grid};
use crate::kernels::gkr::{GKR_NODES, GKR_PARTS, GKR_SHAPE_WORDS, GKR_THREADS};
use crate::CudaError;

type Ext = [u64; 3];

/// One GKR layer's tables on the device. Each fold leaves the entries in
/// place at twice the stride: after `r` rounds, logical entry `j` of a table
/// sits at element `j·2^r`.
pub struct CudaGkrLayer<'a> {
    /// Each tree's factor count and parts (factors, then the denominator).
    trees: Vec<(usize, Vec<Buffer>)>,
    eq: Buffer,
    /// The caller's tables (every part, then the eq table), written at the
    /// last fold.
    host: Vec<&'a mut [u64]>,
    /// One partial sum per message block, sized for the first round.
    partials: Buffer,
    stride: usize,
    /// Pairs of the next round: half the eq table's live length.
    pairs: usize,
}

impl CudaSession {
    /// The device view of one GKR layer: per tree, its parts (at most two
    /// factors, then the denominator; `4·n` elements of three words), and
    /// the eq table (`2·n` elements).
    pub fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Result<CudaGkrLayer<'a>, CudaError> {
        let pairs = eq.len() / 6;
        let parts: usize = trees.iter().map(Vec::len).sum();
        let shaped = |parts: &Vec<&mut [u64]>| {
            (1..=3).contains(&parts.len()) && parts.iter().all(|part| part.len() == 12 * pairs)
        };
        if trees.is_empty() || parts > GKR_PARTS || pairs == 0 || eq.len() != 6 * pairs || !trees.iter().all(shaped) {
            return Err(CudaError::Shape(
                "a GKR layer needs one to three parts per tree, each twice the eq table",
            ));
        }
        let partials = self.buffer(pairs.div_ceil(GKR_THREADS) * GKR_NODES * 3 * size_of::<u64>())?;
        let mut device_trees = Vec::with_capacity(trees.len());
        let mut host = Vec::with_capacity(parts + 1);
        for parts in trees {
            let factors = parts.len() - 1;
            let mut buffers = Vec::with_capacity(parts.len());
            for part in parts {
                buffers.push(self.buffer_from_slice(part)?);
                host.push(part);
            }
            device_trees.push((factors, buffers));
        }
        let device_eq = self.buffer_from_slice(eq)?;
        host.push(eq);
        Ok(CudaGkrLayer {
            trees: device_trees,
            eq: device_eq,
            host,
            partials,
            stride: 1,
            pairs,
        })
    }

    /// The round message at nodes `0, 2, ..., degree`: see `gkr_message` in
    /// `kernels/gkr.rs`. `weights` has one entry per tree.
    pub fn gkr_message(
        &self,
        layer: &CudaGkrLayer<'_>,
        weights: &[Ext],
        mu: Ext,
        degree: usize,
    ) -> Result<Vec<Ext>, CudaError> {
        if layer.pairs == 0 || degree > GKR_NODES || weights.len() != layer.trees.len() {
            return Err(CudaError::Shape(
                "a GKR round needs a pair, a degree up to four and one weight per tree",
            ));
        }
        let mut shape = [0u64; GKR_SHAPE_WORDS];
        shape[..4].copy_from_slice(&[
            layer.pairs as u64,
            layer.stride as u64,
            layer.trees.len() as u64,
            degree as u64,
        ]);
        shape[4..7].copy_from_slice(&mu);
        let mut first = 0;
        for (t, ((factors, parts), weight)) in layer.trees.iter().zip(weights).enumerate() {
            shape[7 + 5 * t..9 + 5 * t].copy_from_slice(&[*factors as u64, first as u64]);
            shape[9 + 5 * t..12 + 5 * t].copy_from_slice(weight);
            first += parts.len();
        }
        let mut addresses = [0u64; GKR_PARTS];
        for (slot, table) in addresses
            .iter_mut()
            .zip(layer.trees.iter().flat_map(|(_, parts)| parts))
        {
            *slot = table.ptr::<u64>() as u64;
        }
        let groups = layer.pairs.div_ceil(GKR_THREADS);
        let kernels = &self.kernels.gkr;
        self.launch(Grid::groups(groups, GKR_THREADS), |stream, config| {
            kernels.gkr_message(stream, config, addresses, layer.eq.ptr(), shape, layer.partials.ptr())
        })?;
        self.finish()?;
        let words = self.read_buffer::<u64>(&layer.partials, groups * GKR_NODES * 3)?;
        let sums = words
            .chunks_exact(GKR_NODES * 3)
            .map(|group| std::array::from_fn::<F, { GKR_NODES * 3 }, _>(|w| F::from_u64(group[w])))
            .fold([F::ZERO; GKR_NODES * 3], |a, b| std::array::from_fn(|w| a[w] + b[w]));
        Ok((0..degree)
            .map(|j| std::array::from_fn(|c| sums[3 * j + c].as_canonical_u64()))
            .collect())
    }

    /// Bind the lowest variable of every table and the eq table to `r`. The
    /// last fold copies the tables back to the caller's memory.
    pub fn gkr_fold(&self, layer: &mut CudaGkrLayer<'_>, r: Ext) -> Result<(), CudaError> {
        if layer.pairs == 0 {
            return Err(CudaError::Shape("a GKR fold needs a pair"));
        }
        // A table has twice the eq table's live entries.
        let shape = |count: usize| [count as u64, layer.stride as u64, r[0], r[1], r[2]];
        let tables = layer.trees.iter().flat_map(|(_, parts)| parts);
        let folds = tables
            .map(|table| (table, 2 * layer.pairs))
            .chain([(&layer.eq, layer.pairs)]);
        let kernels = &self.kernels.gkr;
        for (table, count) in folds {
            self.launch(Grid::Threads(count), |stream, config| {
                kernels.gkr_fold(stream, config, table.ptr(), shape(count))
            })?;
        }
        self.finish()?;
        layer.stride *= 2;
        layer.pairs /= 2;
        if layer.pairs == 0 {
            let tables = layer
                .trees
                .iter()
                .flat_map(|(_, parts)| parts)
                .chain([&layer.eq]);
            for (table, host) in tables.zip(layer.host.iter_mut()) {
                self.read_into(table, 0, host)?;
            }
        }
        Ok(())
    }
}
