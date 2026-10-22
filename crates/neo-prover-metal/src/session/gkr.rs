//! The rounds of one layer of a batched fraction-sum GKR prover on the
//! device.
//!
//! Owns: the round message and the fold over a layer's tables, which stay
//! in the caller's memory (wrapped in place, no copy, no new device
//! allocation). Does not own the protocol or the tables: the caller draws
//! every challenge, builds every layer and reads the children.

use neo_math::F;
use objc2::runtime::ProtocolObject;
use objc2_metal::{MTLBuffer, MTLCommandBuffer, MTLCommandEncoder, MTLComputeCommandEncoder, MTLResourceUsage};
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use rayon::prelude::*;

use super::{dft::WrappedPages, Arg, Buffer, Grid, MetalSession};
use crate::MetalError;

/// Address slots of the table list; `GKR_PARTS` in `gkr.metal`.
const PARTS: usize = 16;
/// Message nodes `0, 2, 3, 4`; `GKR_NODES` in `gkr.metal`.
const NODES: usize = 4;
/// Threads per message group; `GKR_THREADS` in `gkr.metal`.
const THREADS: usize = 256;

type Ext = [u64; 3];

/// One GKR layer's tables on the device, in the caller's memory. Each fold
/// leaves the entries in place at twice the stride: after `r` rounds,
/// logical entry `j` of a table sits at element `j·2^r`.
pub struct MetalGkrLayer<'a> {
    /// Each tree's factor count and parts (factors, then the denominator).
    trees: Vec<(usize, Vec<WrappedPages<&'a mut [u64]>>)>,
    eq: WrappedPages<&'a mut [u64]>,
    /// One partial sum per message group, sized for the first round.
    partials: Buffer,
    stride: usize,
    /// Pairs of the next round: half the eq table's live length.
    pairs: usize,
}

impl MetalSession {
    /// The device view of one GKR layer: per tree, its parts (at most two
    /// factors, then the denominator; `4·n` elements of three words), and
    /// the eq table (`2·n` elements). Fails, with nothing changed, unless
    /// every table starts on a page and spans whole pages. Wrapped memory
    /// counts as device memory, so the layer allocates its only buffer
    /// first and its rounds pass parameters by value.
    pub fn gkr_layer<'a>(
        &self,
        trees: Vec<Vec<&'a mut [u64]>>,
        eq: &'a mut [u64],
    ) -> Result<MetalGkrLayer<'a>, MetalError> {
        let pairs = eq.len() / 6;
        let parts: usize = trees.iter().map(Vec::len).sum();
        let shaped = |parts: &Vec<&mut [u64]>| {
            (1..=3).contains(&parts.len()) && parts.iter().all(|part| part.len() == 12 * pairs)
        };
        if trees.is_empty() || parts > PARTS || pairs == 0 || eq.len() != 6 * pairs || !trees.iter().all(shaped) {
            return Err(MetalError::Shape(
                "a GKR layer needs one to three parts per tree, each twice the eq table",
            ));
        }
        let partials = self.buffer(pairs.div_ceil(THREADS) * NODES * 3 * size_of::<u64>())?;
        let wrap = |table| {
            self.wrap_pages(table)
                .ok_or(MetalError::Shape("a GKR table must span whole pages"))
        };
        let trees = trees
            .into_iter()
            .map(|parts| {
                let factors = parts.len() - 1;
                let parts = parts.into_iter().map(wrap).collect::<Result<Vec<_>, _>>()?;
                Ok((factors, parts))
            })
            .collect::<Result<Vec<_>, MetalError>>()?;
        Ok(MetalGkrLayer {
            trees,
            eq: wrap(eq)?,
            partials,
            stride: 1,
            pairs,
        })
    }

    /// The round message at nodes `0, 2, ..., degree`: see `gkr_message` in
    /// `gkr.metal`. `weights` has one entry per tree.
    pub fn gkr_message(
        &self,
        layer: &MetalGkrLayer<'_>,
        weights: &[Ext],
        mu: Ext,
        degree: usize,
    ) -> Result<Vec<Ext>, MetalError> {
        if layer.pairs == 0 || degree > NODES || weights.len() != layer.trees.len() {
            return Err(MetalError::Shape(
                "a GKR round needs a pair, a degree up to four and one weight per tree",
            ));
        }
        let mut shape = vec![
            layer.pairs as u64,
            layer.stride as u64,
            layer.trees.len() as u64,
            degree as u64,
        ];
        shape.extend(mu);
        let mut first = 0;
        for ((factors, parts), weight) in layer.trees.iter().zip(weights) {
            shape.extend([*factors as u64, first as u64]);
            shape.extend(weight);
            first += parts.len();
        }
        let tables: Vec<_> = layer.trees.iter().flat_map(|(_, parts)| parts).collect();
        let mut addresses = [0u64; PARTS];
        for (slot, table) in addresses.iter_mut().zip(&tables) {
            *slot = table.gpuAddress();
        }
        let groups = layer.pairs.div_ceil(THREADS);
        let command = self.command_buffer("nightstream.gkr.message")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        // Tables reached through addresses must be made resident.
        for table in &tables {
            encoder.useResource_usage(ProtocolObject::from_ref(&***table), MTLResourceUsage::Read);
        }
        self.launch(
            &encoder,
            &self.gkr_message,
            &[
                Arg::Words(0, &addresses),
                Arg::Buffer(1, &layer.eq),
                Arg::Words(2, &shape),
                Arg::Buffer(3, &layer.partials),
            ],
            Grid::Groups {
                groups,
                threads: THREADS,
            },
        );
        encoder.endEncoding();
        self.finish(&command)?;
        let words = self.read_buffer_parallel::<u64>(&layer.partials, groups * NODES * 3);
        let sums = words
            .par_chunks_exact(NODES * 3)
            .map(|group| std::array::from_fn::<F, { NODES * 3 }, _>(|w| F::from_u64(group[w])))
            .reduce(|| [F::ZERO; NODES * 3], |a, b| std::array::from_fn(|w| a[w] + b[w]));
        Ok((0..degree)
            .map(|j| std::array::from_fn(|c| sums[3 * j + c].as_canonical_u64()))
            .collect())
    }

    /// Bind the lowest variable of every table and the eq table to `r`.
    pub fn gkr_fold(&self, layer: &mut MetalGkrLayer<'_>, r: Ext) -> Result<(), MetalError> {
        if layer.pairs == 0 {
            return Err(MetalError::Shape("a GKR fold needs a pair"));
        }
        // A table has twice the eq table's live entries.
        let shape = |count: usize| [count as u64, layer.stride as u64, r[0], r[1], r[2]];
        let (table_shape, eq_shape) = (shape(2 * layer.pairs), shape(layer.pairs));
        let command = self.command_buffer("nightstream.gkr.fold")?;
        let encoder = command.computeCommandEncoder().ok_or(MetalError::Encoder)?;
        let tables = layer.trees.iter().flat_map(|(_, parts)| parts);
        let folds = tables
            .map(|table| (table, &table_shape, 2 * layer.pairs))
            .chain([(&layer.eq, &eq_shape, layer.pairs)]);
        for (table, shape, count) in folds {
            self.launch(
                &encoder,
                &self.gkr_fold,
                &[Arg::Buffer(0, table), Arg::Words(1, shape)],
                Grid::Threads(count),
            );
        }
        encoder.endEncoding();
        self.finish(&command)?;
        layer.stride *= 2;
        layer.pairs /= 2;
        Ok(())
    }
}
