use super::{DenseBlocks, FactoredSlots, RealBlockStorage, SuperneoZBlocks};
use neo_math::{Rq, D, F, K};
use p3_field::{Field, PrimeCharacteristicRing};

pub enum OpeningWitness<'a> {
    Signed {
        positive: &'a [u64],
        negative: &'a [u64],
    },
    Dense(&'a [Rq]),
    Centered(&'a [[i8; D]]),
}

impl<'a> OpeningWitness<'a> {
    pub(super) fn from_blocks(w: &'a SuperneoZBlocks, blocks: usize) -> Option<Self> {
        if !w.imag_all_zero || w.block_len() != blocks {
            return None;
        }
        match &w.re {
            RealBlockStorage::SignedUnit { positive, negative } => Some(Self::Signed { positive, negative }),
            RealBlockStorage::Dense { blocks, .. } => match blocks {
                DenseBlocks::Wide(rows) => Some(Self::Dense(rows)),
                DenseBlocks::Centered(rows) => Some(Self::Centered(rows)),
            },
            RealBlockStorage::Zero { .. } => None,
        }
    }
}

#[non_exhaustive]
pub struct OpeningMatrix<'a> {
    pub layout: FactoredSlots<'a>,
    pub slots: &'a [u32],
    pub offsets: &'a [u32],
    pub rows: &'a [u8],
    pub row_bytes: usize,
    pub row_count: usize,
    pub tags: &'a [u16],
    pub factors: &'a [F],
    pub tiles: &'a OpeningTiles,
}

pub trait TerminalOpeningDevice {
    fn supports(&self, blocks: usize, witnesses: usize) -> bool;
    fn evaluate(
        &self,
        matrices: &[OpeningMatrix<'_>],
        weights: &[K],
        witnesses: &[OpeningWitness<'_>],
        workspace_bytes: usize,
        result: &mut [[K; D]],
    ) -> bool;
}

pub struct OpeningTiles {
    pub first: Box<[u32]>,
    pub tasks: Box<[[u32; 2]]>,
    pub heavy: Box<[[u32; 4]]>,
    pub powers: [F; 257],
}

impl OpeningTiles {
    pub(super) fn sizes(blocks: usize, offsets: &[u32]) -> Option<(usize, usize, usize)> {
        let mut heavy_count = 0usize;
        let mut task_count = 0usize;
        for span in offsets.windows(2) {
            let entries = span[1].checked_sub(span[0])? as usize;
            if entries > 128 {
                heavy_count += 1;
                task_count += entries.div_ceil(256);
            }
        }
        let bytes = size_of::<Self>()
            .checked_add(blocks.checked_add(1)?.checked_mul(4)?)?
            .checked_add(heavy_count.checked_mul(16)?)?
            .checked_add(task_count.checked_mul(8)?)?;
        Some((bytes, heavy_count, task_count))
    }

    pub(super) fn prepare(
        layout: FactoredSlots<'_>,
        blocks: usize,
        slots: &[u32],
        offsets: &[u32],
        heavy_count: usize,
        task_count: usize,
    ) -> Option<Box<Self>> {
        if layout.ratio == F::ZERO || blocks > (u32::MAX as usize) / D || slots.is_empty() {
            return None;
        }
        let mut powers = [F::ONE; 257];
        for i in 1..256 {
            powers[i] = powers[i - 1] * layout.ratio;
        }
        powers[256] = (F::ONE + powers[D]).try_inverse()?;
        let mut first = vec![0u32; blocks + 1];
        let mut previous_end = 0;
        for &slot in slots {
            let i = slot as usize;
            let start = *layout.starts.get(i)?;
            let width = u32::from(*layout.widths.get(i)?);
            if width == 0 || start < previous_end {
                return None;
            }
            previous_end = start.checked_add(width)?;
        }
        if previous_end as usize > blocks * D {
            return None;
        }
        let mut at = 0;
        for (block, cursor) in first.iter_mut().enumerate() {
            while let Some(&slot) = slots.get(at) {
                let s = slot as usize;
                if layout.starts[s] as usize + usize::from(layout.widths[s]) > block * D {
                    break;
                }
                at += 1;
            }
            *cursor = u32::try_from(at).ok()?;
        }
        let mut tasks = Vec::with_capacity(task_count);
        let mut heavy = Vec::with_capacity(heavy_count);
        for (slot, span) in offsets.windows(2).enumerate() {
            if span[1] - span[0] <= 128 {
                continue;
            }
            let start = tasks.len() as u32;
            for from in (span[0]..span[1]).step_by(256) {
                tasks.push([from, from.saturating_add(256).min(span[1])]);
            }
            heavy.push([slot as u32, start, tasks.len() as u32, 0]);
        }
        Some(Box::new(Self {
            first: first.into_boxed_slice(),
            tasks: tasks.into_boxed_slice(),
            heavy: heavy.into_boxed_slice(),
            powers,
        }))
    }

    pub(super) fn bytes(&self) -> usize {
        size_of::<Self>() + size_of_val(&*self.first) + size_of_val(&*self.tasks) + size_of_val(&*self.heavy)
    }
}
