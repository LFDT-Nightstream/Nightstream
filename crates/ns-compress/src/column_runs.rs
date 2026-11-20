use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

#[derive(Clone, Copy)]
pub(crate) struct Run {
    end: u32,
    slot: u32,
    width: u8,
}

pub(crate) struct Runs {
    runs: Vec<Run>,
    widths: [bool; 256],
}

impl Runs {
    pub(crate) fn new(starts: &[u32], widths: &[u8]) -> Option<Self> {
        let mut runs: Vec<Run> = Vec::new();
        let mut seen = [false; 256];
        let mut end = 0;
        let mut old = 0;
        for (slot, (&start, &w)) in starts.iter().zip(widths).enumerate() {
            if w != 0 {
                if start != end {
                    return None;
                }
                end = start.checked_add(u32::from(w))?;
                if w != old {
                    if runs.len() > starts.len() / 8 + 1 {
                        return None;
                    }
                    seen[w as usize] = true;
                    runs.push(Run {
                        end,
                        slot: u32::try_from(slot).ok()?,
                        width: w,
                    });
                } else {
                    runs.last_mut().unwrap().end = end;
                }
            }
            old = w;
        }
        Some(Self { runs, widths: seen })
    }
    fn slices(&self, first: usize, end: usize, mut emit: impl FnMut(usize, Run)) {
        let start = self.runs.partition_point(|r| r.end as usize <= first);
        let mut col = start
            .checked_sub(1)
            .map_or(0, |i| self.runs[i].end as usize);
        for &run in &self.runs[start..] {
            if col >= end {
                break;
            }
            emit(col, run);
            col = run.end as usize;
        }
    }
    pub(crate) fn write(&self, weights: &[K], first: usize, out: &mut [K], powers: &[F; 256]) {
        let end = first + out.len();
        let mut written = 0;
        self.slices(first, end, |col, r| {
            let w = usize::from(r.width);
            let edge = (r.end as usize).min(end);
            let from = col.max(first);
            let mut src = r.slot as usize + (from - col) / w;
            let mut dest = &mut out[from - first..edge - first];
            written = edge - first;
            if w == 1 {
                dest.copy_from_slice(&weights[src..src + dest.len()]);
                return;
            }
            let shift = (from - col) % w;
            if shift > 0 {
                let n = dest.len().min(w - shift);
                let (head, rest) = dest.split_at_mut(n);
                for (val, &p) in head.iter_mut().zip(&powers[shift..]) {
                    *val = weights[src].scale_base(p);
                }
                dest = rest;
                src += 1;
            }
            let mut chunks = dest.chunks_exact_mut(w);
            for (out, &weight) in (&mut chunks).zip(&weights[src..]) {
                for (val, &p) in out.iter_mut().zip(powers) {
                    *val = weight.scale_base(p);
                }
                src += 1;
            }
            let tail = chunks.into_remainder();
            if !tail.is_empty() {
                for (val, &p) in tail.iter_mut().zip(powers) {
                    *val = weights[src].scale_base(p);
                }
            }
        });
        out[written..].fill(K::ZERO);
    }
    pub(crate) fn windows(&self, form: &[K; D], powers: &[F; 256]) -> Windows {
        let mut index = [0; 256];
        let mut weights = Vec::new();
        for (w, &used) in self.widths.iter().enumerate() {
            if !used {
                continue;
            }
            index[w] = weights.len();
            for lane in 0..D {
                let mut slots = [K::ZERO; 5];
                for (j, &p) in powers[..w].iter().enumerate() {
                    slots[(j + lane) / D] += form[(j + lane) % D].scale_base(p);
                }
                weights.push(slots);
            }
        }
        Windows {
            index,
            weights,
            form: *form,
        }
    }
    pub(crate) fn project(&self, weights: &[K], first: usize, out: &mut [K], table: &Windows) {
        let end = first + D * out.len();
        out.fill(K::ZERO);
        self.slices(first, end, |col, r| {
            let w = usize::from(r.width);
            let edge = (r.end as usize).min(end);
            let from = col.max(first);
            let lo = (from - col) / w;
            let hi = (edge - col).div_ceil(w);
            let mut start = col + lo * w;
            let mut src = &weights[r.slot as usize + lo..r.slot as usize + hi];
            if w == 1 {
                let mut lane = start % D;
                for dest in &mut out[(start - first) / D..] {
                    let size = (D - lane).min(src.len());
                    let sum = src[..size]
                        .iter()
                        .zip(&table.form[lane..])
                        .fold(neo_math::ProductSum::default(), |sum, (&v, &f)| sum.add(v, f));
                    *dest += sum.value();
                    src = &src[size..];
                    if src.is_empty() {
                        break;
                    }
                    lane = 0;
                }
                return;
            }
            if w <= D && start >= first {
                let block_weights = &table.weights[table.index[w]..][..D];
                for &weight in src {
                    let lane = start % D;
                    let block = (start - first) / D;
                    let terms = block_weights[lane];
                    out[block] += weight * terms[0];
                    if lane + w > D && block + 1 < out.len() {
                        out[block + 1] += weight * terms[1];
                    }
                    start += w;
                }
                return;
            }
            for &weight in src {
                let lane = start % D;
                let block = start / D;
                let terms = &table.weights[table.index[w] + lane];
                let skip = (first / D).saturating_sub(block);
                let n = (lane + w).div_ceil(D).min(end / D - block);
                for (dest, &window) in out[block + skip - first / D..]
                    .iter_mut()
                    .zip(&terms[skip..n])
                {
                    *dest += weight * window;
                }
                start += w;
            }
        });
    }
}
pub(crate) struct Windows {
    index: [usize; 256],
    weights: Vec<[K; 5]>,
    form: [K; D],
}
