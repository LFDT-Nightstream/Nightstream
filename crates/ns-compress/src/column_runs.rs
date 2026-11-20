use neo_math::{KExtensions, D, F, K};
use p3_field::PrimeCharacteristicRing;

use neo_reductions::superneo_eval::SlotRuns;

pub(crate) fn write(runs: &SlotRuns, weights: &[K], first: usize, out: &mut [K], powers: &[F; 256]) {
    let end = first + out.len();
    let mut written = 0;
    runs.slices(first, end, |col, r| {
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
pub(crate) fn windows(runs: &SlotRuns, form: &[K; D], powers: &[F; 256]) -> Windows {
    let mut index = [0; 256];
    let mut weights = Vec::new();
    for (w, &used) in runs.widths.iter().enumerate() {
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
pub(crate) fn project(runs: &SlotRuns, weights: &[K], first: usize, out: &mut [K], table: &Windows) {
    let end = first + D * out.len();
    out.fill(K::ZERO);
    runs.slices(first, end, |col, r| {
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
pub(crate) struct Windows {
    index: [usize; 256],
    weights: Vec<[K; 5]>,
    form: [K; D],
}

#[cfg(test)]
#[path = "../tests/unit/column_runs.rs"]
mod tests;
