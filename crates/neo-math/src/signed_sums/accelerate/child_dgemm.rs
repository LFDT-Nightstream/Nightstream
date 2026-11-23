use super::{cblas_dgemm, D, F, MAX_BATCH_PLANES};
use p3_field::PrimeCharacteristicRing;

const QUARTER: usize = D / 4;
const LIMBS: usize = 2 * QUARTER;
const TILE: usize = 128;
const GROUP: usize = 1 << 13;
const TERMS: usize = 7;
const _: () = assert!(GROUP as u128 * 225 * (1u128 << 32) <= (1u128 << f64::MANTISSA_DIGITS));

pub(super) fn workspace(planes: usize) -> usize {
    let n = planes * QUARTER;
    (TILE * (LIMBS + n) + LIMBS * n) * TERMS * size_of::<f64>() + super::CHILD_STACK_AND_PAGE_SLACK
}

struct Panels {
    key: Vec<f64>,
    digits: Vec<f64>,
    products: Vec<f64>,
    n: usize,
}

#[inline(always)]
fn eval([a, b, c, d]: [f64; 4]) -> [f64; TERMS] {
    let ev = a + c;
    let od = b + d;
    let ev2 = a + 4.0 * c;
    let od2 = 2.0 * b + 8.0 * d;
    [
        a,
        d,
        ev + od,
        ev - od,
        ev2 + od2,
        ev2 - od2,
        8.0 * a + 4.0 * b + 2.0 * c + d,
    ]
}

fn interpolate([zero, inf, pos, neg, pos2, neg2, half]: [i128; TERMS]) -> [i128; TERMS] {
    let e1 = (pos + neg) / 2 - zero - inf;
    let e2 = ((pos2 + neg2) / 2 - zero - 64 * inf) / 4;
    let c4 = (e2 - e1) / 3;
    let c2 = e1 - c4;
    let sum = (pos - neg) / 2;
    let low = ((pos2 - neg2) / 4 - sum) / 3;
    let high = ((half - 64 * zero - 16 * c2 - 4 * c4 - inf) / 2 - sum) / 3;
    let diff = (low - high) / 5;
    let c1 = (low - sum - 4 * diff) / 3;
    let c5 = c1 + diff;
    [zero, c1, c2, sum - c1 - c5, c4, c5, inf]
}

impl Panels {
    fn new(planes: usize) -> Self {
        let n = QUARTER * planes;
        Self {
            key: vec![0.0; TERMS * TILE * LIMBS],
            digits: vec![0.0; TERMS * TILE * n],
            products: vec![0.0; TERMS * LIMBS * n],
            n,
        }
    }

    fn pack(&mut self, cols: &[u32], keys: &[u64], planes: &[(&[u64], &[u64])]) {
        let [k0, k1, k2, k3, k4, k5, k6] = self
            .key
            .as_mut_slice()
            .as_chunks_mut::<{ TILE * LIMBS }>()
            .0
        else {
            unreachable!()
        };
        let ds: [&mut [f64]; TERMS] = {
            let mut iter = self.digits.chunks_exact_mut(TILE * self.n);
            std::array::from_fn(|_| iter.next().unwrap())
        };
        let [d0, d1, d2, d3, d4, d5, d6] = ds;
        for (r, &col) in cols.iter().enumerate() {
            let c = col as usize;
            let k = &keys[c * D..][..D];
            for lane in 0..QUARTER {
                let w: [u64; 4] = std::array::from_fn(|i| k[QUARTER * i + lane]);
                let lo = eval(w.map(|v| f64::from(v as u32)));
                let hi = eval(w.map(|v| f64::from((v >> 32) as u32)));
                let off = r * LIMBS + lane;
                [k0[off], k1[off], k2[off], k3[off], k4[off], k5[off], k6[off]] = lo;
                let off = off + QUARTER;
                [k0[off], k1[off], k2[off], k3[off], k4[off], k5[off], k6[off]] = hi;
            }
            for (p, (positive, negative)) in planes.iter().enumerate() {
                let (pos, neg) = (positive[c], negative[c]);
                for lane in 0..QUARTER {
                    let a = eval(std::array::from_fn(|q| {
                        let bit = QUARTER * q + lane;
                        f64::from(((pos >> bit) & 1) as i32 - ((neg >> bit) & 1) as i32)
                    }));
                    let off = r * self.n + p * QUARTER + lane;
                    [d0[off], d1[off], d2[off], d3[off], d4[off], d5[off], d6[off]] = a;
                }
            }
        }
    }

    fn multiply(&mut self, count: usize, accumulate: bool) {
        for ((key, digits), out) in self
            .key
            .chunks_exact(TILE * LIMBS)
            .zip(self.digits.chunks_exact(TILE * self.n))
            .zip(self.products.chunks_exact_mut(LIMBS * self.n))
        {
            // SAFETY: The two TILE-row panels and the disjoint LIMBS-by-n output have the indicated row strides.
            #[allow(unsafe_code)]
            unsafe {
                cblas_dgemm(
                    101,
                    112,
                    111,
                    LIMBS as i32,
                    self.n as i32,
                    count as i32,
                    1.0,
                    key.as_ptr(),
                    LIMBS as i32,
                    digits.as_ptr(),
                    self.n as i32,
                    f64::from(u8::from(accumulate)),
                    out.as_mut_ptr(),
                    self.n as i32,
                );
            }
        }
    }

    fn finish(&self, outputs: &mut [[F; D]]) {
        for (plane, out) in outputs.iter_mut().enumerate() {
            let mut points = [[0i128; 2 * QUARTER - 1]; TERMS];
            for (s, part) in points
                .iter_mut()
                .zip(self.products.chunks_exact(LIMBS * self.n))
            {
                for k in 0..QUARTER {
                    for lane in 0..QUARTER {
                        let base = k * self.n + QUARTER * plane + lane;
                        s[k + lane] +=
                            i128::from(part[base] as i64) + (i128::from(part[QUARTER * self.n + base] as i64) << 32);
                    }
                }
            }
            let mut prod = [0i128; 2 * D - 1];
            for j in 0..2 * QUARTER - 1 {
                let parts = interpolate(points.each_ref().map(|s| s[j]));
                for (k, c) in parts.into_iter().enumerate() {
                    prod[k * QUARTER + j] += c;
                }
            }
            for (i, v) in out.iter_mut().enumerate() {
                *v += F::from_i128(prod[i] - prod.get(i + D).copied().unwrap_or(0));
            }
        }
    }
}

pub(super) fn compute(keys: &[u64], columns: &[u32], planes: &[(&[u64], &[u64])], out: &mut [[F; D]]) -> bool {
    if !(1..=MAX_BATCH_PLANES).contains(&planes.len())
        || planes.len() != out.len()
        || !keys.len().is_multiple_of(D)
        || planes
            .iter()
            .any(|(p, n)| p.len() != keys.len() / D || n.len() != p.len())
        || columns.iter().any(|&c| c as usize >= keys.len() / D)
    {
        return false;
    }
    out.fill([F::ZERO; D]);
    let mut panels = Panels::new(planes.len());
    for group in columns.chunks(GROUP) {
        for (i, chunk) in group.chunks(TILE).enumerate() {
            panels.pack(chunk, keys, planes);
            panels.multiply(chunk.len(), i != 0);
        }
        panels.finish(out);
    }
    true
}

#[cfg(test)]
#[path = "../../../tests/unit/child_dgemm.rs"]
mod tests;
