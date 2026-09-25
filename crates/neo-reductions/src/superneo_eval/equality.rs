use neo_math::K;
use p3_field::PrimeCharacteristicRing;

/// Boolean equality weights stored as two tensor factors, in low-bit order.
pub struct EqualityWeights {
    low: Vec<K>,
    high: Vec<K>,
    low_bits: usize,
}

impl EqualityWeights {
    pub fn new(point: &[K]) -> Self {
        fn table(point: &[K]) -> Vec<K> {
            let mut values = vec![K::ONE];
            for &coordinate in point {
                let width = values.len();
                values.resize(width * 2, K::ZERO);
                for index in 0..width {
                    let value = values[index];
                    values[index] = value * (K::ONE - coordinate);
                    values[index + width] = value * coordinate;
                }
            }
            values
        }
        // This split minimizes the combined size of the two stored factors.
        let low_bits = point.len() / 2;
        Self {
            low: table(&point[..low_bits]),
            high: table(&point[low_bits..]),
            low_bits,
        }
    }

    pub fn at(&self, index: usize) -> K {
        self.low[index & (self.low.len() - 1)] * self.high[index >> self.low_bits]
    }
}
