use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

const FOLD_DOMAIN: &[u8] = b"Nightstream/SuperNeo/fold/v2";

pub fn fold_domain_chunk_v1_2() -> [F; p2::RATE] {
    domain_chunk_v1_2(FOLD_DOMAIN)
}

/// One rate chunk that names a protocol: `label`, eight little-endian bytes
/// per word, then zero words. The label fits in one chunk.
pub fn domain_chunk_v1_2(label: &[u8]) -> [F; p2::RATE] {
    assert!(label.len() <= 8 * p2::RATE, "domain label exceeds one rate chunk");
    std::array::from_fn(|word| {
        let bytes = label.get(8 * word..).unwrap_or_default();
        let mut packed = [0u8; 8];
        let length = bytes.len().min(8);
        packed[..length].copy_from_slice(&bytes[..length]);
        F::from_u64(u64::from_le_bytes(packed))
    })
}

pub struct FoldCoin {
    pub value: [F; 2],
    pub absorbed_zero_chunk: bool,
}

#[derive(Clone)]
pub struct Poseidon2Transcript {
    st: [Goldilocks; p2::WIDTH],
    absorbed: usize,
    coins_read: usize,
    perm: &'static Poseidon2Goldilocks<{ p2::WIDTH }>,
}

impl Poseidon2Transcript {
    pub fn new_v1_2() -> Self {
        Self::empty()
    }

    pub fn reset_v1_2(&mut self) {
        self.st = [Goldilocks::ZERO; p2::WIDTH];
        self.absorbed = 0;
        self.coins_read = 0;
    }

    pub fn absorb_v1_2(&mut self, fields: &[F]) {
        assert_eq!(self.absorbed, 0, "v1_2 transcript cannot inherit an absorb cursor");
        for chunk in fields.chunks(p2::RATE) {
            for (lane, &value) in chunk.iter().enumerate() {
                self.st[lane] += value;
            }
            self.permute();
        }
    }

    /// Coin `n` of one state is rate lanes `2n` and `2n + 1`; after the sixth, one zero chunk is
    /// absorbed. This is the Lean `squeezeAt` schedule.
    pub fn read_coin_v1_2(&mut self) -> FoldCoin {
        assert_eq!(self.absorbed, 0, "v1_2 transcript cannot inherit an absorb cursor");
        let pair = self.coins_read;
        let value = [self.st[2 * pair], self.st[2 * pair + 1]];
        self.coins_read += 1;
        let absorbed_zero_chunk = self.coins_read == p2::RATE / 2;
        if absorbed_zero_chunk {
            self.absorb_v1_2(&[F::ZERO; p2::RATE]);
        }
        FoldCoin {
            value,
            absorbed_zero_chunk,
        }
    }

    pub fn squeeze_digest_v1_2(&mut self) -> [F; p2::DIGEST_LEN] {
        assert_eq!(self.absorbed, 0, "v1_2 transcript cannot inherit an absorb cursor");
        let digest = std::array::from_fn(|lane| F::from_u64(self.st[lane].as_canonical_u64()));
        self.permute();
        digest
    }

    pub fn state_prefix_v1_2(&self) -> [u8; 32] {
        let mut output = [0u8; 32];
        for lane in 0..4 {
            output[lane * 8..(lane + 1) * 8].copy_from_slice(&self.st[lane].as_canonical_u64().to_le_bytes());
        }
        output
    }

    #[inline]
    fn empty() -> Self {
        Self {
            st: [Goldilocks::ZERO; p2::WIDTH],
            absorbed: 0,
            coins_read: 0,
            perm: p2::permutation(),
        }
    }

    #[inline]
    fn permute(&mut self) {
        self.st = self.perm.permute(self.st);
        self.absorbed = 0;
        self.coins_read = 0;
    }

    pub fn state(&self) -> [Goldilocks; p2::WIDTH] {
        self.st
    }

    pub fn absorbed(&self) -> usize {
        self.absorbed
    }

    pub fn from_state_and_absorbed(state: [Goldilocks; p2::WIDTH], absorbed: usize) -> Self {
        assert!(absorbed <= p2::RATE, "transcript absorbed cursor out of range");
        Self {
            st: state,
            absorbed,
            coins_read: 0,
            perm: p2::permutation(),
        }
    }
}
