#[derive(Clone, Debug)]
pub struct Event {
    pub op: &'static str,
    pub label: &'static [u8],
    pub len: usize,
    pub st_prefix: [u64; 4],
}

impl Event {
    pub fn new(op: &'static str, label: &'static [u8], len: usize, st: &[p3_goldilocks::Goldilocks; neo_ccs::crypto::poseidon2_goldilocks::WIDTH]) -> Self {
        let mut st_prefix = [0u64; 4];
        for i in 0..4 { st_prefix[i] = st[i].as_canonical_u64(); }
        Self { op, label, len, st_prefix }
    }
}

