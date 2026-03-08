use neo_ccs::crypto::poseidon2_goldilocks::poseidon2_hash;
use p3_field::{Field, PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;

#[derive(Clone, Debug)]
pub struct TransferWitness {
    pub ram_pairs: Vec<(u64, u32)>,
    pub output_layout_words: Vec<(u64, u32)>,
}

type GlDigest = [Goldilocks; 4];

const ZERO_DIGEST: GlDigest = [Goldilocks::ZERO; 4];
const TAG_MT_NODE: u64 = 1;
const TAG_NOTE: u64 = 2;
const TAG_ADDR: u64 = 5;
const TAG_BL_BUCKET: u64 = 7;
const BL_DEPTH: u32 = 16;
const BL_BUCKET_SIZE: usize = 12;

fn gl(v: u64) -> Goldilocks {
    Goldilocks::from_u64(v)
}

fn h(input: &[Goldilocks]) -> GlDigest {
    poseidon2_hash(input)
}

fn gl_digest_to_bytes(d: &GlDigest) -> [u8; 32] {
    let mut out = [0u8; 32];
    for (i, elem) in d.iter().enumerate() {
        out[i * 8..(i + 1) * 8].copy_from_slice(&elem.as_canonical_u64().to_le_bytes());
    }
    out
}

struct RamWriter {
    addr: u64,
    pairs: Vec<(u64, u32)>,
}

impl RamWriter {
    fn new(start: u64) -> Self {
        Self {
            addr: start,
            pairs: Vec::new(),
        }
    }

    fn write_u32(&mut self, val: u32) {
        self.pairs.push((self.addr, val));
        self.addr += 4;
    }

    fn write_u64(&mut self, val: u64) {
        self.write_u32(val as u32);
        self.write_u32((val >> 32) as u32);
    }

    fn write_gl_digest(&mut self, d: &GlDigest) {
        let bytes = gl_digest_to_bytes(d);
        for i in 0..4 {
            let mut buf = [0u8; 8];
            buf.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            self.write_u64(u64::from_le_bytes(buf));
        }
    }
}

fn default_blacklist_root() -> (GlDigest, Vec<GlDigest>) {
    let empty_leaf = {
        let mut input = [Goldilocks::ZERO; 1 + BL_BUCKET_SIZE * 4];
        input[0] = gl(TAG_BL_BUCKET);
        h(&input)
    };

    let mut nodes = Vec::with_capacity(BL_DEPTH as usize + 1);
    nodes.push(empty_leaf);
    for level in 0..BL_DEPTH {
        let prev = nodes[level as usize];
        let mut input = [Goldilocks::ZERO; 10];
        input[0] = gl(TAG_MT_NODE);
        input[1] = gl(level as u64);
        input[2..6].copy_from_slice(&prev);
        input[6..10].copy_from_slice(&prev);
        nodes.push(h(&input));
    }
    (nodes[BL_DEPTH as usize], nodes)
}

fn compute_bucket_inv(id: &GlDigest, entries: &[[Goldilocks; 4]; BL_BUCKET_SIZE]) -> Goldilocks {
    let mut prod = Goldilocks::ONE;
    for entry in entries {
        for i in 0..4 {
            prod *= id[i] - entry[i];
        }
    }
    prod.inverse()
}

pub fn build_note_deposit_witness() -> TransferWitness {
    build_note_deposit_witness_with_addrs(0x104, 0x100)
}

pub fn build_note_deposit_witness_with_addrs(input_addr: u64, output_addr: u64) -> TransferWitness {
    let mut ram = RamWriter::new(input_addr);

    let domain = [gl(1), gl(1), gl(1), gl(1)];
    let value = 777u64;
    let rho = [gl(200), gl(201), gl(202), gl(203)];
    let pk_spend = [gl(42), gl(43), gl(44), gl(45)];
    let pk_ivk = [gl(100), gl(101), gl(102), gl(103)];

    let recipient = {
        let mut input = [Goldilocks::ZERO; 13];
        input[0] = gl(TAG_ADDR);
        input[1..5].copy_from_slice(&domain);
        input[5..9].copy_from_slice(&pk_spend);
        input[9..13].copy_from_slice(&pk_ivk);
        h(&input)
    };

    let cm_out = {
        let mut input = [Goldilocks::ZERO; 18];
        input[0] = gl(TAG_NOTE);
        input[1..5].copy_from_slice(&domain);
        input[5] = gl(value);
        input[6..10].copy_from_slice(&rho);
        input[10..14].copy_from_slice(&recipient);
        input[14..18].copy_from_slice(&recipient);
        h(&input)
    };

    let (blacklist_root, blacklist_nodes) = default_blacklist_root();
    let empty_bucket = [ZERO_DIGEST; BL_BUCKET_SIZE];
    let recipient_bucket_inv = compute_bucket_inv(&recipient, &empty_bucket);

    ram.write_gl_digest(&domain);
    ram.write_u64(value);
    ram.write_gl_digest(&rho);
    ram.write_gl_digest(&pk_spend);
    ram.write_gl_digest(&pk_ivk);
    ram.write_gl_digest(&cm_out);
    ram.write_gl_digest(&blacklist_root);
    for _ in 0..BL_BUCKET_SIZE {
        ram.write_gl_digest(&ZERO_DIGEST);
    }
    ram.write_u64(recipient_bucket_inv.as_canonical_u64());
    for node in blacklist_nodes.iter().take(BL_DEPTH as usize) {
        ram.write_gl_digest(node);
    }

    let mut out = RamWriter::new(output_addr);
    out.write_gl_digest(&domain);
    out.write_u64(value);
    out.write_gl_digest(&recipient);
    out.write_gl_digest(&cm_out);
    out.write_gl_digest(&blacklist_root);

    TransferWitness {
        ram_pairs: ram.pairs,
        output_layout_words: out.pairs,
    }
}
