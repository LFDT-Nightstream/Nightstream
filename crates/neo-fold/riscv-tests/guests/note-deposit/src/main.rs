#![no_std]
#![no_main]

type GlDigest = [u64; 4];

const ZERO_DIGEST: GlDigest = [0, 0, 0, 0];
const TAG_MT_NODE: u64 = 1;
const TAG_NOTE: u64 = 2;
const TAG_ADDR: u64 = 5;
const TAG_BL_BUCKET: u64 = 7;
const BL_DEPTH: u32 = 16;
const BL_BUCKET_SIZE: usize = 12;
const INPUT_ADDR: u32 = 0x104;
const OUTPUT_ADDR: u32 = 0x100;

#[inline]
fn digest_eq(a: &GlDigest, b: &GlDigest) -> bool {
    a[0] == b[0] && a[1] == b[1] && a[2] == b[2] && a[3] == b[3]
}

#[inline]
fn gl_add(a: u64, b: u64) -> u64 {
    a.wrapping_add(b)
}

#[inline]
fn gl_sub(a: u64, b: u64) -> u64 {
    a.wrapping_sub(b)
}

#[inline]
fn gl_mul(a: u64, b: u64) -> u64 {
    a.wrapping_mul(b)
}

#[inline]
fn poseidon2_hash(input: &[u64]) -> GlDigest {
    nightstream_sdk::poseidon2::poseidon2_hash(input)
}

struct RamReader {
    addr: u32,
}

impl RamReader {
    fn new(addr: u32) -> Self {
        Self { addr }
    }

    fn read_u32(&mut self) -> u32 {
        let val = unsafe { core::ptr::read_volatile(self.addr as *const u32) };
        self.addr += 4;
        val
    }

    fn read_u64(&mut self) -> u64 {
        let lo = self.read_u32() as u64;
        let hi = self.read_u32() as u64;
        lo | (hi << 32)
    }

    fn read_digest(&mut self) -> GlDigest {
        [self.read_u64(), self.read_u64(), self.read_u64(), self.read_u64()]
    }
}

struct RamWriter {
    addr: u32,
}

impl RamWriter {
    fn new(addr: u32) -> Self {
        Self { addr }
    }

    fn write_u32(&mut self, val: u32) {
        unsafe { core::ptr::write_volatile(self.addr as *mut u32, val) };
        self.addr += 4;
    }

    fn write_u64(&mut self, val: u64) {
        self.write_u32(val as u32);
        self.write_u32((val >> 32) as u32);
    }

    fn write_digest(&mut self, digest: &GlDigest) {
        for &word in digest {
            self.write_u64(word);
        }
    }
}

fn derive_address(domain: &GlDigest, pk_spend: &GlDigest, pk_ivk: &GlDigest) -> GlDigest {
    let mut input = [0u64; 13];
    input[0] = TAG_ADDR;
    input[1..5].copy_from_slice(domain);
    input[5..9].copy_from_slice(pk_spend);
    input[9..13].copy_from_slice(pk_ivk);
    poseidon2_hash(&input)
}

fn note_commitment(
    domain: &GlDigest,
    value: u64,
    rho: &GlDigest,
    recipient: &GlDigest,
    sender_id: &GlDigest,
) -> GlDigest {
    let mut input = [0u64; 18];
    input[0] = TAG_NOTE;
    input[1..5].copy_from_slice(domain);
    input[5] = value;
    input[6..10].copy_from_slice(rho);
    input[10..14].copy_from_slice(recipient);
    input[14..18].copy_from_slice(sender_id);
    poseidon2_hash(&input)
}

fn mt_node(level: u64, left: &GlDigest, right: &GlDigest) -> GlDigest {
    let mut input = [0u64; 10];
    input[0] = TAG_MT_NODE;
    input[1] = level;
    input[2..6].copy_from_slice(left);
    input[6..10].copy_from_slice(right);
    poseidon2_hash(&input)
}

fn bl_bucket_leaf(entries: &[GlDigest; BL_BUCKET_SIZE]) -> GlDigest {
    let mut input = [0u64; 1 + BL_BUCKET_SIZE * 4];
    input[0] = TAG_BL_BUCKET;
    let mut i = 0usize;
    while i < BL_BUCKET_SIZE {
        let off = 1 + i * 4;
        input[off..off + 4].copy_from_slice(&entries[i]);
        i += 1;
    }
    poseidon2_hash(&input)
}

fn merkle_root(leaf: &GlDigest, pos: u32, reader: &mut RamReader, depth: u32) -> GlDigest {
    let mut cur = *leaf;
    let mut p = pos;
    let mut level = 0u32;
    while level < depth {
        let sibling = reader.read_digest();
        let bit = (p & 1) as u64;
        let mut left = [0u64; 4];
        let mut right = [0u64; 4];
        let mut i = 0usize;
        while i < 4 {
            let delta = gl_mul(bit, gl_sub(sibling[i], cur[i]));
            left[i] = gl_add(cur[i], delta);
            right[i] = gl_sub(sibling[i], delta);
            i += 1;
        }
        cur = mt_node(level as u64, &left, &right);
        p >>= 1;
        level += 1;
    }
    cur
}

fn bl_bucket_pos(id: &GlDigest) -> u32 {
    (id[0] as u32) & ((1u32 << BL_DEPTH) - 1)
}

fn assert_not_blacklisted(id: &GlDigest, blacklist_root: &GlDigest, reader: &mut RamReader) {
    let mut entries = [ZERO_DIGEST; BL_BUCKET_SIZE];
    for entry in entries.iter_mut() {
        *entry = reader.read_digest();
    }
    let bucket_inv = reader.read_u64();

    let mut prod = 1u64;
    for entry in &entries {
        let mut i = 0usize;
        while i < 4 {
            prod = gl_mul(prod, gl_sub(id[i], entry[i]));
            i += 1;
        }
    }
    assert!(gl_mul(prod, bucket_inv) == 1);

    let leaf = bl_bucket_leaf(&entries);
    let root = merkle_root(&leaf, bl_bucket_pos(id), reader, BL_DEPTH);
    assert!(digest_eq(&root, blacklist_root));
}

#[nightstream_sdk::provable]
fn note_deposit() -> ! {
    let mut r = RamReader::new(INPUT_ADDR);
    let mut w = RamWriter::new(OUTPUT_ADDR);

    let domain = r.read_digest();
    let value = r.read_u64();
    let rho = r.read_digest();
    let pk_spend = r.read_digest();
    let pk_ivk = r.read_digest();
    let cm_out_pub = r.read_digest();
    let blacklist_root = r.read_digest();

    let recipient = derive_address(&domain, &pk_spend, &pk_ivk);
    let cm_out = note_commitment(&domain, value, &rho, &recipient, &recipient);
    assert!(digest_eq(&cm_out, &cm_out_pub));
    assert_not_blacklisted(&recipient, &blacklist_root, &mut r);

    w.write_digest(&domain);
    w.write_u64(value);
    w.write_digest(&recipient);
    w.write_digest(&cm_out);
    w.write_digest(&blacklist_root);

    nightstream_sdk::halt();
}
