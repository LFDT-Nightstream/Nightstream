use crate::{Transcript, TranscriptProtocol};
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w8-r4";
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

const OP_APPEND_MESSAGE: u64 = 1;
const OP_APPEND_FIELDS: u64 = 2;
const QUERY_FIELDS: u64 = 0x101;
const QUERY_BYTES: u64 = 0x102;
const QUERY_NONZERO_FIELD: u64 = 0x103;
const QUERY_DIGEST32: u64 = 0x104;

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
    #[cfg(feature = "debug-log")]
    log: Vec<crate::debug::Event>,
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
            #[cfg(feature = "debug-log")]
            log: Vec::new(),
        }
    }

    #[inline]
    fn absorb_elem(&mut self, x: Goldilocks) {
        if self.absorbed >= p2::RATE {
            self.permute();
        }
        self.st[self.absorbed] = x;
        self.absorbed += 1;
    }

    #[inline]
    fn absorb_slice(&mut self, inputs: &[F]) {
        let mut src_idx = 0;
        let len = inputs.len();

        while self.absorbed < p2::RATE && src_idx < len {
            self.st[self.absorbed] = inputs[src_idx];
            self.absorbed += 1;
            src_idx += 1;
        }

        if self.absorbed == p2::RATE {
            self.permute();
        }

        while len - src_idx >= p2::RATE {
            self.st[..p2::RATE].copy_from_slice(&inputs[src_idx..src_idx + p2::RATE]);

            self.permute();
            src_idx += p2::RATE;
        }

        while src_idx < len {
            self.st[self.absorbed] = inputs[src_idx];
            self.absorbed += 1;
            src_idx += 1;
        }
    }

    #[inline]
    fn absorb_packed_bytes_with_len(&mut self, bytes: &[u8]) {
        self.absorb_elem(Goldilocks::from_u64(bytes.len() as u64));

        const BYTES_PER_LIMB: usize = 7;
        const LIMB_CHUNK: usize = 64;
        let mut packed = [Goldilocks::ZERO; LIMB_CHUNK];
        let mut used = 0usize;
        let mut i = 0usize;
        let len = bytes.len();

        while i + BYTES_PER_LIMB <= len {
            let mut limb = [0u8; 8];
            limb[..BYTES_PER_LIMB].copy_from_slice(&bytes[i..i + BYTES_PER_LIMB]);
            packed[used] = Goldilocks::from_u64(u64::from_le_bytes(limb));
            used += 1;
            i += BYTES_PER_LIMB;

            if used == LIMB_CHUNK {
                self.absorb_slice(&packed);
                used = 0;
            }
        }

        if i < len {
            let mut limb = [0u8; 8];
            limb[..(len - i)].copy_from_slice(&bytes[i..]);
            packed[used] = Goldilocks::from_u64(u64::from_le_bytes(limb));
            used += 1;
        }

        if used > 0 {
            self.absorb_slice(&packed[..used]);
        }
    }

    #[inline]
    fn permute(&mut self) {
        self.st = self.perm.permute(self.st);
        self.absorbed = 0;
        self.coins_read = 0;
    }

    #[inline]
    fn bind_query(&mut self, query: u64, output_len: usize) {
        self.absorb_elem(Goldilocks::from_u64(query));
        self.absorb_elem(Goldilocks::from_u64(output_len as u64));
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
            #[cfg(feature = "debug-log")]
            log: Vec::new(),
        }
    }

    pub fn new_raw_fields(fields: &[F]) -> Self {
        let mut tr = Self::empty();
        tr.append_fields_raw(fields);
        tr
    }
}

impl Transcript for Poseidon2Transcript {
    fn new(app_label: &'static [u8]) -> Self {
        let mut tr = Self::empty();
        tr.append_message(APP_DOMAIN, app_label);
        tr
    }

    fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        self.absorb_elem(Goldilocks::from_u64(OP_APPEND_MESSAGE));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_packed_bytes_with_len(msg);
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_message", label, msg.len(), &self.st));
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("append_message", label, msg.len(), &self.st));
    }

    fn append_fields(&mut self, label: &'static [u8], fs: &[F]) {
        self.absorb_elem(Goldilocks::from_u64(OP_APPEND_FIELDS));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_elem(Goldilocks::from_u64(fs.len() as u64));
        self.absorb_slice(fs);
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_fields", label, fs.len(), &self.st));
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("append_fields", label, fs.len(), &self.st));
    }

    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]) {
        self.append_message(b"chal/label", label);
        let mut produced = 0usize;
        while produced < out.len() {
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            for i in 0..4 {
                let limb = self.st[i].as_canonical_u64().to_le_bytes();
                let take = (out.len() - produced).min(8);
                out[produced..produced + take].copy_from_slice(&limb[..take]);
                produced += take;
                if produced >= out.len() {
                    break;
                }
            }
        }
        self.bind_query(QUERY_BYTES, out.len());
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_bytes");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_bytes", label, out.len(), &self.st));
    }

    fn challenge_field(&mut self, label: &'static [u8]) -> F {
        self.append_message(b"chal/label", label);
        self.absorb_elem(Goldilocks::ONE);
        self.permute();
        let out = F::from_u64(self.st[0].as_canonical_u64());
        self.bind_query(QUERY_FIELDS, 1);
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_field");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_field", label, 1, &self.st));
        out
    }

    fn challenge_fields(&mut self, label: &'static [u8], n: usize) -> Vec<F> {
        self.append_message(b"chal/label", label);
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            for i in 0..p2::DIGEST_LEN.min(n - out.len()) {
                out.push(F::from_u64(self.st[i].as_canonical_u64()));
            }
        }
        self.bind_query(QUERY_FIELDS, n);
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_fields");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_fields", label, n, &self.st));
        out
    }

    fn fork(&self, scope: &'static [u8]) -> Self {
        let mut child = self.clone();
        child.append_message(b"fork", scope);
        child
    }

    fn digest32(&mut self) -> [u8; 32] {
        self.absorb_elem(Goldilocks::ONE);
        self.permute();
        let mut out = [0u8; 32];
        for i in 0..4 {
            out[i * 8..(i + 1) * 8].copy_from_slice(&self.st[i].as_canonical_u64().to_le_bytes());
        }
        self.bind_query(QUERY_DIGEST32, out.len());
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("digest32", b"", 0, &self.st));
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("digest32");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("digest32", b"", 0, &self.st));
        out
    }
}

impl TranscriptProtocol for Poseidon2Transcript {
    fn absorb_ccs_header(&mut self, n: usize, m: usize, t: usize) {
        self.append_message(crate::labels::CCS_HEADER, &[]);
        self.append_fields(
            crate::labels::CCS_DIMS,
            &[F::from_u64(n as u64), F::from_u64(m as u64), F::from_u64(t as u64)],
        );
    }
    fn absorb_poly_sparse(&mut self, label: &'static [u8], terms: &[(F, Vec<u32>)]) {
        self.append_message(crate::labels::POLY_SPARSE, label);
        self.append_fields(crate::labels::POLY_LEN, &[F::from_u64(terms.len() as u64)]);
        for (coeff, exps) in terms {
            self.append_fields(crate::labels::POLY_COEFF, &[*coeff]);
            let exps_f: Vec<F> = exps.iter().map(|&e| F::from_u64(e as u64)).collect();
            self.append_fields(crate::labels::POLY_EXPS, &exps_f);
        }
    }
    fn absorb_commit_coords(&mut self, coords: &[F]) {
        self.append_message(crate::labels::COMMIT_COORDS, &[]);
        self.append_fields(crate::labels::COMMIT_COORDS, coords);
    }
    fn absorb_public_fields(&mut self, label: &'static [u8], fs: &[F]) {
        self.append_fields(label, fs);
    }
}

impl Poseidon2Transcript {
    pub fn append_fields_raw(&mut self, fs: &[F]) {
        self.absorb_elem(Goldilocks::from_u64(fs.len() as u64));
        self.absorb_slice(fs);
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_fields_raw", b"", fs.len(), &self.st));
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("append_fields_raw", b"", fs.len(), &self.st));
    }

    pub fn challenge_fields_raw(&mut self, n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            for i in 0..p2::DIGEST_LEN.min(n - out.len()) {
                out.push(F::from_u64(self.st[i].as_canonical_u64()));
            }
        }
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_fields_raw");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_fields_raw", b"", n, &self.st));
        out
    }

    pub fn challenge_nonzero_field(&mut self, label: &'static [u8]) -> F {
        self.append_message(b"chal/label", label);
        loop {
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            let x = F::from_u64(self.st[0].as_canonical_u64());
            if x != F::ZERO {
                self.bind_query(QUERY_NONZERO_FIELD, 1);
                return x;
            }
        }
    }

    pub fn append_fields_iter<I>(&mut self, label: &'static [u8], len: usize, iter: I)
    where
        I: IntoIterator<Item = F>,
    {
        self.absorb_elem(Goldilocks::from_u64(OP_APPEND_FIELDS));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_elem(Goldilocks::from_u64(len as u64));

        const CHUNK: usize = 1024;
        let mut buf = [F::ZERO; CHUNK];
        let mut used = 0usize;
        let mut seen = 0usize;
        for f in iter {
            buf[used] = f;
            used += 1;
            seen += 1;
            if used == CHUNK {
                self.absorb_slice(&buf);
                used = 0;
            }
        }
        if used > 0 {
            self.absorb_slice(&buf[..used]);
        }

        if seen != len {
            panic!("append_fields_iter: iterator length mismatch (seen={seen}, len={len})");
        }

        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_fields_iter", label, len, &self.st));
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("append_fields_iter", label, len, &self.st));
    }

    pub fn append_bytes_packed(&mut self, label: &'static [u8], bytes: &[u8]) {
        self.absorb_elem(Goldilocks::from_u64(OP_APPEND_MESSAGE));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_packed_bytes_with_len(bytes);
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new(
            "append_bytes_packed",
            label,
            bytes.len(),
            &self.st,
        ));
    }
    #[cfg(feature = "debug-log")]
    pub fn dump_and_clear(&mut self, ctx: &str) {
        let tag = std::env::var("NEO_TRANSCRIPT_TAG").ok();
        if let Some(tag) = tag.as_deref() {
            eprintln!("--- [{}] Transcript dump [{}] ---", tag, ctx);
        } else {
            eprintln!("--- Transcript dump [{}] ---", ctx);
        }
        for (i, ev) in self.log.iter().enumerate() {
            let label = String::from_utf8_lossy(ev.label);
            eprintln!(
                "[{:04}] op={} label=\"{}\" len={} st0..3=[{}, {}, {}, {}]",
                i, ev.op, label, ev.len, ev.st_prefix[0], ev.st_prefix[1], ev.st_prefix[2], ev.st_prefix[3]
            );
        }
        self.log.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn append_message_bytes_packing_is_injective_against_modulus_wrap() {
        let msg1 = [0u8; 8];
        let msg2 = [0x01, 0x00, 0x00, 0x00, 0xFF, 0xFF, 0xFF, 0xFF];

        let mut t1 = <Poseidon2Transcript as crate::Transcript>::new(b"test/domain");
        t1.append_message(b"m", &msg1);
        let d1 = t1.digest32();

        let mut t2 = <Poseidon2Transcript as crate::Transcript>::new(b"test/domain");
        t2.append_message(b"m", &msg2);
        let d2 = t2.digest32();

        assert_ne!(d1, d2, "packed byte encoding must be injective");
    }
}
