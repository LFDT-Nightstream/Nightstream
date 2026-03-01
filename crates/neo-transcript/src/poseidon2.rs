use crate::{Transcript, TranscriptProtocol};
use neo_ccs::crypto::poseidon2_goldilocks as p2;
use neo_math::F;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::{Goldilocks, Poseidon2Goldilocks};
use p3_symmetric::Permutation;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w8-r4";

#[derive(Clone)]
pub struct Poseidon2Transcript {
    st: [Goldilocks; p2::WIDTH],
    absorbed: usize,
    perm: &'static Poseidon2Goldilocks<{ p2::WIDTH }>,
    #[cfg(feature = "debug-log")]
    log: Vec<crate::debug::Event>,
}

impl Poseidon2Transcript {
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

        // 1. Fill remaining buffer
        while self.absorbed < p2::RATE && src_idx < len {
            self.st[self.absorbed] = inputs[src_idx];
            self.absorbed += 1;
            src_idx += 1;
        }

        if self.absorbed == p2::RATE {
            self.permute();
        }

        // 2. Process full chunks
        while len - src_idx >= p2::RATE {
            // Manually unroll for p2::RATE = 4
            // We use assignment (overwrite) to match absorb_elem behavior
            self.st[0] = inputs[src_idx];
            self.st[1] = inputs[src_idx + 1];
            self.st[2] = inputs[src_idx + 2];
            self.st[3] = inputs[src_idx + 3];

            self.permute();
            src_idx += p2::RATE;
        }

        // 3. Buffer remaining
        while src_idx < len {
            self.st[self.absorbed] = inputs[src_idx];
            self.absorbed += 1;
            src_idx += 1;
        }
    }

    #[inline]
    fn absorb_packed_bytes_with_len(&mut self, bytes: &[u8]) {
        self.absorb_elem(Goldilocks::from_u64(bytes.len() as u64));

        // Pack 7 bytes per limb so every encoded integer is < 2^56 < p_goldilocks.
        // This preserves injectivity under Goldilocks::from_u64 reduction.
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
    fn absorb_u64_slice(&mut self, values: &[u64]) {
        // Encode each u64 as two 32-bit limbs to preserve full-range injectivity.
        const WORD_CHUNK: usize = 64;
        let mut buf = [Goldilocks::ZERO; WORD_CHUNK * 2];
        let mut i = 0usize;
        while i < values.len() {
            let take = (values.len() - i).min(WORD_CHUNK);
            for j in 0..take {
                let v = values[i + j];
                let lo = (v & 0xFFFF_FFFF) as u64;
                let hi = v >> 32;
                buf[2 * j] = Goldilocks::from_u64(lo);
                buf[2 * j + 1] = Goldilocks::from_u64(hi);
            }
            self.absorb_slice(&buf[..2 * take]);
            i += take;
        }
    }

    #[inline]
    fn permute(&mut self) {
        self.st = self.perm.permute(self.st);
        self.absorbed = 0;
    }

    /// Export current internal state (for RNG binding).
    pub fn state(&self) -> [Goldilocks; p2::WIDTH] {
        self.st
    }
}

impl Transcript for Poseidon2Transcript {
    fn new(app_label: &'static [u8]) -> Self {
        let mut tr = Self {
            st: [Goldilocks::ZERO; p2::WIDTH],
            absorbed: 0,
            perm: p2::permutation(),
            #[cfg(feature = "debug-log")]
            log: Vec::new(),
        };
        tr.append_message(APP_DOMAIN, app_label);
        tr
    }

    fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        self.absorb_packed_bytes_with_len(label);
        self.absorb_packed_bytes_with_len(msg);
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_message", label, msg.len(), &self.st));
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("append_message", label, msg.len(), &self.st));
    }

    fn append_fields(&mut self, label: &'static [u8], fs: &[F]) {
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
            // Domain gate before squeezing to avoid state reuse issues.
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            for i in 0..4 {
                // 4 limbs = 32 bytes per squeeze
                let limb = self.st[i].as_canonical_u64().to_le_bytes();
                let take = (out.len() - produced).min(8);
                out[produced..produced + take].copy_from_slice(&limb[..take]);
                produced += take;
                if produced >= out.len() {
                    break;
                }
            }
        }
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_bytes");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_bytes", label, out.len(), &self.st));
    }

    fn challenge_field(&mut self, label: &'static [u8]) -> F {
        // Uniform over F (including zero).
        self.append_message(b"chal/label", label);
        self.absorb_elem(Goldilocks::ONE);
        self.permute();
        let out = F::from_u64(self.st[0].as_canonical_u64());
        #[cfg(feature = "debug-log")]
        if std::env::var("NEO_TRANSCRIPT_DUMP").ok().as_deref() == Some("1") {
            self.dump_and_clear("challenge_field");
        }
        #[cfg(feature = "fs-guard")]
        crate::fs_guard::record(crate::debug::Event::new("challenge_field", label, 1, &self.st));
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

// Convenience helpers (not in the Transcript trait for minimal surface)
impl Poseidon2Transcript {
    pub fn challenge_nonzero_field(&mut self, label: &'static [u8]) -> F {
        // Rejection sampling to obtain an element in F\{0}.
        self.append_message(b"chal/label", label);
        loop {
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            let x = F::from_u64(self.st[0].as_canonical_u64());
            if x != F::ZERO {
                return x;
            }
        }
    }
    pub fn append_u64s(&mut self, label: &'static [u8], us: &[u64]) {
        self.absorb_packed_bytes_with_len(label);
        self.absorb_elem(Goldilocks::from_u64(us.len() as u64));
        self.absorb_u64_slice(us);
        #[cfg(feature = "debug-log")]
        self.log
            .push(crate::debug::Event::new("append_u64s", label, us.len(), &self.st));
    }

    pub fn append_fields_iter<I>(&mut self, label: &'static [u8], len: usize, iter: I)
    where
        I: IntoIterator<Item = F>,
    {
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

    pub fn challenge_fields(&mut self, label: &'static [u8], n: usize) -> Vec<F> {
        self.append_message(b"chal/label", label);
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
            self.dump_and_clear("challenge_fields");
        }
        out
    }

    pub fn challenge_bytes_vec(&mut self, label: &'static [u8], len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        self.challenge_bytes(label, &mut out);
        out
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

    #[test]
    fn append_u64s_is_injective_over_full_u64_range() {
        const GOLDILOCKS_P: u64 = 0xFFFF_FFFF_0000_0001;

        let mut t1 = <Poseidon2Transcript as crate::Transcript>::new(b"test/domain");
        t1.append_u64s(b"x", &[0u64]);
        let d1 = t1.digest32();

        let mut t2 = <Poseidon2Transcript as crate::Transcript>::new(b"test/domain");
        t2.append_u64s(b"x", &[GOLDILOCKS_P]);
        let d2 = t2.digest32();

        assert_ne!(d1, d2, "u64 encoding must be injective");
    }
}
