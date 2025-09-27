use p3_goldilocks::Goldilocks;
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_symmetric::Permutation;
use neo_math::F;
use crate::{Transcript, TranscriptProtocol};
use neo_ccs::crypto::poseidon2_goldilocks as p2;

const APP_DOMAIN: &[u8] = b"neo/transcript/v1|poseidon2-goldilocks-w12-r8";

#[derive(Clone)]
pub struct Poseidon2Transcript {
    st: [Goldilocks; p2::WIDTH],
    absorbed: usize,
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
    fn permute(&mut self) {
        self.st = p2::permutation().permute(self.st);
        self.absorbed = 0;
    }

    /// Export current internal state (for RNG binding).
    pub fn state(&self) -> [Goldilocks; p2::WIDTH] { self.st }
}

impl Transcript for Poseidon2Transcript {
    fn new(app_label: &'static [u8]) -> Self {
        let mut tr = Self {
            st: [Goldilocks::ZERO; p2::WIDTH],
            absorbed: 0,
            #[cfg(feature = "debug-log")]
            log: Vec::new(),
        };
        tr.append_message(APP_DOMAIN, app_label);
        tr
    }

    fn append_message(&mut self, label: &'static [u8], msg: &[u8]) {
        for &b in label { self.absorb_elem(Goldilocks::from_u64(b as u64)); }
        self.absorb_elem(Goldilocks::from_u64(msg.len() as u64));
        for &b in msg { self.absorb_elem(Goldilocks::from_u64(b as u64)); }
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new("append_message", label, msg.len(), &self.st));
    }

    fn append_fields(&mut self, label: &'static [u8], fs: &[F]) {
        for &b in label { self.absorb_elem(Goldilocks::from_u64(b as u64)); }
        self.absorb_elem(Goldilocks::from_u64(fs.len() as u64));
        for &x in fs { self.absorb_elem(Goldilocks::from_u64(x.as_canonical_u64())); }
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new("append_fields", label, fs.len(), &self.st));
    }

    fn challenge_bytes(&mut self, label: &'static [u8], out: &mut [u8]) {
        self.append_message(b"chal/label", label);
        let mut produced = 0usize;
        while produced < out.len() {
            // Domain gate before squeezing to avoid state reuse issues.
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            for i in 0..4 { // 4 limbs = 32 bytes per squeeze
                let limb = self.st[i].as_canonical_u64().to_le_bytes();
                let take = (out.len() - produced).min(8);
                out[produced..produced + take].copy_from_slice(&limb[..take]);
                produced += take;
                if produced >= out.len() { break; }
            }
        }
    }

    fn challenge_field(&mut self, label: &'static [u8]) -> F {
        // Append the label once, then rejection sample until non-zero
        // to obtain a uniform element from F\{0}.
        self.append_message(b"chal/label", label);
        loop {
            // gate + permute
            self.absorb_elem(Goldilocks::ONE);
            self.permute();
            let limb = self.st[0].as_canonical_u64();
            if limb != 0 {
                return F::from_u64(limb);
            }
        }
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
        for i in 0..4 { out[i*8..(i+1)*8].copy_from_slice(&self.st[i].as_canonical_u64().to_le_bytes()); }
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new("digest32", b"", 0, &self.st));
        out
    }
}

impl TranscriptProtocol for Poseidon2Transcript {
    fn absorb_ccs_header(&mut self, n: usize, m: usize, t: usize) {
        self.append_message(crate::labels::CCS_HEADER, &[]);
        self.append_fields(crate::labels::CCS_DIMS, &[
            F::from_u64(n as u64), F::from_u64(m as u64), F::from_u64(t as u64)
        ]);
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
    pub fn append_u64s(&mut self, label: &'static [u8], us: &[u64]) {
        for &b in label { self.absorb_elem(Goldilocks::from_u64(b as u64)); }
        self.absorb_elem(Goldilocks::from_u64(us.len() as u64));
        for &u in us { self.absorb_elem(Goldilocks::from_u64(u)); }
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new("append_u64s", label, us.len(), &self.st));
    }

    pub fn append_bytes_packed(&mut self, label: &'static [u8], bytes: &[u8]) {
        // Pack 8-bytes per field + absorb length at end for framing
        for &b in label { self.absorb_elem(Goldilocks::from_u64(b as u64)); }
        // store byte len for framing
        self.absorb_elem(Goldilocks::from_u64(bytes.len() as u64));
        for chunk in bytes.chunks(8) {
            let mut buf = [0u8; 8];
            buf[..chunk.len()].copy_from_slice(chunk);
            self.absorb_elem(Goldilocks::from_u64(u64::from_le_bytes(buf)));
        }
        #[cfg(feature = "debug-log")]
        self.log.push(crate::debug::Event::new("append_bytes_packed", label, bytes.len(), &self.st));
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
        out
    }

    pub fn challenge_bytes_vec(&mut self, label: &'static [u8], len: usize) -> Vec<u8> {
        let mut out = vec![0u8; len];
        self.challenge_bytes(label, &mut out);
        out
    }
}
