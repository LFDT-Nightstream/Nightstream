//! Bit-backed Poseidon2 transcript trace builder.
//!
//! Owns the low-norm trace image for sponge-mode transcript operations:
//! raw/labelled absorbs, raw/labelled squeezes, and digest squeezes. It
//! mirrors `neo_transcript::Poseidon2Transcript` state evolution while
//! reusing the standalone bit-backed permutation builder for each
//! permutation trace.
//!
//! Does not own any protocol decision about what gets absorbed. Callers
//! drive the same operation sequence as the verifier they are mirroring.

use neo_math::F;
use neo_transcript::{Poseidon2Transcript, Transcript as _};
use p3_field::PrimeCharacteristicRing;

use crate::engine::ccs_native::poseidon2::{
    build_bit_backed_poseidon2_permutation, BITS_PER_PERMUTATION, POSEIDON2_DIGEST_LEN, POSEIDON2_GOLDILOCKS_BITS,
    POSEIDON2_RATE, POSEIDON2_WIDTH,
};

const BYTES_PER_LIMB: usize = 7;
const OP_APPEND_MESSAGE: u64 = 1;
const OP_APPEND_FIELDS: u64 = 2;
const QUERY_FIELDS: u64 = 0x101;
const QUERY_DIGEST32: u64 = 0x104;

#[derive(Clone, Debug, Default, PartialEq, Eq)]
pub struct SpongeTraceLayout {
    pub permute_offsets: Vec<usize>,
    pub squeeze_lane_offsets: Vec<usize>,
}

#[derive(Clone, Debug)]
pub struct SpongeTraceImage {
    pub layout: SpongeTraceLayout,
    /// Low-norm assignment fragment. `values[0] = F::ONE`; all later
    /// coordinates are Poseidon2 trace bits.
    pub values: Vec<F>,
    pub final_state: [F; POSEIDON2_WIDTH],
    pub absorbed: usize,
    pub squeezed_values: Vec<F>,
}

pub struct SpongeTraceBuilder {
    state: [F; POSEIDON2_WIDTH],
    absorbed: usize,
    values: Vec<F>,
    layout: SpongeTraceLayout,
    squeezed_values: Vec<F>,
}

impl SpongeTraceBuilder {
    /// Start from the same post-init state as
    /// `Poseidon2Transcript::new(app_label)`. The init itself is not
    /// traced; this matches `TranscriptGadget::new`, which binds the
    /// native init state directly.
    pub fn new(app_label: &'static [u8]) -> Self {
        let native = Poseidon2Transcript::new(app_label);
        Self::from_native_state(native.state(), native.absorbed())
    }

    pub fn from_native_state(state: [F; POSEIDON2_WIDTH], absorbed: usize) -> Self {
        assert!(absorbed <= POSEIDON2_RATE, "transcript absorbed cursor out of range");
        Self {
            state,
            absorbed,
            values: vec![F::ONE],
            layout: SpongeTraceLayout::default(),
            squeezed_values: Vec::new(),
        }
    }

    pub fn append_message(&mut self, label: &[u8], msg: &[u8]) {
        self.absorb_elem(F::from_u64(OP_APPEND_MESSAGE));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_packed_bytes_with_len(msg);
    }

    pub fn append_fields_raw(&mut self, fs: &[F]) {
        self.absorb_elem(F::from_u64(fs.len() as u64));
        self.absorb_slice(fs);
    }

    pub fn append_fields(&mut self, label: &[u8], fs: &[F]) {
        self.absorb_elem(F::from_u64(OP_APPEND_FIELDS));
        self.absorb_packed_bytes_with_len(label);
        self.absorb_elem(F::from_u64(fs.len() as u64));
        self.absorb_slice(fs);
    }

    pub fn challenge_field(&mut self, label: &[u8]) -> F {
        self.append_message(b"chal/label", label);
        let out = self.squeeze_n_raw(1)[0];
        self.bind_query(QUERY_FIELDS, 1);
        out
    }

    pub fn challenge_fields(&mut self, label: &[u8], n: usize) -> Vec<F> {
        self.append_message(b"chal/label", label);
        let out = self.squeeze_n_raw(n);
        self.bind_query(QUERY_FIELDS, n);
        out
    }

    pub fn challenge_fields_raw(&mut self, n: usize) -> Vec<F> {
        self.squeeze_n_raw(n)
    }

    pub fn digest_fields(&mut self) -> [F; POSEIDON2_DIGEST_LEN] {
        let lanes = self.squeeze_once(POSEIDON2_DIGEST_LEN);
        let out = std::array::from_fn(|i| lanes[i]);
        self.bind_query(QUERY_DIGEST32, POSEIDON2_DIGEST_LEN * 8);
        out
    }

    pub fn finish(self) -> SpongeTraceImage {
        SpongeTraceImage {
            layout: self.layout,
            values: self.values,
            final_state: self.state,
            absorbed: self.absorbed,
            squeezed_values: self.squeezed_values,
        }
    }

    fn squeeze_n_raw(&mut self, n: usize) -> Vec<F> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            let take = POSEIDON2_DIGEST_LEN.min(n - out.len());
            out.extend(self.squeeze_once(take));
        }
        out
    }

    fn squeeze_once(&mut self, take: usize) -> Vec<F> {
        assert!(take <= POSEIDON2_DIGEST_LEN);
        self.absorb_elem(F::ONE);
        self.permute();
        let lane_offsets = self.current_final_state_lane_offsets();
        let mut out = Vec::with_capacity(take);
        for (lane, offset) in lane_offsets.iter().copied().enumerate().take(take) {
            self.layout.squeeze_lane_offsets.push(offset);
            out.push(self.state[lane]);
            self.squeezed_values.push(self.state[lane]);
        }
        out
    }

    fn absorb_elem(&mut self, x: F) {
        if self.absorbed >= POSEIDON2_RATE {
            self.permute();
        }
        self.state[self.absorbed] = x;
        self.absorbed += 1;
    }

    fn bind_query(&mut self, query: u64, output_len: usize) {
        self.absorb_elem(F::from_u64(query));
        self.absorb_elem(F::from_u64(output_len as u64));
    }

    fn absorb_slice(&mut self, xs: &[F]) {
        let len = xs.len();
        let mut i = 0;

        while self.absorbed < POSEIDON2_RATE && i < len {
            self.state[self.absorbed] = xs[i];
            self.absorbed += 1;
            i += 1;
        }
        if self.absorbed == POSEIDON2_RATE {
            self.permute();
        }

        while len - i >= POSEIDON2_RATE {
            self.state[..POSEIDON2_RATE].copy_from_slice(&xs[i..i + POSEIDON2_RATE]);
            self.permute();
            i += POSEIDON2_RATE;
        }

        while i < len {
            self.state[self.absorbed] = xs[i];
            self.absorbed += 1;
            i += 1;
        }
    }

    fn absorb_packed_bytes_with_len(&mut self, bytes: &[u8]) {
        self.absorb_elem(F::from_u64(bytes.len() as u64));
        for chunk in bytes.chunks(BYTES_PER_LIMB) {
            let mut limb = [0u8; 8];
            limb[..chunk.len()].copy_from_slice(chunk);
            self.absorb_elem(F::from_u64(u64::from_le_bytes(limb)));
        }
    }

    fn permute(&mut self) {
        let offset = self.values.len();
        let bundle = build_bit_backed_poseidon2_permutation(self.state);
        assert_eq!(
            bundle.z.len(),
            BITS_PER_PERMUTATION + 1,
            "permutation trace length drifted"
        );
        self.values.extend_from_slice(&bundle.z[1..]);
        self.layout.permute_offsets.push(offset);
        self.state = bundle.output_state;
        self.absorbed = 0;
    }

    fn current_final_state_lane_offsets(&self) -> [usize; POSEIDON2_DIGEST_LEN] {
        let start = *self
            .layout
            .permute_offsets
            .last()
            .expect("squeeze output requested before first permutation");
        let final_state_start = start + BITS_PER_PERMUTATION - POSEIDON2_WIDTH * POSEIDON2_GOLDILOCKS_BITS;
        std::array::from_fn(|lane| final_state_start + lane * POSEIDON2_GOLDILOCKS_BITS)
    }
}

pub fn decode_squeezed_lanes(image: &SpongeTraceImage) -> Vec<F> {
    image
        .layout
        .squeeze_lane_offsets
        .iter()
        .map(|&offset| decode_lane(&image.values, offset))
        .collect()
}

pub fn decode_lane(values: &[F], offset: usize) -> F {
    let mut acc = F::ZERO;
    let mut pow = F::ONE;
    for bit in 0..POSEIDON2_GOLDILOCKS_BITS {
        let v = values[offset + bit];
        assert!(
            v == F::ZERO || v == F::ONE,
            "sponge trace bit out of range: offset={offset} bit={bit} value={v:?}"
        );
        if v == F::ONE {
            acc += pow;
        }
        pow *= F::from_u64(2);
    }
    acc
}
