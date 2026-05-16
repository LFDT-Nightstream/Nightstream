//! In-circuit Poseidon2 sponge transcript — byte-for-byte state mirror of
//! `neo_transcript::Poseidon2Transcript`.
//!
//! Owns: the `(state: [Var; 8], absorbed: 0..RATE)` sponge in R1CS form,
//! and the absorb/squeeze primitives that match the native transcript's
//! state evolution.
//!
//! Does not own: any decision about *what* a verifier absorbs. Callers
//! drive the absorb/squeeze sequence to match their native verifier path.
//!
//! ## Soundness contract
//!
//! For any sequence of absorbs/squeezes that a native `Poseidon2Transcript`
//! performs, the same sequence on `TranscriptGadget` must produce wires
//! whose witness values equal the native field outputs. This is enforced
//! by the parity tests in `tests/gadgets/transcript.rs`.
//!
//! ## Layout invariants
//!
//! - `state[i]` holds the i-th lane of the sponge state (Goldilocks F).
//! - `absorbed ∈ 0..=RATE`. When it equals RATE, the next absorb triggers
//!   a `permute()` before writing.
//! - Constant absorbs (labels, length headers, the squeeze-domain `F::ONE`)
//!   are computed at gadget-emit time and bound to fresh wires via an
//!   equality constraint, so an adversarial prover cannot deviate.
//!
//! ## Cost
//!
//! Each `permute()` invokes [`enforce_poseidon2_permutation`], adding the
//! per-permutation row/wire cost from `paper/poseidon2_gadget` tests.
//! `new()` collapses the constant init phase by running the native
//! transcript offline and binding the result, which saves several permutes
//! per session at no soundness cost (the bindings make the state values
//! verifier-checkable).

use neo_math::F;
use neo_transcript::Poseidon2Transcript;
use p3_field::PrimeCharacteristicRing;

use crate::engine::r1cs_circuit::builder::{Lc, R1csBuilder, Var};
use crate::engine::r1cs_circuit::field_ext::KVar;
use crate::engine::r1cs_circuit::poseidon2::enforce_poseidon2_permutation;

const WIDTH: usize = 8;
const RATE: usize = 4;
const DIGEST_LEN: usize = 4;

/// In-circuit Poseidon2 sponge state.
pub struct TranscriptGadget {
    state: [Var; WIDTH],
    /// Compile-time-tracked rate cursor. Not a witness — its value follows
    /// deterministically from the absorb sequence emitted so far.
    absorbed: usize,
}

impl TranscriptGadget {
    /// Initialize a fresh transcript with the given application label.
    ///
    /// Mirrors `Poseidon2Transcript::new(app_label)`, which absorbs the
    /// fixed `APP_DOMAIN` then `app_label`. Since both are static, we run
    /// the absorb offline and bind the resulting state to wires — saving
    /// several permutes without losing soundness (the bindings are
    /// constraints, so the prover cannot start from a different state).
    pub fn new(builder: &mut R1csBuilder, app_label: &'static [u8]) -> Self {
        use neo_transcript::Transcript as _;
        let native = Poseidon2Transcript::new(app_label);
        Self::from_native_state(builder, native.state(), native.absorbed())
    }

    /// Wrap a pre-computed sponge state into a `TranscriptGadget`.
    ///
    /// **Use for production wiring**, when F' already ran a native session
    /// init before reaching the in-circuit verifier. The state values are
    /// bound to wires via equality constraints.
    pub fn from_native_state(builder: &mut R1csBuilder, state_vals: [F; WIDTH], absorbed: usize) -> Self {
        assert!(absorbed <= RATE, "absorbed cursor out of range");
        let mut state = [Var::ONE; WIDTH];
        for (slot, &v) in state.iter_mut().zip(state_vals.iter()) {
            *slot = alloc_constant(builder, v);
        }
        Self { state, absorbed }
    }

    // ── Public API mirrors `neo_transcript::Transcript` ─────────────────

    /// Replicates `Poseidon2Transcript::append_message(label, msg)`.
    pub fn append_message(&mut self, builder: &mut R1csBuilder, label: &[u8], msg: &[u8]) {
        self.absorb_packed_bytes_with_len(builder, label);
        self.absorb_packed_bytes_with_len(builder, msg);
    }

    /// Replicates `Poseidon2Transcript::append_fields_raw(fs)` for constant
    /// field values — absorbs `len(fs)` as a length header, then each `F` as
    /// a constant-bound wire. Use for protocol markers like the
    /// `[F::from_u64(0), F::from_u64(i)]` domain separators in the native
    /// ρ-sampler.
    pub fn append_fields_raw_const(&mut self, builder: &mut R1csBuilder, fs: &[F]) {
        self.absorb_const_elem(builder, F::from_u64(fs.len() as u64));
        for &v in fs {
            self.absorb_const_elem(builder, v);
        }
    }

    /// Replicates `Poseidon2Transcript::append_fields_raw(fs)` for witness
    /// `Var` values — absorbs `len(fs)` as a constant-bound length header,
    /// then the variable wires verbatim. This is the sumcheck-round absorb
    /// shape used inside `verify_sumcheck_rounds_poseidon_v3` and the
    /// engine-side `append_fields_raw` calls in
    /// `optimized_verify_with_cache_and_public_instance_digest_impl`.
    pub fn append_fields_raw_vars(&mut self, builder: &mut R1csBuilder, fs: &[Var]) {
        self.absorb_const_elem(builder, F::from_u64(fs.len() as u64));
        self.absorb_slice(builder, fs);
    }

    /// Replicates `Poseidon2Transcript::append_fields(label, fs)`.
    pub fn append_fields(&mut self, builder: &mut R1csBuilder, label: &[u8], fs: &[Var]) {
        self.absorb_packed_bytes_with_len(builder, label);
        self.absorb_const_elem(builder, F::from_u64(fs.len() as u64));
        self.absorb_slice(builder, fs);
    }

    /// Replicates `Poseidon2Transcript::challenge_field(label)`. Returns a
    /// `Var` whose witness equals the native squeezed F.
    pub fn challenge_field(&mut self, builder: &mut R1csBuilder, label: &[u8]) -> Var {
        self.append_message(builder, b"chal/label", label);
        self.absorb_const_elem(builder, F::ONE);
        self.permute(builder);
        self.state[0]
    }

    /// Replicates `Poseidon2Transcript::challenge_fields(label, n)`. Returns
    /// `n` Vars whose witnesses equal the native squeezed fields.
    pub fn challenge_fields(&mut self, builder: &mut R1csBuilder, label: &[u8], n: usize) -> Vec<Var> {
        self.append_message(builder, b"chal/label", label);
        self.squeeze_n_raw(builder, n)
    }

    /// Replicates `Poseidon2Transcript::challenge_fields_raw(n)` — no
    /// `chal/label` prefix, just the raw squeeze loop. Required for the
    /// sumcheck-round challenge and `sample_k_batch` paths in
    /// `engines::utils` and `sumcheck::verify_sumcheck_rounds_poseidon_v3`.
    pub fn challenge_fields_raw(&mut self, builder: &mut R1csBuilder, n: usize) -> Vec<Var> {
        self.squeeze_n_raw(builder, n)
    }

    /// Shared squeeze loop used by `challenge_fields` and
    /// `challenge_fields_raw`. The native sponge appends `Goldilocks::ONE`
    /// before each permutation and emits at most `DIGEST_LEN` lanes from the
    /// rate region.
    fn squeeze_n_raw(&mut self, builder: &mut R1csBuilder, n: usize) -> Vec<Var> {
        let mut out = Vec::with_capacity(n);
        while out.len() < n {
            self.absorb_const_elem(builder, F::ONE);
            self.permute(builder);
            let take = DIGEST_LEN.min(n - out.len());
            for i in 0..take {
                out.push(self.state[i]);
            }
        }
        out
    }

    // ── K-extension helpers (paired F absorbs/squeezes) ────────────────

    /// Absorb a slice of K-elements under `label`, by appending each `KVar`
    /// as the pair `[c0, c1]`. Mirrors the native `append_fields(label,
    /// round_coeff_fields(&[K]))` pattern used in `neo-reductions::sumcheck`.
    pub fn append_k_slice(&mut self, builder: &mut R1csBuilder, label: &[u8], ks: &[KVar]) {
        let mut packed: Vec<Var> = Vec::with_capacity(ks.len() * 2);
        for k in ks {
            packed.push(k.c0);
            packed.push(k.c1);
        }
        self.append_fields(builder, label, &packed);
    }

    /// Squeeze one K-element under `label`. Mirrors the native
    /// `from_complex(c[0], c[1])` after `challenge_fields(label, 2)`.
    pub fn challenge_k(&mut self, builder: &mut R1csBuilder, label: &[u8]) -> KVar {
        let lanes = self.challenge_fields(builder, label, 2);
        KVar::new(lanes[0], lanes[1])
    }

    /// Squeeze `n` K-elements under one shared `label`. Single
    /// `append_message(b"chal/label", label)` prefix, then a stream of
    /// `2·n` F squeezes packed into K elements.
    pub fn challenge_k_vec(&mut self, builder: &mut R1csBuilder, label: &[u8], n: usize) -> Vec<KVar> {
        let lanes = self.challenge_fields(builder, label, 2 * n);
        lanes
            .chunks_exact(2)
            .map(|p| KVar::new(p[0], p[1]))
            .collect()
    }

    /// Replicates `Poseidon2Transcript::digest32()` but returns the 4 F lanes
    /// directly. Bytes are not the natural in-circuit currency; callers
    /// that need bytes can bit-decompose each lane.
    pub fn digest_fields(&mut self, builder: &mut R1csBuilder) -> [Var; DIGEST_LEN] {
        self.absorb_const_elem(builder, F::ONE);
        self.permute(builder);
        let mut out = [Var::ONE; DIGEST_LEN];
        out.copy_from_slice(&self.state[..DIGEST_LEN]);
        out
    }

    // ── Sponge primitives (private) ─────────────────────────────────────

    /// Absorb a single variable F into the sponge rate. Mirrors
    /// `Poseidon2Transcript::absorb_elem`.
    fn absorb_elem(&mut self, builder: &mut R1csBuilder, x: Var) {
        if self.absorbed >= RATE {
            self.permute(builder);
        }
        self.state[self.absorbed] = x;
        self.absorbed += 1;
    }

    /// Absorb a Goldilocks constant. Allocates a wire bound to `c`, then
    /// proceeds via `absorb_elem`.
    fn absorb_const_elem(&mut self, builder: &mut R1csBuilder, c: F) {
        let v = alloc_constant(builder, c);
        self.absorb_elem(builder, v);
    }

    /// Mirrors `Poseidon2Transcript::absorb_slice` — fill the buffer, then
    /// for each full RATE-sized chunk overwrite `state[0..RATE]` and
    /// permute, then buffer the remainder.
    fn absorb_slice(&mut self, builder: &mut R1csBuilder, xs: &[Var]) {
        let len = xs.len();
        let mut i = 0;

        // 1. Fill remaining buffer.
        while self.absorbed < RATE && i < len {
            self.state[self.absorbed] = xs[i];
            self.absorbed += 1;
            i += 1;
        }
        if self.absorbed == RATE {
            self.permute(builder);
        }

        // 2. Bulk-absorb full chunks (overwrite rate slots, preserve capacity).
        while len - i >= RATE {
            for j in 0..RATE {
                self.state[j] = xs[i + j];
            }
            self.permute(builder);
            i += RATE;
        }

        // 3. Buffer the remainder.
        while i < len {
            self.state[self.absorbed] = xs[i];
            self.absorbed += 1;
            i += 1;
        }
    }

    /// Mirrors `Poseidon2Transcript::absorb_packed_bytes_with_len`. The bytes
    /// are static (or otherwise known to the verifier), so each absorb is a
    /// constant binding.
    fn absorb_packed_bytes_with_len(&mut self, builder: &mut R1csBuilder, bytes: &[u8]) {
        self.absorb_const_elem(builder, F::from_u64(bytes.len() as u64));
        const BYTES_PER_LIMB: usize = 7;
        let mut i = 0;
        while i + BYTES_PER_LIMB <= bytes.len() {
            let mut limb = [0u8; 8];
            limb[..BYTES_PER_LIMB].copy_from_slice(&bytes[i..i + BYTES_PER_LIMB]);
            self.absorb_const_elem(builder, F::from_u64(u64::from_le_bytes(limb)));
            i += BYTES_PER_LIMB;
        }
        if i < bytes.len() {
            let mut limb = [0u8; 8];
            limb[..bytes.len() - i].copy_from_slice(&bytes[i..]);
            self.absorb_const_elem(builder, F::from_u64(u64::from_le_bytes(limb)));
        }
    }

    /// Apply the Poseidon2 permutation to the state and reset the rate
    /// cursor. Materializes 8 fresh wires for the new state.
    fn permute(&mut self, builder: &mut R1csBuilder) {
        self.state = enforce_poseidon2_permutation(builder, &self.state);
        self.absorbed = 0;
    }
}

/// Allocate a fresh wire bound to a Goldilocks constant.
fn alloc_constant(builder: &mut R1csBuilder, c: F) -> Var {
    let v = builder.alloc(c);
    builder.enforce_eq(&Lc::from_var(v), &Lc::from_const(c));
    v
}

#[cfg(test)]
mod tests {
    use super::*;
    use neo_transcript::Transcript as _;

    const APP: &[u8] = b"neo.fold.clean.unit.transcript/v1";

    fn digest_lanes(bytes: [u8; 32]) -> [F; 4] {
        std::array::from_fn(|i| {
            let mut limb = [0u8; 8];
            limb.copy_from_slice(&bytes[i * 8..(i + 1) * 8]);
            F::from_u64(u64::from_le_bytes(limb))
        })
    }

    #[test]
    fn raw_absorb_and_digest_match_native_across_rate_boundary() {
        let fields: Vec<F> = (0..9).map(|i| F::from_u64(100 + i)).collect();

        let mut native = Poseidon2Transcript::new(APP);
        native.append_fields_raw(&fields);
        let expected = digest_lanes(native.digest32());

        let mut builder = R1csBuilder::new();
        let mut gadget = TranscriptGadget::new(&mut builder, APP);
        let vars = fields
            .iter()
            .copied()
            .map(|v| builder.alloc(v))
            .collect::<Vec<_>>();
        gadget.append_fields_raw_vars(&mut builder, &vars);
        let got = gadget.digest_fields(&mut builder);

        for (wire, expected_lane) in got.into_iter().zip(expected) {
            assert_eq!(builder.witness()[wire.col()], expected_lane);
            builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(expected_lane));
        }
        assert_eq!(gadget.absorbed, 0, "digest squeeze must reset the rate cursor");
        assert!(
            builder.is_satisfied(),
            "raw absorb + digest parity failed (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );
    }

    #[test]
    fn raw_challenge_stream_matches_native_for_more_than_one_digest_block() {
        let prefix: Vec<F> = (0..6).map(|i| F::from_u64(17 * i + 3)).collect();

        let mut native = Poseidon2Transcript::new(APP);
        native.append_fields_raw(&prefix);
        let expected = native.challenge_fields_raw(11);

        let mut builder = R1csBuilder::new();
        let mut gadget = TranscriptGadget::new(&mut builder, APP);
        let vars = prefix
            .iter()
            .copied()
            .map(|v| builder.alloc(v))
            .collect::<Vec<_>>();
        gadget.append_fields_raw_vars(&mut builder, &vars);
        let got = gadget.challenge_fields_raw(&mut builder, 11);

        for (wire, expected_lane) in got.into_iter().zip(expected) {
            assert_eq!(builder.witness()[wire.col()], expected_lane);
            builder.enforce_eq(&Lc::from_var(wire), &Lc::from_const(expected_lane));
        }
        assert!(
            builder.is_satisfied(),
            "raw challenge stream parity failed (first bad row: {:?})",
            builder.first_unsatisfied_row()
        );
    }

    #[test]
    fn constant_absorb_is_bound_to_the_declared_constant() {
        let mut builder = R1csBuilder::new();
        let mut gadget = TranscriptGadget::new(&mut builder, APP);
        let first_const_after_init = builder.witness().len();
        gadget.append_fields_raw_const(&mut builder, &[F::from_u64(9), F::from_u64(10)]);
        assert!(builder.is_satisfied(), "baseline should satisfy");

        // The first newly allocated constant in append_fields_raw_const is
        // the length header `2`. Tampering it must violate the equality
        // constraint emitted by `alloc_constant`.
        builder.tamper_witness(first_const_after_init, F::from_u64(999));
        assert!(
            !builder.is_satisfied(),
            "constant absorb wires must be equality-bound, not prover-controlled"
        );
    }
}
