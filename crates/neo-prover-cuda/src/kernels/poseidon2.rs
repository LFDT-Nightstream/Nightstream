//! Poseidon2 (Goldilocks, width 8, rate 4) permutation and transcript ops.
//!
//! Contract: value-identical to `neo_ccs::crypto::poseidon2_goldilocks`
//! (permutation) and to `neo_transcript::Poseidon2Transcript` (absorb cursor
//! and challenge squeeze semantics). Round constants are uploaded from
//! `round_constants()` in the `RC_*` layout below — nothing is transcribed
//! by hand. The `parity transcript` gate asserts bit-equality against both.

use std::sync::Arc;

use cuda_core::{CudaContext, CudaStream, DeviceBuffer, DriverError, LaunchConfig};
use cuda_device::{cuda_module, kernel, thread, warp, DisjointSlice};
use cuda_host::EmbeddedModuleError;

use crate::kernels::goldilocks::{Gl, GOLDILOCKS_MODULUS};

pub const WIDTH: usize = 8;
pub const RATE: usize = 4;
pub const DIGEST_LEN: usize = 4;
pub const EXTERNAL_HALF_ROUNDS: usize = 4;
pub const INTERNAL_ROUNDS: usize = 22;

/// Round-constant buffer layout (u64 words): initial external rounds row by
/// row, internal constants, terminal external rounds, internal-matrix diag.
pub const RC_INITIAL: usize = 0;
pub const RC_INTERNAL: usize = RC_INITIAL + EXTERNAL_HALF_ROUNDS * WIDTH;
pub const RC_TERMINAL: usize = RC_INTERNAL + INTERNAL_ROUNDS;
pub const RC_DIAG: usize = RC_TERMINAL + EXTERNAL_HALF_ROUNDS * WIDTH;
pub const RC_WORDS: usize = RC_DIAG + WIDTH;

/// Transcript op stream: `[code, arg]` u64 pairs for host-payload ops.
pub const OP_ABSORB: u64 = 0;
pub const OP_CHALLENGE: u64 = 1;
/// Extended transcript IO stream: `[code, len, offset]` u64 triples.
/// Device variants read/write the device payload/output buffers.
pub const OP_ABSORB_DEVICE: u64 = 2;
pub const OP_CHALLENGE_DEVICE: u64 = 3;

/// Sponge state buffer: WIDTH lanes then the absorb cursor.
pub const ST_WORDS: usize = WIDTH + 1;
pub const MAX_DEVICE_FIELD_BIND_PREFIX_WORDS: usize = 8;

/// Packed label prefix for `Transcript::append_fields`, passed by value so a
/// device-resident field payload needs no host-to-device command buffer.
#[repr(C)]
#[derive(Clone, Copy)]
pub struct DeviceFieldBindPrefix {
    pub(crate) words: [u64; MAX_DEVICE_FIELD_BIND_PREFIX_WORDS],
    pub(crate) word_count: u64,
    pub(crate) field_count: u64,
}

pub use poseidon2_kernels::LoadedModule as Poseidon2KernelModule;

pub fn load_poseidon2_kernels(ctx: &Arc<CudaContext>) -> Result<Poseidon2KernelModule, EmbeddedModuleError> {
    poseidon2_kernels::load(ctx)
}

/// Permute `n` packed width-8 states in place, one thread per state.
pub fn launch_permute_states(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    n: usize,
    states: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if n == 0 {
        return Ok(());
    }
    module.p2_permute_states(stream, LaunchConfig::for_num_elems(n as u32), states, rc)
}

/// Hash `count` field slices with the canonical `poseidon2_hash` sponge.
///
/// `offsets[i]` and `lengths[i]` select the i-th preimage inside `fields`.
/// One thread owns each sponge.
pub fn launch_hash_fields(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    count: usize,
    fields: &DeviceBuffer<u64>,
    offsets: &DeviceBuffer<u64>,
    lengths: &DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if count == 0 {
        return Ok(());
    }
    module.p2_hash_fields(
        stream,
        LaunchConfig::for_num_elems(count as u32),
        fields,
        offsets,
        lengths,
        out,
        rc,
    )
}

/// Hash `count` field slices with one eight-lane warp tile per sponge.
///
/// The canonical sponge is unchanged; the eight Poseidon state lanes execute
/// each permutation cooperatively instead of serially inside one CUDA thread.
pub fn launch_hash_fields_cooperative(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    count: usize,
    fields: &DeviceBuffer<u64>,
    offsets: &DeviceBuffer<u64>,
    lengths: &DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if count == 0 {
        return Ok(());
    }
    module.p2_hash_fields_cooperative(
        stream,
        cooperative_hash_launch(count),
        fields,
        offsets,
        lengths,
        out,
        rc,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_hash_fields_cooperative_plan(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    count: usize,
    fields: &DeviceBuffer<u64>,
    plan: &DeviceBuffer<u64>,
    offsets_start: usize,
    lengths_start: usize,
    out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    if count == 0 {
        return Ok(());
    }
    module.p2_hash_fields_cooperative_plan(
        stream,
        cooperative_hash_launch(count),
        fields,
        plan,
        offsets_start as u32,
        lengths_start as u32,
        out,
        rc,
    )
}

pub fn launch_hash_contiguous_cooperative(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    fields: &DeviceBuffer<u64>,
    len: usize,
    out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_hash_contiguous_cooperative(stream, cooperative_hash_launch(1), fields, len as u64, out, rc)
}

fn cooperative_hash_launch(count: usize) -> LaunchConfig {
    const WARP_SIZE: u32 = 32;
    LaunchConfig {
        // One dependency-heavy sponge per block lets CUDA distribute the
        // independent hashes across SMs instead of pinning them to one SM.
        grid_dim: (count as u32, 1, 1),
        block_dim: (WARP_SIZE, 1, 1),
        shared_mem_bytes: 0,
    }
}

/// Run a transcript op stream against the device sponge. Single thread: the
/// sponge is inherently sequential, and the volumes are tiny.
pub fn launch_transcript_ops(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    st: &mut DeviceBuffer<u64>,
    ops: &DeviceBuffer<u64>,
    payload: &DeviceBuffer<u64>,
    out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_transcript_ops(stream, LaunchConfig::for_num_elems(1), st, ops, payload, out, rc)
}

/// Run a transcript op stream where absorbs/challenges can use host or
/// device buffers. This is the primitive needed for device-driven sumcheck
/// challenges: round coeff kernels write into a device payload buffer, this
/// kernel absorbs them, then writes challenges into a device output buffer
/// consumed by the next fold kernel.
pub fn launch_transcript_io_ops(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    st: &mut DeviceBuffer<u64>,
    ops: &DeviceBuffer<u64>,
    host_payload: &DeviceBuffer<u64>,
    device_payload: &DeviceBuffer<u64>,
    host_out: &mut DeviceBuffer<u64>,
    device_out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_transcript_io_ops(
        stream,
        LaunchConfig::for_num_elems(1),
        st,
        ops,
        host_payload,
        device_payload,
        host_out,
        device_out,
        rc,
    )
}

#[allow(clippy::too_many_arguments)]
pub fn launch_transcript_absorb_device_challenge(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    st: &mut DeviceBuffer<u64>,
    len_prefix: u64,
    device_payload: &DeviceBuffer<u64>,
    payload_offset: usize,
    payload_len: usize,
    device_out: &mut DeviceBuffer<u64>,
    out_offset: usize,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_transcript_absorb_device_challenge(
        stream,
        LaunchConfig::for_num_elems(1),
        st,
        len_prefix,
        device_payload,
        payload_offset as u32,
        payload_len as u32,
        device_out,
        out_offset as u32,
        rc,
    )
}

pub fn launch_transcript_sample_rlc_rhos(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    st: &mut DeviceBuffer<u64>,
    count: usize,
    coeffs_out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_transcript_sample_rlc_rhos(stream, LaunchConfig::for_num_elems(1), st, count as u32, coeffs_out, rc)
}

pub fn launch_transcript_bind_device_fields_sample_rlc_rhos(
    module: &Poseidon2KernelModule,
    stream: &Arc<CudaStream>,
    st: &mut DeviceBuffer<u64>,
    prefix: DeviceFieldBindPrefix,
    device_fields: &DeviceBuffer<u64>,
    count: usize,
    coeffs_out: &mut DeviceBuffer<u64>,
    rc: &DeviceBuffer<u64>,
) -> Result<(), DriverError> {
    module.p2_transcript_bind_device_fields_sample_rlc_rhos(
        stream,
        LaunchConfig::for_num_elems(1),
        st,
        prefix,
        device_fields,
        count as u32,
        coeffs_out,
        rc,
    )
}

// ── permutation core (plain Rust, named lanes: no local-memory arrays) ──────

#[derive(Clone, Copy)]
pub struct P2State {
    pub s0: Gl,
    pub s1: Gl,
    pub s2: Gl,
    pub s3: Gl,
    pub s4: Gl,
    pub s5: Gl,
    pub s6: Gl,
    pub s7: Gl,
}

fn sbox(x: Gl) -> Gl {
    let x2 = x * x;
    let x3 = x2 * x;
    x3 * x3 * x
}

/// `apply_mat4`: multiply by `[2 3 1 1; 1 2 3 1; 1 1 2 3; 3 1 1 2]`.
fn mat4(x0: Gl, x1: Gl, x2: Gl, x3: Gl) -> (Gl, Gl, Gl, Gl) {
    let t01 = x0 + x1;
    let t23 = x2 + x3;
    let t0123 = t01 + t23;
    let t01123 = t0123 + x1;
    let t01233 = t0123 + x3;
    (t01123 + t01, t01123 + x2 + x2, t01233 + t23, t01233 + x0 + x0)
}

/// External linear layer: `mat4` per half, then the outer circulant sums.
fn mds_light(st: P2State) -> P2State {
    let (a0, a1, a2, a3) = mat4(st.s0, st.s1, st.s2, st.s3);
    let (b0, b1, b2, b3) = mat4(st.s4, st.s5, st.s6, st.s7);
    let (m0, m1, m2, m3) = (a0 + b0, a1 + b1, a2 + b2, a3 + b3);
    P2State {
        s0: a0 + m0,
        s1: a1 + m1,
        s2: a2 + m2,
        s3: a3 + m3,
        s4: b0 + m0,
        s5: b1 + m1,
        s6: b2 + m2,
        s7: b3 + m3,
    }
}

fn external_round(st: P2State, rc: &[u64], base: usize) -> P2State {
    let st = P2State {
        s0: sbox(st.s0 + Gl::from_u64(rc[base])),
        s1: sbox(st.s1 + Gl::from_u64(rc[base + 1])),
        s2: sbox(st.s2 + Gl::from_u64(rc[base + 2])),
        s3: sbox(st.s3 + Gl::from_u64(rc[base + 3])),
        s4: sbox(st.s4 + Gl::from_u64(rc[base + 4])),
        s5: sbox(st.s5 + Gl::from_u64(rc[base + 5])),
        s6: sbox(st.s6 + Gl::from_u64(rc[base + 6])),
        s7: sbox(st.s7 + Gl::from_u64(rc[base + 7])),
    };
    mds_light(st)
}

/// One full Poseidon2 permutation. `rc` is the `RC_*`-layout constant buffer.
pub fn permute(st: P2State, rc: &[u64]) -> P2State {
    let mut st = mds_light(st);
    for r in 0..EXTERNAL_HALF_ROUNDS {
        st = external_round(st, rc, RC_INITIAL + WIDTH * r);
    }
    let d0 = Gl::from_u64(rc[RC_DIAG]);
    let d1 = Gl::from_u64(rc[RC_DIAG + 1]);
    let d2 = Gl::from_u64(rc[RC_DIAG + 2]);
    let d3 = Gl::from_u64(rc[RC_DIAG + 3]);
    let d4 = Gl::from_u64(rc[RC_DIAG + 4]);
    let d5 = Gl::from_u64(rc[RC_DIAG + 5]);
    let d6 = Gl::from_u64(rc[RC_DIAG + 6]);
    let d7 = Gl::from_u64(rc[RC_DIAG + 7]);
    for r in 0..INTERNAL_ROUNDS {
        st.s0 = sbox(st.s0 + Gl::from_u64(rc[RC_INTERNAL + r]));
        let sum = st.s0 + st.s1 + st.s2 + st.s3 + st.s4 + st.s5 + st.s6 + st.s7;
        st = P2State {
            s0: st.s0 * d0 + sum,
            s1: st.s1 * d1 + sum,
            s2: st.s2 * d2 + sum,
            s3: st.s3 * d3 + sum,
            s4: st.s4 * d4 + sum,
            s5: st.s5 * d5 + sum,
            s6: st.s6 * d6 + sum,
            s7: st.s7 * d7 + sum,
        };
    }
    for r in 0..EXTERNAL_HALF_ROUNDS {
        st = external_round(st, rc, RC_TERMINAL + WIDTH * r);
    }
    st
}

#[cuda_module]
pub mod poseidon2_kernels {
    use super::*;

    #[kernel]
    pub fn p2_permute_states(mut states: DisjointSlice<u64>, rc: &[u64]) {
        let i = thread::index_1d().get();
        let base = WIDTH * i;
        if base + WIDTH > states.len() {
            return;
        }
        let st = unsafe {
            P2State {
                s0: Gl::from_u64(*states.get_unchecked_mut(base)),
                s1: Gl::from_u64(*states.get_unchecked_mut(base + 1)),
                s2: Gl::from_u64(*states.get_unchecked_mut(base + 2)),
                s3: Gl::from_u64(*states.get_unchecked_mut(base + 3)),
                s4: Gl::from_u64(*states.get_unchecked_mut(base + 4)),
                s5: Gl::from_u64(*states.get_unchecked_mut(base + 5)),
                s6: Gl::from_u64(*states.get_unchecked_mut(base + 6)),
                s7: Gl::from_u64(*states.get_unchecked_mut(base + 7)),
            }
        };
        let st = permute(st, rc);
        unsafe {
            *states.get_unchecked_mut(base) = st.s0.as_canonical_u64();
            *states.get_unchecked_mut(base + 1) = st.s1.as_canonical_u64();
            *states.get_unchecked_mut(base + 2) = st.s2.as_canonical_u64();
            *states.get_unchecked_mut(base + 3) = st.s3.as_canonical_u64();
            *states.get_unchecked_mut(base + 4) = st.s4.as_canonical_u64();
            *states.get_unchecked_mut(base + 5) = st.s5.as_canonical_u64();
            *states.get_unchecked_mut(base + 6) = st.s6.as_canonical_u64();
            *states.get_unchecked_mut(base + 7) = st.s7.as_canonical_u64();
        }
    }

    /// One canonical `poseidon2_hash` per thread. This is separate from the
    /// transcript kernels: digests use the stateless sponge hash, not
    /// transcript absorb/challenge semantics.
    #[kernel]
    pub fn p2_hash_fields(fields: &[u64], offsets: &[u64], lengths: &[u64], mut out: DisjointSlice<u64>, rc: &[u64]) {
        let i = thread::index_1d().get();
        if i >= offsets.len() || i >= lengths.len() || (i + 1) * DIGEST_LEN > out.len() {
            return;
        }
        let offset = offsets[i] as usize;
        let len = lengths[i] as usize;
        if offset + len > fields.len() {
            return;
        }
        let digest = hash_fields(fields, offset, len, rc);
        let dst = i * DIGEST_LEN;
        unsafe {
            *out.get_unchecked_mut(dst) = digest[0];
            *out.get_unchecked_mut(dst + 1) = digest[1];
            *out.get_unchecked_mut(dst + 2) = digest[2];
            *out.get_unchecked_mut(dst + 3) = digest[3];
        }
    }

    /// One canonical `poseidon2_hash` per warp. Lanes 0..8 own the eight
    /// sponge lanes; the rest of each warp are intentionally idle.
    #[kernel]
    pub fn p2_hash_fields_cooperative(
        fields: &[u64],
        offsets: &[u64],
        lengths: &[u64],
        out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        const WARP_SIZE: usize = 32;
        let thread = thread::index_1d().get();
        let lane = thread % WARP_SIZE;
        if lane >= WIDTH {
            return;
        }
        let i = thread / WARP_SIZE;
        if i >= offsets.len() || i >= lengths.len() || (i + 1) * DIGEST_LEN > out.len() {
            return;
        }
        hash_fields_cooperative_at(fields, offsets[i] as usize, lengths[i] as usize, out, i, lane, rc);
    }

    #[kernel]
    pub fn p2_hash_fields_cooperative_plan(
        fields: &[u64],
        plan: &[u64],
        offsets_start: u32,
        lengths_start: u32,
        out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        const WARP_SIZE: usize = 32;
        let thread = thread::index_1d().get();
        let lane = thread % WARP_SIZE;
        if lane >= WIDTH {
            return;
        }
        let i = thread / WARP_SIZE;
        let offsets_start = offsets_start as usize;
        let lengths_start = lengths_start as usize;
        if offsets_start + i >= plan.len() || lengths_start + i >= plan.len() || (i + 1) * DIGEST_LEN > out.len() {
            return;
        }
        hash_fields_cooperative_at(
            fields,
            plan[offsets_start + i] as usize,
            plan[lengths_start + i] as usize,
            out,
            i,
            lane,
            rc,
        );
    }

    #[kernel]
    pub fn p2_hash_contiguous_cooperative(fields: &[u64], len: u64, out: DisjointSlice<u64>, rc: &[u64]) {
        let lane = thread::index_1d().get();
        if lane >= WIDTH {
            return;
        }
        hash_fields_cooperative_at(fields, 0, len as usize, out, 0, lane, rc);
    }

    fn hash_fields_cooperative_at(
        fields: &[u64],
        offset: usize,
        len: usize,
        mut out: DisjointSlice<u64>,
        i: usize,
        lane: usize,
        rc: &[u64],
    ) {
        if offset + len > fields.len() || (i + 1) * DIGEST_LEN > out.len() {
            return;
        }

        let mut state = Gl::ZERO;
        let mut pos = 0usize;
        while pos < len {
            let remaining = len - pos;
            let take = if remaining < RATE { remaining } else { RATE };
            if lane < take {
                state = state + Gl::from_u64(fields[offset + pos + lane]);
            }
            state = permute_cooperative(state, lane, rc);
            pos += take;
        }
        if lane == 0 {
            state = state + Gl::ONE;
        }
        state = permute_cooperative(state, lane, rc);
        if lane < DIGEST_LEN {
            unsafe {
                *out.get_unchecked_mut(i * DIGEST_LEN + lane) = state.as_canonical_u64();
            }
        }
    }

    /// Mirrors `Poseidon2Transcript`: `absorb_elem` permutes lazily when the
    /// cursor reaches RATE; `challenge` absorbs ONE, permutes, squeezes up to
    /// DIGEST_LEN lanes per iteration.
    #[kernel]
    pub fn p2_transcript_ops(
        mut st: DisjointSlice<u64>,
        ops: &[u64],
        payload: &[u64],
        mut out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        if thread::index_1d().get() != 0 || st.len() < ST_WORDS {
            return;
        }
        let mut state = unsafe {
            P2State {
                s0: Gl::from_u64(*st.get_unchecked_mut(0)),
                s1: Gl::from_u64(*st.get_unchecked_mut(1)),
                s2: Gl::from_u64(*st.get_unchecked_mut(2)),
                s3: Gl::from_u64(*st.get_unchecked_mut(3)),
                s4: Gl::from_u64(*st.get_unchecked_mut(4)),
                s5: Gl::from_u64(*st.get_unchecked_mut(5)),
                s6: Gl::from_u64(*st.get_unchecked_mut(6)),
                s7: Gl::from_u64(*st.get_unchecked_mut(7)),
            }
        };
        let mut cursor = unsafe { *st.get_unchecked_mut(WIDTH) } as usize;
        let mut pay = 0usize;
        let mut o = 0usize;
        for k in 0..ops.len() / 2 {
            let code = ops[2 * k];
            let arg = ops[2 * k + 1] as usize;
            if code == OP_ABSORB {
                for _ in 0..arg {
                    if pay >= payload.len() {
                        return;
                    }
                    let v = payload[pay];
                    pay += 1;
                    absorb_word(&mut state, &mut cursor, v, rc);
                }
            } else if code == OP_CHALLENGE {
                if !squeeze_challenges(&mut state, &mut cursor, arg, o, &mut out, rc) {
                    return;
                }
                o += arg;
            }
        }
        store_state(st, state, cursor);
    }

    /// Extended IO op stream. Ops are triples `[code, len, offset]`.
    #[kernel]
    pub fn p2_transcript_io_ops(
        mut st: DisjointSlice<u64>,
        ops: &[u64],
        host_payload: &[u64],
        device_payload: &[u64],
        mut host_out: DisjointSlice<u64>,
        mut device_out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        if thread::index_1d().get() != 0 || st.len() < ST_WORDS {
            return;
        }
        let mut state = load_state(&mut st);
        let mut cursor = unsafe { *st.get_unchecked_mut(WIDTH) } as usize;

        for k in 0..ops.len() / 3 {
            let code = ops[3 * k];
            let len = ops[3 * k + 1] as usize;
            let offset = ops[3 * k + 2] as usize;
            if code == OP_ABSORB {
                if offset + len > host_payload.len() {
                    return;
                }
                for i in 0..len {
                    absorb_word(&mut state, &mut cursor, host_payload[offset + i], rc);
                }
            } else if code == OP_ABSORB_DEVICE {
                if offset + len > device_payload.len() {
                    return;
                }
                for i in 0..len {
                    absorb_word(&mut state, &mut cursor, device_payload[offset + i], rc);
                }
            } else if code == OP_CHALLENGE {
                if !squeeze_challenges(&mut state, &mut cursor, len, offset, &mut host_out, rc) {
                    return;
                }
            } else if code == OP_CHALLENGE_DEVICE
                && !squeeze_challenges(&mut state, &mut cursor, len, offset, &mut device_out, rc)
            {
                return;
            }
        }
        store_state(st, state, cursor);
    }

    /// Specialized sumcheck step: append_fields_raw(device_payload slice),
    /// then challenge_fields_raw(2) into `device_out[out_offset..]`.
    #[kernel]
    pub fn p2_transcript_absorb_device_challenge(
        mut st: DisjointSlice<u64>,
        len_prefix: u64,
        device_payload: &[u64],
        payload_offset: u32,
        payload_len: u32,
        mut device_out: DisjointSlice<u64>,
        out_offset: u32,
        rc: &[u64],
    ) {
        let lane = thread::index_1d().get();
        if lane >= WIDTH || st.len() < ST_WORDS {
            return;
        }
        let offset = payload_offset as usize;
        let len = payload_len as usize;
        let out_offset = out_offset as usize;
        if offset + len > device_payload.len() || out_offset + 2 > device_out.len() {
            return;
        }
        let mut state = Gl::from_u64(unsafe { *st.get_unchecked_mut(lane) });
        let cursor_word = if lane == 0 {
            unsafe { *st.get_unchecked_mut(WIDTH) }
        } else {
            0
        };
        let mut cursor = warp::shuffle_u64_sync(POSEIDON_TILE_MASK, cursor_word, 0) as usize;
        absorb_word_cooperative(&mut state, lane, &mut cursor, len_prefix, rc);
        for i in 0..len {
            absorb_word_cooperative(&mut state, lane, &mut cursor, device_payload[offset + i], rc);
        }
        if cursor >= RATE {
            state = permute_cooperative(state, lane, rc);
            cursor = 0;
        }
        if lane == cursor {
            state = Gl::ONE;
        }
        state = permute_cooperative(state, lane, rc);
        cursor = 0;
        if lane < 2 {
            unsafe {
                *device_out.get_unchecked_mut(out_offset + lane) = state.as_canonical_u64();
            }
        }
        unsafe {
            *st.get_unchecked_mut(lane) = state.as_canonical_u64();
            if lane == 0 {
                *st.get_unchecked_mut(WIDTH) = cursor as u64;
            }
        }
    }

    /// Device mirror of `sample_rot_rhos_n` for the production Goldilocks
    /// profile: alphabet `[-2, -1, 0, 1, 2]`, D = 54. Writes only the
    /// rotation-ring coefficients (`cf(a_i)`); callers build any host matrix
    /// view from these same coefficients.
    #[kernel]
    pub fn p2_transcript_sample_rlc_rhos(
        mut st: DisjointSlice<u64>,
        count: u32,
        mut coeffs_out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        let lane = thread::index_1d().get();
        if lane >= WIDTH || st.len() < ST_WORDS {
            return;
        }
        let count = count as usize;
        if coeffs_out.len() < count * 54 {
            return;
        }

        let mut state = Gl::from_u64(unsafe { *st.get_unchecked_mut(lane) });
        let cursor_word = if lane == 0 {
            unsafe { *st.get_unchecked_mut(WIDTH) }
        } else {
            0
        };
        let mut cursor = warp::shuffle_u64_sync(POSEIDON_TILE_MASK, cursor_word, 0) as usize;

        sample_rlc_rhos_cooperative(&mut state, lane, &mut cursor, count, &mut coeffs_out, rc);

        unsafe {
            *st.get_unchecked_mut(lane) = state.as_canonical_u64();
            if lane == 0 {
                *st.get_unchecked_mut(WIDTH) = cursor as u64;
            }
        }
    }

    /// Canonical `append_fields(label, resident_fields)` followed immediately
    /// by Π_RLC rho sampling on the same device sponge.
    #[kernel]
    pub fn p2_transcript_bind_device_fields_sample_rlc_rhos(
        mut st: DisjointSlice<u64>,
        prefix: DeviceFieldBindPrefix,
        device_fields: &[u64],
        count: u32,
        mut coeffs_out: DisjointSlice<u64>,
        rc: &[u64],
    ) {
        let lane = thread::index_1d().get();
        if lane >= WIDTH
            || st.len() < ST_WORDS
            || prefix.word_count as usize > MAX_DEVICE_FIELD_BIND_PREFIX_WORDS
            || prefix.field_count as usize > device_fields.len()
            || coeffs_out.len() < count as usize * 54
        {
            return;
        }

        let mut state = Gl::from_u64(unsafe { *st.get_unchecked_mut(lane) });
        let cursor_word = if lane == 0 {
            unsafe { *st.get_unchecked_mut(WIDTH) }
        } else {
            0
        };
        let mut cursor = warp::shuffle_u64_sync(POSEIDON_TILE_MASK, cursor_word, 0) as usize;
        for i in 0..prefix.word_count as usize {
            absorb_word_cooperative(&mut state, lane, &mut cursor, prefix.words[i], rc);
        }
        absorb_word_cooperative(&mut state, lane, &mut cursor, prefix.field_count, rc);
        for i in 0..prefix.field_count as usize {
            absorb_word_cooperative(&mut state, lane, &mut cursor, device_fields[i], rc);
        }
        sample_rlc_rhos_cooperative(&mut state, lane, &mut cursor, count as usize, &mut coeffs_out, rc);

        unsafe {
            *st.get_unchecked_mut(lane) = state.as_canonical_u64();
            if lane == 0 {
                *st.get_unchecked_mut(WIDTH) = cursor as u64;
            }
        }
    }

    fn sample_rlc_rhos_cooperative(
        state: &mut Gl,
        lane: usize,
        cursor: &mut usize,
        count: usize,
        coeffs_out: &mut DisjointSlice<u64>,
        rc: &[u64],
    ) {
        for rho in 0..count {
            absorb_word_cooperative(state, lane, cursor, 2, rc);
            absorb_word_cooperative(state, lane, cursor, 0, rc);
            absorb_word_cooperative(state, lane, cursor, rho as u64, rc);
            let mut written = 0usize;
            let mut ctr = rho as u64;
            while written < 54 {
                absorb_word_cooperative(state, lane, cursor, 2, rc);
                absorb_word_cooperative(state, lane, cursor, 1, rc);
                absorb_word_cooperative(state, lane, cursor, ctr, rc);
                absorb_word_cooperative(state, lane, cursor, 1, rc);
                *state = permute_cooperative(*state, lane, rc);
                *cursor = 0;

                for digest_lane in 0..DIGEST_LEN {
                    let limb = warp::shuffle_u64_sync(POSEIDON_TILE_MASK, state.as_canonical_u64(), digest_lane as u32);
                    let mut shift = 0u32;
                    while shift < 64 && written < 54 {
                        let chunk = ((limb >> shift) & 0xffff) as u32;
                        if chunk < 65_535 {
                            let word = match chunk % 5 {
                                0 => GOLDILOCKS_MODULUS - 2,
                                1 => GOLDILOCKS_MODULUS - 1,
                                2 => 0,
                                3 => 1,
                                _ => 2,
                            };
                            if lane == 0 {
                                unsafe {
                                    *coeffs_out.get_unchecked_mut(rho * 54 + written) = word;
                                }
                            }
                            written += 1;
                        }
                        shift += 16;
                    }
                }
                ctr = ctr.wrapping_add(1);
            }
        }
    }

    fn load_state(st: &mut DisjointSlice<u64>) -> P2State {
        unsafe {
            P2State {
                s0: Gl::from_u64(*st.get_unchecked_mut(0)),
                s1: Gl::from_u64(*st.get_unchecked_mut(1)),
                s2: Gl::from_u64(*st.get_unchecked_mut(2)),
                s3: Gl::from_u64(*st.get_unchecked_mut(3)),
                s4: Gl::from_u64(*st.get_unchecked_mut(4)),
                s5: Gl::from_u64(*st.get_unchecked_mut(5)),
                s6: Gl::from_u64(*st.get_unchecked_mut(6)),
                s7: Gl::from_u64(*st.get_unchecked_mut(7)),
            }
        }
    }

    fn store_state(mut st: DisjointSlice<u64>, state: P2State, cursor: usize) {
        unsafe {
            *st.get_unchecked_mut(0) = state.s0.as_canonical_u64();
            *st.get_unchecked_mut(1) = state.s1.as_canonical_u64();
            *st.get_unchecked_mut(2) = state.s2.as_canonical_u64();
            *st.get_unchecked_mut(3) = state.s3.as_canonical_u64();
            *st.get_unchecked_mut(4) = state.s4.as_canonical_u64();
            *st.get_unchecked_mut(5) = state.s5.as_canonical_u64();
            *st.get_unchecked_mut(6) = state.s6.as_canonical_u64();
            *st.get_unchecked_mut(7) = state.s7.as_canonical_u64();
            *st.get_unchecked_mut(WIDTH) = cursor as u64;
        }
    }

    fn hash_fields(fields: &[u64], offset: usize, len: usize, rc: &[u64]) -> [u64; DIGEST_LEN] {
        let mut state = P2State {
            s0: Gl::ZERO,
            s1: Gl::ZERO,
            s2: Gl::ZERO,
            s3: Gl::ZERO,
            s4: Gl::ZERO,
            s5: Gl::ZERO,
            s6: Gl::ZERO,
            s7: Gl::ZERO,
        };
        let mut pos = 0usize;
        while pos < len {
            let remaining = len - pos;
            let take = if remaining < RATE { remaining } else { RATE };
            for lane in 0..take {
                add_rate_lane(&mut state, lane, fields[offset + pos + lane]);
            }
            state = permute(state, rc);
            pos += take;
        }
        state.s0 = state.s0 + Gl::ONE;
        state = permute(state, rc);
        [
            state.s0.as_canonical_u64(),
            state.s1.as_canonical_u64(),
            state.s2.as_canonical_u64(),
            state.s3.as_canonical_u64(),
        ]
    }

    const POSEIDON_TILE_MASK: u32 = (1 << WIDTH) - 1;

    fn absorb_word_cooperative(state: &mut Gl, lane: usize, cursor: &mut usize, word: u64, rc: &[u64]) {
        if *cursor >= RATE {
            *state = permute_cooperative(*state, lane, rc);
            *cursor = 0;
        }
        if lane == *cursor {
            *state = Gl::from_u64(word);
        }
        *cursor += 1;
    }

    fn permute_cooperative(mut state: Gl, lane: usize, rc: &[u64]) -> Gl {
        state = mds_light_cooperative(state, lane);
        for r in 0..EXTERNAL_HALF_ROUNDS {
            state = sbox(state + Gl::from_u64(rc[RC_INITIAL + WIDTH * r + lane]));
            state = mds_light_cooperative(state, lane);
        }
        for r in 0..INTERNAL_ROUNDS {
            if lane == 0 {
                state = sbox(state + Gl::from_u64(rc[RC_INTERNAL + r]));
            }
            let mut sum = state;
            sum = sum + shuffle_xor_gl(sum, 1);
            sum = sum + shuffle_xor_gl(sum, 2);
            sum = sum + shuffle_xor_gl(sum, 4);
            state = state * Gl::from_u64(rc[RC_DIAG + lane]) + sum;
        }
        for r in 0..EXTERNAL_HALF_ROUNDS {
            state = sbox(state + Gl::from_u64(rc[RC_TERMINAL + WIDTH * r + lane]));
            state = mds_light_cooperative(state, lane);
        }
        state
    }

    fn mds_light_cooperative(state: Gl, lane: usize) -> Gl {
        let half = lane & 4;
        let x0 = shuffle_gl(state, half);
        let x1 = shuffle_gl(state, half + 1);
        let x2 = shuffle_gl(state, half + 2);
        let x3 = shuffle_gl(state, half + 3);
        let t01 = x0 + x1;
        let t23 = x2 + x3;
        let t0123 = t01 + t23;
        let t01123 = t0123 + x1;
        let t01233 = t0123 + x3;
        let local = match lane & 3 {
            0 => t01123 + t01,
            1 => t01123 + x2 + x2,
            2 => t01233 + t23,
            _ => t01233 + x0 + x0,
        };
        let paired = shuffle_gl(local, lane ^ 4);
        local + local + paired
    }

    fn shuffle_gl(value: Gl, source_lane: usize) -> Gl {
        Gl::from_u64(warp::shuffle_u64_sync(
            POSEIDON_TILE_MASK,
            value.as_canonical_u64(),
            source_lane as u32,
        ))
    }

    fn shuffle_xor_gl(value: Gl, lane_mask: u32) -> Gl {
        Gl::from_u64(warp::shuffle_xor_u64_sync(
            POSEIDON_TILE_MASK,
            value.as_canonical_u64(),
            lane_mask,
        ))
    }

    fn add_rate_lane(state: &mut P2State, lane: usize, word: u64) {
        let value = Gl::from_u64(word);
        match lane {
            0 => state.s0 = state.s0 + value,
            1 => state.s1 = state.s1 + value,
            2 => state.s2 = state.s2 + value,
            _ => state.s3 = state.s3 + value,
        }
    }

    fn absorb_word(state: &mut P2State, cursor: &mut usize, word: u64, rc: &[u64]) {
        let v = Gl::from_u64(word);
        if *cursor >= RATE {
            *state = permute(*state, rc);
            *cursor = 0;
        }
        match *cursor {
            0 => state.s0 = v,
            1 => state.s1 = v,
            2 => state.s2 = v,
            _ => state.s3 = v,
        }
        *cursor += 1;
    }

    fn squeeze_challenges(
        state: &mut P2State,
        cursor: &mut usize,
        len: usize,
        offset: usize,
        out: &mut DisjointSlice<u64>,
        rc: &[u64],
    ) -> bool {
        let mut produced = 0usize;
        while produced < len {
            if *cursor >= RATE {
                *state = permute(*state, rc);
                *cursor = 0;
            }
            match *cursor {
                0 => state.s0 = Gl::ONE,
                1 => state.s1 = Gl::ONE,
                2 => state.s2 = Gl::ONE,
                _ => state.s3 = Gl::ONE,
            }
            *state = permute(*state, rc);
            *cursor = 0;
            let rem = len - produced;
            let take = if rem < DIGEST_LEN { rem } else { DIGEST_LEN };
            let dst = offset + produced;
            if dst + take > out.len() {
                return false;
            }
            unsafe {
                *out.get_unchecked_mut(dst) = state.s0.as_canonical_u64();
                if take > 1 {
                    *out.get_unchecked_mut(dst + 1) = state.s1.as_canonical_u64();
                }
                if take > 2 {
                    *out.get_unchecked_mut(dst + 2) = state.s2.as_canonical_u64();
                }
                if take > 3 {
                    *out.get_unchecked_mut(dst + 3) = state.s3.as_canonical_u64();
                }
            }
            produced += take;
        }
        true
    }
}
