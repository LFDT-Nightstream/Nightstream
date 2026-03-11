use std::sync::atomic::{AtomicUsize, Ordering};

#[repr(C)]
pub struct DeviceRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
pub struct DeviceResponse {
    pub status: i32,
    pub available: i32,
}

#[repr(C)]
pub struct SessionRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
pub struct SnapshotRequest {
    pub session: usize,
    pub snapshot_ptr: *const u8,
    pub snapshot_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct FlatK {
    pub re: u64,
    pub im: u64,
}

#[repr(C)]
pub struct EvalsRequest {
    pub session: usize,
    pub evaluator: usize,
    pub points_ptr: *const FlatK,
    pub points_len: usize,
    pub out_ptr: *mut FlatK,
    pub out_len: usize,
}

#[repr(C)]
pub struct FoldRequest {
    pub session: usize,
    pub evaluator: usize,
    pub challenge: FlatK,
}

#[repr(C)]
pub struct FlatFq {
    pub limb: u64,
}

static POSEIDON2_BATCH_CALLS: AtomicUsize = AtomicUsize::new(0);
static SESSION_OPEN_CALLS: AtomicUsize = AtomicUsize::new(0);

#[no_mangle]
pub extern "C" fn nightstream_gpu_abi_version() -> u32 {
    1
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_device_probe(_req: *const DeviceRequest, out: *mut DeviceResponse) -> i32 {
    unsafe {
        if let Some(out) = out.as_mut() {
            out.status = 0;
            out.available = 1;
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_session_open(req: *const SessionRequest, out_handle: *mut usize) -> i32 {
    SESSION_OPEN_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        let Some(req) = req.as_ref() else {
            return -1;
        };
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        *out_handle = ((req.api as usize) << 32) | (req.device_id as usize + 1);
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_session_close(_session: usize) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_create(req: *const SnapshotRequest, out_handle: *mut usize) -> i32 {
    unsafe {
        let Some(req) = req.as_ref() else {
            return -1;
        };
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        *out_handle = req.session ^ 0xFE;
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_destroy(_session: usize, _evaluator: usize) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_evals_at(req: *const EvalsRequest) -> i32 {
    unsafe {
        let Some(req) = req.as_ref() else {
            return -1;
        };
        let n = req.points_len.min(req.out_len);
        if n == 0 {
            return 0;
        }
        let points = std::slice::from_raw_parts(req.points_ptr, n);
        let out = std::slice::from_raw_parts_mut(req.out_ptr, n);
        out.copy_from_slice(points);
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_fe_fold(_req: *const FoldRequest) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_create(req: *const SnapshotRequest, out_handle: *mut usize) -> i32 {
    unsafe {
        let Some(req) = req.as_ref() else {
            return -1;
        };
        let Some(out_handle) = out_handle.as_mut() else {
            return -2;
        };
        *out_handle = req.session ^ 0xAC;
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_destroy(_session: usize, _evaluator: usize) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_evals_at(req: *const EvalsRequest) -> i32 {
    nightstream_gpu_fe_evals_at(req)
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_nc_fold(_req: *const FoldRequest) -> i32 {
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_poseidon2_permute_u64x8(
    _session: usize,
    state_ptr: *mut FlatFq,
    width: u32,
) -> i32 {
    unsafe {
        if width != 8 {
            return -2;
        }
        let Some(_state_ptr) = state_ptr.as_ref() else {
            return -3;
        };
        let state = std::slice::from_raw_parts_mut(state_ptr, width as usize);
        let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();

        let mut input = [Goldilocks::ZERO; 8];
        for (dst, src) in input.iter_mut().zip(state.iter()) {
            *dst = Goldilocks::from_u64(src.limb);
        }
        let result = perm.permute(input);
        for (dst, src) in state.iter_mut().zip(result.iter()) {
            dst.limb = src.as_canonical_u64();
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_poseidon2_permute_batch_u64x8(
    _session: usize,
    state_ptr: *mut FlatFq,
    num_states: u32,
    width: u32,
) -> i32 {
    POSEIDON2_BATCH_CALLS.fetch_add(1, Ordering::Relaxed);
    unsafe {
        if width != 8 {
            return -2;
        }
        let Some(_state_ptr) = state_ptr.as_ref() else {
            return -3;
        };
        let total_words = width as usize * num_states as usize;
        let state_words = std::slice::from_raw_parts_mut(state_ptr, total_words);
        let perm = neo_ccs::crypto::poseidon2_goldilocks::permutation();

        for state in state_words.chunks_exact_mut(width as usize) {
            let mut input = [Goldilocks::ZERO; 8];
            for (dst, src) in input.iter_mut().zip(state.iter()) {
                *dst = Goldilocks::from_u64(src.limb);
            }
            let result = perm.permute(input);
            for (dst, src) in state.iter_mut().zip(result.iter()) {
                dst.limb = src.as_canonical_u64();
            }
        }
    }
    0
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_reset_counters() {
    POSEIDON2_BATCH_CALLS.store(0, Ordering::Relaxed);
    SESSION_OPEN_CALLS.store(0, Ordering::Relaxed);
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_poseidon2_batch_calls() -> usize {
    POSEIDON2_BATCH_CALLS.load(Ordering::Relaxed)
}

#[no_mangle]
pub extern "C" fn nightstream_gpu_test_session_open_calls() -> usize {
    SESSION_OPEN_CALLS.load(Ordering::Relaxed)
}
use p3_field::{PrimeCharacteristicRing, PrimeField64};
use p3_goldilocks::Goldilocks;
use p3_symmetric::Permutation;
