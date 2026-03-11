pub const ABI_VERSION: u32 = 1;

pub const ABI_VERSION_SYMBOL: &[u8] = b"nightstream_gpu_abi_version\0";
pub const DEVICE_PROBE_SYMBOL: &[u8] = b"nightstream_gpu_device_probe\0";
pub const SESSION_OPEN_SYMBOL: &[u8] = b"nightstream_gpu_session_open\0";
pub const SESSION_CLOSE_SYMBOL: &[u8] = b"nightstream_gpu_session_close\0";
pub const FE_CREATE_SYMBOL: &[u8] = b"nightstream_gpu_fe_create\0";
pub const FE_DESTROY_SYMBOL: &[u8] = b"nightstream_gpu_fe_destroy\0";
pub const FE_EVALS_AT_SYMBOL: &[u8] = b"nightstream_gpu_fe_evals_at\0";
pub const FE_FOLD_SYMBOL: &[u8] = b"nightstream_gpu_fe_fold\0";
pub const NC_CREATE_SYMBOL: &[u8] = b"nightstream_gpu_nc_create\0";
pub const NC_DESTROY_SYMBOL: &[u8] = b"nightstream_gpu_nc_destroy\0";
pub const NC_EVALS_AT_SYMBOL: &[u8] = b"nightstream_gpu_nc_evals_at\0";
pub const NC_FOLD_SYMBOL: &[u8] = b"nightstream_gpu_nc_fold\0";
pub const POSEIDON2_PERMUTE_SYMBOL: &[u8] = b"nightstream_gpu_poseidon2_permute_u64x8\0";
pub const POSEIDON2_PERMUTE_BATCH_SYMBOL: &[u8] = b"nightstream_gpu_poseidon2_permute_batch_u64x8\0";
pub const POSEIDON2_STATE_WIDTH: usize = 8;

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatFq {
    pub limb: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FlatK {
    pub re: u64,
    pub im: u64,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct FlatRq {
    pub coeffs: [u64; 54],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct DeviceResponse {
    pub status: i32,
    pub available: i32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SessionRequest {
    pub api: u32,
    pub device_id: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct SnapshotRequest {
    pub session: usize,
    pub snapshot_ptr: *const u8,
    pub snapshot_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct EvalsRequest {
    pub session: usize,
    pub evaluator: usize,
    pub snapshot_ptr: *const u8,
    pub snapshot_len: usize,
    pub points_ptr: *const FlatK,
    pub points_len: usize,
    pub out_ptr: *mut FlatK,
    pub out_len: usize,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq)]
pub struct FoldRequest {
    pub session: usize,
    pub evaluator: usize,
    pub snapshot_ptr: *mut u8,
    pub snapshot_len: usize,
    pub challenge: FlatK,
}
