from nightstream_gpu import ffi, sumcheck


@export("nightstream_gpu_abi_version", ABI="C")
fn nightstream_gpu_abi_version() -> UInt32:
    return ffi.abi_version()


@export("nightstream_gpu_device_probe", ABI="C")
fn nightstream_gpu_device_probe(req_addr: UInt, out_words: UnsafePointer[mut=True, UInt64]) -> Int32:
    return ffi.device_probe(req_addr, out_words)


@export("nightstream_gpu_session_open", ABI="C")
fn nightstream_gpu_session_open(req_addr: UInt, handle_ptr: UnsafePointer[mut=True, UInt64]) -> Int32:
    return ffi.session_open(req_addr, handle_ptr)


@export("nightstream_gpu_session_close", ABI="C")
fn nightstream_gpu_session_close(session: UInt) -> Int32:
    return ffi.session_close(session)


@export("nightstream_gpu_fe_create", ABI="C")
fn nightstream_gpu_fe_create(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    return ffi.fe_create(session, snapshot_words, snapshot_len, out_handle)


@export("nightstream_gpu_fe_destroy", ABI="C")
fn nightstream_gpu_fe_destroy(session: UInt, evaluator: UInt) -> Int32:
    return ffi.fe_destroy(session, evaluator)


@export("nightstream_gpu_fe_evals_at", ABI="C")
fn nightstream_gpu_fe_evals_at(
    session: UInt64,
    evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    return ffi.fe_evals_at(session, evaluator, snapshot_words, snapshot_len, points_words, points_len, out_ptr, out_len)


@export("nightstream_gpu_fe_fold", ABI="C")
fn nightstream_gpu_fe_fold(session: UInt, evaluator: UInt, challenge_re: UInt64, challenge_im: UInt64) -> Int32:
    return ffi.fe_fold(session, evaluator, sumcheck.KVal(challenge_re, challenge_im))


@export("nightstream_gpu_nc_create", ABI="C")
fn nightstream_gpu_nc_create(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    return ffi.nc_create(session, snapshot_words, snapshot_len, out_handle)


@export("nightstream_gpu_nc_destroy", ABI="C")
fn nightstream_gpu_nc_destroy(session: UInt, evaluator: UInt) -> Int32:
    return ffi.nc_destroy(session, evaluator)


@export("nightstream_gpu_nc_evals_at", ABI="C")
fn nightstream_gpu_nc_evals_at(
    session: UInt64,
    evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    return ffi.nc_evals_at(session, evaluator, snapshot_words, snapshot_len, points_words, points_len, out_ptr, out_len)


@export("nightstream_gpu_nc_fold", ABI="C")
fn nightstream_gpu_nc_fold(session: UInt, evaluator: UInt, challenge_re: UInt64, challenge_im: UInt64) -> Int32:
    return ffi.nc_fold(session, evaluator, sumcheck.KVal(challenge_re, challenge_im))


@export("nightstream_gpu_debug_snapshot_head", ABI="C")
fn nightstream_gpu_debug_snapshot_head(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_words: UnsafePointer[mut=True, UInt64],
    out_len: UInt32,
) -> Int32:
    return ffi.debug_snapshot_head(session, snapshot_words, snapshot_len, out_words, out_len)


@export("nightstream_gpu_poseidon2_permute_u64x8", ABI="C")
fn nightstream_gpu_poseidon2_permute_u64x8(
    session: UInt,
    state: UnsafePointer[mut=True, UInt64],
    width: UInt32,
) -> Int32:
    return ffi.poseidon2_permute_u64x8(session, state, width)


@export("nightstream_gpu_poseidon2_permute_batch_u64x8", ABI="C")
fn nightstream_gpu_poseidon2_permute_batch_u64x8(
    session: UInt,
    state_words: UnsafePointer[mut=True, UInt64],
    num_states: UInt32,
    width: UInt32,
) -> Int32:
    return ffi.poseidon2_permute_batch_u64x8(session, state_words, num_states, width)
