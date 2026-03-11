from nightstream_gpu import ffi


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
fn nightstream_gpu_fe_create(req_addr: UInt, out_addr: UInt) -> Int32:
    return ffi.fe_create(req_addr, out_addr)


@export("nightstream_gpu_fe_destroy", ABI="C")
fn nightstream_gpu_fe_destroy(session: UInt, evaluator: UInt) -> Int32:
    return ffi.fe_destroy(session, evaluator)


@export("nightstream_gpu_fe_evals_at", ABI="C")
fn nightstream_gpu_fe_evals_at(req_addr: UInt) -> Int32:
    return ffi.fe_evals_at(req_addr)


@export("nightstream_gpu_fe_fold", ABI="C")
fn nightstream_gpu_fe_fold(req_addr: UInt) -> Int32:
    return ffi.fe_fold(req_addr)


@export("nightstream_gpu_nc_create", ABI="C")
fn nightstream_gpu_nc_create(req_addr: UInt, out_addr: UInt) -> Int32:
    return ffi.nc_create(req_addr, out_addr)


@export("nightstream_gpu_nc_destroy", ABI="C")
fn nightstream_gpu_nc_destroy(session: UInt, evaluator: UInt) -> Int32:
    return ffi.nc_destroy(session, evaluator)


@export("nightstream_gpu_nc_evals_at", ABI="C")
fn nightstream_gpu_nc_evals_at(req_addr: UInt) -> Int32:
    return ffi.nc_evals_at(req_addr)


@export("nightstream_gpu_nc_fold", ABI="C")
fn nightstream_gpu_nc_fold(req_addr: UInt) -> Int32:
    return ffi.nc_fold(req_addr)


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
