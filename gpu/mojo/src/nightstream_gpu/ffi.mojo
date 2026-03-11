from memory import UnsafePointer
from sys import has_accelerator
from nightstream_gpu import field, poseidon, ring, sumcheck, superneo


alias DEVICE_API_CPU = 0
alias DEVICE_API_METAL = 1
alias DEVICE_API_CUDA = 2
alias DEVICE_API_HIP = 3
alias STATUS_OK = 0
alias STATUS_UNAVAILABLE = -1
alias SESSION_HANDLE_MAGIC = UInt(0x4E53000000000000)


fn request_word(req_addr: UInt) -> UInt64:
    var req_words = UnsafePointer[UInt64](unchecked_downcast_value=Int(req_addr))
    return req_words[0]


fn accelerator_requested(req_word: UInt64) -> Bool:
    return req_word != UInt64(0)


fn device_api_available(req_word: UInt64) -> Bool:
    if not accelerator_requested(req_word):
        return True

    return has_accelerator()


fn unpack_api(req_word: UInt64) -> UInt32:
    return UInt32(req_word & UInt64(0xFFFF_FFFF))


fn unpack_device_id(req_word: UInt64) -> UInt32:
    return UInt32(req_word >> 32)


fn pack_probe_response(status: Int32, available: Bool) -> UInt64:
    var available_flag = UInt64(0)
    if available:
        available_flag = 1
    return UInt64(UInt32(status)) | (available_flag << 32)


fn abi_version() -> UInt32:
    return 1


fn device_probe(
    req_addr: UInt,
    out_words: UnsafePointer[mut=True, UInt64],
) -> Int32:
    var available = device_api_available(request_word(req_addr))
    out_words[0] = pack_probe_response(Int32(STATUS_OK), available)
    return STATUS_OK


fn session_open(
    req_addr: UInt,
    handle_ptr: UnsafePointer[mut=True, UInt64],
) -> Int32:
    var req_word = request_word(req_addr)
    var api = unpack_api(req_word)
    var device_id = unpack_device_id(req_word)
    if not device_api_available(req_word):
        handle_ptr[0] = 0
        return STATUS_UNAVAILABLE

    if accelerator_requested(req_word) and api == UInt32(DEVICE_API_CPU):
        api = UInt32(DEVICE_API_CUDA)

    handle_ptr[0] = (
        UInt64(SESSION_HANDLE_MAGIC)
        | (UInt64(api) << 32)
        | UInt64(device_id)
    )
    return STATUS_OK


fn session_close(_session: UInt) -> Int32:
    return STATUS_OK


fn fe_create(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    return sumcheck.fe_create(session, snapshot_words, snapshot_len, out_handle)


fn fe_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return sumcheck.fe_destroy(_session, _evaluator)


fn fe_evals_at(
    session: UInt64,
    evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    return sumcheck.fe_evals_at(session, evaluator, snapshot_words, snapshot_len, points_words, points_len, out_ptr, out_len)


fn fe_fold(session: UInt, evaluator: UInt, challenge: sumcheck.KVal) -> Int32:
    return sumcheck.fe_fold(session, evaluator, challenge)


fn nc_create(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    return sumcheck.nc_create(session, snapshot_words, snapshot_len, out_handle)


fn nc_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return sumcheck.nc_destroy(_session, _evaluator)


fn nc_evals_at(
    session: UInt64,
    evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    return sumcheck.nc_evals_at(session, evaluator, snapshot_words, snapshot_len, points_words, points_len, out_ptr, out_len)


fn nc_fold(session: UInt, evaluator: UInt, challenge: sumcheck.KVal) -> Int32:
    return sumcheck.nc_fold(session, evaluator, challenge)


fn scaffold_sanity() -> Bool:
    return (
        field.scaffold_ready()
        and ring.scaffold_ready()
        and superneo.scaffold_ready()
        and sumcheck.scaffold_ready()
        and poseidon.scaffold_ready()
    )


fn debug_snapshot_head(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_words: UnsafePointer[mut=True, UInt64],
    out_len: UInt32,
) -> Int32:
    return sumcheck.debug_snapshot_head(session, snapshot_words, snapshot_len, out_words, out_len)


fn poseidon2_permute_u64x8(
    session: UInt,
    state: UnsafePointer[mut=True, UInt64],
    width: UInt32,
) -> Int32:
    return poseidon.poseidon2_permute_u64x8(session, state, width)


fn poseidon2_permute_batch_u64x8(
    session: UInt,
    state_words: UnsafePointer[mut=True, UInt64],
    num_states: UInt32,
    width: UInt32,
) -> Int32:
    return poseidon.poseidon2_permute_batch_u64x8(session, state_words, num_states, width)
