from memory import UnsafePointer
from nightstream_gpu import field, poseidon, ring, sumcheck, superneo


alias DEVICE_API_CPU = 0
alias DEVICE_API_METAL = 1
alias DEVICE_API_CUDA = 2
alias DEVICE_API_HIP = 3
alias STATUS_OK = 0
alias STATUS_UNAVAILABLE = -1
alias SESSION_HANDLE_MAGIC = UInt(0x4E53000000000000)


fn device_api_available(api: UInt32) -> Bool:
    if api == UInt32(DEVICE_API_CPU):
        return True
    if api == UInt32(DEVICE_API_METAL):
        return True
    if api == UInt32(DEVICE_API_CUDA):
        return True
    if api == UInt32(DEVICE_API_HIP):
        return True
    return False


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
    var req_words = UnsafePointer[UInt64](unchecked_downcast_value=Int(req_addr))
    var available = device_api_available(unpack_api(req_words[0]))
    out_words[0] = pack_probe_response(Int32(STATUS_OK), available)
    return STATUS_OK


fn session_open(
    req_addr: UInt,
    handle_ptr: UnsafePointer[mut=True, UInt64],
) -> Int32:
    var req_words = UnsafePointer[UInt64](unchecked_downcast_value=Int(req_addr))
    var req_word = req_words[0]
    var api = unpack_api(req_word)
    var device_id = unpack_device_id(req_word)
    if not device_api_available(api):
        handle_ptr[0] = 0
        return STATUS_UNAVAILABLE

    handle_ptr[0] = (
        UInt64(SESSION_HANDLE_MAGIC)
        | (UInt64(api) << 32)
        | UInt64(device_id)
    )
    return STATUS_OK


fn session_close(_session: UInt) -> Int32:
    return STATUS_OK


fn fe_create(
    _req_addr: UInt,
    _out_addr: UInt,
) -> Int32:
    return -1


fn fe_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return 0


fn fe_evals_at(_req_addr: UInt) -> Int32:
    return -1


fn fe_fold(_req_addr: UInt) -> Int32:
    return -1


fn nc_create(
    _req_addr: UInt,
    _out_addr: UInt,
) -> Int32:
    return -1


fn nc_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return 0


fn nc_evals_at(_req_addr: UInt) -> Int32:
    return -1


fn nc_fold(_req_addr: UInt) -> Int32:
    return -1


fn scaffold_sanity() -> Bool:
    return (
        field.scaffold_ready()
        and ring.scaffold_ready()
        and superneo.scaffold_ready()
        and sumcheck.scaffold_ready()
        and poseidon.scaffold_ready()
    )


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
