from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from nightstream_gpu import field


alias POSEIDON2_WIDTH = 8
alias EXTERNAL_ROUNDS_HALF = 4
alias INTERNAL_ROUNDS = 22
alias GPU_BLOCK_SIZE = 64
alias POSEIDON2_GPU_MIN_STATES = 128
alias DEVICE_API_CPU = 0
alias SESSION_HANDLE_MAGIC = UInt(0x4E53000000000000)


fn scaffold_ready() -> Bool:
    return True


fn poseidon2_permute_u64x8(
    _session: UInt,
    state: UnsafePointer[mut=True, UInt64],
    width: UInt32,
) -> Int32:
    if width != UInt32(POSEIDON2_WIDTH):
        return -2

    permute_state_in_place(state)
    return 0


fn poseidon2_permute_batch_u64x8(
    session: UInt,
    state_words: UnsafePointer[mut=True, UInt64],
    num_states: UInt32,
    width: UInt32,
) -> Int32:
    if width != UInt32(POSEIDON2_WIDTH):
        return -2

    var num_states_int = Int(num_states)
    if num_states_int <= 0:
        return 0
    if not session_prefers_gpu(session) or num_states_int < POSEIDON2_GPU_MIN_STATES:
        permute_batch_cpu_in_place(state_words, num_states_int)
        return 0

    try:
        permute_batch_gpu_in_place(state_words, num_states_int)
    except:
        permute_batch_cpu_in_place(state_words, num_states_int)
    return 0


fn permute_state_in_place(state: UnsafePointer[mut=True, UInt64]):
    permute_state_at_offset(state, 0)


fn words_for_states(num_states: Int) -> Int:
    return num_states * POSEIDON2_WIDTH


fn grid_dim_for(num_states: Int) -> Int:
    return (num_states + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE


fn poseidon2_gpu_batch_kernel(state_words: UnsafePointer[mut=True, UInt64], num_states: Int):
    var state_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if state_idx < num_states:
        permute_state_at_offset(state_words, state_idx * POSEIDON2_WIDTH)


fn permute_batch_cpu_in_place(state_words: UnsafePointer[mut=True, UInt64], num_states: Int):
    for state_idx in range(num_states):
        permute_state_at_offset(state_words, state_idx * POSEIDON2_WIDTH)


fn permute_batch_gpu_in_place(state_words: UnsafePointer[mut=True, UInt64], num_states: Int) raises:
    var word_count = words_for_states(num_states)
    var ctx = DeviceContext()
    var host = ctx.enqueue_create_host_buffer[DType.uint64](word_count)
    var dev = ctx.enqueue_create_buffer[DType.uint64](word_count)
    ctx.synchronize()

    for i in range(word_count):
        host[i] = state_words[i]

    ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    var kernel = ctx.compile_function[poseidon2_gpu_batch_kernel]()
    ctx.enqueue_function(
        kernel,
        dev.unsafe_ptr(),
        num_states,
        grid_dim=grid_dim_for(num_states),
        block_dim=GPU_BLOCK_SIZE,
    )
    ctx.enqueue_copy(src_buf=dev, dst_buf=host)
    ctx.synchronize()

    for i in range(word_count):
        state_words[i] = host[i]


fn session_prefers_gpu(session: UInt) -> Bool:
    if session <= UInt(1):
        return False
    if (session & SESSION_HANDLE_MAGIC) != SESSION_HANDLE_MAGIC:
        return False
    return UInt32((session >> 32) & UInt(0xFFFF_FFFF)) != UInt32(DEVICE_API_CPU)


fn permute_state_at_offset(state_words: UnsafePointer[mut=True, UInt64], base: Int):
    for i in range(POSEIDON2_WIDTH):
        state_words[base + i] = field.fq_canonicalize(state_words[base + i])

    mds_light_permutation_at_offset(state_words, base)

    for round in range(EXTERNAL_ROUNDS_HALF):
        apply_external_round_at_offset(state_words, base, round, True)

    for round in range(INTERNAL_ROUNDS):
        state_words[base + 0] = field.fq_exp7(
            field.fq_add(state_words[base + 0], internal_rc(round))
        )
        internal_linear_layer_at_offset(state_words, base)

    for round in range(EXTERNAL_ROUNDS_HALF):
        apply_external_round_at_offset(state_words, base, round, False)


fn apply_external_round(state: UnsafePointer[mut=True, UInt64], round: Int, initial: Bool):
    apply_external_round_at_offset(state, 0, round, initial)


fn apply_external_round_at_offset(
    state_words: UnsafePointer[mut=True, UInt64],
    base: Int,
    round: Int,
    initial: Bool,
):
    for idx in range(POSEIDON2_WIDTH):
        var rc = terminal_rc(round, idx)
        if initial:
            rc = initial_rc(round, idx)
        state_words[base + idx] = field.fq_exp7(field.fq_add(state_words[base + idx], rc))
    mds_light_permutation_at_offset(state_words, base)


fn mds_light_permutation(state: UnsafePointer[mut=True, UInt64]):
    mds_light_permutation_at_offset(state, 0)


fn mds_light_permutation_at_offset(state_words: UnsafePointer[mut=True, UInt64], base: Int):
    apply_mat4_chunk_at_offset(state_words, base, 0)
    apply_mat4_chunk_at_offset(state_words, base, 4)

    var sum0 = field.fq_add(state_words[base + 0], state_words[base + 4])
    var sum1 = field.fq_add(state_words[base + 1], state_words[base + 5])
    var sum2 = field.fq_add(state_words[base + 2], state_words[base + 6])
    var sum3 = field.fq_add(state_words[base + 3], state_words[base + 7])

    state_words[base + 0] = field.fq_add(state_words[base + 0], sum0)
    state_words[base + 1] = field.fq_add(state_words[base + 1], sum1)
    state_words[base + 2] = field.fq_add(state_words[base + 2], sum2)
    state_words[base + 3] = field.fq_add(state_words[base + 3], sum3)
    state_words[base + 4] = field.fq_add(state_words[base + 4], sum0)
    state_words[base + 5] = field.fq_add(state_words[base + 5], sum1)
    state_words[base + 6] = field.fq_add(state_words[base + 6], sum2)
    state_words[base + 7] = field.fq_add(state_words[base + 7], sum3)


fn apply_mat4_chunk(state: UnsafePointer[mut=True, UInt64], offset: Int):
    apply_mat4_chunk_at_offset(state, 0, offset)


fn apply_mat4_chunk_at_offset(
    state_words: UnsafePointer[mut=True, UInt64],
    base: Int,
    offset: Int,
):
    var x0 = state_words[base + offset + 0]
    var x1 = state_words[base + offset + 1]
    var x2 = state_words[base + offset + 2]
    var x3 = state_words[base + offset + 3]

    var t01 = field.fq_add(x0, x1)
    var t23 = field.fq_add(x2, x3)
    var t0123 = field.fq_add(t01, t23)
    var t01123 = field.fq_add(t0123, x1)
    var t01233 = field.fq_add(t0123, x3)

    state_words[base + offset + 3] = field.fq_add(t01233, field.fq_double(x0))
    state_words[base + offset + 1] = field.fq_add(t01123, field.fq_double(x2))
    state_words[base + offset + 0] = field.fq_add(t01123, t01)
    state_words[base + offset + 2] = field.fq_add(t01233, t23)


fn internal_linear_layer(state: UnsafePointer[mut=True, UInt64]):
    internal_linear_layer_at_offset(state, 0)


fn internal_linear_layer_at_offset(state_words: UnsafePointer[mut=True, UInt64], base: Int):
    var acc = UInt64(0)
    for idx in range(POSEIDON2_WIDTH):
        acc = field.fq_add(acc, state_words[base + idx])
    for idx in range(POSEIDON2_WIDTH):
        state_words[base + idx] = field.fq_add(
            field.fq_mul(state_words[base + idx], matrix_diag(idx)),
            acc,
        )


fn matrix_diag(idx: Int) -> UInt64:
    if idx == 0:
        return 0xA98811A1FED4E3A5
    if idx == 1:
        return 0x1CC48B54F377E2A0
    if idx == 2:
        return 0xE40CD4F6C5609A26
    if idx == 3:
        return 0x11DE79EBCA97A4A3
    if idx == 4:
        return 0x9177C73D8B7E929C
    if idx == 5:
        return 0x2A6FE8085797E791
    if idx == 6:
        return 0x3DE6E93329F8D5AD
    return 0x3F7AF9125DA962FE


fn internal_rc(round: Int) -> UInt64:
    if round == 0:
        return 0x67D6184D5EA5FCFE
    if round == 1:
        return 0x302EDF3A1B784AB0
    if round == 2:
        return 0x7AC5AEA122DA27F2
    if round == 3:
        return 0x3D7E234FAD5CC287
    if round == 4:
        return 0xC7996CE7C8310E86
    if round == 5:
        return 0xD90059CA8EEE0FC8
    if round == 6:
        return 0xC98879052E16D8E7
    if round == 7:
        return 0x3C622EE5557474DB
    if round == 8:
        return 0xC3F3C7222F3AEB69
    if round == 9:
        return 0x07548FA82C00F654
    if round == 10:
        return 0xD56EA1123363578C
    if round == 11:
        return 0xBC59A21856ABB7EB
    if round == 12:
        return 0x0AD3E0E2A5A3203D
    if round == 13:
        return 0x92FB9EE729612129
    if round == 14:
        return 0x19AAC61CC077ED02
    if round == 15:
        return 0x65F011723421BBA6
    if round == 16:
        return 0xCA59B23C7001BA57
    if round == 17:
        return 0x79CF23880B0BBBA6
    if round == 18:
        return 0x18687250BB553AE7
    if round == 19:
        return 0x86A27245417A1134
    if round == 20:
        return 0xD9AE528F43C0EDAC
    return 0x4EB82BA4DA413ECB


fn initial_rc(round: Int, idx: Int) -> UInt64:
    if round == 0:
        if idx == 0:
            return 15504881536434223753
        if idx == 1:
            return 2212164856944708396
        if idx == 2:
            return 1885257220781225929
        if idx == 3:
            return 17531637481572944510
        if idx == 4:
            return 16769640728293682348
        if idx == 5:
            return 445908668462176974
        if idx == 6:
            return 1308472042479836079
        return 17465001500823438575
    if round == 1:
        if idx == 0:
            return 1922033642430128704
        if idx == 1:
            return 2657514617275794404
        if idx == 2:
            return 17238706657248448792
        if idx == 3:
            return 7348277157222259646
        if idx == 4:
            return 10777112892842897939
        if idx == 5:
            return 1771261721914735482
        if idx == 6:
            return 9409693344407549465
        return 16619731096074499912
    if round == 2:
        if idx == 0:
            return 1922036059108268922
        if idx == 1:
            return 2681686362645798986
        if idx == 2:
            return 12432722052283819565
        if idx == 3:
            return 2826979200512189741
        if idx == 4:
            return 5080805286413226676
        if idx == 5:
            return 16827966425431695029
        if idx == 6:
            return 9196241087337510154
        return 2350771591198563053
    if idx == 0:
        return 2989012136977041732
    if idx == 1:
        return 4359939046747977080
    if idx == 2:
        return 16089932437481530267
    if idx == 3:
        return 6601984573273403484
    if idx == 4:
        return 13005272261058756234
    if idx == 5:
        return 17128237926164276121
    if idx == 6:
        return 8240789415616872849
    return 8676316357341090631


fn terminal_rc(round: Int, idx: Int) -> UInt64:
    if round == 0:
        if idx == 0:
            return 16452552554259143025
        if idx == 1:
            return 17874550554210084887
        if idx == 2:
            return 3031715677034868367
        if idx == 3:
            return 18215520516675091549
        if idx == 4:
            return 18186005068527139405
        if idx == 5:
            return 11138995707668647102
        if idx == 6:
            return 15098195648006184282
        return 2025927025270509469
    if round == 1:
        if idx == 0:
            return 9957669227203243937
        if idx == 1:
            return 11554336633716867616
        if idx == 2:
            return 9729067570563846225
        if idx == 3:
            return 4239770196713589268
        if idx == 4:
            return 4390607796152185292
        if idx == 5:
            return 17647511975646925721
        if idx == 6:
            return 7671337049037340193
        return 4209452938403606590
    if round == 2:
        if idx == 0:
            return 6593973666654839090
        if idx == 1:
            return 8390781086037206386
        if idx == 2:
            return 7324343054784993307
        if idx == 3:
            return 17780748563735894140
        if idx == 4:
            return 15974082699116886783
        if idx == 5:
            return 13213371256836887512
        if idx == 6:
            return 7312926934405385057
        return 10393853239698468203
    if idx == 0:
        return 2710107888698774842
    if idx == 1:
        return 2801523468128575786
    if idx == 2:
        return 15894340394120906162
    if idx == 3:
        return 13510783799941644149
    if idx == 4:
        return 7917164295139071913
    if idx == 5:
        return 13839801071899888959
    if idx == 6:
        return 6672989303670154677
    return 4519956214037211385
