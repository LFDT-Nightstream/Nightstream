from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from memory import alloc
from sys import has_accelerator

from nightstream_gpu import field, poseidon


alias RATE = 4
alias DIGEST_LEN = 4
alias GPU_BLOCK_SIZE = 64
alias COMPARE_BATCH_STATES = 256


fn words_for_states(num_states: Int) -> Int:
    return num_states * poseidon.POSEIDON2_WIDTH


fn grid_dim_for(num_states: Int) -> Int:
    return (num_states + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE


fn fill_state_words_at(state_words: UnsafePointer[mut=True, UInt64], state_idx: Int):
    var base = state_idx * poseidon.POSEIDON2_WIDTH
    var seed = UInt64(state_idx * 31)
    state_words[base + 0] = seed + 3
    state_words[base + 1] = seed + 5
    state_words[base + 2] = seed + 7
    state_words[base + 3] = seed + 11
    state_words[base + 4] = seed + 13
    state_words[base + 5] = seed + 17
    state_words[base + 6] = seed + 19
    state_words[base + 7] = seed + 23


fn poseidon2_gpu_batch_kernel(state_words: UnsafePointer[mut=True, UInt64], num_states: Int):
    var state_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if state_idx < num_states:
        poseidon.permute_state_at_offset(
            state_words,
            state_idx * poseidon.POSEIDON2_WIDTH,
        )


fn run_gpu_permutations_in_place(state_words: UnsafePointer[mut=True, UInt64], num_states: Int) raises:
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


fn compare_batched_states(num_states: Int) raises:
    var word_count = words_for_states(num_states)
    var cpu_words = alloc[UInt64](word_count)
    var gpu_words = alloc[UInt64](word_count)

    for state_idx in range(num_states):
        fill_state_words_at(cpu_words, state_idx)
        fill_state_words_at(gpu_words, state_idx)
        poseidon.permute_state_at_offset(
            cpu_words,
            state_idx * poseidon.POSEIDON2_WIDTH,
        )

    run_gpu_permutations_in_place(gpu_words, num_states)

    for i in range(word_count):
        if cpu_words[i] != gpu_words[i]:
            raise Error("poseidon2 batched permutation mismatch")

    cpu_words.free()
    gpu_words.free()


fn compare_sponge(length: Int) raises:
    var cpu_state = alloc[UInt64](poseidon.POSEIDON2_WIDTH)
    var gpu_state = alloc[UInt64](poseidon.POSEIDON2_WIDTH)
    for i in range(poseidon.POSEIDON2_WIDTH):
        cpu_state[i] = 0
        gpu_state[i] = 0

    for start in range(0, length, RATE):
        for off in range(RATE):
            var idx = start + off
            if idx < length:
                var word = UInt64(idx) * 17 + 3
                cpu_state[off] = field.fq_add(cpu_state[off], word)
                gpu_state[off] = field.fq_add(gpu_state[off], word)
        poseidon.permute_state_in_place(cpu_state)
        run_gpu_permutations_in_place(gpu_state, 1)
    cpu_state[0] = field.fq_add(cpu_state[0], 1)
    gpu_state[0] = field.fq_add(gpu_state[0], 1)
    poseidon.permute_state_in_place(cpu_state)
    run_gpu_permutations_in_place(gpu_state, 1)

    for i in range(DIGEST_LEN):
        if cpu_state[i] != gpu_state[i]:
            raise Error("poseidon2 sponge mismatch")

    cpu_state.free()
    gpu_state.free()


fn main() raises:
    @parameter
    if not has_accelerator():
        raise Error("No compatible GPU found")

    compare_batched_states(COMPARE_BATCH_STATES)

    for n in range(9):
        compare_sponge(n)

    print("poseidon_gpu_compare_ok")
