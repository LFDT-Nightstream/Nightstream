from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from memory import alloc
from python import Python
from sys import has_accelerator

from nightstream_gpu import poseidon


alias GPU_BLOCK_SIZE = 64
alias WIDTH = poseidon.POSEIDON2_WIDTH
alias TARGET_TOTAL_STATES = 131072
alias TARGET_ROUNDTRIP_STATES = 4096


fn words_for_states(num_states: Int) -> Int:
    return num_states * WIDTH


fn grid_dim_for(num_states: Int) -> Int:
    return (num_states + GPU_BLOCK_SIZE - 1) // GPU_BLOCK_SIZE


fn iter_count_for_target(num_states: Int, target_total_states: Int) -> Int:
    var count = target_total_states // num_states
    if count < 10:
        return 10
    return count


fn fill_state_words_at(state_words: UnsafePointer[mut=True, UInt64], state_idx: Int):
    var base = state_idx * WIDTH
    var seed = UInt64(state_idx * 31)
    state_words[base + 0] = seed + 3
    state_words[base + 1] = seed + 5
    state_words[base + 2] = seed + 7
    state_words[base + 3] = seed + 11
    state_words[base + 4] = seed + 13
    state_words[base + 5] = seed + 17
    state_words[base + 6] = seed + 19
    state_words[base + 7] = seed + 23


fn init_state_words(state_words: UnsafePointer[mut=True, UInt64], num_states: Int):
    for state_idx in range(num_states):
        fill_state_words_at(state_words, state_idx)


fn poseidon2_gpu_batch_kernel(state_words: UnsafePointer[mut=True, UInt64], num_states: Int):
    var state_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    if state_idx < num_states:
        poseidon.permute_state_at_offset(state_words, state_idx * WIDTH)


fn print_stats(label: String, num_states: Int, iters: Int, elapsed_ns: Int):
    var total_states = num_states * iters
    var ns_per_state = elapsed_ns // total_states
    var states_per_sec = (total_states * 1_000_000_000) // elapsed_ns
    print(
        label
        + " batch="
        + String(num_states)
        + " iters="
        + String(iters)
        + " total_ns="
        + String(elapsed_ns)
        + " ns_per_state="
        + String(ns_per_state)
        + " states_per_sec="
        + String(states_per_sec)
    )


fn bench_batch(num_states: Int) raises:
    var time = Python.import_module("time")
    var iters = iter_count_for_target(num_states, TARGET_TOTAL_STATES)
    var roundtrip_iters = iter_count_for_target(num_states, TARGET_ROUNDTRIP_STATES)
    var word_count = words_for_states(num_states)

    var cpu_words = alloc[UInt64](word_count)
    init_state_words(cpu_words, num_states)

    var cpu_start = time.perf_counter_ns()
    for _ in range(iters):
        for state_idx in range(num_states):
            poseidon.permute_state_at_offset(cpu_words, state_idx * WIDTH)
    var cpu_end = time.perf_counter_ns()
    print_stats("cpu", num_states, iters, Int(cpu_end - cpu_start))
    cpu_words.free()

    @parameter
    if not has_accelerator():
        print("gpu_available=0")
        return

    var ctx = DeviceContext()
    var host = ctx.enqueue_create_host_buffer[DType.uint64](word_count)
    var dev = ctx.enqueue_create_buffer[DType.uint64](word_count)
    ctx.synchronize()
    for state_idx in range(num_states):
        var base = state_idx * WIDTH
        var seed = UInt64(state_idx * 31)
        host[base + 0] = seed + 3
        host[base + 1] = seed + 5
        host[base + 2] = seed + 7
        host[base + 3] = seed + 11
        host[base + 4] = seed + 13
        host[base + 5] = seed + 17
        host[base + 6] = seed + 19
        host[base + 7] = seed + 23

    ctx.enqueue_copy(src_buf=host, dst_buf=dev)
    var kernel = ctx.compile_function[poseidon2_gpu_batch_kernel]()
    ctx.enqueue_function(
        kernel,
        dev.unsafe_ptr(),
        num_states,
        grid_dim=grid_dim_for(num_states),
        block_dim=GPU_BLOCK_SIZE,
    )
    ctx.synchronize()

    var gpu_start = time.perf_counter_ns()
    for _ in range(iters):
        ctx.enqueue_function(
            kernel,
            dev.unsafe_ptr(),
            num_states,
            grid_dim=grid_dim_for(num_states),
            block_dim=GPU_BLOCK_SIZE,
        )
    ctx.synchronize()
    var gpu_end = time.perf_counter_ns()
    print_stats("gpu_steady", num_states, iters, Int(gpu_end - gpu_start))

    for state_idx in range(num_states):
        var base = state_idx * WIDTH
        var seed = UInt64(state_idx * 31)
        host[base + 0] = seed + 3
        host[base + 1] = seed + 5
        host[base + 2] = seed + 7
        host[base + 3] = seed + 11
        host[base + 4] = seed + 13
        host[base + 5] = seed + 17
        host[base + 6] = seed + 19
        host[base + 7] = seed + 23
    var roundtrip_start = time.perf_counter_ns()
    for _ in range(roundtrip_iters):
        ctx.enqueue_copy(src_buf=host, dst_buf=dev)
        ctx.enqueue_function(
            kernel,
            dev.unsafe_ptr(),
            num_states,
            grid_dim=grid_dim_for(num_states),
            block_dim=GPU_BLOCK_SIZE,
        )
        ctx.enqueue_copy(src_buf=dev, dst_buf=host)
        ctx.synchronize()
    var roundtrip_end = time.perf_counter_ns()
    print_stats(
        "gpu_roundtrip",
        num_states,
        roundtrip_iters,
        Int(roundtrip_end - roundtrip_start),
    )


fn main() raises:
    bench_batch(1)
    bench_batch(32)
    bench_batch(128)
    bench_batch(512)
    bench_batch(2048)
