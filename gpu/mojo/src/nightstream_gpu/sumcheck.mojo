from gpu.host import DeviceContext
from gpu import block_dim, block_idx, thread_idx
from memory import UnsafePointer
from nightstream_gpu import field


alias SPLIT_NC_SNAPSHOT_MAGIC = UInt64(0x4E53504C49544E43)
alias SPLIT_NC_SNAPSHOT_VERSION = UInt64(1)
alias SPLIT_NC_FE_ROW_V1 = UInt64(1)
alias SPLIT_NC_NC_COL_V1 = UInt64(2)
alias FE_HEADER_WORDS = 16
alias NC_HEADER_WORDS = 13
alias D_WIDTH = 54
alias SUMCHECK_GPU_BLOCK_SIZE = 64
alias SUMCHECK_GPU_MIN_TASKS = 256
alias DEVICE_API_CPU = 0
alias DEVICE_API_CUDA = 2
alias SESSION_HANDLE_MAGIC = UInt64(0x4E53000000000000)


struct KVal(Copyable, ImplicitlyCopyable, Movable):
    var re: UInt64
    var im: UInt64

    fn __init__(out self, re: UInt64, im: UInt64):
        self.re = re
        self.im = im

    fn __copyinit__(out self, existing: Self):
        self.re = existing.re
        self.im = existing.im


fn scaffold_ready() -> Bool:
    return True


fn k_zero() -> KVal:
    return KVal(0, 0)


fn k_one() -> KVal:
    return KVal(1, 0)


fn k_is_zero(x: KVal) -> Bool:
    return x.re == 0 and x.im == 0


fn k_add(a: KVal, b: KVal) -> KVal:
    return KVal(field.fq_add(a.re, b.re), field.fq_add(a.im, b.im))


fn k_sub(a: KVal, b: KVal) -> KVal:
    return KVal(field.fq_sub(a.re, b.re), field.fq_sub(a.im, b.im))


fn k_mul(a: KVal, b: KVal) -> KVal:
    var ac = field.fq_mul(a.re, b.re)
    var bd = field.fq_mul(a.im, b.im)
    var ad = field.fq_mul(a.re, b.im)
    var bc = field.fq_mul(a.im, b.re)
    return KVal(field.fq_sub(ac, bd), field.fq_add(ad, bc))


fn k_pow_small(x: KVal, exp: UInt64) -> KVal:
    if exp == 0:
        return k_one()
    var acc = x
    for _ in range(Int(exp - 1)):
        acc = k_mul(acc, x)
    return acc


fn k_interp(lo: KVal, hi: KVal, x: KVal, one_minus: KVal) -> KVal:
    return k_add(k_mul(one_minus, lo), k_mul(x, hi))


fn k_store(words: UnsafePointer[mut=True, UInt64], word_idx: Int, value: KVal):
    words[word_idx] = value.re
    words[word_idx + 1] = value.im


fn k_load(words: UnsafePointer[UInt64], word_idx: Int) -> KVal:
    return KVal(words[word_idx], words[word_idx + 1])


fn session_prefers_gpu(session: UInt64) -> Bool:
    if session <= 1:
        return False
    if (session & SESSION_HANDLE_MAGIC) != SESSION_HANDLE_MAGIC:
        return False
    var api = UInt32((session >> 32) & UInt64(0xFFFF_FFFF))
    if api == UInt32(DEVICE_API_CPU):
        return False
    # Metal FE/NC shared-library launches are still unstable; keep them on the host path.
    return api == UInt32(DEVICE_API_CUDA)


fn grid_dim_for(num_tasks: Int) -> Int:
    return (num_tasks + SUMCHECK_GPU_BLOCK_SIZE - 1) // SUMCHECK_GPU_BLOCK_SIZE


fn read_snapshot_word(snapshot_ptr: UnsafePointer[UInt64], word_idx: Int) -> UInt64:
    return snapshot_ptr[word_idx]


fn read_snapshot_k(snapshot_ptr: UnsafePointer[UInt64], word_idx: Int) -> KVal:
    return KVal(
        read_snapshot_word(snapshot_ptr, word_idx),
        read_snapshot_word(snapshot_ptr, word_idx + 1),
    )


fn load_point(points_ptr: UnsafePointer[UInt64], point_idx: Int) -> KVal:
    return KVal(points_ptr[point_idx * 2], points_ptr[point_idx * 2 + 1])


fn store_out(out_ptr: UnsafePointer[mut=True, UInt64], point_idx: Int, value: KVal):
    out_ptr[point_idx * 2] = value.re
    out_ptr[point_idx * 2 + 1] = value.im


fn snapshot_kind(snapshot_ptr: UnsafePointer[UInt64], snapshot_len: Int) -> UInt64:
    var normalized_len = snapshot_len
    if normalized_len < 24 or normalized_len % 8 != 0:
        normalized_len = normalized_len * 8
    if normalized_len < 24 or normalized_len % 8 != 0:
        return UInt64(0)
    if read_snapshot_word(snapshot_ptr, 0) != SPLIT_NC_SNAPSHOT_MAGIC:
        return UInt64(0)
    if read_snapshot_word(snapshot_ptr, 1) != SPLIT_NC_SNAPSHOT_VERSION:
        return UInt64(0)
    return read_snapshot_word(snapshot_ptr, 2)


fn snapshot_status(snapshot_ptr: UnsafePointer[UInt64], snapshot_len: Int, expected_kind: UInt64) -> Int32:
    var normalized_len = snapshot_len
    if normalized_len < 24 or normalized_len % 8 != 0:
        normalized_len = normalized_len * 8
    if normalized_len < 24 or normalized_len % 8 != 0:
        return -10
    if read_snapshot_word(snapshot_ptr, 0) != SPLIT_NC_SNAPSHOT_MAGIC:
        return -11
    if read_snapshot_word(snapshot_ptr, 1) != SPLIT_NC_SNAPSHOT_VERSION:
        return -12
    if read_snapshot_word(snapshot_ptr, 2) != expected_kind:
        return -13
    return 0


fn fe_terms_offset(snapshot_ptr: UnsafePointer[UInt64]) -> Int:
    var eq_beta_len = Int(read_snapshot_word(snapshot_ptr, 6))
    var eq_r_inputs_len = Int(read_snapshot_word(snapshot_ptr, 7))
    var gamma_pow_len = Int(read_snapshot_word(snapshot_ptr, 8))
    return FE_HEADER_WORDS + ((eq_beta_len + eq_r_inputs_len + gamma_pow_len) * 2)


fn fe_tables_offset(snapshot_ptr: UnsafePointer[UInt64]) -> Int:
    var term_count = Int(read_snapshot_word(snapshot_ptr, 9))
    var offset = fe_terms_offset(snapshot_ptr)
    for _ in range(term_count):
        var vars_len = Int(read_snapshot_word(snapshot_ptr, offset + 2))
        offset = offset + 3 + (vars_len * 2)
    return offset


fn fe_eval_one(snapshot_ptr: UnsafePointer[UInt64], x: KVal) -> KVal:
    var cur_len = Int(read_snapshot_word(snapshot_ptr, 5))
    var tail_len = cur_len // 2
    var total = k_zero()
    for t in range(tail_len):
        total = k_add(total, fe_eval_pair(snapshot_ptr, x, t))
    return total


fn fe_eval_pair(snapshot_ptr: UnsafePointer[UInt64], x: KVal, t: Int) -> KVal:
    var cur_len = Int(read_snapshot_word(snapshot_ptr, 5))
    var eq_beta_len = Int(read_snapshot_word(snapshot_ptr, 6))
    var eq_r_inputs_len = Int(read_snapshot_word(snapshot_ptr, 7))
    var gamma_pow_len = Int(read_snapshot_word(snapshot_ptr, 8))
    var term_count = Int(read_snapshot_word(snapshot_ptr, 9))
    var num_mcs = Int(read_snapshot_word(snapshot_ptr, 10))
    var num_vars = Int(read_snapshot_word(snapshot_ptr, 11))
    var table_len = Int(read_snapshot_word(snapshot_ptr, 12))
    var eval_len = Int(read_snapshot_word(snapshot_ptr, 13))
    var gamma_to_k = read_snapshot_k(snapshot_ptr, 14)

    var eq_beta_off = FE_HEADER_WORDS
    var eq_r_inputs_off = eq_beta_off + (eq_beta_len * 2)
    var gamma_pow_off = eq_r_inputs_off + (eq_r_inputs_len * 2)
    var tables_off = fe_tables_offset(snapshot_ptr)
    var eval_off = tables_off + (num_mcs * num_vars * table_len * 2)

    var one_minus = k_sub(k_one(), x)
    if t >= cur_len // 2:
        return k_zero()
    var pair_idx = 2 * t
    var eq_beta = k_interp(
        read_snapshot_k(snapshot_ptr, eq_beta_off + (pair_idx * 2)),
        read_snapshot_k(snapshot_ptr, eq_beta_off + ((pair_idx + 1) * 2)),
        x,
        one_minus,
    )
    var f_prime = k_zero()

    for mcs_idx in range(num_mcs):
        var f_i = k_zero()
        var term_off = gamma_pow_off + (gamma_pow_len * 2)
        for _ in range(term_count):
            var acc = read_snapshot_k(snapshot_ptr, term_off)
            var vars_len = Int(read_snapshot_word(snapshot_ptr, term_off + 2))
            var entry_off = term_off + 3
            for _ in range(vars_len):
                var var_pos = Int(read_snapshot_word(snapshot_ptr, entry_off))
                var exp = read_snapshot_word(snapshot_ptr, entry_off + 1)
                var table_off = tables_off + (((mcs_idx * num_vars + var_pos) * table_len + pair_idx) * 2)
                var xi = k_interp(
                    read_snapshot_k(snapshot_ptr, table_off),
                    read_snapshot_k(snapshot_ptr, table_off + 2),
                    x,
                    one_minus,
                )
                acc = k_mul(acc, k_pow_small(xi, exp))
                entry_off = entry_off + 2
            f_i = k_add(f_i, acc)
            term_off = entry_off

        var gamma = k_one()
        if mcs_idx < gamma_pow_len:
            gamma = read_snapshot_k(snapshot_ptr, gamma_pow_off + (mcs_idx * 2))
        f_prime = k_add(f_prime, k_mul(gamma, f_i))

    var acc = k_mul(eq_beta, f_prime)
    if eq_r_inputs_len > 0 and eval_len > 0:
        var eq_r = k_interp(
            read_snapshot_k(snapshot_ptr, eq_r_inputs_off + (pair_idx * 2)),
            read_snapshot_k(snapshot_ptr, eq_r_inputs_off + ((pair_idx + 1) * 2)),
            x,
            one_minus,
        )
        if not k_is_zero(eq_r):
            var eval_val = k_interp(
                read_snapshot_k(snapshot_ptr, eval_off + (pair_idx * 2)),
                read_snapshot_k(snapshot_ptr, eval_off + ((pair_idx + 1) * 2)),
                x,
                one_minus,
            )
            acc = k_add(acc, k_mul(eq_r, k_mul(gamma_to_k, eval_val)))
    return acc


fn range_product_cached(snapshot_ptr: UnsafePointer[UInt64], range_off: Int, range_len: Int, y: KVal) -> KVal:
    if range_len == 0:
        return y
    var y2 = k_mul(y, y)
    var prod = y
    for idx in range(range_len):
        prod = k_mul(prod, k_sub(y2, read_snapshot_k(snapshot_ptr, range_off + (idx * 2))))
    return prod


fn nc_eval_one(snapshot_ptr: UnsafePointer[UInt64], x: KVal) -> KVal:
    var cur_len = Int(read_snapshot_word(snapshot_ptr, 5))
    var tail_len = cur_len // 2
    var total = k_zero()
    for t in range(tail_len):
        total = k_add(total, nc_eval_pair(snapshot_ptr, x, t))
    return total


fn nc_eval_pair(snapshot_ptr: UnsafePointer[UInt64], x: KVal, t: Int) -> KVal:
    var cur_len = Int(read_snapshot_word(snapshot_ptr, 5))
    var eq_beta_len = Int(read_snapshot_word(snapshot_ptr, 6))
    var num_tables = Int(read_snapshot_word(snapshot_ptr, 7))
    var table_len = Int(read_snapshot_word(snapshot_ptr, 8))
    var d_width = Int(read_snapshot_word(snapshot_ptr, 9))
    var weights_tables = Int(read_snapshot_word(snapshot_ptr, 10))
    var weights_width = Int(read_snapshot_word(snapshot_ptr, 11))
    var range_len = Int(read_snapshot_word(snapshot_ptr, 12))

    var eq_beta_off = NC_HEADER_WORDS
    var digits_off = eq_beta_off + (eq_beta_len * 2)
    var weights_off = digits_off + (num_tables * table_len * d_width * 2)
    var range_off = weights_off + (weights_tables * weights_width * 2)

    var one_minus = k_sub(k_one(), x)
    if t >= cur_len // 2:
        return k_zero()
    var pair_idx = 2 * t
    var eq_beta = k_interp(
        read_snapshot_k(snapshot_ptr, eq_beta_off + (pair_idx * 2)),
        read_snapshot_k(snapshot_ptr, eq_beta_off + ((pair_idx + 1) * 2)),
        x,
        one_minus,
    )
    var nc_sum = k_zero()

    for table_idx in range(num_tables):
        for rho in range(d_width):
            var lo_off = digits_off + (((table_idx * table_len + pair_idx) * d_width + rho) * 2)
            var hi_off = digits_off + (((table_idx * table_len + pair_idx + 1) * d_width + rho) * 2)
            var y = k_interp(
                read_snapshot_k(snapshot_ptr, lo_off),
                read_snapshot_k(snapshot_ptr, hi_off),
                x,
                one_minus,
            )
            var weight = read_snapshot_k(snapshot_ptr, weights_off + ((table_idx * weights_width + rho) * 2))
            nc_sum = k_add(
                nc_sum,
                k_mul(weight, range_product_cached(snapshot_ptr, range_off, range_len, y)),
            )

    return k_mul(eq_beta, nc_sum)


fn fe_partial_gpu_kernel(
    snapshot_words: UnsafePointer[mut=True, UInt64],
    points_words: UnsafePointer[mut=True, UInt64],
    partial_words: UnsafePointer[mut=True, UInt64],
    points_len: Int,
):
    var cur_len = Int(read_snapshot_word(snapshot_words, 5))
    var tail_len = cur_len // 2
    var task_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total_tasks = points_len * tail_len
    if task_idx >= total_tasks:
        return
    var point_idx = task_idx // tail_len
    var pair_idx = task_idx % tail_len
    k_store(partial_words, task_idx * 2, fe_eval_pair(snapshot_words, load_point(points_words, point_idx), pair_idx))


fn nc_partial_gpu_kernel(
    snapshot_words: UnsafePointer[mut=True, UInt64],
    points_words: UnsafePointer[mut=True, UInt64],
    partial_words: UnsafePointer[mut=True, UInt64],
    points_len: Int,
):
    var cur_len = Int(read_snapshot_word(snapshot_words, 5))
    var tail_len = cur_len // 2
    var task_idx = Int(block_idx.x * block_dim.x + thread_idx.x)
    var total_tasks = points_len * tail_len
    if task_idx >= total_tasks:
        return
    var point_idx = task_idx // tail_len
    var pair_idx = task_idx % tail_len
    k_store(partial_words, task_idx * 2, nc_eval_pair(snapshot_words, load_point(points_words, point_idx), pair_idx))


fn reduce_partials_host(
    partial_words: UnsafePointer[mut=True, UInt64],
    points_len: Int,
    tail_len: Int,
    out_ptr: UnsafePointer[mut=True, UInt64],
):
    for point_idx in range(points_len):
        var acc = k_zero()
        for pair_idx in range(tail_len):
            acc = k_add(acc, k_load(partial_words, (point_idx * tail_len + pair_idx) * 2))
        store_out(out_ptr, point_idx, acc)


fn fe_evals_at_gpu(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
) raises:
    var snapshot_word_count = Int(UInt(snapshot_len)) // 8
    var point_count = Int(UInt(points_len))
    var tail_len = Int(read_snapshot_word(snapshot_words, 5)) // 2
    var partial_word_count = point_count * tail_len * 2
    var ctx = DeviceContext()
    var host_snapshot = ctx.enqueue_create_host_buffer[DType.uint64](snapshot_word_count)
    var host_points = ctx.enqueue_create_host_buffer[DType.uint64](point_count * 2)
    var host_partials = ctx.enqueue_create_host_buffer[DType.uint64](partial_word_count)
    var dev_snapshot = ctx.enqueue_create_buffer[DType.uint64](snapshot_word_count)
    var dev_points = ctx.enqueue_create_buffer[DType.uint64](point_count * 2)
    var dev_partials = ctx.enqueue_create_buffer[DType.uint64](partial_word_count)
    ctx.synchronize()

    for idx in range(snapshot_word_count):
        host_snapshot[idx] = snapshot_words[idx]
    for idx in range(point_count * 2):
        host_points[idx] = points_words[idx]

    ctx.enqueue_copy(src_buf=host_snapshot, dst_buf=dev_snapshot)
    ctx.enqueue_copy(src_buf=host_points, dst_buf=dev_points)
    var kernel = ctx.compile_function[fe_partial_gpu_kernel]()
    ctx.enqueue_function(
        kernel,
        dev_snapshot.unsafe_ptr(),
        dev_points.unsafe_ptr(),
        dev_partials.unsafe_ptr(),
        point_count,
        grid_dim=grid_dim_for(point_count * tail_len),
        block_dim=SUMCHECK_GPU_BLOCK_SIZE,
    )
    ctx.enqueue_copy(src_buf=dev_partials, dst_buf=host_partials)
    ctx.synchronize()
    reduce_partials_host(host_partials.unsafe_ptr(), point_count, tail_len, out_ptr)


fn nc_evals_at_gpu(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
) raises:
    var snapshot_word_count = Int(UInt(snapshot_len)) // 8
    var point_count = Int(UInt(points_len))
    var tail_len = Int(read_snapshot_word(snapshot_words, 5)) // 2
    var partial_word_count = point_count * tail_len * 2
    var ctx = DeviceContext()
    var host_snapshot = ctx.enqueue_create_host_buffer[DType.uint64](snapshot_word_count)
    var host_points = ctx.enqueue_create_host_buffer[DType.uint64](point_count * 2)
    var host_partials = ctx.enqueue_create_host_buffer[DType.uint64](partial_word_count)
    var dev_snapshot = ctx.enqueue_create_buffer[DType.uint64](snapshot_word_count)
    var dev_points = ctx.enqueue_create_buffer[DType.uint64](point_count * 2)
    var dev_partials = ctx.enqueue_create_buffer[DType.uint64](partial_word_count)
    ctx.synchronize()

    for idx in range(snapshot_word_count):
        host_snapshot[idx] = snapshot_words[idx]
    for idx in range(point_count * 2):
        host_points[idx] = points_words[idx]

    ctx.enqueue_copy(src_buf=host_snapshot, dst_buf=dev_snapshot)
    ctx.enqueue_copy(src_buf=host_points, dst_buf=dev_points)
    var kernel = ctx.compile_function[nc_partial_gpu_kernel]()
    ctx.enqueue_function(
        kernel,
        dev_snapshot.unsafe_ptr(),
        dev_points.unsafe_ptr(),
        dev_partials.unsafe_ptr(),
        point_count,
        grid_dim=grid_dim_for(point_count * tail_len),
        block_dim=SUMCHECK_GPU_BLOCK_SIZE,
    )
    ctx.enqueue_copy(src_buf=dev_partials, dst_buf=host_partials)
    ctx.synchronize()
    reduce_partials_host(host_partials.unsafe_ptr(), point_count, tail_len, out_ptr)


fn fe_create(
    _session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    var status = snapshot_status(snapshot_words, Int(UInt(snapshot_len)), SPLIT_NC_FE_ROW_V1)
    if status != 0:
        return status
    out_handle[0] = 1
    return 0


fn fe_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return 0


fn fe_evals_at(
    _session: UInt64,
    _evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    var status = snapshot_status(snapshot_words, Int(UInt(snapshot_len)), SPLIT_NC_FE_ROW_V1)
    if status != 0:
        return status
    if out_len < UInt(points_len):
        return -2
    var point_count = Int(UInt(points_len))
    var tail_len = Int(read_snapshot_word(snapshot_words, 5)) // 2
    var total_tasks = point_count * tail_len
    if session_prefers_gpu(_session) and total_tasks >= SUMCHECK_GPU_MIN_TASKS:
        try:
            fe_evals_at_gpu(_session, snapshot_words, snapshot_len, points_words, points_len, out_ptr)
            return 0
        except:
            pass
    for idx in range(Int(UInt(points_len))):
        store_out(out_ptr, idx, fe_eval_one(snapshot_words, load_point(points_words, idx)))
    return 0


fn fe_fold(_session: UInt, _evaluator: UInt, _challenge: KVal) -> Int32:
    return 0


fn nc_create(
    _session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_handle: UnsafePointer[mut=True, UInt64],
) -> Int32:
    var status = snapshot_status(snapshot_words, Int(UInt(snapshot_len)), SPLIT_NC_NC_COL_V1)
    if status != 0:
        return status
    out_handle[0] = 2
    return 0


fn nc_destroy(_session: UInt, _evaluator: UInt) -> Int32:
    return 0


fn nc_evals_at(
    _session: UInt64,
    _evaluator: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    points_words: UnsafePointer[mut=True, UInt64],
    points_len: UInt64,
    out_ptr: UnsafePointer[mut=True, UInt64],
    out_len: UInt,
) -> Int32:
    var status = snapshot_status(snapshot_words, Int(UInt(snapshot_len)), SPLIT_NC_NC_COL_V1)
    if status != 0:
        return status
    if out_len < UInt(points_len):
        return -2
    var point_count = Int(UInt(points_len))
    var tail_len = Int(read_snapshot_word(snapshot_words, 5)) // 2
    var total_tasks = point_count * tail_len
    if session_prefers_gpu(_session) and total_tasks >= SUMCHECK_GPU_MIN_TASKS:
        try:
            nc_evals_at_gpu(_session, snapshot_words, snapshot_len, points_words, points_len, out_ptr)
            return 0
        except:
            pass
    for idx in range(Int(UInt(points_len))):
        store_out(out_ptr, idx, nc_eval_one(snapshot_words, load_point(points_words, idx)))
    return 0


fn nc_fold(_session: UInt, _evaluator: UInt, _challenge: KVal) -> Int32:
    return 0


fn debug_snapshot_head(
    session: UInt64,
    snapshot_words: UnsafePointer[mut=True, UInt64],
    snapshot_len: UInt64,
    out_words: UnsafePointer[mut=True, UInt64],
    out_len: UInt32,
) -> Int32:
    if out_len < 3:
        return -2
    out_words[0] = session
    out_words[1] = UInt64(Int(snapshot_words))
    out_words[2] = snapshot_len
    if snapshot_len < 8:
        return 0
    var max_words = Int(UInt(out_len)) - 3
    var snapshot_word_count = Int(UInt(snapshot_len)) // 8
    if max_words > snapshot_word_count:
        max_words = snapshot_word_count
    for idx in range(max_words):
        out_words[idx + 3] = snapshot_words[idx]
    return 0
