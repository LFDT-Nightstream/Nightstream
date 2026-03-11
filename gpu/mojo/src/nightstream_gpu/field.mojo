alias GOLDILOCKS_MODULUS = UInt64(0xFFFFFFFF00000001)
alias GOLDILOCKS_EPSILON = UInt64(0x00000000FFFFFFFF)
alias MASK32 = UInt64(0x00000000FFFFFFFF)


fn scaffold_ready() -> Bool:
    return True


fn fq_canonicalize(x: UInt64) -> UInt64:
    var y = x
    if y >= GOLDILOCKS_MODULUS:
        y = y - GOLDILOCKS_MODULUS
    return y


fn fq_add(a: UInt64, b: UInt64) -> UInt64:
    var sum = a + b
    if sum < a:
        sum = sum + GOLDILOCKS_EPSILON
    if sum >= GOLDILOCKS_MODULUS:
        sum = sum - GOLDILOCKS_MODULUS
    return sum


fn fq_neg(x: UInt64) -> UInt64:
    var y = fq_canonicalize(x)
    if y == 0:
        return 0
    return GOLDILOCKS_MODULUS - y


fn fq_sub(a: UInt64, b: UInt64) -> UInt64:
    return fq_add(a, fq_neg(b))


fn low32(x: UInt64) -> UInt64:
    return x & MASK32


fn high32(x: UInt64) -> UInt64:
    return x >> 32


fn fq_mul(a: UInt64, b: UInt64) -> UInt64:
    var a0 = low32(a)
    var a1 = high32(a)
    var b0 = low32(b)
    var b1 = high32(b)

    var p0 = a0 * b0
    var p1 = a0 * b1
    var p2 = a1 * b0
    var p3 = a1 * b1

    var cross = high32(p0) + low32(p1) + low32(p2)
    var lo = low32(p0) | (low32(cross) << 32)
    var hi = p3 + high32(p1) + high32(p2) + high32(cross)

    var result = fq_canonicalize(lo)
    var hi_lo = low32(hi)
    var hi_hi = high32(hi)

    result = fq_add(result, hi_lo << 32)
    result = fq_sub(result, hi_lo)
    result = fq_sub(result, hi_hi)
    return result


fn fq_square(x: UInt64) -> UInt64:
    return fq_mul(x, x)


fn fq_double(x: UInt64) -> UInt64:
    return fq_add(x, x)


fn fq_exp7(x: UInt64) -> UInt64:
    var x2 = fq_square(x)
    var x4 = fq_square(x2)
    return fq_mul(fq_mul(x4, x2), x)
