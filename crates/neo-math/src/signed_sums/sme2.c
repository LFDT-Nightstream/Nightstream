#include <arm_neon.h>
#include <arm_sme.h>
#include <stdbool.h>
#include <stddef.h>
#include <stdint.h>
#include <string.h>

enum {
    RING_DEGREE = 64,
    STREAMING_BYTES = 64,
    TILE_COLUMNS = 128,
    DOT_LANES = 4,
    TILE_ROWS = STREAMING_BYTES / sizeof(int32_t),
    KEY_BYTES = RING_DEGREE * sizeof(uint64_t),
    KEY_PANELS = KEY_BYTES / TILE_ROWS,
    MAX_COLUMNS_PER_SUM = 1 << 22,
};

typedef struct {
    uint8_t keys[KEY_PANELS][TILE_COLUMNS / DOT_LANES][STREAMING_BYTES];
    int8_t digits[TILE_COLUMNS / DOT_LANES][RING_DEGREE * DOT_LANES];
    int32_t products[KEY_BYTES][RING_DEGREE];
} Workspace;

_Static_assert(sizeof(Workspace) == 204800, "SME2 row workspace size");
_Static_assert(_Alignof(Workspace) <= _Alignof(uint64_t), "SME2 row workspace alignment");
_Static_assert(UINT8_MAX * (int64_t)MAX_COLUMNS_PER_SUM <= INT32_MAX, "SME2 byte dot product bound");
_Static_assert(__BYTE_ORDER__ == __ORDER_LITTLE_ENDIAN__, "key byte significance");

size_t ns_sme2_streaming_bytes(void) {
    return svcntsb();
}

static int8x8_t signed_digits(uint64_t positive, uint64_t negative, unsigned shift) {
    static const int8_t bit_shifts[8] = {0, -1, -2, -3, -4, -5, -6, -7};
    int8x8_t shifts = vld1_s8(bit_shifts);
    uint8x8_t pos = vshl_u8(vdup_n_u8((uint8_t)(positive >> shift)), shifts);
    uint8x8_t neg = vshl_u8(vdup_n_u8((uint8_t)(negative >> shift)), shifts);
    return vreinterpret_s8_u8(vsub_u8(vand_u8(pos, vdup_n_u8(1)), vand_u8(neg, vdup_n_u8(1))));
}

static void pack_tile(Workspace *workspace, size_t columns, const uint64_t *keys,
                      const uint64_t *positive, const uint64_t *negative) {
    static const uint8_t zero_key[KEY_BYTES] = {0};
    for (size_t first = 0; first < columns; first += DOT_LANES) {
        const uint8_t *key[DOT_LANES];
        uint64_t pos[DOT_LANES] = {0}, neg[DOT_LANES] = {0};
        for (size_t j = 0; j < DOT_LANES; ++j) {
            size_t column = first + j;
            if (column < columns) {
                key[j] = (const uint8_t *)(keys + column * RING_DEGREE);
                pos[j] = positive[column];
                neg[j] = negative[column];
            } else {
                key[j] = zero_key;
            }
        }
        size_t group = first / DOT_LANES;
        for (size_t panel = 0; panel < KEY_PANELS; ++panel) {
            uint8x16x4_t packed;
            for (size_t j = 0; j < DOT_LANES; ++j) {
                packed.val[j] = vld1q_u8(key[j] + panel * TILE_ROWS);
            }
            vst4q_u8(workspace->keys[panel][group], packed);
        }
        for (unsigned lane = 0; lane < RING_DEGREE; lane += 8) {
            int8x8x4_t packed;
            for (size_t j = 0; j < DOT_LANES; ++j) {
                packed.val[j] = signed_digits(pos[j], neg[j], lane);
            }
            vst4_s8(workspace->digits[group] + lane * DOT_LANES, packed);
        }
    }
}

static inline void store_slice(int32_t *dest, svint32_t value, bool accumulate) __arm_streaming __arm_in("za") {
    svbool_t lanes = svptrue_b32();
    if (accumulate) {
        value = svadd_s32_x(lanes, value, svld1_s32(lanes, dest));
    }
    svst1_s32(lanes, dest, value);
}

__arm_locally_streaming __arm_new("za")
static void multiply_tile(Workspace *workspace, size_t groups, bool accumulate) {
    svbool_t bytes = svptrue_b8();
    svbool_t words = svptrue_b32();
    for (size_t panel = 0; panel < KEY_PANELS; ++panel) {
        svzero_za();
        for (size_t group = 0; group < groups; ++group) {
            svuint8_t key = svld1_u8(bytes, workspace->keys[panel][group]);
            const int8_t *digits = workspace->digits[group];
            svusmopa_za32_u8_m(0, bytes, bytes, key, svld1_s8(bytes, digits));
            svusmopa_za32_u8_m(1, bytes, bytes, key, svld1_s8(bytes, digits + STREAMING_BYTES));
            svusmopa_za32_u8_m(2, bytes, bytes, key, svld1_s8(bytes, digits + 2 * STREAMING_BYTES));
            svusmopa_za32_u8_m(3, bytes, bytes, key, svld1_s8(bytes, digits + 3 * STREAMING_BYTES));
        }
        for (unsigned row = 0; row < TILE_ROWS; ++row) {
            int32_t *dest = workspace->products[panel * TILE_ROWS + row];
            svint32_t zero = svdup_s32(0);
            store_slice(dest, svread_hor_za32_s32_m(zero, words, 0, row), accumulate);
            store_slice(dest + TILE_ROWS, svread_hor_za32_s32_m(zero, words, 1, row), accumulate);
            store_slice(dest + 2 * TILE_ROWS, svread_hor_za32_s32_m(zero, words, 2, row), accumulate);
            store_slice(dest + 3 * TILE_ROWS, svread_hor_za32_s32_m(zero, words, 3, row), accumulate);
        }
    }
}

void ns_sme2_signed_dot(size_t columns, const uint64_t *keys, const uint64_t *positive,
                        const uint64_t *negative, void *scratch, __int128 *out) {
    Workspace *workspace = scratch;
    memset(out, 0, RING_DEGREE * sizeof(*out));
    if (columns == 0) {
        return;
    }
    for (size_t first = 0; first < columns; first += TILE_COLUMNS) {
        size_t count = columns - first < TILE_COLUMNS ? columns - first : TILE_COLUMNS;
        pack_tile(workspace, count, keys + first * RING_DEGREE, positive + first, negative + first);
        multiply_tile(workspace, (count + DOT_LANES - 1) / DOT_LANES, first != 0);
    }
    for (size_t byte = 0; byte < KEY_BYTES; ++byte) {
        size_t key_degree = byte / sizeof(uint64_t);
        __int128 weight = (__int128)1 << (8 * (byte % sizeof(uint64_t)));
        for (size_t lane = 0; lane < RING_DEGREE; ++lane) {
            size_t degree = key_degree + lane;
            __int128 product = weight * workspace->products[byte][lane];
            out[degree % RING_DEGREE] += degree < RING_DEGREE ? product : -product;
        }
    }
}

enum {
    QUARTER = RING_DEGREE / 4,
    QUARTER_PANELS = QUARTER * sizeof(uint64_t) / TILE_ROWS,
    KARATSUBA_PRODUCTS = 9,
    QUARTER_PRODUCT = 2 * QUARTER - 1,
    HALF_PRODUCT = 4 * QUARTER - 1,
    CHILD_GROUP_COLUMNS = 1 << 20,
    MAX_CHILD_PLANES = 6,
};

typedef int32_t ChildProducts[KARATSUBA_PRODUCTS][QUARTER_PANELS][TILE_ROWS][QUARTER];

typedef struct {
    uint8_t keys[KARATSUBA_PRODUCTS][QUARTER_PANELS][TILE_COLUMNS / DOT_LANES][STREAMING_BYTES];
    int8_t digits[KARATSUBA_PRODUCTS][TILE_COLUMNS / DOT_LANES][STREAMING_BYTES];
    ChildProducts products[];
} ChildWorkspace;

_Static_assert(sizeof(ChildWorkspace) == 165888, "SME2 child tile bytes");
_Static_assert(sizeof(ChildProducts) == 73728, "SME2 per-plane product bytes");
_Static_assert(_Alignof(ChildWorkspace) <= _Alignof(uint64_t), "SME2 child workspace alignment");
_Static_assert(4 * UINT8_MAX * (int64_t)CHILD_GROUP_COLUMNS <= INT32_MAX, "transformed digit dot product bound");

static inline uint64x2_t canonical_key_pair(const uint64_t *src) {
    uint64x2_t words = vld1q_u64(src);
    uint64x2_t reduce = vcgeq_u64(words, vdupq_n_u64(UINT64_MAX - 58));
    return vaddq_u64(words, vandq_u64(reduce, vdupq_n_u64(59)));
}

static inline uint64x2_t add_mod_q(uint64x2_t left, uint64x2_t right) {
    uint64x2_t sum = vaddq_u64(left, right);
    uint64x2_t reduce = vorrq_u64(vcltq_u64(sum, left), vcgeq_u64(sum, vdupq_n_u64(UINT64_MAX - 58)));
    return vaddq_u64(sum, vandq_u64(reduce, vdupq_n_u64(59)));
}

static void pack_child_keys(ChildWorkspace *workspace, size_t count, const uint32_t *columns, const uint64_t *keys) {
    static const uint64_t zero[RING_DEGREE] = {0};
    for (size_t first = 0; first < count; first += DOT_LANES) {
        const uint64_t *key[DOT_LANES];
        for (size_t j = 0; j < DOT_LANES; ++j) {
            key[j] = first + j < count ? keys + (size_t)columns[first + j] * RING_DEGREE : zero;
        }
        for (size_t panel = 0; panel < QUARTER_PANELS; ++panel) {
            uint8x16x4_t packed[KARATSUBA_PRODUCTS];
            for (size_t j = 0; j < DOT_LANES; ++j) {
                const uint64_t *src = key[j] + 2 * panel;
                uint64x2_t a = canonical_key_pair(src);
                uint64x2_t b = canonical_key_pair(src + QUARTER);
                uint64x2_t c = canonical_key_pair(src + 2 * QUARTER);
                uint64x2_t d = canonical_key_pair(src + 3 * QUARTER);
                uint64x2_t ab = add_mod_q(a, b), cd = add_mod_q(c, d);
                uint64x2_t terms[KARATSUBA_PRODUCTS] = {a, b, ab, c, d, cd, add_mod_q(a, c),
                                                        add_mod_q(b, d), add_mod_q(ab, cd)};
                for (size_t k = 0; k < KARATSUBA_PRODUCTS; ++k) {
                    packed[k].val[j] = vreinterpretq_u8_u64(terms[k]);
                }
            }
            for (size_t k = 0; k < KARATSUBA_PRODUCTS; ++k) {
                vst4q_u8(workspace->keys[k][panel][first / DOT_LANES], packed[k]);
            }
        }
    }
}

static bool pack_child_digits(ChildWorkspace *workspace, size_t count, const uint32_t *columns,
                              const uint64_t *positive, const uint64_t *negative) {
    uint64_t active = 0;
    for (size_t first = 0; first < count; first += DOT_LANES) {
        uint64_t pos[DOT_LANES] = {0}, neg[DOT_LANES] = {0};
        for (size_t j = 0; j < DOT_LANES && first + j < count; ++j) {
            pos[j] = positive[columns[first + j]];
            neg[j] = negative[columns[first + j]];
            active |= pos[j] ^ neg[j];
        }
        for (unsigned lane = 0; lane < QUARTER; lane += 8) {
            int8x8x4_t packed[KARATSUBA_PRODUCTS];
            for (size_t j = 0; j < DOT_LANES; ++j) {
                int8x8_t a = signed_digits(pos[j], neg[j], lane);
                int8x8_t b = signed_digits(pos[j], neg[j], lane + QUARTER);
                int8x8_t c = signed_digits(pos[j], neg[j], lane + 2 * QUARTER);
                int8x8_t d = signed_digits(pos[j], neg[j], lane + 3 * QUARTER);
                int8x8_t ab = vadd_s8(a, b), cd = vadd_s8(c, d);
                int8x8_t terms[KARATSUBA_PRODUCTS] = {a, b, ab, c, d, cd, vadd_s8(a, c),
                                                      vadd_s8(b, d), vadd_s8(ab, cd)};
                for (size_t k = 0; k < KARATSUBA_PRODUCTS; ++k) {
                    packed[k].val[j] = terms[k];
                }
            }
            for (size_t k = 0; k < KARATSUBA_PRODUCTS; ++k) {
                vst4_s8(workspace->digits[k][first / DOT_LANES] + DOT_LANES * lane, packed[k]);
            }
        }
    }
    return active != 0;
}

__arm_locally_streaming __arm_new("za")
static void multiply_child_tile(ChildWorkspace *workspace, size_t plane, size_t groups, bool accumulate) {
    svbool_t bytes = svptrue_b8();
    svbool_t words = svptrue_b32();
    for (size_t k = 0; k < KARATSUBA_PRODUCTS; ++k) {
        for (size_t panel = 0; panel < QUARTER_PANELS; panel += 4) {
            svzero_za();
            for (size_t group = 0; group < groups; ++group) {
                svint8_t digits = svld1_s8(bytes, workspace->digits[k][group]);
                svusmopa_za32_u8_m(0, bytes, bytes, svld1_u8(bytes, workspace->keys[k][panel][group]), digits);
                svusmopa_za32_u8_m(1, bytes, bytes, svld1_u8(bytes, workspace->keys[k][panel + 1][group]), digits);
                svusmopa_za32_u8_m(2, bytes, bytes, svld1_u8(bytes, workspace->keys[k][panel + 2][group]), digits);
                svusmopa_za32_u8_m(3, bytes, bytes, svld1_u8(bytes, workspace->keys[k][panel + 3][group]), digits);
            }
            for (unsigned row = 0; row < TILE_ROWS; ++row) {
                svint32_t zero = svdup_s32(0);
                int32_t (*dest)[TILE_ROWS][QUARTER] = &workspace->products[plane][k][panel];
                store_slice(dest[0][row], svread_hor_za32_s32_m(zero, words, 0, row), accumulate);
                store_slice(dest[1][row], svread_hor_za32_s32_m(zero, words, 1, row), accumulate);
                store_slice(dest[2][row], svread_hor_za32_s32_m(zero, words, 2, row), accumulate);
                store_slice(dest[3][row], svread_hor_za32_s32_m(zero, words, 3, row), accumulate);
            }
        }
    }
}

static void recombine_child_products(const ChildProducts products, __int128 *out) {
    __int128 halves[3][HALF_PRODUCT] = {0};
    for (size_t h = 0; h < 3; ++h) {
        __int128 quarters[3][QUARTER_PRODUCT] = {0};
        for (size_t k = 0; k < 3; ++k) {
            for (size_t byte = 0; byte < QUARTER * sizeof(uint64_t); ++byte) {
                __int128 weight = (__int128)1 << (8 * (byte % sizeof(uint64_t)));
                const int32_t *row = products[3 * h + k][byte / TILE_ROWS][byte % TILE_ROWS];
                for (size_t j = 0; j < QUARTER; ++j) {
                    quarters[k][byte / sizeof(uint64_t) + j] += weight * row[j];
                }
            }
        }
        for (size_t j = 0; j < QUARTER_PRODUCT; ++j) {
            halves[h][j] += quarters[0][j];
            halves[h][QUARTER + j] += quarters[2][j] - quarters[0][j] - quarters[1][j];
            halves[h][2 * QUARTER + j] += quarters[1][j];
        }
    }
    for (size_t j = 0; j < HALF_PRODUCT; ++j) {
        __int128 middle = halves[2][j] - halves[0][j] - halves[1][j];
        out[j] += halves[0][j] - halves[1][j];
        size_t degree = 2 * QUARTER + j;
        out[degree % RING_DEGREE] += degree < RING_DEGREE ? middle : -middle;
    }
}

void ns_sme2_child_batch(size_t count, const uint32_t *columns, const uint64_t *keys, size_t planes,
                         const uint64_t *const *positive, const uint64_t *const *negative,
                         void *scratch, __int128 *out) {
    ChildWorkspace *workspace = scratch;
    bool initialized[MAX_CHILD_PLANES] = {false};
    memset(out, 0, planes * RING_DEGREE * sizeof(*out));
    for (size_t first = 0; first < count; first += TILE_COLUMNS) {
        size_t tile = count - first < TILE_COLUMNS ? count - first : TILE_COLUMNS;
        pack_child_keys(workspace, tile, columns + first, keys);
        for (size_t plane = 0; plane < planes; ++plane) {
            if (pack_child_digits(workspace, tile, columns + first, positive[plane], negative[plane])) {
                multiply_child_tile(workspace, plane, (tile + DOT_LANES - 1) / DOT_LANES, initialized[plane]);
                initialized[plane] = true;
            }
        }
    }
    for (size_t plane = 0; plane < planes; ++plane) {
        if (initialized[plane]) {
            recombine_child_products(workspace->products[plane], out + RING_DEGREE * plane);
        }
    }
}
