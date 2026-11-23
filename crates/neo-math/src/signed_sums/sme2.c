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
