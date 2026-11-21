#include <arm_sme.h>
#include <stdbool.h>
#include <stdint.h>
#include <sys/sysctl.h>

enum { DEGREE = 64, TILE = 320, WIDTH = 8, VECTOR_WORDS = 16 };

static bool has_cpu_feature(const char *name) {
    int present = 0;
    size_t size = sizeof(present);
    return sysctlbyname(name, &present, &size, NULL, 0) == 0 && present;
}

bool nightstream_has_signed_sme(void) {
    return has_cpu_feature("hw.optional.arm.FEAT_SME") && has_cpu_feature("hw.optional.arm.SME_I8I32");
}

__attribute__((target("sme")))
static svuint8_t sign_mask(uint64_t positive, uint64_t negative) __arm_streaming {
    svbool_t all = svptrue_b8();
    svuint8_t lane = svindex_u8(0, 1);
    svuint8_t bytes = svlsr_n_u8_x(all, lane, 3);
    svuint8_t shift = svand_n_u8_x(all, lane, 7);
    svuint8_t p = svtbl_u8(svreinterpret_u8_u64(svdup_u64(positive)), bytes);
    svuint8_t n = svtbl_u8(svreinterpret_u8_u64(svdup_u64(negative)), bytes);
    p = svand_n_u8_x(all, svlsr_u8_x(all, p, shift), 1);
    n = svand_n_u8_x(all, svlsr_u8_x(all, n, shift), 1);
    return svsub_u8_x(all, p, n);
}

__attribute__((target("sme")))
static void unpack_signs(uint64_t *scratch, size_t count, uint64_t *columns) __arm_streaming {
    svbool_t all = svptrue_b8();
    uint8_t *signs = (uint8_t *)scratch;
    for (size_t end = (count + 3) & ~(size_t)3; end != 0; end -= 4) {
        size_t first = end - 4;
        uint64_t masks[4][2] = {{0}};
        for (size_t k = 0; k < 4; ++k) {
            size_t i = first + k;
            if (i < count) {
                columns[i] = scratch[3 * i];
                masks[k][0] = scratch[3 * i + 1];
                masks[k][1] = scratch[3 * i + 2];
            } else {
                columns[i] = 0;
            }
        }
        svst4_u8(all, signs + first * DEGREE,
                 svcreate4_u8(sign_mask(masks[0][0], masks[0][1]), sign_mask(masks[1][0], masks[1][1]),
                              sign_mask(masks[2][0], masks[2][1]), sign_mask(masks[3][0], masks[3][1])));
    }
}

__attribute__((target("sme")))
static void pack_key(const uint64_t key[][DEGREE], const uint64_t columns[4], unsigned start,
                     uint8_t *digits) __arm_streaming {
    svbool_t all = svptrue_b8();
    svst4_u8(all, digits,
             svcreate4_u8(svld1_u8(all, (const uint8_t *)&key[columns[0]][start]),
                          svld1_u8(all, (const uint8_t *)&key[columns[1]][start]),
                          svld1_u8(all, (const uint8_t *)&key[columns[2]][start]),
                          svld1_u8(all, (const uint8_t *)&key[columns[3]][start])));
}

#define ACCUMULATE(tile) svsumopa_za32_s8_m(tile, bytes, bytes, sign, svld1_u8(bytes, key + 64 * (tile)))
#define STORE(tile) svst1_hor_za32(tile, r, words, product[r] + 16 * (tile))

__attribute__((target("sme")))
static void convolve(const int8_t *signs, const uint8_t *digits, unsigned groups,
                     unsigned start, int32_t sums[DEGREE][8]) __arm_streaming __arm_inout("za") {
    svbool_t bytes = svptrue_b8(), words = svptrue_b32();
    int32_t product[16][64];
    for (unsigned lane = 0; lane < DEGREE; lane += 16) {
        svzero_za();
        for (unsigned g = 0; g < groups; ++g) {
            svint8_t sign = svld1_s8(bytes, signs + g * 256 + lane * 4);
            const uint8_t *key = digits + g * WIDTH * 32;
            ACCUMULATE(0);
            ACCUMULATE(1);
            ACCUMULATE(2);
            ACCUMULATE(3);
        }
        for (unsigned r = 0; r < 16; ++r) {
            STORE(0);
            STORE(1);
            STORE(2);
            STORE(3);
        }
        for (unsigned r = 0; r < 16; ++r) {
            for (unsigned k = 0; k < 8; ++k) {
                unsigned degree = lane + r + start + k;
                int32_t *target = sums[degree % DEGREE];
                for (unsigned j = 0; j < 8; ++j) {
                    int32_t value = product[r][8 * k + j];
                    target[j] += degree < DEGREE ? value : -value;
                }
            }
        }
    }
}

__attribute__((target("sme"))) __arm_locally_streaming __arm_new("za")
bool nightstream_signed_tile(const uint64_t key[][DEGREE], uint64_t *scratch,
                             size_t count, int64_t out[2 * DEGREE]) {
    if (svcntw() != VECTOR_WORDS)
        return false;
    int8_t *signs = (int8_t *)scratch;
    uint8_t *digits = (uint8_t *)scratch + TILE * DEGREE;
    uint64_t *columns = scratch + TILE * (DEGREE / 8 + WIDTH);
    int32_t sums[DEGREE][8];
    svbool_t limb_words = svwhilelt_b32_u32(0, 8);
    for (unsigned lane = 0; lane < DEGREE; ++lane)
        svst1_s32(limb_words, sums[lane], svdup_s32(0));
    unpack_signs(scratch, count, columns);
    for (unsigned start = 0; start < DEGREE; start += WIDTH) {
        for (unsigned b = 0; b < count; b += 4)
            pack_key(key, columns + b, start, digits + b * WIDTH * 8);
        convolve(signs, digits, (count + 3) / 4, start, sums);
    }
    for (unsigned lane = 0; lane < DEGREE; ++lane) {
        for (unsigned limb = 0; limb < 2; ++limb) {
            const int32_t *s = sums[lane] + 4 * limb;
            out[2 * lane + limb] += s[0] + s[1] * INT64_C(256) + s[2] * INT64_C(65536) + s[3] * INT64_C(16777216);
        }
    }
    return true;
}
