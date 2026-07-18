#ifndef NEO_METAL_BENCH_H
#define NEO_METAL_BENCH_H

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

int32_t neo_metal_benchmark_run_json(
    const uint8_t *config_ptr,
    size_t config_len,
    uint8_t **out_ptr,
    size_t *out_len,
    uint8_t **error_ptr,
    size_t *error_len);

void neo_metal_benchmark_free_bytes(uint8_t *ptr, size_t len);

#ifdef __cplusplus
}
#endif

#endif
