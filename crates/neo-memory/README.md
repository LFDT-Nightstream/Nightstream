# neo-memory

Memory and lookup argument implementation for Nightstream.

## Overview

This crate provides:

- **Twist**: Read/write memory arguments via sparse increment recurrence
- **Shout**: Read-only lookup table arguments
- **CPU Constraints**: Binding constraints between CPU semantics and the shared memory/lookup bus
- **Address helpers**: Bit-address validation + encoding utilities
- **Witness Building**: Tools for constructing memory/lookup witnesses from VM traces

## Modules

| Module | Description |
|--------|-------------|
| `twist.rs` | Twist (R/W memory) Route A proof metadata |
| `twist_oracle.rs` | Sum-check oracles for Twist |
| `shout.rs` | Shout (lookup) Route A proof metadata |
| `addr.rs` | Bit-address validation + helpers |
| `cpu/` | CPU integration module (constraints + R1CS adapter) |
| `cpu/bus_layout.rs` | Canonical shared-bus layout (single source of truth) |
| `cpu/constraints.rs` | CPU-to-bus binding constraints (adapted from Jolt) |
| `cpu/r1cs_adapter.rs` | R1CS-based CPU adapter for the shared bus |
| `builder.rs` | Shard/step witness construction |
| `witness.rs` | Witness data structures |

## CPU Constraints

The `cpu/constraints.rs` module provides security-critical constraints that bind CPU instruction semantics to the shared memory/lookup bus. Without these constraints, a malicious prover could create divergent CPU and memory states.

### Credits

The constraint logic in `cpu/constraints.rs` is **adapted from the Jolt zkVM project**:

- **Repository**: [https://github.com/a16z/jolt](https://github.com/a16z/jolt)
- **Original file**: `jolt-core/src/zkvm/r1cs/constraints.rs`
- **License**: Apache-2.0 / MIT

### Constraints Implemented (Core)

| Constraint | Formula | Purpose |
|------------|---------|---------|
| Load Value Binding | `is_load * (rd_value - bus_rv) = 0` | CPU load result matches memory |
| Store Value Binding | `is_store * (rs2_value - bus_wv) = 0` | CPU store value matches memory |
| Load Selector Binding | `is_load - has_read = 0` | CPU load flag matches bus |
| Store Selector Binding | `is_store - has_write = 0` | CPU store flag matches bus |
| Load Address Binding | `is_load * (effective_addr - pack(ra_bits)) = 0` | CPU load addr matches bus bits |
| Store Address Binding | `is_store * (effective_addr - pack(wa_bits)) = 0` | CPU store addr matches bus bits |
| Read Value Padding | `(1 - has_read) * rv = 0` | Zero inactive read values |
| Write Value Padding | `(1 - has_write) * wv = 0` | Zero inactive write values |
| Read Address Padding | `(1 - has_read) * ra_bits[i] = 0` | Zero inactive read addresses |
| Write Address Padding | `(1 - has_write) * wa_bits[i] = 0` | Zero inactive write addresses |
| Increment Padding | `(1 - has_write) * inc = 0` | Zero inactive increments |
| Lookup Value Binding | `is_lookup * (lookup_out - bus_val) = 0` | CPU lookup result matches table |
| Lookup Selector Binding | `is_lookup - has_lookup = 0` | CPU lookup flag matches bus |
| Lookup Key Binding | `is_lookup * (lookup_key - pack(addr_bits)) = 0` | CPU lookup key matches bus bits |
| Lookup Value Padding | `(1 - has_lookup) * val = 0` | Zero inactive lookup values |

### Usage Example

```rust
use neo_memory::cpu::constraints::{CpuColumnLayout, CpuConstraintBuilder, TwistBusConfig};

// Define where CPU columns are in the witness
let cpu_layout = CpuColumnLayout {
    is_load: 1,
    is_store: 2,
    rd_write_value: 3,
    rs2_value: 4,
    effective_addr: 5,
    is_lookup: 6,
    lookup_key: 7,
    lookup_output: 8,
};

// Create constraint builder (const_one_col must be fixed to 1)
let mut builder = CpuConstraintBuilder::new(n, m, bus_base, /*const_one_col=*/ 0);

// Add constraints for each memory instance
let twist_cfg = TwistBusConfig::new(4); // 4 address bits
builder.add_twist_instance(&twist_cfg, &cpu_layout);

// Build CCS or extend existing CCS
let ccs = builder.build()?;
```

## Twist (R/W Memory)

Twist models memory as a recurrence via sparse updates:

```
Val_{t+1} = Val_t + Inc_t
```

**Shared-bus mode:** Twist consumes bus fields opened from the CPU commitment (the tail of the CPU witness `z`):
- `ra_bits`, `wa_bits`
- `has_read`, `has_write`
- `rv`, `wv`
- `inc_at_write_addr`

## Shout (Read-Only Lookups)

Shout proves that when `has_lookup[t] = 1`, the committed `val[t]` matches `table[addr[t]]`.

**Shared-bus mode:** Shout consumes bus fields opened from the CPU commitment:
- `addr_bits`
- `has_lookup`
- `val`

## Address Encoding

Addresses use compact **bit-decomposition** instead of one-hot vectors:
- Each address is `d` components in base `n_side`
- Each component commits `ell = ceil(log2(n_side))` bit-columns
- Address width: `d * ell` columns instead of `d * n_side` (~32× reduction)

## Tests

```bash
# Run all neo-memory tests
cargo test -p neo-memory --release

# CPU constraints tests
cargo test -p neo-memory --test cpu_constraints_tests --release -- --nocapture
```

## License

Licensed under the [Apache License, Version 2.0](../../LICENSE).

### Third-Party Notices

- **Jolt zkVM**: CPU constraint logic adapted under Apache-2.0 / MIT license.
  See [https://github.com/a16z/jolt](https://github.com/a16z/jolt).
