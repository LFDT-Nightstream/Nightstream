# neo-quickcheck-tests

Property-based tests that verify critical protocol invariants using randomized generation to explore mathematical identities and security properties fundamental to Nightstream's correctness.

## Running Tests

**By design, these tests only run when explicitly requested:**

```bash
# Skip all quickcheck tests (default - compiles 0 tests)
cargo test -p neo-quickcheck-tests

# Run property-based tests (with feature flag)
cargo test -p neo-quickcheck-tests --features quickcheck
```

**Why this flag?**
Nightstream's property-based tests are intended to be invoked explicitly in a "security/property verification" mode rather than during routine development cycles.

## ✅ Final Test Suite

| Test File | Tests | Type | Purpose |
|-----------|-------|------|---------|
| `bridge_parity.rs` | 1 | Proptest | Header encoding consistency |
| `dec_props.rs` | 3 | 2 Proptest + 1 QuickCheck | Base-b decomposition properties |
| `header_sensitivity.rs` | 1 | QuickCheck | Digest bit-flip detection |
| `rb_props.rs` | 1 | QuickCheck | MLE partition of unity |
| `rlc_props.rs` | 1 | QuickCheck | Π_RLC linearity |
| `security_red_team.rs` | 2 | Regular tests | Security guard rails |

**Total: 9 tests providing comprehensive property verification**

## 🎯 Key Mathematical Properties Now Tested

- **Π_RLC Linearity**: `combine(evaluate(A), evaluate(B)) == evaluate(combine(A,B))`
- **MLE Partition of Unity**: `∑_j w_j(r) = 1` for Boolean hypercube tensor weights  
- **Base-b Decomposition**: Recomposition correctness over both F and K
- **Range Constraints**: Polynomial vanishing on correct digit sets `v(v-1)(v+1) = 0`
- **Bridge Consistency**: Header encoding matches circuit public values exactly
- **Security Guard Rails**: Fail-closed behavior when assumptions violated
- **Digest Sensitivity**: Any bit flip in digest changes header encoding

## Usage

```bash
# Run all property-based tests
cargo test -p neo-quickcheck-tests --features quickcheck

# Run specific mathematical property tests
cargo test -p neo-quickcheck-tests --features quickcheck rlc_linearity_holds
cargo test -p neo-quickcheck-tests --features quickcheck rb_is_partition_of_unity

# Run bridge consistency tests
cargo test -p neo-quickcheck-tests --features quickcheck header_encoding_matches_public_values
cargo test -p neo-quickcheck-tests --features quickcheck header_digest_flip_changes_bytes

# Run field arithmetic property tests  
cargo test -p neo-quickcheck-tests --features quickcheck recomposition_f_roundtrip_and_neg
cargo test -p neo-quickcheck-tests --features quickcheck b2_range_poly_vanishes_on_digits

# Run security guard rail tests
cargo test -p neo-quickcheck-tests --features quickcheck bridge_rejects_missing_ajtai_rows
```