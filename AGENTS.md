# AGENTS.md

## Code
- Avoid extending files over 2,000 LOC

## Testing
- Never add tests in the same implementation file, always prefer to add them to a file inside tests/ (current or new)
- If you add a test to catch a problem, the test should fail if aims to confirm a problem.

## Build & Test Commands
- When running tests use --release eg cargo test --workspace --release
- For extra debugs use debug-logs eg --features paper-exact,debug-logs

## Profiling

- CPU + Memory profiling: ./scripts/profile_for_ai.sh neo-fold test_sha256_single_step test_sha256_preimage_256_bytes --ignored
- Deep memory profiling (allocation sites with full call context): ./scripts/profile_memory_deep.sh neo-fold test_sha256_single_step test_sha256_preimage_256_bytes --ignored
