# Prepared fixed-source bound

Counted on 2026-09-22 after CPU capture 91637 ended with exit 0. No Rust build, test, proof, or profile was run for this count.

The input was `/Users/nijaar/Library/Application Support/Nightstream/runs/metal-norm-20260921-bgylh03h/prepared-poseidon2.nsc` (127,306,104 bytes). Its inner magic at bytes 12..20 was `NSFPREP1`. The little-endian length at bytes 20..28 was 124,458,756. Only that fixed JSON section, starting at byte 100, was counted; binary application records were excluded.

| Quantity | Exact count |
| --- | ---: |
| Array nodes, including empty arrays | 13,612,550 |
| Unsigned numeric nodes | 18,424,983 |
| Native reference nodes `N_ref` | **32,037,533** |
| Constant part `N0 = N_ref - 12` | **32,037,521** |
| Maximum compiler nodes `N_max = N_ref + 7,696` | **32,045,229** |
| Compact source byte ceiling `21 * N_max` | **672,949,809** |

The count used a buffered byte stream with Python's standard `io.DEFAULT_BUFFER_SIZE`. It counted `[` and decimal digit runs, carried digit state across buffers, required matching opening/closing array counts, and rejected bytes outside numeric-array JSON syntax. It did not allocate a JSON Value tree. The complete reference application metadata occurred once in the fixed section at artifact byte 120,107,802 and matched `W=4`, `L=7,696`, `R=7,700`.

## Compiler derivation

Here W is the private-input count, L is the generated-local count, and R is the application-row count. Count every array as one node and every number as one node.

The exact formula for the selected compiler is:

`N(W,L,R) = N0 + W + 4*[W>0] + 4*[L>0]`.

The source evidence is:

- `src/assembly/application.rs` emits W witness columns. Its input/output port arrays always contain four entries, and its seven dynamic payload arrays are empty.
- `src/assembly/source.rs::replace_application` removes the same reference application rows/instructions/batches for every compiled application. Its remaining changes replace numeric atoms. The terminal and next-preimage replacements have fixed shape.
- `src/assembly/connect.rs::matrix` changes numeric relocation targets and replaces the selected connector with the same number and shape of blocks.
- `src/assembly/connect.rs::assignment` recompresses run lists. In `artifacts/shared-verifier-v1.json`, only blocks 28 and 29 have variable run counts: each has one step-one source run of length W or L. It produces no run when its length is zero, otherwise one run. Each emitted run is one array plus three numbers, hence four nodes. Other assignment blocks retain their run counts.
- The only shifted Phi81 value-source run is index 17, with start `29,336,447+W+L` and length 270. Its neighbors are `46,312..47,499` and `1,286..1,555`. It cannot merge with either for any supported W/L. All 68 Phi81 source-run templates have step one and length at least 108, so this shift does not alter their compressed list length.

The selected key supplies `4,685,394*54 = 253,011,276` scalar coordinates (`neo-ajtai/src/nightstream_fprime_setup.rs`). The manifest width is `252,695,531+41*(W+L)`. Existing key-prefix validation therefore gives `W+L <= 7,701`.

The variable part is largest at **W=7,700, L=1**: `7,700+4+4 = 7,708`. With L=0 its maximum is only `7,701+4 = 7,705`. The native reference contributes `4+4+4 = 12`, so `N_max = N_ref-12+7,708 = N_ref+7,696`. R only changes numeric atoms. This maximum is derived from the compiler source; no maximum-shape artifact was built for this count.

## Byte ceiling and scope

Compiler output is compact numeric-array JSON. A u64 atom needs at most 20 decimal bytes; an array adds two brackets; total commas are at most `N-1`. If P and A count numeric and array nodes, its length is at most `20P+2A+(N-1) <= 21N`. Thus **672,949,809 bytes** is a format-derived preflight bound, not an arbitrary storage cap. A source-length check alone is insufficient: the decoder must also enforce **32,045,229 nodes** before growing a Value tree. Neither count is authority for the circuit's saved identities.

These checks bound the cardinality of accepted fixed metadata, including redundant terms and unused templates. They do not change supported compiler output, b=2, or k_rho=16. They also do not establish an exact universal RSS bound: decoded element sizes, vector capacities, simultaneous owners, allocator/runtime storage, and driver residency remain separate accounting terms. The source-byte ceiling must not be reported as a resident-memory ceiling.

## Subsequent compiler regression

The separate `assembled_fixed_source_reaches_the_compiler_node_bound` regression in [tests/assembly_internal/encoding.rs](../../assembly_internal/encoding.rs) passed in 2.89 seconds. It assembles both W=7,700/L=1 and W=L=0, and checks exactly **32,045,229** and **32,037,521** nodes. This later check validates the constants against compiler output; the original count above did not build a maximum-shape artifact. Final-image measurement remains pending.
