# WASM/Nebula Metal demo

Branch: `enzo/wasm-vm-host-abi-impl-metal-demo`, based on `acb4b9a1b`
(August 6) with the host ABI and neo-application work rebased onto it.
The September integration remains on `enzo/wasm-vm-host-abi-impl-metal`.

Run on an Apple machine with the Metal toolchain and Python 3:

```sh
python3 scripts/run_wasm_metal_demo.py
```

The runner compiles first, then applies the AGENTS.md 300-second hard cap
to the test process group. Run it outside a sandbox that blocks Metal access.
Compilation is separate from the timings below.

## Execution covered

The component's `run(21)` export calls the `double` host import, stores its
result at linear-memory address zero, reloads it, and returns 42. Three
Poseidon2 event blocks bind the export input, import argument/result, and
export output. The host computation is represented by the claimed event
transcript; this proof does not verify the host's implementation of doubling.

The 88 normalized rows occupy four application steps of 22 rows each. The
runtime log must show `Base`, `BootstrapRecursive`, and `Recursive` branches.
The latter branches construct the in-circuit NIFS verifier witness. Nebula
authenticates the memory accesses and completes its segment scan, and the
delayed terminal fold is present. The test requires actual Metal dispatches
in both proving and verification, with no CPU fallback. It also checks
rejection of changed output and import-transcript claims.

## Measured run

September 14, 2026, on the local development machine:

| Phase | Seconds |
|---|---:|
| Preprocessing | 48.371 |
| Metal static preparation | 3.989 |
| Proving, including the terminal fold | 38.521 |
| Terminal verification | 2.485 |
| Cold total through successful verification | 94.214 |

There were 931 Metal dispatches through successful verification. The full
test, including the two rejection checks and cleanup, passed in 99.84 seconds.
Prove plus verify was 41.006 seconds; this is the measured online phase, not
a separately repeated warm-run benchmark.

## Scope

This is the historical uncompressed recursive folding proof. Terminal
verification checks the folded witness openings without replaying the full
history. It does not produce a succinct Spartan/WHIR final proof or establish
zero knowledge for the exported proof artifact.

The demo uses test parameters (`kappa=1`, `k_rho=14`, `lambda=20`) and a
64-word linear-memory domain. ROM and RAM each have 4096 cells, with 2048
cells scanned per application step. The fixture declares one WASM page but
only accesses address zero; the reduced test entrypoint does not promise
the capacity of a full WASM page. Normal preprocessing retains its page
capacity checks. Imported memories and globals remain rejected.

The assembled relation has 13,355,201 rows and 18,558,774 columns. These
timings apply to this fixed fixture and historical protocol revision;
they are not production-security or current SuperNeo v1.1 conformance results.
