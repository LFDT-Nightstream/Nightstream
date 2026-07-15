# Nightstream WASM folding demo

A local browser UI backed by the native `neo-wasm` tracer. The default program
increments a counter in a short loop and uses no linear memory. The WAT source
is editable; the server compiles and executes it through Wasmtime, then returns
the normalized rows consumed by the proof frontend. The example selector
includes a loop with a direct call and a mutable function table that exercises
`ref.func`, `table.set`, and nested `call_indirect` execution. A third example
updates mutable global state through a parameterized helper function. The
division-by-zero example ends in a proved terminal trap.
The branch-table example exposes a three-way `br_table` control choice.

Run it from the repository root:

```sh
cargo run --release -p nightstream-wasm-folding-demo
```

Open <http://127.0.0.1:3000>.

Click any trace row to inspect the exact normalized JSON object returned by the
native tracer API.

The proof flow has separate “Preprocess” and “Fold & verify” actions. The server
retains the exact normalized trace, preprocessing consumes that retained trace
without executing Wasmtime again, and the proof action reuses the most recently
prepared relation. Both long-running actions show activity in their buttons.
Editing the WAT requires a new trace. There are three explicit proof modes:

- **Folding audit · no NIFS.V** is the fastest sanity-check path. It uses
  batch 64, does not constrain `NIFS.V` inside F′, and verifies by replaying
  the full retained audit history. It is not a succinct recursive proof.

- **Recursive IVC · no memory consistency** uses batch 32 and the authoritative
  recursive F′ relation with constrained `NIFS.V`. It proves the normalized
  WASM transition relation but does not authenticate ROM or RAM consistency.
- **Recursive IVC · Nebula memory consistency** additionally proves the
  Nebula memory relation and complete initial/final scan. It is substantially
  slower.

Both modes show the Poseidon2-derived verifier-key digest. The Nebula mode also
shows its memory-plan and initial-RAM digests. These identify verifier material
created during preprocessing; they are not standalone evidence that an edited
program is valid. The demo calls an argumentless `main`; change constants
directly in the editable WAT before tracing or proving.
