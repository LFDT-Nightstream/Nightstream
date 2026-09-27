# Applications

`nightstream::application` owns the application program and its builder.
`Circuit::compile` combines that program with the single shared recursive
verifier. The same general assembly path serves the Poseidon2 hash chain and
other supported application programs.

The compiler validates the program, builds its rows and witness recipes, and
binds it into the package identity. Runtime witness generation must satisfy
those rows; an application output or digest is not accepted as authority.
See the [current API](../../crates/nightstream/README.md) and
[`application`](../../crates/nightstream/src/application/mod.rs).

The old WASM, Nebula, and direct-CCS lifecycle consumers are retired. They do
not define additional maintained application paths.
