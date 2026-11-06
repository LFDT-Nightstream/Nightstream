# Applications

`nightstream::application` owns the application program and its builder.
`Circuit::compile` puts that program in place of the Lean application in the
light and refresh step packages that Lean emits. The Poseidon2 hash chain and
the other supported application programs use this same path.

The compiler validates the program, builds its rows and witness recipes, and
binds it into the package identity. Runtime witness generation must satisfy
those rows; an application output or digest is not accepted as authority.
See the [current API](../../crates/nightstream/README.md) and
[`application`](../../crates/nightstream/src/application/mod.rs).

`nightstream::nebula` builds Nebula memory applications on the same step
packages. A memory circuit needs `nebula::Verifier`.
