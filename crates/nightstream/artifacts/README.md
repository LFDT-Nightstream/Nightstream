Library users need these saved artifacts:

| Artifact | Delivery and use |
| --- | --- |
| `shared-verifier-v1.json` | Included by the `nightstream` Rust build. Declares mandatory components, application ports, dimensions, and relocation rules. |
| `nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | Included in this package. Pass these bytes to `Circuit::compile` as verifier configuration. |
| `shared-formulas-v1.json` | Included by the `nightstream-fprime` dependency's Rust build. Supplies the shared Poseidon2 and Phi81 formulas. |

Application preparation, proving, verification, and Rust builds do not run
Lean. The selected blueprint is required even for a different Rust application:
the assembler retains its shared verifier and replaces only its application.
It checks the original package and key identities before that replacement.

In this repository, the blueprint path is a symbolic link to the existing
file under `formal/nightstream-fprime/artifacts`. This avoids another large
copy in Git. The explicit Cargo `include` list includes the artifact, and
[Cargo packaging replaces symbolic links with their target files](https://doc.rust-lang.org/cargo/commands/cargo-package.html).
An extracted Cargo package therefore contains the JSON file and needs no
`formal` directory. A source checkout must contain the full Git LFS payload
before packaging; an LFS pointer is not a usable blueprint.

Saved execution results and the small independent application reference are
test inputs under `tests/fixtures`. They are included for conformance tests.
They are not producer inputs or prerequisites for a consumer's proof run.
Those tests use only package-local paths. Their source records are in
`tests/fixtures/README.md`.

The selected blueprint uses the wide PiRLC sampler and the general application
layout. It has 3,256,394 logical rows and 137,646,810 committed coordinates.
The manifest source is
`formal/nightstream-fprime/NightstreamFPrime/Export/SharedVerifier.lean`.
Its definitions and contract names are recorded in the JSON. The maintainer
toolchain is `leanprover/lean4:v4.32.2`, as set by the formal project's
`lean-toolchain` file. All artifacts use the Nightstream Goldilocks profile
with `b = 2`, `k_rho = 16`, and `B = 2^16`.

The reference application has four witness words, 7,696 local words, and
7,700 rows. Manifest version 3 has one application reference and one local
index. The selected application and other applications use the same assembly
code and witness executor. Application state has four input words and four
output words. The manifest exports the source-row, source-column, and
retained-carrier conditions for the `2^28` profile.

The Rust `Circuit` supports at most 2,851,939 witness and local words together.
This bound follows from the
[approved key capacity](../../neo-ajtai/src/nightstream_fprime_setup.rs) of
22 × 4,708,530 ring columns and the exported width
`137331104 + 41 * (witness_words + local_words)`. Each package binds its exact
key prefix. The selected reference uses 2,549,015 ring columns.
Assembly-only dimension checks
also require the source rows, source columns, and padded retained carrier to fit
the declared domain.

From `formal/nightstream-fprime`, run the maintainer commands one at a time.
`validate.sh` applies the 1,500-second cap required by that project's `AGENTS.md`.

```sh
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh build emitSharedVerifier
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh build checkSharedVerifier
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh file tests/SharedVerifier.lean
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh lean-executable .lake/build/bin/checkSharedVerifier
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh lean-executable .lake/build/bin/emitSharedVerifier ../../crates/nightstream/artifacts/shared-verifier-v1.json
```

The native check compares the manifest with both existing Lean applications.
The file check audits the referenced theorems; it does not run that comparison.
`tests.SharedVerifier` is an explicit test-library root and is imported by
`tests/Axioms.lean`. The runtime comparison uses the `checkSharedVerifier` target.
These results do not prove the Rust assembler or arbitrary application semantics.

To regenerate a separate copy of the selected blueprint for comparison, use
the existing production emitter from the same directory:

```sh
elan run leanprover/lean4:v4.32.2 bash scripts/validate.sh emit /tmp/nightstream-reference.json
```

Keep the selected blueprint and pins until the complete circuit and execution
comparisons pass. The test-only per-application emitter selects another
application and is not a replacement for this command.

Before a Cargo release, inspect `cargo package --list -p nightstream` and test
the extracted package with the workspace's released dependencies and no Lean
tools. Registry publication is separate work: the current workspace uses path
dependencies without release-version keys. This artifact preparation does not
choose a registry, publish dependencies, or claim that a registry release exists.
