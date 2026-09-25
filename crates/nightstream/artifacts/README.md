Library users need these saved artifacts:

| Artifact | Delivery and use |
| --- | --- |
| `shared-verifier-v1.json` | Included by the `nightstream` Rust build. Declares mandatory components, application ports, dimensions, and relocation rules. |
| `nightstream-fprime-stage1-poseidon2-hash-chain-v1.json` | Included in this package. Pass these bytes to `Circuit::prepare` as verifier configuration. |
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

The selected blueprint is retained from repository snapshot
`9787d8e77069246e3e2afc7dcfab755556fd5023`. Its data was last changed by
`dd38a22f9e213538966bf0d5c867e477b3bc016d`. The manifest source is
`formal/nightstream-fprime/NightstreamFPrime/Export/SharedVerifier.lean`.
Its definitions and contract names are recorded in the JSON. The maintainer
toolchain is `leanprover/lean4:v4.32.2`, as set by the formal project's
`lean-toolchain` file. All artifacts use the selected Nightstream Goldilocks
profile with `b = 2`, `k_rho = 16`, and `B = 2^16`.

The reference application has four witness words, 7,696 local words, and 7,700
rows. Application state has four input words and four output words. The manifest
exports the existing source-row, source-column, and retained-carrier conditions
for the `2^28` Nightstream Goldilocks profile with `k_rho = 16`.

The current Rust `Circuit` supports at most 7,701 witness and local words
together. This bound follows from the
[selected key's](../../neo-ajtai/src/nightstream_fprime_setup.rs)
253,011,276-coordinate carrier and the exported width
`252695531 + 41 * (witness_words + local_words)`. Assembly-only dimension checks
also require the source rows, source columns, and padded retained carrier to fit
the declared domain.

From `formal/nightstream-fprime`, run the maintainer commands one at a time.
The outer 1,500-second cap follows that project's `AGENTS.md`.

```sh
timeout --signal=KILL 1500 bash scripts/validate.sh build emitSharedVerifier
timeout --signal=KILL 1500 bash scripts/validate.sh build checkSharedVerifier
timeout --signal=KILL 1500 bash scripts/validate.sh file tests/SharedVerifier.lean
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/checkSharedVerifier
timeout --signal=KILL 1500 bash scripts/validate.sh lean-executable .lake/build/bin/emitSharedVerifier ../../crates/nightstream/artifacts/shared-verifier-v1.json
```

The native check compares the manifest with both existing Lean applications.
The file check audits the referenced theorems; it does not run that comparison.
`tests.SharedVerifier` is an explicit test-library root and is imported by
`tests/Axioms.lean`. The runtime comparison uses the `checkSharedVerifier` target.
These results do not prove the Rust assembler or arbitrary application semantics.

To regenerate a separate copy of the selected blueprint for comparison, use
the existing production emitter from the same directory:

```sh
timeout --signal=KILL 1500 bash scripts/validate.sh emit-poseidon2-hash-chain-v1 /tmp/nightstream-reference.json
```

Keep the selected blueprint and pins until the complete circuit and execution
comparisons pass. The test-only per-application emitter selects another
application and is not a replacement for this command.

Before a Cargo release, inspect `cargo package --list -p nightstream` and test
the extracted package with the workspace's released dependencies and no Lean
tools. Registry publication is separate work: the current workspace uses path
dependencies without release-version keys. This artifact preparation does not
choose a registry, publish dependencies, or claim that a registry release exists.
