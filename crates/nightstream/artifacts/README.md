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

The blueprint and manifest come from the current canonical Lean package.
The manifest source is
`formal/nightstream-fprime/NightstreamFPrime/Export/SharedVerifier.lean`.
It emits version 3 with one reference application and fourteen phase children.
The repository toolchain remains `leanprover/lean4:v4.32.2`. A compatible local
compiler can be selected with `elan run TOOLCHAIN`. All artifacts use
`b = 2`, `k_rho = 16`, and `B = 65536`.

The reference application has four witness words, 5,480 local words, and 5,484
rows. Application state has four input words and four output words. The manifest
exports the existing source-row, source-column, and retained-carrier conditions
for the `2^28` Nightstream Goldilocks profile with `k_rho = 16`.

The current selected application has 43,963,750 logical coordinates and
43,963,776 padded coordinates, using 814,144 columns of the unchanged fixed
key. The approved maximum is 4,708,530 columns. With logical width
`43738906 + 41 * (witness_words + local_words)`, that maximum permits 5,134,675
application witness and local fields together. Source rows, source columns,
and the padded carrier must also fit the declared domain.

From `formal/nightstream-fprime`, run the maintainer commands one at a time.
The wrapper enforces that project's 1,500-second cap.

```sh
bash scripts/validate.sh build emitSharedVerifier
bash scripts/validate.sh build checkSharedVerifier
bash scripts/validate.sh file tests/SharedVerifier.lean
bash scripts/validate.sh lean-executable .lake/build/bin/checkSharedVerifier
bash scripts/validate.sh lean-executable .lake/build/bin/emitSharedVerifier ../../crates/nightstream/artifacts/shared-verifier-v1.json
```

The native check compares the manifest with both existing Lean applications.
The file check audits the referenced theorems; it does not run that comparison.
`tests.SharedVerifier` is an explicit test-library root and is imported by
`tests/Axioms.lean`. The runtime comparison uses the `checkSharedVerifier` target.
These results do not prove the Rust assembler or arbitrary application semantics.

To regenerate a separate copy of the selected blueprint for comparison, use
the existing production emitter from the same directory:

```sh
bash scripts/validate.sh emit /tmp/nightstream-reference.json
```

Check emitted candidates before installing matching local blueprint and pins.
Promote saved fold fixtures only after the fresh native and Lean comparisons
pass. The test-only per-application emitter selects another
application and is not a replacement for this command.

Before a Cargo release, inspect `cargo package --list -p nightstream` and test
the extracted package with the workspace's released dependencies and no Lean
tools. Registry publication is separate work: the current workspace uses path
dependencies without release-version keys. This artifact preparation does not
choose a registry, publish dependencies, or claim that a registry release exists.
