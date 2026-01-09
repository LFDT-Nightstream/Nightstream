# whir-p3

Vendored copy of the upstream `whir-p3` crate (https://github.com/WizardOfMenlo/whir-p3),
which implements the WHIR protocol using the Plonky3 (`p3-*`) libraries.

Upstream snapshot: `53caa719ccf135a574071500552b878fb9bd0a12`.

## Why this is vendored in this workspace

`neo-closure-proof`’s WHIR backend needs WHIR and Neo to share the *exact same* Plonky3 crates
(`p3-field`, `p3-goldilocks`, etc). If WHIR pulls Plonky3 from a different source/revision than the
rest of the workspace, you end up with two distinct `p3-goldilocks::Goldilocks` types, which:

- forces lossy/bug-prone conversions between “two Goldilocks types”, and
- turns some operations into large `Vec` copies (bad for performance and memory at Neo sizes).

Upstream `whir-p3` is currently pinned to Plonky3 via a git rev, while this workspace uses crates.io
`p3-* = 0.4.1`. We vendor a known-good snapshot so we can keep WHIR’s code pinned while aligning
its Plonky3 dependencies to the workspace.

## What changed vs upstream

This vendored copy is intended to be source-identical to upstream, except for dependency plumbing:

- `Cargo.toml`: Plonky3 `p3-*` dependencies are switched from git to crates.io `0.4.1`.

## Licensing

Upstream is dual-licensed under MIT or Apache-2.0; see `LICENSE-MIT` and `LICENSE-APACHE`.

## Updating

If you want to refresh this vendor:

1. Sync the source from upstream `whir-p3`.
2. Keep `p3-*` dependency versions aligned with the workspace.
3. Sanity check: `cargo tree -p neo-closure-proof --features whir-p3-backend | rg p3-goldilocks`
   should show a single version.
