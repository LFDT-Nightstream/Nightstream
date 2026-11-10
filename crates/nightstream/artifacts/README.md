The `nightstream` Rust build embeds three Lean packages as zstd copies:

| Copy | Lean output | Use |
| --- | --- | --- |
| `nightstream-fprime-bounded-depth-light-v1.json.zst` | `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json` | The light step circuit. |
| `nightstream-fprime-bounded-depth-refresh-v1.json.zst` | `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json` | The refresh step circuit. |
| `nightstream-fprime-bounded-depth-finish-v1.json.zst` | `formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-finish-v1.json` | The finish circuit of the final proof. |

The crate decodes the copies once per process. Their identities are their pins
(`BOUNDED_DEPTH_PACKAGES_IDENTITY` and `BOUNDED_DEPTH_FINISH_IDENTITY` in
`nightstream-fprime`) without a hash at run time, because the bytes are part of
the binary. `embedded_packages_have_their_pinned_identities` checks that each
copy hashes to its pin. Every `Circuit::compile` then puts the Rust
application in place of Lean's.
Proving, verification and Rust builds do not run Lean.

The copies are normal Git files, so a Git dependency of `nightstream` builds
and proves without any file of the repository. Together they add about
2.2 MB to a binary, and each unpacks once. After a Lean regeneration of a
package, make its copy again from this directory, then run
`cargo test --release -p nightstream --lib artifacts::`, which checks that
each copy is the Lean output and hashes to its pin:

```sh
zstd -19 -q -f -o nightstream-fprime-bounded-depth-light-v1.json.zst < ../../../formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-light-v1.json
zstd -19 -q -f -o nightstream-fprime-bounded-depth-refresh-v1.json.zst < ../../../formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-refresh-v1.json
zstd -19 -q -f -o nightstream-fprime-bounded-depth-finish-v1.json.zst < ../../../formal/nightstream-fprime/artifacts/nightstream-fprime-bounded-depth-finish-v1.json
```

The Nebula memory packages of Lean
(`nightstream-fprime-bounded-depth-nebula-{v1,n1}-{light,refresh}-v1.json`) are
test inputs only. The internal `nebula::Package` loads them for the parity
tests.
