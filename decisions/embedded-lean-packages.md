# Embedded Lean Packages

**Status:** Accepted

## Problem

Every circuit needs two Lean packages: the Stage 1 reference package, whose
shared F′ verifier `Circuit::compile` keeps, and the compression terminal
package. They are 50 MB and 40 MB of JSON in Git LFS. Cargo does not fetch
LFS files for a Git dependency, so a dependency such as Starstream got only
the pointer files, and it had to ship the reference package and pass its
bytes.

## SuperNeo

SuperNeo v1.2 fixes no package format or way to deliver one.

## Decision

The owner approved this on 2026-11-03.

1. **Embedded zstd copies.** `crates/nightstream/artifacts` holds a zstd copy
   of each package as a normal Git file. The `nightstream` build embeds both,
   and `ruzstd`, a pure-Rust decoder, unpacks each once.
2. **No reference argument.** `Circuit::compile` and `nebula::Circuit::compile`
   use the embedded reference package. Its identity pins accept only that
   package, so an argument gave a caller no choice.
3. **The Lean outputs stay the source.** The raw JSON files stay in Git LFS
   for Lean and for the tests. A test checks that each copy unpacks to its
   Lean output, and the pins of each package still decide acceptance.
