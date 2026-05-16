# Rv64IMNarrowMemoryHelpers Spec

## Purpose

- **What it is**: The theorem-facing contract for the Stage-1 arithmetic helper
  relations used by lowered narrow loads and narrow stores.
- **What it is not**: It is not the Stage-2 RAM history theorem and it does not
  own aligned 64-bit RAM read/write authentication.
- **Protocol role**: It fixes the exact aligned-address, byte-offset,
  `extract_extend`, and `blend` semantics consumed by `VExtractLoad` and
  `VBlendStore`.

## Target Formulas

Define byte extraction:

$$
\mathrm{byteAt}(word, k) := \left\lfloor \frac{word}{2^{8k}} \right\rfloor \bmod 256.
$$

Define aligned-address and byte-offset derivation:

$$
\mathrm{alignDown8}(addr) := addr - (addr \bmod 8),
$$

$$
\mathrm{byteOffset8}(addr) := addr \bmod 8.
$$

The exact alignment decomposition theorem target is:

$$
\mathrm{alignDown8}(addr) + \mathrm{byteOffset8}(addr) = addr,
$$

with

$$
\mathrm{byteOffset8}(addr) < 8.
$$

Define the raw extracted subword:

$$
\mathrm{extractRaw}(word, off, width)
:=
\left\lfloor \frac{word}{2^{8 \cdot off}} \right\rfloor \bmod 2^{8 \cdot width}.
$$

Define sign fill:

$$
\mathrm{signFill}(word, off, width)
:=
\begin{cases}
255 & \text{if } \mathrm{byteAt}(word, off + width - 1) \ge 128, \\
0 & \text{otherwise.}
\end{cases}
$$

Define the exact Stage-1 narrow-load helper:

$$
\mathrm{extractExtend}(word, off, width, unsigned)
:=
\begin{cases}
\mathrm{extractRaw}(word, off, width) & \text{if } unsigned = 1, \\
\mathrm{extractRaw}(word, off, width)
+ \sum_{k=width}^{7} 2^{8k} \cdot \mathrm{signFill}(word, off, width) & \text{otherwise.}
\end{cases}
$$

Define the exact Stage-1 narrow-store helper:

$$
\mathrm{blend}(word, src, off, width)
:=
\sum_{k=0}^{7}
2^{8k}
\cdot
\begin{cases}
\mathrm{byteAt}(src, k - off) & \text{if } off \le k < off + width, \\
\mathrm{byteAt}(word, k) & \text{otherwise.}
\end{cases}
$$

The theorem-facing proof packages are:

- `NarrowMemoryExtractProofPackage`, which binds `off = byteOffset8(addr)` and
  `out = extractExtend(word, off, width, unsigned)`,
- `NarrowMemoryBlendProofPackage`, which binds `off = byteOffset8(addr)` and
  `out = blend(word, src, off, width)`.

## Paper Anchors

- **Source**: `./crates/neo-fold-prototype/specs/riscv-kernel.md`
- Anchors:
  - `VExtractLoad`
  - `VBlendStore`
  - narrow-memory helper arithmetic
  - Stage-2 memory-width handling

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Rv64IM/Stage1/NarrowMemoryHelpers.lean` | Exact narrow-memory helper arithmetic surfaces |
| `Nightstream/Rv64IM/Stage1/NarrowMemoryHelpersInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Byte access | `byteAt` | def | Definitional | Extracts the `k`th byte of a 64-bit word |
| Address derivation | `alignDown8` | def | Definitional | Computes the aligned 8-byte base address |
| Address derivation | `byteOffset8` | def | Definitional | Computes the byte offset within the aligned word |
| Theorem | `byteOffset8_lt` | theorem | Theorem-Target | The byte offset is always in `{0..7}` |
| Theorem | `alignDown8_add_byteOffset8` | theorem | Theorem-Target | Aligned address plus byte offset reconstructs the address |
| Extraction | `extractRaw` | def | Definitional | Computes the raw width-byte slice before extension |
| Extraction | `signFill` | def | Definitional | Computes the sign-fill byte from the extracted sign bit |
| Extraction | `extractExtend` | def | Definitional | Computes the exact narrow-load helper result |
| Blend | `blend` | def | Definitional | Computes the exact narrow-store helper result |
| Package | `NarrowMemoryExtractProofPackage` | structure | Definitional | Binds byte offset and extract/extend result |
| Package | `NarrowMemoryBlendProofPackage` | structure | Definitional | Binds byte offset and blended aligned word |

## Proof Obligations

- Narrow-memory helper semantics are exact arithmetic relations, not prover-side
  intuition and not implicit backend code.
- The helper surfaces own only Stage-1 arithmetic over already-authenticated
  aligned RAM words; they do not replace Stage-2 RAM authentication.
- Any lowered narrow load/store accepted by the kernel must instantiate these
  helper relations together with the surrounding committed-sequence theorem.

## Out of Scope

- aligned 64-bit RAM read/write authentication
- RAM `Val`-from-`Inc`
- committed-sequence soundness for the enclosing opcode
