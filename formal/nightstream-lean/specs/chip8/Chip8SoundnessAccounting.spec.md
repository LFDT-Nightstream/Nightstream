# Chip8SoundnessAccounting Spec

## Purpose

- **What it is**: The theorem-facing owner for the exact parameterized
  soundness-accounting surface of the 3-stage CHIP-8 kernel.
- **Key property**: `KernelSoundnessAccounting.negligible_epsTotal`: once every
  paper-level and instantiation-level primitive error surface is fixed and each
  primitive term is negligible, the published kernel total error term is also
  negligible and is upper-bounded by the exact Stage-1 / Stage-2 / Stage-3 /
  batching / PCS / Fiat-Shamir / outer-composition decomposition required by
  the kernel spec.
- **Protocol role**: This module owns only the exact budget decomposition from
  `chip8-kernel.md` §10 together with the Twist/Shout composition rules from
  `twist-and-shout-requirements.md`. It does not own the underlying Shout or
  Twist theorem statements, PCS binding, or Fiat-Shamir security proofs.

## Target Formulas

### Primitive index families

Define the exact Stage-1 Shout channels:

$$
\mathrm{Stage1ShoutChannel}
:=
\{\mathrm{fetch}, \mathrm{decode}, \mathrm{alu}, \mathrm{eq4}\}.
$$

Define the exact address families:

$$
\mathrm{AddressFamily}
:=
\{\mathrm{fetch}, \mathrm{decode}, \mathrm{alu}, \mathrm{eq4},
\mathrm{RegRaX}, \mathrm{RegRaY}, \mathrm{RegRaI}, \mathrm{RegWa},
\mathrm{RamRa}, \mathrm{RamWa}\}.
$$

Define the exact Twist read families:

$$
\mathrm{TwistReadFamily}
:=
\{\mathrm{RegX}, \mathrm{RegY}, \mathrm{RegI}, \mathrm{Ram}\}.
$$

Define the exact Twist write / `Val` families:

$$
\mathrm{TwistMemoryFamily}
:=
\{\mathrm{Reg}, \mathrm{Ram}\}.
$$

These finite families are the theorem-facing index sets that prevent an
implementation from silently deleting a Stage-1 or Stage-2 surface from the
accounting.

### Primitive error surfaces

Define `KernelSoundnessTerms` to package the exact primitive budget terms:

$$
\varepsilon_{\mathrm{shout\_core}}(c),
\quad
\varepsilon_{\mathrm{addr}}(f),
\quad
\varepsilon_{\mathrm{twist\_read}}(p),
\quad
\varepsilon_{\mathrm{twist\_write}}(m),
\quad
\varepsilon_{\mathrm{twist\_val}}(m),
$$

plus the scalar terms:

$$
\varepsilon_{\mathrm{ram\_raf\_read}},
\varepsilon_{\mathrm{ram\_raf\_write}},
\varepsilon_{\mathrm{shift\_reduce}},
\varepsilon_{\mathrm{continuity}},
\varepsilon_{\mathrm{reg\_rw\_batch}},
\varepsilon_{\mathrm{ram\_rw\_batch}},
\varepsilon_{\mathrm{lookup\_link}},
\varepsilon_{\mathrm{twist\_link}},
\varepsilon_{\mathrm{PCS}},
\varepsilon_{\mathrm{FS}},
\varepsilon_{\mathrm{outer}}.
$$

Define `PrimitiveNegligibility terms` to mean that every primitive surface
above is negligible at the theorem boundary.

### Exact stage accounting

Define the exact Stage-1 budget:

$$
\varepsilon_{\mathrm{stage1}}
=
\sum_{c \in \{\mathrm{fetch},\mathrm{decode},\mathrm{alu},\mathrm{eq4}\}}
\left(
\varepsilon_{\mathrm{shout\_core}}(c)
+
\varepsilon_{\mathrm{addr}}(c)
\right).
$$

Define the exact Stage-2 budget:

$$
\varepsilon_{\mathrm{stage2}}
=
\sum_{p \in \{\mathrm{RegX},\mathrm{RegY},\mathrm{RegI}\}}
\varepsilon_{\mathrm{twist\_read}}(p)
+
\varepsilon_{\mathrm{twist\_write}}(\mathrm{Reg})
+
\varepsilon_{\mathrm{twist\_val}}(\mathrm{Reg})
+
\sum_{f \in \{\mathrm{RegRaX},\mathrm{RegRaY},\mathrm{RegRaI},\mathrm{RegWa}\}}
\varepsilon_{\mathrm{addr}}(f)
$$

$$
+
\varepsilon_{\mathrm{twist\_read}}(\mathrm{Ram})
+
\varepsilon_{\mathrm{twist\_write}}(\mathrm{Ram})
+
\varepsilon_{\mathrm{twist\_val}}(\mathrm{Ram})
+
\varepsilon_{\mathrm{ram\_raf\_read}}
+
\varepsilon_{\mathrm{ram\_raf\_write}}
+
\varepsilon_{\mathrm{addr}}(\mathrm{RamRa})
+
\varepsilon_{\mathrm{addr}}(\mathrm{RamWa}).
$$

Define the exact Stage-3 budget:

$$
\varepsilon_{\mathrm{stage3}}
=
\varepsilon_{\mathrm{shift\_reduce}}
+
\varepsilon_{\mathrm{continuity}}.
$$

Define the exact batching budget:

$$
\varepsilon_{\mathrm{batch}}
=
\varepsilon_{\mathrm{reg\_rw\_batch}}
+
\varepsilon_{\mathrm{ram\_rw\_batch}}
+
\varepsilon_{\mathrm{lookup\_link}}
+
\varepsilon_{\mathrm{twist\_link}}.
$$

Define the exact kernel upper envelope:

$$
\varepsilon_{\mathrm{total\_upper}}
=
\varepsilon_{\mathrm{stage1}}
+
\varepsilon_{\mathrm{stage2}}
+
\varepsilon_{\mathrm{stage3}}
+
\varepsilon_{\mathrm{batch}}
+
\varepsilon_{\mathrm{PCS}}
+
\varepsilon_{\mathrm{FS}}
+
\varepsilon_{\mathrm{outer}}.
$$

### Kernel accounting package

Define `KernelSoundnessAccounting` to package:

- one `KernelSoundnessTerms`,
- one `PrimitiveNegligibility terms`,
- one claimed total error function `epsTotal`,
- and the exact pointwise upper bound

$$
\forall n,\;
\varepsilon_{\mathrm{total}}(n)
\le
\varepsilon_{\mathrm{total\_upper}}(n).
$$

This is the exact theorem-facing budget boundary for the CHIP-8 kernel.

### Theorem targets

The accounting owner must expose:

$$
\mathrm{PrimitiveNegligibility}(terms)
\Longrightarrow
\mathrm{IsNegligible}(\varepsilon_{\mathrm{stage1}}),
$$

$$
\mathrm{PrimitiveNegligibility}(terms)
\Longrightarrow
\mathrm{IsNegligible}(\varepsilon_{\mathrm{stage2}}),
$$

$$
\mathrm{PrimitiveNegligibility}(terms)
\Longrightarrow
\mathrm{IsNegligible}(\varepsilon_{\mathrm{stage3}}),
$$

$$
\mathrm{PrimitiveNegligibility}(terms)
\Longrightarrow
\mathrm{IsNegligible}(\varepsilon_{\mathrm{batch}}),
$$

$$
\mathrm{PrimitiveNegligibility}(terms)
\Longrightarrow
\mathrm{IsNegligible}(\varepsilon_{\mathrm{total\_upper}}).
$$

Finally, the main theorem target is:

$$
\mathrm{KernelSoundnessAccounting}(A)
\Longrightarrow
\mathrm{IsNegligible}(A.\varepsilon_{\mathrm{total}}).
$$

This is the honest closure step needed before the CHIP-8 kernel can claim the
full §10 soundness surface in Lean.

## Paper Anchors

- **Sources**:
  - `./crates/neo-fold-prototype/specs/chip8-kernel.md`
  - `./docs/soundness-specs/twist-and-shout-requirements.md`
  - `./docs/superneo-paper/07_7_Neo_s_folding_scheme_for_CCS.md`
- Anchors:
  - exact Stage-1 / Stage-2 / Stage-3 decomposition from kernel spec §10
  - explicit preservation of every Shout / Twist theorem surface used by the
    kernel
  - separate batching, PCS, Fiat-Shamir, and outer-composition terms
  - no silent collapse of multiple logical read/write/address families into one
    vague theorem label

## Module Mapping

| Lean file | Local owner |
|---|---|
| `Nightstream/Chip8/SoundnessAccounting.lean` | Exact parameterized CHIP-8 kernel soundness budget |
| `Nightstream/Chip8/SoundnessAccountingInterface.lean` | Theorem-facing re-export surface |

## Contract Surface

| Group | Lean surface | Kind | Role | Guarantee |
|---|---|---|---|---|
| Indices | `Stage1ShoutChannel` | def | Definitional | Exact Stage-1 Shout channels used in accounting |
| Indices | `AddressFamily` | def | Definitional | Exact address-correctness families used in accounting |
| Indices | `TwistReadFamily` | def | Definitional | Exact Twist read families used in accounting |
| Indices | `TwistMemoryFamily` | def | Definitional | Exact Twist write / `Val` families used in accounting |
| Terms | `KernelSoundnessTerms` | def | Definitional | Packages every primitive error term required by kernel spec §10 |
| Terms | `PrimitiveNegligibility` | def | Definitional | Packages negligible boundaries for every primitive term |
| Terms | `epsStage1` | def | Definitional | Exact Stage-1 accounting formula |
| Terms | `epsStage2` | def | Definitional | Exact Stage-2 accounting formula |
| Terms | `epsStage3` | def | Definitional | Exact Stage-3 accounting formula |
| Terms | `epsBatch` | def | Definitional | Exact batching accounting formula |
| Terms | `epsTotalUpper` | def | Definitional | Exact total upper envelope from stage, batch, PCS, FS, and outer terms |
| Package | `KernelSoundnessAccounting` | def | Definitional | Packages one claimed total bound against the exact upper envelope |
| Theorem | `negligible_epsStage1` | theorem | Theorem-Target | Stage-1 budget is negligible from primitive negligibility |
| Theorem | `negligible_epsStage2` | theorem | Theorem-Target | Stage-2 budget is negligible from primitive negligibility |
| Theorem | `negligible_epsStage3` | theorem | Theorem-Target | Stage-3 budget is negligible from primitive negligibility |
| Theorem | `negligible_epsBatch` | theorem | Theorem-Target | Batching budget is negligible from primitive negligibility |
| Theorem | `negligible_epsTotalUpper` | theorem | Theorem-Target | Exact total upper envelope is negligible |
| Theorem | `KernelSoundnessAccounting.negligible_epsTotal` | theorem | Theorem-Target | Claimed kernel total error is negligible from the exact upper bound |

## Proof Obligations

- This owner must not replace the exact Stage-2 family decomposition with a
  single vague "Twist memory theorem" term.
- Theorem surfaces for Shout core, address correctness, Twist read, Twist
  write, and Twist `Val` must remain distinct because the paper and the
  instantiation requirements treat them as distinct sources of soundness error.
- PCS, Fiat-Shamir, and outer-composition terms must remain separate from the
  PIOP-side batching and stage-local terms.
- This owner may package negligible primitive terms, but it must not pretend to
  prove the underlying PCS or Fiat-Shamir theorem statements itself.
