# Digest Parent Binding Specification

## Mathematical Target

This module models digest-compressed parent binding for the direct CCS/F'
reduced-source strategy.

The digest-only source carries:

```text
source.digest
```

and authorizes a parent residue vector when:

```text
source.digest = hashParent(parent)
```

This relation is functional only under `ParentDigestBinding`. In the symbolic
model this has the same logical shape as injectivity over accepted parent
residue vectors. In the computational implementation, it is the parent-digest
binding assumption: no feasible prover can produce two different accepted
parent encodings with the same protocol hash digest.

## Required Theorems

- If the parent digest is binding, then digest-source parent binding is
  functional.
- Under that binding assumption, accepted Goldilocks child-table
  authorizations for the same digest source have equal next `Pi_CCS` inputs.
- A deterministic challenge over the same digest source cannot authorize
  different next inputs under that binding assumption.
- A constant digest is not functional parent binding.

## Soundness Role

This module states the exact condition under which a hash digest may
replace explicit parent residues in a reduced Fiat-Shamir source. The digest
must be recomputed from the authoritative parent residue data, and the security
argument must rely on the appropriate collision-resistance/preimage-binding
assumption. A digest supplied as self-consistent advice is not authority.
