# Lean semantic-model contract

Status: **selected theorem interface; G2 evidence open**.

Lean owns mathematical definitions, verifier relations, refinements, and
reduction theorems. It does not own Rust layout, generated answers, circuit
hints, or a proof-supplied digest.

## Required model boundary

The Lean model must represent the 104 normative rules at their stated
generality or at the exact v1 profile. Each evidence claim must cite a file and
named declaration. A directory or nearby theorem is only an owner location.

The required object families are:

| Family | Required content |
|---|---|
| Algebra | Goldilocks, `K_ext`, Phi81 rings, coefficient order, centered norm |
| Relation | Structure, `CCS`, `CE`, strict bounds, commitment projection |
| Paper core | SumCheck, joint PiCCS, PiRLC, PiDEC, strong and weak reductions |
| V1 refinement | Prefix injection, zero padding, padded identity, lifted `f` |
| Non-interactive | Exact frame/squeeze schedule, Poseidon2 duplex, bounded sampler |
| Assurance | Canonical decoders and independent Rust-evidence replay |

## G2 theorem groups

### LEAN-PAPER-CORE

The model must prove the corrected paper statements, including:

```text
P_b(a)=0 iff abs(centered(a))<b
D_Q=max(D_f+1,2b,2)
T_abs is the absolute joint target
joint PiCCS terminal equality
joint strong extraction into CE(B_amb,L)^15
separate SumCheck and Schwartz-Zippel losses
unconditioned first extraction and success-gated retry
global sqrt(delta) witness-disagreement loss
PiRLC coordinate-fork loss 15/|C|, or conservative 16/|C|
PiDEC recomposition and output relation
```

### LEAN-REDUCTION-FRAMEWORK

The model must prove the common interactive-reduction interface, shared
commitment projection, PiCCS strong conditions, PiRLC weak conditions and
witness agreement, strong-weak composition, and sequential PiDEC composition.
The ambient family is `CE(B_amb,L)^15`. It is not `CE(b,L)^15` or
`CE(B,L)^15`.

### LEAN-PADDED-IDENTITY

For a logical assignment `z` of length 11,437,038, define `pad(z)` as its prefix
in a vector of length `2^24` followed by zeros. For
`M_0=[I_11437038;0]`, prove:

```text
M_0 z = pad(z)
MLE(M_0 z)(r) = MLE(pad(z))(r)
ct(y_(i,0)) = MLE(pad(z_i))(r_new)
```

The final equality must use the `y_(i,0)` field in the same CE instance whose
commitment is `L(z_i)`. No separate norm-opening witness or optional field is
permitted.

### LEAN-PADDING-REFINEMENT

For each of the 13 application matrices, prove that its padded output is the
logical output followed by zeros. Prove `P_2(0)=0` and:

```text
logical CCS acceptance iff padded CCS acceptance
logical joint polynomial identity iff padded joint polynomial identity
f_app(0,...,0)=0
the lifted f_v1 ignores exactly M_0
the commitment projection is unchanged
```

These theorems must establish that the paper identity-based PiCCS proof applies
to `PaddedRowIdentity`; shape equality alone is not enough.

### LEAN-NIGHTSTREAM-SECURITY

The model must instantiate:

- the Phi81 five-element strong set and expansion `T=216`;
- the guard `15*216*(2-1)=3240<16384`;
- the exact signed-binary split;
- the 24-round joint SumCheck and its 210 terminal ring values;
- the bounded three-attempt sampler and its distribution;
- the exact transcript event dependencies;
- the canonical Structure stream and verifier-key prehash;
- the selected strong, weak, and sequential reductions.

The model must keep the algebraic planning count separate from a Fiat-Shamir or
end-to-end production theorem. Its query census must include 2,457 fold
transcript squeezes and one public-image squeeze per fold, or at most 157,313
tagged squeezes per verifier key.

## Encoding interface

Lean must define canonical decoders for base fields, extension fields, ring
values, commitments, the sparse Structure stream, statement sections, proof
sections, and the circuit public-image tuple. Required results are:

```text
decode(encode(x)) = x
successful decode is canonical
two successful encodings of one value are byte-equal
the public-image vector decodes to the exact nine-field tuple
the circuit recomputes its digest from the selected verifier key and statement
```

The theorem must expose byte order, coefficient order, extension basis, index
order, section order, and failure for trailing or unknown content. A distinct
statement with the same digest is a Poseidon2 collision event, not a decoder
alias.

## Implementation evidence boundary

For each Rust-origin artifact, Lean or an independently checked semantic
oracle must decode the full canonical input and compute the expected semantic
result. Rust acceptance bits and hand-written expected results are not
premises. Finite artifact replay proves only those executions. Universal Rust
refinement remains a separate theorem or an explicit final trust assumption.

## Closure rule

G2 closes only after the exact Lean build passes and a proposition review maps
every required atomic rule to its named theorem. Existing paper or legacy
rectangular files can provide reusable lemmas, but they do not prove the
selected padded profile by location alone.
