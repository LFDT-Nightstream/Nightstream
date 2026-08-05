# Nightstream end-to-end security-reduction contract

Status: **theorem interface selected; G5 evidence open**.

This document defines the required reduction. It does not claim 96-bit
end-to-end security for the current implementation.

## 1. Final theorem shape

Let `Accept_deployed` be acceptance by the selected on-chain parser and
verifier for one concrete backend manifest. Let `ValidExecution` be the
normative Nightstream statement under the selected verifier key and lifecycle
limits. The final theorem must have this form:

```text
Pr[Accept_deployed and not ValidExecution]
  <= eps_terminal
   + eps_circuit
   + eps_implementation
   + eps_encoding
   + eps_fiat_shamir
   + eps_poseidon2
   + eps_sampler
   + eps_fold.
```

Each nonzero term must have a named theorem or assumption, exact v1
substitution, adversary resource bound, and owner. A term cannot be omitted
because a test passed.

## 2. Backward reduction path

The proof must proceed from the real acceptance predicate:

```text
deployed verifier acceptance
  -> selected terminal proof relation
  -> selected current-circuit R1CS
  -> normative verifier acceptance
  -> interactive PaddedRowIdentity fold acceptance
  -> extracted witnesses for the reviewed relations
  -> ValidExecution or a named assumption failure.
```

The first three arrows require the concrete backend manifest, public image,
parser, verifier key, circuit correspondence, and Rust refinement. A theorem
about an abstract fixture cannot replace these arrows.

## 3. Interactive fold core

For one fold, the interactive proof must compose:

1. corrected joint PiCCS strong extraction;
2. weak PiRLC extraction and witness agreement for the same commitment
   projection;
3. sequential PiDEC extraction.

The relation pair is:

```text
CCS(b,L)^1 * CE(b,L)^14
  --PiCCS--> CE(b,L)^15, ambient CE(B_amb,L)^15
  --PiRLC--> CE(B,L)
  --PiDEC--> CE(b,L)^14.
```

Here `b=2`, `B=2^14=16384`, and
`B_amb=floor(q/2)+1`. These bounds have different purposes and must not be
merged.

## 4. Padded-identity bridge

Before the paper strong theorem is reused, the proof must establish:

```text
M_0 z = z || 0
ct(y_(i,0)) = MLE(z_i || 0)(r_new)
P_2(0)=0
logical application outputs = padded outputs on the prefix
padded application outputs = 0 after the logical row count
f_app(0,...,0)=0
logical CCS acceptance iff padded CCS acceptance
the commitment projection is unchanged.
```

This bridge binds the norm terminal to the witness that opens the same
commitment. There is no independent NC claim, carrier opening, or extra
challenge family in v1.

## 5. Exact algebraic planning terms

For the selected profile:

```text
q = 2^64-2^32+1
|K_ext| = q^2
ell = 24
D_Q = 9
N_SC = ell*D_Q = 216
D_SZ = max(24,39,10599) = 10599
N_field = N_SC+D_SZ = 10815
|C| = 5^54
coordinate-fork numerator <= 16.
```

The one-fold algebraic planning term is
`10815/q^2 + 16/5^54`. Its first component is about 114.60 bits. A simple
64-fold union bound lowers that component to about 108.60 bits. These are
component planning values, not production security levels.

The proof must keep these events separate:

- false SumCheck acceptance: `216/q^2`;
- independent alpha/gamma root event: `10599/q^2`;
- coordinate fork: exact `15/5^54` or conservative `16/5^54`;
- strong-extractor witness disagreement: the reviewed global
  `sqrt(delta)` term;
- relaxed commitment binding and its Module-SIS reduction.

The extractor must execute the first run without conditioning. A second run is
allowed only after the first ambient success.

## 6. Strong set and commitment

The selected strong set uses 54 coefficients from
`{-2,-1,0,1,2}` in Phi81. The reduction must prove:

```text
3 | 81
q = 1 mod 3
ord_81(q) = 27
max difference norm = 4 < q^(1/2)/sqrt(3)
T <= 2*phi(81)*2 = 216
15*T*(b-1) = 3240 < B.
```

The commitment reduction must use `kappa=18`, message width 211,797 ring
elements, and infinity bound `8*T*B=28,311,552`. It must state whether the
Ajtai matrix is uniform or reduce the 32-byte seeded generation procedure to
an explicit PRG assumption. The exact ChaCha8 row, chunk, coefficient, and
rejection schedule is fixed in `src/profile/ajtai-setup-v1.toml`. A seed digest
does not prove uniformity.

## 7. Fiat-Shamir and sampler terms

The Fiat-Shamir theorem must use the exact 12-event fold transcript, the four
challenge families, the stated one-session and one-proof key scope, at most
262,144 adaptive oracle queries, and the selected width-8 Poseidon2 permutation.
It must use the machine-readable nested frame and squeeze schedule. It must
state the classical random-oracle programming and extraction costs.

The fold transcript uses at most 2,457 prescribed tagged squeezes per fold:
alpha, gamma, 24 SumCheck challenges, 2,430 candidate attempts, and one final
fold transcript digest. The public-image prehash adds one fresh tagged squeeze
per fold. Across 64 folds these are 157,312 squeezes, plus one cached
verifier-key prehash, or 157,313 per key. The remaining query allowance belongs
to the adversary.

For each of 810 PiRLC coefficients per fold, the sampler tries at most three
base-field candidates. It rejects only candidate `q-1`, so accepted residues
are exactly uniform on five digits. The explicit abort terms are:

```text
per fold:       810/q^3
at most 64:   51840/q^3.
```

These are about 182.34 and 176.34 bits. The reduction must also prove that
native and circuit implementations use the same counter order, rejection
condition, and exhaustion behavior.

## 8. Implementation and terminal transfer

The following terms remain separate until proved:

| Term | Required evidence |
|---|---|
| `eps_encoding` | Canonical statement, proof, Structure, and public-image tuple decoders |
| `eps_implementation` | Pinned Rust refinement and Rust-origin evidence |
| `eps_circuit` | Arbitrary-assignment circuit soundness and lowering |
| `eps_terminal` | Concrete Spartan/WHIR manifest, parser, public image, and deployed verifier theorem |
| `eps_poseidon2` | Exact permutation security assumption or bound |
| `eps_fiat_shamir` | Named ROM theorem with query and extraction costs |

If one arrow remains a trusted implementation assumption, the final theorem
must name it. It must not appear as a zero term.

The circuit public image is the exact nine-field tuple from `public_image_v1`.
Its fresh Poseidon2 duplex binds the contract domain, profile version,
verifier-key digest, and 39,848-field canonical statement stream. Canonical
tuple decoding is an encoding property. Substitution of another canonical
preimage with the same digest belongs to `eps_poseidon2`; it is not an
injective-decoder claim.

## 9. Lifetime composition

The final proof must cover at most 64 adaptive folds and 262,144 oracle queries.
A union bound needs a uniform conditional bound for each accepted prefix. The
proof-of-knowledge result must state a concrete expected-polynomial-time
extractor bound. It must include session, fold, sampler-abort, forking, retry,
and terminal-proof costs.

## 10. Release rule

G5 closes only when every table entry has a theorem or explicit assumption and
the evaluated total is at most `2^-96` for the selected limits. Until then,
114-bit and 108-bit values may be reported only as algebraic planning counts.
