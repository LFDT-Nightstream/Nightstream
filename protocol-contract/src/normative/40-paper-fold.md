## 4. Reviewed PiRLC, PiDEC, and fold composition

### SN-PIRLC-DOMAIN — PiRLC input family

PiRLC MUST take exactly the `K_fresh+k` PiCCS outputs. All inputs MUST use one
Structure and one evaluation point.

Source: PAPER-PIRLC-001.

### SN-PIRLC-EQUATIONS — Ring-linear combination

After all bound inputs exist, the verifier MUST sample one `rho_i in C` per
source and compute `z`, `c`, `x`, and every `y_j` with the same ordered
`R_F`-module linear combination.

Source: PAPER-PIRLC-001 and PAPER-PIRLC-002.

### SN-PIRLC-OUTPUT — PiRLC output bound

The Structure and evaluation point MUST remain unchanged. The result MUST be
one claim in `CE(B,L)`.

Source: PAPER-PIRLC-001.

### SN-PIRLC-FORK-LOSS — Coordinate-fork probability

For `L=K_fresh+k` independent coordinates, the exact coordinate-fork loss
MUST be `L/abs(C)`. A proof MAY use the conservative `(L+1)/abs(C)` bound and
MUST NOT divide by `abs(C)^L`.

Source: PAPER-PIRLC-001, PAPER-FORK-001, and ERR-COORD-FORK.

### SN-PIRLC-FORK-SET — Complete fork shape

The fork extractor MUST produce one base challenge vector and `L` neighbours,
where neighbour `i` changes only coordinate `i`. Its expected query count is
at most `L+1`.

Source: PAPER-FORK-001 and ERR-PIRLC-PROJECTION.

### SN-PIRLC-AGREEMENT — Weak extractor agreement

The PiRLC proof MUST establish the witness-agreement condition from
SN-RED-WEAK-CONDITIONS separately from coordinate forking and relaxed binding.

Source: PAPER-PIRLC-002 and ERR-PIRLC-PROJECTION.

### SN-PIDEC-SPLIT — PiDEC witness decomposition

PiDEC MUST take one `CE(B,L)` claim and apply the selected exact `split_b` to
its witness. The verifier MUST derive the same ordered public-input split.

Source: PAPER-PIDEC-001.

### SN-PIDEC-EQUATIONS — PiDEC recomposition

The verifier MUST derive `(x_0,...,x_(k-1))=split_b(x)` and MUST check

```text
c = sum_h b^h*c_h
y_j = sum_h b^h*y_(h,j) for every j.
```

The derived public split MUST satisfy `x=sum_h b^h*x_h`; it is not a prover
message.

Source: PAPER-PIDEC-001, ERR-PIDEC-EQUATIONS, and ERR-EVALUATION-NOTATION.

### SN-PIDEC-OUTPUT — PiDEC output family

PiDEC MUST enforce the child count, common Structure, common point, and
canonical public split. Its output MUST be exactly `k` claims in `CE(b,L)`.

Source: PAPER-PIDEC-001.

### SN-COMP-ORDER — Fold stage order

One fold MUST execute PiCCS, then PiRLC, then PiDEC.

Source: PAPER-COMP-001.

### SN-COMP-BINDING — Phase boundary equality

Each phase output MUST bind exactly to the next phase input for every
relation-authoritative field, and the strong and weak stages MUST share the
commitment projection.

Source: PAPER-COMP-001 and PAPER-COMP-002.

### SN-FOLD-TYPE — Composed folding type

The composed fold MUST have type

```text
PiDEC o PiRLC o PiCCS :
  CCS(b,L)^K_fresh * CE(b,L)^k -> CE(b,L)^k.
```

The running width `k` MUST be unchanged.

Source: PAPER-RED-006, PAPER-COMP-002, and PAPER-COMP-003.

### SN-FOLD-PROOF — Composition proof structure

The proof MUST use strong-weak composition for `PiRLC o PiCCS`, then
sequential composition with PiDEC. PiDEC MUST NOT enter the strong-weak step.

Source: PAPER-RED-006, PAPER-COMP-002, and PAPER-COMP-003.

### SN-SEC-ABSTRACT — Paper security boundary

The abstract reduction MUST include SumCheck soundness, field root bounds,
strong-set extraction, relaxed binding, and the reviewed extractor
corrections. It does not establish Fiat-Shamir, an implementation, a circuit,
a backend proof, or an on-chain verifier.

Source: PAPER-COMP-001, PAPER-COMP-002, ERR-COORD-FORK, and
ERR-STRONG-EXTRACT.
