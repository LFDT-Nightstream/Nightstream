## PAPER-EMB — coefficient embedding

`PAPER-EMB-001` Partition a field vector into consecutive blocks of `d`
coefficients. Embed each block into one ring element. The inverse map preserves
this order.

`PAPER-EMB-002` The transformed inner product satisfies
`ct(bar(a)*bar(b))=<a,b>`. The transform extends to vectors and matrices. Thus
`Mz=ct(bar(M)z)`.

`PAPER-EMB-003` Commitment, input projection, and multilinear evaluation
commute with the ring linear combinations used by PiRLC.

Source: `SRC-PAPER-05`.

## PAPER-REL — Structure, CCS, and CE

`PAPER-REL-001` A Structure is `s=({M_j},f)`. Each matrix has shape
`m*n_F`. Define `D_f=deg_tot(f)` and require `D_f<=u`.

`PAPER-REL-002` For `z=x||w`, `CCS(b,L)` holds when:

```text
c = L(z)
norm(z) < b
f(bar(M_1 z),...,bar(M_t z)) vanishes on the Boolean row cube.
```

`PAPER-REL-003` `CE(b,L)` holds for `(s;c,x,r,{y_j};z)` when:

```text
c = L(z)
x = L_in(z)
norm(z) < b
y_j = MLE(bar(M_j)z)(r) for each j.
```

The point `r` has `log m` extension-field coordinates. `L_in` projects the
first `n_R,in` ring columns under coefficient embedding.

Source: `SRC-PAPER-07`, lines 3-29.

## PAPER-BATCH — relation powers and shared evaluation points

`PAPER-BATCH-001` Appendix C defines `R^N` as the ordinary Cartesian product of
`N` relation instances. Since `r` is part of each CE instance, the literal
type `CE(a,L)^N` permits independent points `r_1,...,r_N`. The protocol text,
however, uses one unindexed point in each batch: PiCCS has one `eq(X,r)`, its
outputs share `r'`, PiRLC retains one `r`, and PiDEC copies one `r` to every
child. This is a formal type mismatch in the printed paper.

Source: `SRC-PAPER-02`, lines 13-27; `SRC-PAPER-07`, PiCCS, PiRLC, and PiDEC;
and `SRC-PAPER-12`, line 3.
