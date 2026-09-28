## PAPER-RED — interactive reductions and composition

`PAPER-RED-001` Definition 5 defines an interactive reduction as the PPT
algorithms `(G,K,P,V)`. `G(1^lambda,sz)` outputs `pp`. `K(pp,s)` deterministically
outputs `(pk,vk)`. `P` and `V` interact and output the verifier instance `u_2`
and the prover witness `w_2`.

The reduction is a reduction of knowledge when it has all three properties:

1. completeness: an honest run on `(pp,s,u_1,w_1) in R_1` gives equal prover and
   verifier output instances, and its result is in `R_2`;
2. knowledge soundness: for every EPT adversary with success probability at
   least `1/poly(lambda)`, an EPT extractor returns `w_1` with
   `(pp,s,u_1,w_1) in R_1` with probability at least the success probability
   minus a negligible term;
3. public coin: every verifier message is a uniformly random string, and the
   verifier messages contain all verifier randomness.

`PAPER-RED-002` Lemma 2 gives sequential composition. For reductions of
knowledge `Pi_1:R_1->R_2` and `Pi_2:R_2->R_3` that share `G` and `K`, the
composition `Pi_2 o Pi_1:R_1->R_3` is a reduction of knowledge.

`PAPER-RED-003` Definition 9 defines a weak reduction `Pi:R_1->R_2` for
relations with `R_1 subset R'_1` and a function `phi` on the ambient instance
space of `R_1`. The reduction must be complete and public coin. For each EPT
adversary, an EPT extractor must satisfy both conditions:

1. it returns `w_1` with `(pp,s,u_1,w_1) in R'_1` with probability at least the
   adversary success probability minus a negligible term;
2. when the adversary gives two input instances with equal `phi` image, the
   extractor returns two different witnesses with at most negligible
   probability.

`PAPER-RED-004` Definition 10 defines a strong reduction `Pi:R_1->R_2` for
relations with `R_2 subset R'_2` and a function `phi` on the ambient instance
space of `R_2`. The reduction must be complete and public coin, and:

1. for every EPT adversary, two independent runs of the same prover give output
   instances with equal `phi` image with probability one;
2. for every EPT adversary whose relaxed success probability for `R'_2` is at
   least `1/poly(lambda)`, and whose two runs give different output witnesses
   with at most negligible probability, an EPT extractor returns `w_1` with
   `(pp,s,u_1,w_1) in R_1` with probability at least that relaxed success
   probability minus a negligible term.

`PAPER-RED-005` Theorem 6 gives strong-weak composition. Let `R_2 subset R'_2`.
Let `Pi_1:R_1->R_2 (R'_2)` be strong for `phi`, and let
`Pi_2:R_2 (R'_2)->R_3` be weak for the **same** `phi`. Then `Pi_2 o Pi_1` is a
reduction of knowledge.

The proof of Theorem 6 uses the strong condition (i) of `Pi_1` to supply the
`phi`-agreement premise of `Pi_2`, and the extractor-agreement conclusion of
`Pi_2` to supply the witness-agreement premise of `Pi_1`. Neither direction is
optional.

Source: `SRC-PAPER-04`, lines 66-95, and `SRC-PAPER-06`.

## PAPER-FOLDING — folding-scheme definition

`PAPER-RED-006` Definition 19 defines a folding scheme for a relation `R` as a
reduction of knowledge of type `R * R_ACC -> R_ACC`.

Source: `SRC-PAPER-12`, line 32.

## PAPER-FORK — special sets and coordinate-wise extraction

`PAPER-FORK-001` Definition 20 defines the special set `SS(C,ell)`. It contains
each tuple of one base challenge vector and `ell` neighbours, where neighbour
`i` differs from the base vector in coordinate `i` only.

Theorem 10 gives coordinate-wise extraction. For a challenge space `C^ell` and
an adversary with success probability `eps`, an EPT extractor makes at most
`ell+1` queries in expectation. With probability at least `eps-ell/abs(C)`, it
returns `ell+1` accepted pairs whose challenge vectors form a special set. The
theorem also gives the weaker bound `eps-(ell+1)/abs(C)`, and Appendix D.5 uses
that weaker form.

Source: `SRC-PAPER-12`, lines 34-56.

## PAPER-SUMCHECK — public-coin polynomial sum

`PAPER-SUMCHECK-001` For an `ell`-variable polynomial with maximum individual
degree at most `D`, SumCheck checks a claimed Boolean-cube sum `T`. It outputs
a random point `r`, a claimed value `v`, and the terminal claim `v=Q(r)` for
the verifier to check. The paper states that the protocol is public coin, has
zero completeness error, and has soundness error at most
`ell*D/abs(K_ext)` over the extension field. It refers to an external note for
the full round protocol.

Source: `SRC-PAPER-04`, line 97.
