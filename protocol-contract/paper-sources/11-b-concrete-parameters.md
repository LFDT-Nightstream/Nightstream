## B Concrete parameters

This section gives three efficient ring and commitment parameterizations over fields whose moduli fit in at most 64 bits. Appendix D.7 and Appendix D.8 give the corresponding Sage scripts. In Definition 14, the Ajtai commitment has ring-message length  $n_R$  and must be  $(2B,\mathcal C)$ -relaxed binding. By Theorem 2, this follows from the hardness of  $\operatorname{MSIS}_{n_R,8TB}^{\infty,\kappa,q}$ .

For the single-domain protocol in Section 7.3, let  $m$  be the power-of-two SumCheck row length and set

$$n_R:=\left\lfloor\frac m d\right\rfloor,
\qquad
n_{\mathbb F}:=dn_R\le m.$$

The padding injection  $M_1$  from Section 7.3 maps a logical witness of length  $n_{\mathbb F}$  into the  $m$ -entry SumCheck domain. Thus the exact coefficient-packing identity  $n_{\mathbb F}=dn_R$  holds even when  $d\nmid m$ .

The tables below use the protocol bound  $B=b^k<q/2$ . The universal ambient bound  $B_{\mathrm{amb}}=\lfloor q/2\rfloor+1$  is used only in the ambient relation  $\operatorname{CE}(B_{\mathrm{amb}},\mathcal L)$  in the security reduction. It does not replace  $B$  in the protocol, relaxed-binding assumption, or Module-SIS parameters.

The stated security estimates are heuristic work-factor estimates for the indicated Module-SIS instances only; they are not proved upper bounds on adversarial advantage and are not end-to-end knowledge-soundness estimates. The source used for the reported figures did not record the exact lattice-estimator revision and configuration. Treat these figures as indicative until they are regenerated with a pinned revision. For a structure with total-degree bound  $u$ , the corrected single-domain statistical terms include

$$\epsilon_{\mathrm{field}}
\le
\frac{\max(u+1,2b,2)\log m+2K+k-1+\max(\log m,ktd)}{|\mathbb K|},
\qquad
\epsilon_{\mathrm{fork}}
\le
\frac{K+k+1}{|\mathcal C|},$$

before any other applicable error terms. These terms depend on the concrete CCS structure and batch size and must be added as probabilities.

For the normalized compilation of R1CS, the three original matrices are preceded by the injection matrix, so  $u=2$  and  $t=4$ . At the maximum stated batch sizes,  $\epsilon_{\mathrm{field}}+\epsilon_{\mathrm{fork}}$  is approximately  $2^{-116.17}$  for Almost Goldilocks,  $2^{-116.13}$  for Goldilocks, and  $2^{-110.32}$  for Mersenne 61. These are one-activation statistical terms. They exclude the other losses and computational assumptions needed for an end-to-end statement.

In particular, the strong-extractor analysis in Appendix D.4 also contains a  $\sqrt\delta$  witness-disagreement loss. To make this term at most  $2^{-\lambda_s}$ , the underlying disagreement bound must satisfy  $\delta\le2^{-2\lambda_s}$ .

### B.1 Almost Goldilocks: $(2^{64} - 2^{32} + 1) - 32$

We provide a new field, which we refer to as *Almost Goldilocks*. This field’s order is  $q = (2^{64} - 2^{32} + 1) - 32$ , which is close to the order of the Goldilocks field  $2^{64} - 2^{32} + 1$ . Because of this, the field admits an efficient implementation with a small change to the Solinas prime reduction algorithm typically used for the Goldilocks field.

$\eta = 128$ ,  $\Phi = X^{64} + 1$ ,  $d = 64$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 15$ ,  $m=2^{33}$ ,  $n_R=2^{27}$ ,  $n_{\mathbb F}=2^{33}$ ,  $b = 2$ ,  $k = 13$ ,  $K \in [50]$ ,  $B = 2^{13}$ . Define  $\mathcal{C}$  to be the set of polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-1, 0, 1, 2]$ . For  $\Phi=X^{64}+1$ , the direct negacyclic convolution bound gives  $T\le d\max_{\rho\in\mathcal C}\|\rho\|_\infty=64\cdot2=128$ ; the generic bound in Theorem 9 is looser. By Theorem 8,  $b_{\mathrm{inv}} \approx 4$ .  $\mathbb{K} = \mathbb{F}_{q^2}$ .  $|\mathcal{C}| = 2^{128}$ ,  $|\mathbb{K}| \approx 2^{128}$ , and the lattice-estimator output for  $\operatorname{MSIS}_{n_R,8TB}^{\infty,\kappa,q}$  is approximately 129 bits.

### B.2 Goldilocks: $(2^{64} - 2^{32} + 1)$

This is a popular choice of field for SNARKs as the field admits an efficient implementation: field operations can be implemented with essentially only bit-shifts and the field has high 2-adicity ( $2^{32} \mid (q-1)$ ), which is useful for compressing Neo’s IVC proofs with SNARKs.

$\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $m=2^{30}$ ,  $n_R=\lfloor2^{30}/54\rfloor=19{,}884{,}107$ ,  $n_{\mathbb F}=54n_R=1{,}073{,}741{,}778$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set of polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 9,  $T \le 216$ ; we use this upper bound. By Theorem 8,  $b_{\mathrm{inv}} \approx 2.5 \cdot 10^9$ .  $\mathbb{K} = \mathbb{F}_{q^2}$ .

$|\mathcal{C}| = 5^{54}\approx 2^{125.38}$ ,  $|\mathbb{K}| \approx 2^{128}$ , and the lattice-estimator output for  $\operatorname{MSIS}_{n_R,8TB}^{\infty,\kappa,q}$  is approximately 129 bits.

*Remark 4 (Incompatibility with LatticeFold [14]).* In LatticeFold [14], the constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  with  $d$  being a power of two. Since the Goldilocks field has high 2-adicity, the cyclotomic polynomial completely factors into linear terms. This means that the ring  $\mathbb R_{\mathbb F}$  is isomorphic to  $\mathbb{F}_q^d$  (the NTT representation). The security of LatticeFold’s construction depends on the size of the field in the NTT representation [14, Sec. 3.3], which here is only 64 bits.

### B.3 Mersenne 61: $2^{61} - 1$

This field admits an efficient implementation because its modulus is one less than a power of two. Modular arithmetic over this field can use simple bit shifts.

$q=2^{61}-1$ ,  $\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $\mathbb R_{\mathbb F} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $m=2^{28}$ ,  $n_R=\lfloor2^{28}/54\rfloor=4{,}971{,}026$ ,  $n_{\mathbb F}=54n_R=268{,}435{,}404$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set of polynomials in  $\mathbb R_{\mathbb F}$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 9,  $T \le 216$ ; we use this upper bound. By Theorem 8,  $b_{\mathrm{inv}} \approx 383$ .  $\mathbb{K} = \mathbb{F}_{q^2}$  and  $|\mathbb{K}| = q^2$ .

$|\mathcal{C}| = 5^{54}\approx 2^{125.38}$ ,  $|\mathbb{K}| \approx 2^{122}$ , and the lattice-estimator output for  $\operatorname{MSIS}_{n_R,8TB}^{\infty,\kappa,q}$  is approximately 129 bits.

*Remark 5 (Incompatibility with LatticeFold [14]).* As stated earlier, LatticeFold’s constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  for  $d$  being a power of two. For Mersenne 61, there is no choice of power-of-two cyclotomic polynomial that satisfies the requirements of Theorem 8. Hence, it cannot be determined whether a choice of parameters with  $\Phi = X^d + 1$  leads to a secure construction.
