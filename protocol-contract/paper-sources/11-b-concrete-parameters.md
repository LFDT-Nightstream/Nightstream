## B Concrete parameters

This section provides three efficient parameterizations over  $\le 64$ -bit fields. Additionally, Appendix D.7 and Appendix D.8 provide the corresponding sage scripts that we used to determine valid parameterizations. In Definition 14, we require the commitment scheme to be  $(d, m, 2B, \mathcal{C})$ -relaxed binding (Definition 4). Thus, we need the commitment scheme to be  $(d, m, 4TB)$ -binding (Definition 4). Finally, Ajtai’s commitment scheme is  $(d, m, 4TB)$ -binding if  $\text{MSIS}_{m, 8TB}^{\infty, \kappa, q}$  is hard. We estimate the hardness of Module-SIS using the lattice estimator library provided by [4] using our script (Appendix D.8).

The tables below use the protocol bound  $B=b^k<q/2$ . The universal ambient bound  $B_{\mathrm{amb}}=\lfloor q/2\rfloor+1$  is used only in the ambient relation  $\text{CE}(B_{\mathrm{amb}},\mathcal{L})$  in the security reduction. It does not replace  $B$  in the protocol, relaxed-binding assumption, or Module-SIS parameters.

### B.1 Almost Goldilocks: $(2^{64} - 2^{32} + 1) - 32$

We provide a new field, which we refer to as *Almost Goldilocks*. This field’s order is  $q = (2^{64} - 2^{32} + 1) - 32$ , which is close to the order of the Goldilocks field  $2^{64} - 2^{32} + 1$ . Because of this, the field admits an efficient implementation with a small change to the Solinas prime reduction algorithm (which is typically used for the Goldilocks field).

$\eta = 128$ ,  $\Phi = X^{64} + 1$ ,  $d = 64$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 15$ ,  $n_{\mathbb{F}} = 2^{33}$ ,  $b = 2$ ,  $k = 13$ ,  $K \in [50]$ ,  $B = 2^{13}$ . Define  $\mathcal{C}$  to be the set polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-1, 0, 1, 2]$ . By Theorem 9,  $T = 128$ . By Theorem 8,  $b_{\text{inv}} \approx 4$ .  $\mathbb{K} = \mathbb{F}_{q^2}$ .  $|\mathcal{C}| \approx 2^{128}$ ,  $|\mathbb{K}| \approx 2^{128}$ ,  $\text{MSIS}_{m, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

### B.2 Goldilocks: $(2^{64} - 2^{32} + 1)$

This is a popular choice of field for SNARKs as the field admits an efficient implementation: field operations can be implemented with essentially only bit-shifts and the field has high 2-adicity ( $2^{32} \mid (p-1)$ ), which is useful for compressing Neo’s IVC proofs with SNARKs.

$\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $n_{\mathbb{F}} = 2^{30}$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 9,  $T = 216$ . By Theorem 8,  $b_{\text{inv}} \approx 2.5 \cdot 10^9$ .  $\mathbb{K} = \mathbb{F}_{q^2}$ .

$|\mathcal{C}| \approx 2^{125}$ ,  $|\mathbb{K}| \approx 2^{128}$ ,  $\text{MSIS}_{m, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

*Remark 4 (Incompatibility with LatticeFold [14]).* In LatticeFold [14], the constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  with  $d$  being a power-of-two. Since the Goldilocks field has high 2-adicity, the cyclotomic polynomial completely factors into linear terms. This means that the ring  $R_F$  is isomorphic to  $\mathbb{F}_q^d$  (the NTT representation). The security of LatticeFold’s construction depends on the size of the field in the NTT representation [14, Sec 3.3], which here is only 64 bits.

### B.3 Mersenne 61: $2^{61} - 1$

This field admits an incredibly efficient implementation as it is only one off from a power-of-two. Specifically, modular arithmetic over this field can be implemented with simple bit-shifts with an algorithm more efficient than Goldilocks.

$\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $R_F := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $n_F = 2^{28}$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set polynomials in  $R_F$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 9,  $T = 216$ . By Theorem 8,  $b_{\text{inv}} \approx 383$ .  $\mathbb{K} = \mathbb{F}_{q^2}$  and  $|\mathbb{K}| = q^2$ .

$|\mathcal{C}| \approx 2^{125}$ ,  $|\mathbb{K}| \approx 2^{122}$ ,  $\text{MSIS}_{m, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

*Remark 5 (Incompatibility with Latticefold [14]).* As stated earlier, LatticeFold’s constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  for  $d$  being a power-of-two. For Mersenne 61, there is no choice of power-of-two cyclotomic polynomials, which satisfies the requirements of Theorem 8. Hence, it cannot be determined whether a choice of parameters with  $\Phi = X^d + 1$  leads to a secure construction.
