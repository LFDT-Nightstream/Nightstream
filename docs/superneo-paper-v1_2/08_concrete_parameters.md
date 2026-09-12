# 8 Concrete parameters

This section provides three efficient parameterizations over  $\leq 64$ -bit fields. For all three parameterizations, we set  $\tau = 1$ , so that  $\mathbb{L} = \mathbb{F}$ , and take  $\mathbb{K} := \mathbb{F}_{q^2}$  and  $\mathbb{A} := \mathbb{K}$ . Additionally, Appendix B.5 and Appendix B.6 provide the corresponding sage scripts that we used to determine valid parameterizations. In Definition 22, we require the commitment scheme to be  $(2B, \mathcal{C})$ -relaxed binding (Definition 7). Thus, we need the commitment scheme to be  $4TB$ -binding (Definition 7). Finally, Ajtai’s commitment scheme is  $4TB$ -binding if  $\text{MSIS}_{n_R, 8TB}^{\infty, \kappa, q}$  is hard. We estimate the hardness of Module-SIS using the lattice estimator library provided by [4] using our script (Appendix B.6), and compute the invertibility bounds that determine challenge set size using Appendix B.5.

## 8.1 Almost Goldilocks: $(2^{64} - 2^{32} + 1) - 32$

We provide a new field, which we refer to as *Almost Goldilocks*. This field’s order is  $q = (2^{64} - 2^{32} + 1) - 32$ , which is close to the order of the Goldilocks field  $2^{64} - 2^{32} + 1$ . Because of this, the field admits an efficient implementation with a small change to the Solinas prime reduction algorithm (which is typically used for the Goldilocks field).

$\eta = 128$ ,  $\Phi = X^{64} + 1$ ,  $d = 64$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 15$ ,  $m = n_L = n_{\mathbb{F}} = 2^{33}$ ,  $n_R = 2^{27}$ ,  $n_{\text{pad}} = 0$ ,  $b = 2$ ,  $k = 13$ ,  $K \in [50]$ ,  $B = 2^{13}$ . Define  $\mathcal{C}$  to be the set polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-1, 0, 1, 2]$ . By Theorem 5,  $T = 128$ . By Theorem 4,  $\mathbf{b}_{\text{inv}} \approx 4$ .

$|\mathcal{C}| = 2^{128}$ ,  $|\mathbb{K}| \approx 2^{128}$ ,  $\text{MSIS}_{n_R, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

## 8.2 Goldilocks: $(2^{64} - 2^{32} + 1)$

This is a popular choice of field for SNARKs as the field admits an efficient implementation: field operations can be implemented with essentially only bit-shifts and the field has high 2-adicity ( $2^{32} \mid (p - 1)$ ), which is useful for compressing Neo and SuperNeo IVC proofs with SNARKs.

$\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $m = 2^{30}$ ,  $n_L = n_{\mathbb{F}} = 2^{30} - 46$ ,  $n_R = 19,884,107$ ,  $n_{\text{pad}} = 46$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 5,  $T = 216$ . By Theorem 4,  $\mathbf{b}_{\text{inv}} \approx 2.5 \cdot 10^9$ .

$|\mathcal{C}| \approx 2^{125}$ ,  $|\mathbb{K}| \approx 2^{128}$ ,  $\text{MSIS}_{n_R, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

*Remark 5 (Incompatibility with LatticeFold [14]).* In LatticeFold [14], the constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  with  $d$  being a power-of-two. Since the Goldilocks field has high 2-adicity, the cyclotomic polynomial completely factors into linear terms. This means that the ring  $\mathbb{R}_{\mathbb{F}}$  is isomorphic to  $\mathbb{F}_q^d$  (the NTT representation). The security of LatticeFold’s construction depends on the size of the field in the NTT representation [14, Sec 3.3], which here is only 64 bits.

## 8.3 Mersenne 61: $2^{61} - 1$

This field admits an incredibly efficient implementation as it is only one off from a power-of-two. Specifically, modular arithmetic over this field can be implemented with simple bit-shifts with an algorithm more efficient than Goldilocks.

$\eta = 81$ ,  $\Phi = X^{54} + X^{27} + 1$ ,  $d = 54$ ,  $\mathbb{R}_{\mathbb{F}} := \mathbb{F}[X]/(\Phi)$ ,  $\kappa = 18$ ,  $m = 2^{28}$ ,  $n_L = n_{\mathbb{F}} = 2^{28} - 52$ ,  $n_R = 4,971,026$ ,  $n_{\text{pad}} = 52$ ,  $b = 2$ ,  $k = 14$ ,  $K \in [61]$ ,  $B = 2^{14}$ . Define  $\mathcal{C}$  to be the set polynomials in  $\mathbb{R}_{\mathbb{F}}$  whose coefficients belong to  $[-2, -1, 0, 1, 2]$ . By Theorem 5,  $T = 216$ . By Theorem 4,  $\mathbf{b}_{\text{inv}} \approx 383$ .

$|\mathcal{C}| \approx 2^{125}$ ,  $|\mathbb{K}| \approx 2^{122}$ ,  $\text{MSIS}_{n_R, 8TB}^{\infty, \kappa, q} \approx 129$  bits of security.

*Remark 6 (Incompatibility with LatticeFold [14]).* As stated earlier, LatticeFold’s constructions and analysis are limited to power-of-two cyclotomic polynomials, namely of the form  $X^d + 1$  for  $d$  being a power-of-two. For Mersenne 61, there is no choice of power-of-two cyclotomic polynomials, which satisfies the requirements of Theorem 4. Hence, it cannot be determined whether a choice of parameters with  $\Phi = X^d + 1$  leads to a secure construction.

