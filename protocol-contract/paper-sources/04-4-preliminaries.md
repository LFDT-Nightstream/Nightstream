## 4 Preliminaries

For brevity, we defer some additional background to Appendix C.

*Notation.* We let  $\lambda$  denote the security parameter and  $\text{negl}(\lambda)$  denote a negligible function in  $\lambda$ . We let PPT denote probabilistic polynomial time and EPT denote expected probabilistic polynomial time. All adversaries and extractors in this paper are classical. We let  $[n]$  denote the set  $\{1, \dots, n\}$ , and  $\{u_i\}_{i \in [n]}$  denote the set  $\{u_1, \dots, u_n\}$ .

*Asymptotic-family convention.* Every statement that uses  $\text{negl}(\lambda)$  is conditional on a uniformly and efficiently constructible family  $\{\mathcal P_\lambda\}_{\lambda\in\mathbb N}$ . For each  $\lambda$ , the member  $\mathcal P_\lambda$  fixes the fields  $\mathbb F_\lambda\subseteq\mathbb K_\lambda$ , cyclotomic rings, dimensions, structure bounds, batch sizes, norm bounds, Module-SIS rank, and challenge set

$$\mathcal P_\lambda=(q_\lambda,\eta_\lambda,d_\lambda,\kappa_\lambda,m_\lambda,n_{R,\lambda},u_\lambda,t_\lambda,k_\lambda,K_\lambda,b_\lambda,B_\lambda,T_\lambda,\mathcal C_\lambda).$$

All parameter encodings, exact challenge samplers, and protocol algorithms must run in time polynomial in  $\lambda$ , and all encoded dimensions must be polynomially bounded in  $\lambda$ . Let

$$D_{Q,\lambda}:=\max(D_{f,\lambda}+1,2b_\lambda,2).$$

The asymptotic knowledge-soundness statements require each of

$$\frac{D_{Q,\lambda}\log m_\lambda}{|\mathbb K_\lambda|},\qquad
\frac{2K_\lambda+k_\lambda-1+\max(\log m_\lambda,k_\lambda t_\lambda d_\lambda)}{|\mathbb K_\lambda|},\qquad
\frac{K_\lambda+k_\lambda+1}{|\mathcal C_\lambda|},$$

and the relaxed-binding error  $\epsilon_{\mathrm{rk},\lambda}(2B_\lambda,\mathcal C_\lambda)$  to be negligible in  $\lambda$ . The corresponding family of Module-SIS instances is assumed hard against classical PPT adversaries. All asymptotic theorems in this paper are conditional on these requirements. Appendix B gives fixed concrete parameter sets and finite error estimates. It neither defines nor proves a family that satisfies this convention.

*Polynomials* We write  $\mathbb{F}^d[X_1, \dots, X_n]$  to denote multivariate polynomials over field  $\mathbb{F}$  in the variables  $(X_1, \dots, X_n)$  with degree bound  $\le d$  for each variable. We omit the superscript if there is no individual-degree bound. For a polynomial  $g$ , we write  $\deg_{\mathrm{tot}}(g)$  for its total degree and  $\deg_{\mathrm{ind}}(g)$  for its maximum individual degree. We define  $\mathbb{ZS}_\ell$  as the set of all multivariate polynomials  $F \in \mathbb{F}[X_1, \dots, X_\ell]$  such that for all  $x \in \{0, 1\}^\ell$ ,  $F(x) = 0$  (i.e. vanish over the Boolean hypercube). We denote the polynomial  $\text{eq}(x, y) = \prod_{i=1}^\ell (x_i \cdot y_i + (1-x_i) \cdot (1-y_i))$ , which outputs 1 if  $x = y$  and 0 otherwise for  $x, y \in \{0, 1\}^\ell$ . For a field vector  $v \in \mathbb{F}^n$ , we let  $\widetilde v \in \mathbb{F}^1[X_1, \dots, X_{\log n}]$  denote its multilinear extension. For a ring vector  $\mathbf v \in \mathbb R_{\mathbb F}^n$ , we use  $\widehat{\mathbf v} \in \mathbb R_{\mathbb K}[X_1,\dots,X_{\log n}]$  for the coefficient-wise multilinear extension after embedding  $\mathbb R_{\mathbb F}$  into  $\mathbb R_{\mathbb K}$ . Thus

$$\widetilde v(\vec X)=\sum_{j\in\{0,1\}^{\log n}}\operatorname{eq}(\vec X,j)v_j,
\qquad
\widehat{\mathbf v}(\vec X)=\sum_{j\in\{0,1\}^{\log n}}\operatorname{eq}(\vec X,j)\mathbf v_j.$$

#### Definition 1 (Fields, Rings, and Dimensions).

Fields: Let  $\mathbb{F}$  be a finite field of prime order  $q$  and  $\mathbb{K}$  to be the lowest degree extension of  $\mathbb{F}$  such that  $1/|\mathbb{K}| = \text{negl}(\lambda)$ . We identify  $\mathbb{F}$  as a subfield of  $\mathbb{K}$ .

Rings: Let  $\Phi(X) := X^d + \Phi_{d-1}X^{d-1} + \dots + \Phi_1X + \Phi_0 \in \mathbb{F}[X]$  be the  $\eta$ -th cyclotomic polynomial with degree  $d$ . We define the ring  $\mathbb{R}_\mathbb{F} := \mathbb{F}[X]/(\Phi(X))$  and the ring  $\mathbb{R}_\mathbb{K} := \mathbb{K}[X]/(\Phi(X))$ . We identify  $\mathbb{F}$  as a sub-ring of  $\mathbb{R}_\mathbb{F}$  and  $\mathbb{R}_\mathbb{F}$  as a sub-ring of  $\mathbb{R}_\mathbb{K}$ .

Dimensions: Let  $m, n_R, n_\mathbb{F} = d \cdot n_R, n_{\mathbb{F}, \text{in}} = d \cdot n_{R, \text{in}} \in \mathbb{N}_{\ge 1}$ . We use  $m$  to denote the number of constraints,  $n_\mathbb{F}$  to denote the length of a field vector,  $n_R$  to denote the length of a ring vector, and  $n_{\mathbb{F}, \text{in}}$  and  $n_{R, \text{in}}$  to denote input lengths. Let  $u, t \in \mathbb{N}_{\ge 1}$ , where  $u$  denotes an upper bound on the total degree of the CCS constraint polynomial and  $t$  denotes a number of matrices. Let  $k, K \in \mathbb{N}_{\ge 1}$ , where  $k$  and  $K$  indicate the number of instances.

Norm bounds: Let  $b, B = b^k \in \mathbb{N}_{\ge 2}$  be protocol norm bounds satisfying  $B < q/2$ . This restriction implies that the prime  $q$  is odd. Define the universal ambient bound

$$B_{\mathrm{amb}} := \left\lfloor \frac q2 \right\rfloor + 1 = \frac{q+1}{2} \in \mathbb{N}.$$

The bound  $B_{\mathrm{amb}}$  is used only for the universal ambient relation; it does not replace the protocol bound  $B=b^k$  or the binding assumptions parameterized by  $B$ .

**Definition 2 (Coefficient maps).** We denote the **coefficient vector** of an element  $a \in \mathbb{R}_\mathbb{F}$  as  $\text{cf}(a) \in \mathbb{F}^d$ . Given a vector  $z \in \mathbb{R}_\mathbb{F}^N$ , we denote  $\text{cf}(z)$  to be the **coefficient matrix**  $[\text{cf}(z_1), \text{cf}(z_2), \dots, \text{cf}(z_N)] \in \mathbb{F}^{d \times N}$ . We define  $\text{cf}(z)_\ell$  to be the  $\ell$ -th row of the coefficient matrix  $\text{cf}(z)$  (i.e. the  $\ell$ -th coefficient vector of  $z$ ).

We denote the **constant term** of an element  $a \in \mathbb{R}_\mathbb{F}$  as  $\text{ct}(a) \in \mathbb{F}$ . Given a vector  $z \in \mathbb{R}_\mathbb{F}^N$ , we denote  $\text{ct}(z)$  to be the vector  $(\text{ct}(z_1), \text{ct}(z_2), \dots, \text{ct}(z_N)) \in \mathbb{F}^N$ .

We analogously define these maps for elements and vectors in  $\mathbb{R}_\mathbb{K}$ .

**Definition 3 (Centered representative and norm).** For an integer  $r \in \mathbb{Z}$ , let  $\iota_q(r) \in \mathbb{F}$  denote reduction modulo  $q$ . For  $a \in \mathbb{F}$ , let  $a' \in \{0, \dots, q-1\}$  be its standard integer representative and define its centered integer representative by

$$\operatorname{ctr}_q(a) := \begin{cases} a' & \text{if } a' \le (q-1)/2, \\ a'-q & \text{if } a' > (q-1)/2. \end{cases}$$

We define  $\|a\|_\infty := |\operatorname{ctr}_q(a)|$ . Hence every field element satisfies

$$\|a\|_\infty \le \left\lfloor \frac q2 \right\rfloor < B_{\mathrm{amb}}.$$

For a vector  $z \in \mathbb{F}^N$ , we define the  $\ell_\infty$ -norm  $\|z\|_\infty$  to be the maximum norm of its elements. For an element  $a \in \mathbb{R}_\mathbb{F}$ , we define  $\|a\|_\infty$  to be the  $\ell_\infty$ -norm of the vector  $\text{cf}(a)$ . For a vector  $z \in \mathbb{R}_\mathbb{F}^N$ , we define  $\|z\|_\infty$  to be the maximum  $\ell_\infty$ -norm of its elements.

*Decomposition* For fixed  $b,k,N\in\mathbb N_{\ge 1}$ , define the total, coordinate-wise map

$$\operatorname{split}_{b,k}:\mathbb F^N\longrightarrow (\mathbb F^N)^k\cup\{\perp\}.$$

For each coordinate  $a\in\mathbb F$ , apply ordinary signed base- $b$  division to its centered representative  $\operatorname{ctr}_q(a)$ . At every step, use the remainder with the same sign as the current dividend. This gives a deterministic digit in  $\{-(b-1),\dots,b-1\}$ . Return  $\perp$  if a nonzero quotient remains after  $k$  digits. Otherwise, embed the signed digits in  $\mathbb F$  coordinate-wise and return  $(z_1,\dots,z_k)$ . In particular,

$$\operatorname{split}_{b,k}(z)=(z_1,\dots,z_k)
\quad\Longrightarrow\quad
z=\sum_{i=1}^k b^{i-1}z_i
\quad\text{and}\quad
\|z_i\|_\infty<b\ \text{ for every }i.$$

Every vector with  $\|z\|_\infty<b^k$  has such a canonical  $k$ -digit decomposition. The map commutes with coordinate projection.

**Definition 4 (Ring Commitment Scheme).** A ring commitment scheme  $\text{com} := (\text{Setup}, \text{Commit})$  consists of two PPT algorithms:

- $\text{Setup}(1^\lambda, n) \to \text{pp}$ : Takes as input a security parameter  $1^\lambda$  and ring-message length  $n \in \mathbb{N}_{\ge 1}$ , outputs public parameters  $\text{pp}$ .
- $\text{Commit}(\text{pp}, z) \to c$ : Takes as input public parameters  $\text{pp}$  and a vector  $z \in \mathbb{R}_\mathbb{F}^n$ , outputs a commitment  $c \in \mathbb{C}$ .

A ring commitment scheme can satisfy the following properties:

**$B$ -binding:** For every length  $n = \text{poly}(\lambda)$  and every EPT adversary  $\mathcal{A}$ , a ring commitment scheme is  $B$ -binding (for  $B \in \mathbb{N}$ ) if the following probability holds:

$$\Pr \left[ \begin{array}{l} \text{Commit}(\text{pp}, z_1) = \text{Commit}(\text{pp}, z_2) \\ \wedge \|z_1\|_\infty, \|z_2\|_\infty < B, \\ \wedge z_1 \neq z_2 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \text{Setup}(1^\lambda, n) \\ z_1, z_2 \in \mathbb{R}_\mathbb{F}^n \leftarrow \mathcal{A}(\text{pp}) \end{array} \right] \le \epsilon_{\text{bind}}(B)$$

for  $\epsilon_{\text{bind}}(B) \le \text{negl}(\lambda)$ . We refer to a pair of vectors  $(z_1, z_2)$  which satisfies the conditions in the probability as a  **$B$ -binding collision**.

**$(B, \mathcal{C})$ -relaxed binding:** For every length  $n = \text{poly}(\lambda)$  and every EPT adversary  $\mathcal{A}$ , a ring commitment scheme is  $(B, \mathcal{C})$ -relaxed binding (for  $B \in \mathbb{N}$  and set  $\mathcal{C} \subseteq \mathbb{R}_\mathbb{F}$ ) if the following probability holds:

$$\Pr \left[ \begin{array}{l} \Delta_1 \cdot c = \text{Commit}(\text{pp}, z_1) \\ \wedge \Delta_2 \cdot c = \text{Commit}(\text{pp}, z_2) \\ \wedge \|z_1\|_\infty, \|z_2\|_\infty < B, \\ \wedge \Delta_1 z_2 \neq \Delta_2 z_1 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \text{Setup}(1^\lambda, n) \\ \left( \begin{array}{c} c \in \mathbb{C}, \\ \Delta_1, \Delta_2 \in (\mathcal{C} - \mathcal{C}), \\ z_1, z_2 \in \mathbb{R}_\mathbb{F}^n \end{array} \right) \leftarrow \mathcal{A}(\text{pp}) \end{array} \right] \le \epsilon_{\text{rk}}(B, \mathcal{C})$$

for  $\epsilon_{\text{rk}}(B, \mathcal{C}) \le \text{negl}(\lambda)$ . We refer to a tuple of elements  $(c, \Delta_1, \Delta_2, z_1, z_2)$  which satisfies the conditions in the probability as a  **$(B, \mathcal{C})$ -relaxed binding collision**.

**Homomorphic:** For every  $n \in \mathbb{N}$  and  $\text{pp} \in \text{Setup}(1^\lambda, n)$ , the commitment algorithm,  $\text{Commit}(\text{pp}, \cdot) : \mathbb{R}_\mathbb{F}^n \to \mathbb{C}$ , is an  $\mathbb{R}_\mathbb{F}$ -module homomorphism.

**Theorem 2 (Properties [2, 7, 9, 14]).** The Ajtai commitment scheme (Definition 18) with ring-message length  $n$  is a ring commitment scheme (Definition 4) that is **homomorphic**,  **$B$ -binding** (assuming the  $\text{MSIS}_{n, 2B}^{\infty, \kappa, q}$  problem (Definition 16) is hard), and  **$(B, \mathcal{C})$ -relaxed binding** (assuming the  $\text{MSIS}_{n, 4TB}^{\infty, \kappa, q}$  problem is hard and  $\mathcal{C}$  is a strong sampling set (Definition 17) with expansion factor  $T$  (Theorem 9)).

**Definition 5 (Interactive Reductions [50, 52]).** Consider relations  $\mathcal{R}_1$  and  $\mathcal{R}_2$  over parameters, structure, instance, and witness tuples. An **interactive reduction** from  $\mathcal{R}_1$  to  $\mathcal{R}_2$  is defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  called the generator, encoder, prover, and verifier respectively with the following interface.

- $\mathcal{G}(1^\lambda, \text{sz}) \to \text{pp}$ : Takes as input a security parameter  $1^\lambda$  and size parameters  $\text{sz}$ . Outputs public parameters  $\text{pp}$ .
- $\mathcal{K}(\text{pp}, \mathbf{s}) \to (\text{pk}, \text{vk})$ : Takes as input public parameters  $\text{pp}$  and a structure  $\mathbf{s}$ . Deterministically, outputs a prover key  $\text{pk}$  and a verifier key  $\text{vk}$ .
- $\mathcal{P}(\text{pk}, u_1, w_1) \to (u_2, w_2)$ : Takes as input a proving key  $\text{pk}$  and an instance-witness pair  $(u_1, w_1)$ . Interactively reduces the task of checking  $(\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$  to the task of checking  $(\text{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}_2$ .
- $\mathcal{V}(\text{vk}, u_1) \to u_2$ : Takes as input a verifier key  $\text{vk}$  and an instance  $u_1$  in  $\mathcal{R}_1$ . Interactively reduces the task of checking the instance  $u_1$  to the task of checking a new instance  $u_2$  in  $\mathcal{R}_2$ .

Let  $\langle \mathcal{P}, \mathcal{V} \rangle$  denote the interaction between  $\mathcal{P}$  and  $\mathcal{V}$ . We treat  $\langle \mathcal{P}, \mathcal{V} \rangle$  as a function that takes as input  $((\text{pk}, \text{vk}), u_1, w_1)$  and runs the interaction on the prover's input  $(\text{pk}, u_1, w_1)$  and the verifier's input  $(\text{vk}, u_1)$ . At the end of the interaction,  $\langle \mathcal{P}, \mathcal{V} \rangle$  outputs the verifier's instance  $u_2$  and the prover's witness  $w_2$ .

A **reduction of knowledge** [51] is an interactive reduction,  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$ , that satisfies the following properties:

- (i) **Completeness**: For any EPT adversary  $\mathcal{A}$ , given  $\text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz})$ ,  $(\mathbf{s}, u_1, w_1) \leftarrow \mathcal{A}(\text{pp})$  such that  $(\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$ , we have that the prover's output instance is equal to the verifier's output instance  $u_2$ , and that

$$(\text{pp}, \mathbf{s}, \langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1)) \in \mathcal{R}_2.$$

- (ii) **Knowledge soundness**: For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , there exists an EPT extractor  $\mathcal{E}$  such that if the success probability of the adversary

$$\epsilon(\mathcal{A}, \mathcal{P}^*) := \Pr \left[ (\text{pp}, \mathbf{s}, \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st})) \in \mathcal{R}_2 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s}) \end{array} \right. \right]$$

$\ge 1/\text{poly}(\lambda)$ , then we have that

$$\Pr \left[ (\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}) \end{array} \right. \right] \ge \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

- (iii) **Public Coin**: Each verifier message is sampled uniformly from its prescribed finite challenge set, and the verifier messages reveal all randomness used by the verifier.<sup>7</sup> A bit-string implementation must sample the prescribed distribution exactly or account for its statistical distance from that distribution.

<sup>7</sup> If a reduction of knowledge is public-coin, then it trivially satisfies the property of **public reducibility** described in [52] as the execution of the verifier  $\mathcal{V}$  can be emulated using the randomness from the transcript.

**Lemma 2 (Sequential composition [50, 52]).** *For reductions of knowledge  $\Pi_1 = (\mathcal{G}, \mathcal{K}, \mathcal{P}_1, \mathcal{V}_1) : \mathcal{R}_1 \to \mathcal{R}_2$  and  $\Pi_2 = (\mathcal{G}, \mathcal{K}, \mathcal{P}_2, \mathcal{V}_2) : \mathcal{R}_2 \to \mathcal{R}_3$ , we have that  $\Pi_2 \circ \Pi_1 = (\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V}) : \mathcal{R}_1 \to \mathcal{R}_3$  is a reduction of knowledge where  $\mathcal{K}(\mathbf{pp}, \mathbf{s})$  computes  $(\mathbf{pk}, \mathbf{vk})$  and where*

$$\begin{aligned}\mathcal{P}(\mathbf{pk}, u_1, w_1) &= \mathcal{P}_2(\mathbf{pk}, \mathcal{P}_1(\mathbf{pk}, u_1, w_1)) \\ \mathcal{V}(\mathbf{vk}, u_1) &= \mathcal{V}_2(\mathbf{vk}, \mathcal{V}_1(\mathbf{vk}, u_1))\end{aligned}$$

**Definition 6 (The sum-check protocol [62]).** *The sum-check protocol SumCheck( $T; Q$ ) is a classic interactive proof protocol between two PPT algorithms  $(\mathcal{P}, \mathcal{V})$  that checks that the sum of evaluations of a  $\ell$ -variate polynomial  $Q \in \mathbb{F}^{\le d}[X_1, \dots, X_\ell]$  on the Boolean hypercube results in some claimed value  $T$ . The output of the sum-check protocol is a claim that  $v \stackrel{?}{=} Q(r)$  for some random point  $r \in \mathbb{F}^\ell$  and claimed evaluations  $v$ , which the verifier  $\mathcal{V}$  can query  $Q$  to check. The protocol is public-coin, has a completeness error of 0, and has a soundness error of  $\le \ell d / |\mathbb{F}|$ . More generally, the field can be chosen to be an extension field  $\mathbb{K}$ . In this case, the soundness error is  $\le \ell d / |\mathbb{K}|$ . A self-contained description of the sum-check protocol can be found in this note [81].*
