# 4 Preliminaries

**Notation** We let  $\lambda$  denote the security parameter and  $\text{negl}(\lambda)$  denote a negligible function in  $\lambda$ . Throughout the paper, the depicted asymptotics depend on  $\lambda$ , but we elide this for brevity. We let PPT denote probabilistic polynomial time and EPT denote expected probabilistic polynomial time. We let  $[n]$  denote the set  $\{1, \dots, n\}$ , and  $\{u_i\}_{i \in [n]}$  denote the set  $\{u_1, \dots, u_n\}$ .

**Polynomials** Let  $\mathbb{B}$  be a field. We write  $\mathbb{B}^d[X_1, \dots, X_n]$  to denote multivariate polynomials over  $\mathbb{B}$  in the variables  $(X_1, \dots, X_n)$  with degree bound  $\leq d$  for each variable. We omit the superscript if there is no degree bound. For  $\ell \in \mathbb{N}_{\geq 1}$ , define

$$\text{ZS}_\ell(\mathbb{B}) := \{P \in \mathbb{B}[X_1, \dots, X_\ell] : P(x) = 0 \text{ for every } x \in \{0, 1\}^\ell\}.$$

When the coefficient field is clear from context, we write  $\text{ZS}_\ell$ . We denote the polynomial  $\text{eq}(x, y) = \prod_{i=1}^\ell (x_i \cdot y_i + (1 - x_i) \cdot (1 - y_i))$ , which outputs 1 if  $x = y$  and 0 otherwise for  $x, y \in \{0, 1\}^\ell$ . For a vector  $v \in \mathbb{B}^n$ , where  $n$  is a power of two, we let  $\tilde{v} \in \mathbb{B}^1[X_1, \dots, X_{\log n}]$  denote the multilinear polynomial extension of  $v$ :  $\tilde{v} = \sum_{b \in \{0, 1\}^{\log n}} \text{eq}(X_1, \dots, X_{\log n}, b) \cdot v_b$ . Let  $\mathbb{B}$  and  $\mathbb{K}$  be subfields of a common field  $\mathbb{A}$ . For  $v \in \mathbb{B}^n$  and  $r \in \mathbb{K}^{\log n}$ , we evaluate  $\tilde{v}$  at  $r$  through the natural inclusions into  $\mathbb{A}$ ; explicitly,  $\tilde{v}(r) := \sum_{b \in \{0, 1\}^{\log n}} \text{eq}(r, b) \cdot v_b \in \mathbb{A}$ . More generally, let  $R_{\mathbb{B}}$  and  $R_{\mathbb{K}}$  be subrings of a common commutative ring  $R_{\mathbb{A}}$ . For  $h \in R_{\mathbb{B}}^n$  and  $r \in R_{\mathbb{K}}^{\log n}$ , define  $\tilde{h}(r) := \sum_{b \in \{0, 1\}^{\log n}} \text{eq}(r, b) \cdot h_b \in R_{\mathbb{A}}$ .

**Lemma 2 (Schwartz-Zippel [77]).** *let  $g : \mathbb{F}^\ell \rightarrow \mathbb{F}$  be a nonzero  $\ell$ -variate polynomial of total degree at most  $d$ . Then, on any finite set  $S \subseteq \mathbb{F}$ ,*

$$\Pr_{x \leftarrow S^\ell} [g(x) = 0] \leq d/|S|.$$

**Lemma 3.** *Let  $\mathbb{B}$  be a field and  $Q \in \mathbb{B}[X_1, \dots, X_\ell]$  be an arbitrary multivariate polynomial. Define multivariate polynomial  $Q'(\vec{X}, \vec{Z}) := \text{eq}(\vec{X}, \vec{Z}) \cdot Q(\vec{X})$ .*

$$0 = \sum_{\vec{x} \in \{0, 1\}^\ell} Q'(\vec{x}, \vec{Z}) \quad \text{if and only if} \quad Q(\vec{X}) \in \text{ZS}_\ell$$

### **Definition 1 (Fields, Rings, and Dimensions).**

Fields: We identify the following fields  $\mathbb{F}$ ,  $\mathbb{L}$ , and  $\mathbb{K}$  with their corresponding subfields of  $\mathbb{A}$ . As such, we treat each field as a subfield of the larger fields under the natural inclusions  $\mathbb{F} \subseteq \mathbb{L} \subseteq \mathbb{A}$  and  $\mathbb{F} \subseteq \mathbb{K} \subseteq \mathbb{A}$ .

- **Base field:** Let  $\mathbb{F} := \mathbb{F}_q$  be a finite field of prime order  $q$ .
- **Native relation field:** Let  $\mathbb{L} := \mathbb{F}_{q^\tau}$  be a degree- $\tau$  extension of  $\mathbb{F}$ .
- **Sum-check challenge field:** Let  $\mathbb{K} := \mathbb{F}_{q^\nu}$  be a degree- $\nu$  extension of  $\mathbb{F}$  such that  $1/|\mathbb{K}| = \text{negl}(\lambda)$ .
- **Ambient field:** Let  $\mathbb{A} := \mathbb{F}_{q^{\text{lcm}(\tau, \nu)}}$  be the degree- $\text{lcm}(\tau, \nu)$  extension of  $\mathbb{F}$  containing both  $\mathbb{L}$  and  $\mathbb{K}$ .

Rings: Let  $\Phi(X) := X^d + \Phi_{d-1}X^{d-1} + \dots + \Phi_1X + \Phi_0 \in \mathbb{F}[X]$  be the  $\eta$ -th cyclotomic polynomial with degree  $d$ . For every field  $\mathbb{B} \in \{\mathbb{F}, \mathbb{L}, \mathbb{K}, \mathbb{A}\}$ , define the ring  $R_{\mathbb{B}} := \mathbb{B}[X]/(\Phi(X))$ . We treat each ring as a subring of the larger rings under the natural inclusions  $R_{\mathbb{F}} \subseteq R_{\mathbb{L}} \subseteq R_{\mathbb{A}}$  and  $R_{\mathbb{F}} \subseteq R_{\mathbb{K}} \subseteq R_{\mathbb{A}}$ .

Dimensions: Let  $m, n_L, n_{\mathbb{F}}, n_{\mathbb{F}, \text{in}}, n_{\mathbb{R}}, n_{\mathbb{R}, \text{in}}, u, t, k, K \in \mathbb{N}_{\geq 1}$ .

- **CCS parameters:**  $m$  denotes the number of constraints and is a power of two,  $u$  denotes a strict upper bound on the total degree of the CCS polynomial, and  $t$  denotes the number of CCS matrices.
- **Vector lengths:**  $n_L$  denotes the length of a relation vector  $z \in \mathbb{L}^{n_L}$ ,  $n_{\mathbb{F}}$  denotes the length of the corresponding flattened base-field vector  $z^b \in \mathbb{F}^{n_{\mathbb{F}}}$  (defined shortly), and  $n_{\mathbb{R}}$  denotes the length of its packed ring-vector representation  $\mathbf{z} \in R_{\mathbb{F}}^{n_{\mathbb{R}}}$ . These dimensions satisfy  $n_{\mathbb{F}} := \tau \cdot n_L = d \cdot n_{\mathbb{R}} \leq m$ . Define the padding length  $n_{\text{pad}} := m - n_{\mathbb{F}} \in \mathbb{N}_{\geq 0}$ .
- **Input lengths:**  $n_{L, \text{in}}, n_{\mathbb{F}, \text{in}}$ , and  $n_{\mathbb{R}, \text{in}}$  denote the corresponding input lengths and satisfy  $n_{\mathbb{F}, \text{in}} := \tau \cdot n_{L, \text{in}} = d \cdot n_{\mathbb{R}, \text{in}}$  and  $n_{L, \text{in}} \leq n_L$ .
- **Instance counts:**  $k$  and  $K$  denote the numbers of instances.

Norm bounds: Let  $b, B = b^k < q/2 \in \mathbb{N}_{\geq 2}$  be norm bounds.

*Remark 1 (Compatible dimensions).* For a target native vector length  $n$ , set

$$s := \frac{d}{\gcd(d, \tau)}, \quad n_L := s \left\lceil \frac{n}{s} \right\rceil, \quad n_F := \tau n_L, \quad n_R := \frac{n_F}{d}.$$

This chooses the smallest  $n_L \geq n$  satisfying  $d \mid \tau n_L$ , with minimal overhead  $0 \leq n_L - n \leq s - 1$ .

**Definition 2 (Coefficient maps).** For every field  $\mathbb{B} \in \{\mathbb{F}, \mathbb{L}, \mathbb{K}, \mathbb{A}\}$ , consider the ring  $R_{\mathbb{B}}$  defined in Definition 1. For an element  $a = \sum_{\ell=1}^d a_{\ell} X^{\ell-1} \in R_{\mathbb{B}}$ , define its **coefficient vector** and **constant term** as

$$\text{cf}(a) := (a_1, \dots, a_d) \in \mathbb{B}^d, \quad \text{ct}(a) := a_1 \in \mathbb{B}.$$

Both maps are trivially  $\mathbb{B}$ -linear, and the coefficient map is bijective.

For a vector  $z = (z_1, \dots, z_m) \in R_{\mathbb{B}}^m$ , define its **coefficient matrix** and **constant-term vector** as

$$\text{cf}(z) := [\text{cf}(z_1) \cdots \text{cf}(z_m)] \in \mathbb{B}^{d \times m}, \quad \text{ct}(z) := (\text{ct}(z_1), \dots, \text{ct}(z_m)) \in \mathbb{B}^m.$$

Concretely,  $\text{cf}(z)_{\ell, j} := \text{cf}(z_j)_{\ell}$  for  $\ell \in [d]$  and  $j \in [m]$ . We denote the  $\ell$ -th row of  $\text{cf}(z)$  by  $\text{cf}(z)_{\ell} \in \mathbb{B}^m$ .

**Definition 3 (Norm).** For an element  $a \in \mathbb{F}$ , let  $a' \in \{0, \dots, q-1\}$  denote its integer representative. Define  $\|a\|_{\infty} := \min\{a', q - a'\}$ . For a vector  $z = (z_1, \dots, z_n) \in \mathbb{F}^n$ , define its  $\ell_{\infty}$ -norm as  $\|z\|_{\infty} := \max_{i \in [n]} \|z_i\|_{\infty}$ . For an element  $a \in R_{\mathbb{F}}$  and a vector  $z = (z_1, \dots, z_m) \in R_{\mathbb{F}}^m$ , define  $\|a\|_{\infty} := \|\text{cf}(a)\|_{\infty}$  and  $\|z\|_{\infty} := \max_{i \in [m]} \|z_i\|_{\infty}$ .

**Decomposition** For  $b, k$  as in Definition 1 and every  $s \in \mathbb{N}_{\geq 1}$ , let  $\text{split}_b : \{z \in \mathbb{F}^s : \|z\|_{\infty} < b^k\} \rightarrow (\mathbb{F}^s)^k$  be the coordinate-wise signed  $b$ -ary decomposition map such that, for  $\text{split}_b(z) = (z_1, \dots, z_k)$ ,

$$z = \sum_{i=1}^k b^{i-1} \cdot z_i \quad \text{and} \quad \|z_i\|_{\infty} < b \quad \text{for all } i \in [k].$$

**Definition 4 (Module Homomorphism).** Modules are a generalization of vector spaces for which the field of scalars is replaced by a ring  $R$ . Suppose  $R$  is a commutative ring with identity 1 and  $G$  is an abelian (commutative) group. The group  $G$  is an  $R$ -module if there is an operation  $\cdot : R \times G \rightarrow G$  such that for all  $r, s \in R$  and  $x, y \in G$ ,  $r \cdot (x + y) = r \cdot x + r \cdot y$ ,  $(r + s) \cdot x = r \cdot x + s \cdot x$ ,  $(rs) \cdot x = r \cdot (s \cdot x)$ ,  $1 \cdot x = x$ . Suppose  $G_1$  and  $G_2$  are  $R$ -modules. Similarly, an  $R$ -module homomorphism is a map  $\mathcal{L} : G_1 \rightarrow G_2$  that is a generalization of a linear map of vector spaces.  $\mathcal{L}$  is an  $R$ -module homomorphism if for all  $x, y \in G_1$  and  $r \in R$ ,  $\mathcal{L}(x + y) = \mathcal{L}(x) + \mathcal{L}(y)$  and  $\mathcal{L}(r \cdot x) = r \cdot \mathcal{L}(x)$ .

**Definition 5 (Module short integer solution [61, 64, 74]).** Define the ring  $R_{\mathbb{Z}} := \mathbb{Z}[X]/(\Phi(X))$ . The  $\text{MSIS}_{m, B}^{\infty, \kappa, q}$  problem is defined as follows: Given a matrix  $M \stackrel{\$}{\leftarrow} R_{\mathbb{F}}^{\kappa \times m}$  sampled uniformly at random, find a non-zero vector  $z \in R_{\mathbb{Z}}^m$  such that  $Mz = 0 \pmod{q}$  and  $\|z\|_{\infty} < B$ , where  $\|z\|_{\infty} := \max_{i \in [m], \ell \in [d]} |z_{i, \ell}|$  for  $z_i = \sum_{\ell=1}^d z_{i, \ell} X^{\ell-1}$ .

**Theorem 4 (Low norm invertibility [66, Theorem 1.1, Conjecture 2.6]).** Let  $z \in \mathbb{N}$  such that  $z \mid \eta$ ,  $q \equiv 1 \pmod{z}$ , and  $\text{ord}_{\eta}(q) = \eta/z$ . Define  $b_{\text{inv}} := 1/\sqrt{\tau(z)} \cdot q^{1/\phi(z)}$  where  $\tau(z) := z$  if  $z$  is odd, otherwise  $\tau(z) = z/2$ . For an arbitrary  $a \in R_{\mathbb{F}}$ , if  $0 < \|a\|_{\infty} < b_{\text{inv}}$ , then  $a$  is invertible in  $R_{\mathbb{F}}$ .

**Definition 6 (Strong sampling sets [3, 28]).** Define  $\mathcal{C} \subseteq R_{\mathbb{F}}$  to be any set of ring elements such that for any distinct elements  $a, b \in \mathcal{C}$ ,  $\|a - b\|_{\infty} < b_{\text{inv}}$  (Theorem 4). Furthermore, we define the

$$\text{expansion factor of } \mathcal{C} := \max_{\substack{v \in R_{\mathbb{F}} \setminus \{0\} \\ \rho \in \mathcal{C}}} \frac{\|\rho v\|_{\infty}}{\|v\|_{\infty}}$$

**Theorem 5 (Expansion factors [3]).** Let  $\mathcal{C}$  be a strong sampling set over the cyclotomic ring  $R_{\mathbb{F}}$  (Definition 6). We denote the Euler totient function as  $\phi$ . We must have that the expansion factor of  $\mathcal{C}$  is  $\leq 2 \cdot \phi(\eta) \cdot \max_{\rho \in \mathcal{C}} \|\rho\|_{\infty}$ . If  $\Phi(X) = X^d + 1$  for a power-of-two  $d$ , then the factor of 2 can be removed.

**Definition 7 (Ring Commitment Scheme).** A ring commitment scheme  $\text{com} := (\text{Setup}, \text{Commit})$  consists of two PPT algorithms:

- $\text{Setup}(1^{\lambda}, m) \rightarrow \text{pp}$ : Takes as input a security parameter  $1^{\lambda}$  and length  $m \in \mathbb{N}_{\geq 1}$ , outputs public parameters  $\text{pp}$ .
- $\text{Commit}(\text{pp}, z) \rightarrow c$ : Takes as input public parameters  $\text{pp}$  and a vector  $z \in R_{\mathbb{F}}^m$ , outputs a commitment  $c \in \mathbb{C}$ .

A ring commitment scheme can satisfy the following properties:

**B-binding:** For every length  $m = \text{poly}(\lambda)$  and every EPT adversary  $\mathcal{A}$ , a ring commitment scheme is B-binding (for  $B \in \mathbb{N}$ ) if the following probability holds:

$$\Pr \left[ \begin{array}{l} \text{Commit}(\text{pp}, z_1) = \text{Commit}(\text{pp}, z_2) \\ \wedge \|z_1\|_{\infty}, \|z_2\|_{\infty} < B, \\ \wedge z_1 \neq z_2 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \text{Setup}(1^{\lambda}, m) \\ z_1, z_2 \in R_{\mathbb{F}}^m \leftarrow \mathcal{A}(\text{pp}) \end{array} \right] \leq \epsilon_{\text{bind}}(B)$$

for  $\epsilon_{\text{bind}}(B) \leq \text{negl}(\lambda)$ . We refer to a pair of vectors  $(z_1, z_2)$  which satisfies the conditions in the probability as a **B-binding collision**.

**(B,C)-relaxed binding:** For every length  $m = \text{poly}(\lambda)$  and every EPT adversary  $\mathcal{A}$ , a ring commitment scheme is (B,C)-relaxed binding (for  $B \in \mathbb{N}$  and set  $\mathcal{C} \subseteq R_{\mathbb{F}}$ ) if the following probability holds:

$$\Pr \left[ \begin{array}{l} \Delta_1 \cdot c = \text{Commit}(\text{pp}, z_1) \\ \wedge \Delta_2 \cdot c = \text{Commit}(\text{pp}, z_2) \\ \wedge \|z_1\|_{\infty}, \|z_2\|_{\infty} < B, \\ \wedge \Delta_1 z_2 \neq \Delta_2 z_1 \end{array} \middle| \begin{array}{l} \text{pp} \leftarrow \text{Setup}(1^{\lambda}, m) \\ \left( \begin{array}{l} c \in \mathbb{C}, \\ \Delta_1, \Delta_2 \in (\mathcal{C} - \mathcal{C}), \\ z_1, z_2 \in R_{\mathbb{F}}^m \end{array} \right) \leftarrow \mathcal{A}(\text{pp}) \end{array} \right] \leq \epsilon_{\text{rlx}}(B, \mathcal{C})$$

for  $\epsilon_{\text{rlx}}(B, \mathcal{C}) \leq \text{negl}(\lambda)$ . We refer to a tuple of elements  $(c, \Delta_1, \Delta_2, z_1, z_2)$  which satisfies the conditions in the probability as a **(B,C)-relaxed binding collision**.

**Homomorphic:** For every  $m \in \mathbb{N}$  and  $\text{pp} \in \text{Setup}(1^{\lambda}, m)$ , the commitment algorithm,  $\text{Commit}(\text{pp}, \cdot) : R_{\mathbb{F}}^m \rightarrow \mathbb{C}$ , is an  $R_{\mathbb{F}}$ -module homomorphism.

**Definition 8 (Ajtai commitment scheme [2]).** Let message length  $m \in \mathbb{N}$ , and let  $\lambda$  be a security parameter with  $\kappa := \kappa(\lambda)$  and  $q := q(\lambda)$ , where  $|\mathbb{F}| = q$ . Define  $\mathbb{C} := R_{\mathbb{F}}^{\kappa}$ . The Ajtai commitment scheme  $\text{com} := (\text{Setup}, \text{Commit})$  consists of the following PPT algorithms:

- $\text{Setup}(1^{\lambda}, m) \rightarrow \text{pp}$ : Sample a random matrix  $M \stackrel{\$}{\leftarrow} R_{\mathbb{F}}^{\kappa \times m}$ . Output  $\text{pp} \leftarrow M$ .
- $\text{Commit}(\text{pp}, z) \rightarrow c$ : Given parameters  $\text{pp} := M$  and vector  $z \in R_{\mathbb{F}}^m$ , output  $c \leftarrow Mz \in \mathbb{C}$ .

**Theorem 6 (Properties [2, 7, 9, 14]).** The Ajtai commitment scheme (Definition 8) is a ring commitment scheme (Definition 7) that is **homomorphic**, **B-binding** (assuming the  $\text{MSIS}_{m, 2B}^{\infty, \kappa, q}$  problem (Definition 5) is hard), and **(B,C)-relaxed binding** (assuming the  $\text{MSIS}_{m, 4TB}^{\infty, \kappa, q}$  problem is hard and  $\mathcal{C}$  is a strong sampling set (Definition 6) with expansion factor  $T$  (Theorem 5)).

**Relation Products** For relations  $\mathcal{R}_1$  and  $\mathcal{R}_2$  over public parameter, structure, instance, and witness pairs we define the relation  $\mathcal{R}_1 \times \mathcal{R}_2$  such that  $(\mathbf{pp}, \mathbf{s}, (u_1, u_2), (w_1, w_2)) \in \mathcal{R}_1 \times \mathcal{R}_2$  if and only if  $(\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$ , and  $(\mathbf{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}_2$ . We let  $\mathcal{R}^n$  denote  $\mathcal{R} \times \dots \times \mathcal{R}$  for  $n$  times.

By a slight abuse of notation, whenever we write  $\mathbf{CE}(b, \mathcal{L})^n$  (Definition 21), we mean the subrelation of the ordinary  $n$ -fold product in which all  $n$  instances contain the same evaluation point  $r \in \mathbb{K}^{\log m}$ . All other relation products retain the ordinary product meaning defined above.

**Definition 9 (Interactive Reductions [51,53]).** Consider relations  $\mathcal{R}_1$  and  $\mathcal{R}_2$  over parameters, structure, instance, and witness tuples. An **interactive reduction** from  $\mathcal{R}_1$  to  $\mathcal{R}_2$  is defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  called the generator, encoder, prover, and verifier respectively with the following interface.

- $\mathcal{G}(1^\lambda, \mathbf{sz}) \rightarrow \mathbf{pp}$ : Takes as input a security parameter  $1^\lambda$  and size parameters  $\mathbf{sz}$ . Outputs public parameters  $\mathbf{pp}$ .
- $\mathcal{K}(\mathbf{pp}, \mathbf{s}) \rightarrow (\mathbf{pk}, \mathbf{vk})$ : Takes as input public parameters  $\mathbf{pp}$  and a structure  $\mathbf{s}$ . Deterministically, outputs a prover key  $\mathbf{pk}$  and a verifier key  $\mathbf{vk}$ .
- $\mathcal{P}(\mathbf{pk}, u_1, w_1) \rightarrow (u_2, w_2)$ : Takes as input a proving key  $\mathbf{pk}$  and an instance-witness pair  $(u_1, w_1)$ . Interactively reduces the task of checking  $(\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$  to the task of checking  $(\mathbf{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}_2$ .
- $\mathcal{V}(\mathbf{vk}, u_1) \rightarrow u_2$ : Takes as input a verifier key  $\mathbf{vk}$  and an instance  $u_1$  in  $\mathcal{R}_1$ . Interactively reduces the task of checking the instance  $u_1$  to the task of checking a new instance  $u_2$  in  $\mathcal{R}_2$ .

Let  $\langle \mathcal{P}, \mathcal{V} \rangle$  denote the interaction between  $\mathcal{P}$  and  $\mathcal{V}$ . We treat  $\langle \mathcal{P}, \mathcal{V} \rangle$  as a function that takes as input  $((\mathbf{pk}, \mathbf{vk}), u_1, w_1)$  and runs the interaction on the prover's input  $(\mathbf{pk}, u_1, w_1)$  and the verifier's input  $(\mathbf{vk}, u_1)$ . At the end of the interaction,  $\langle \mathcal{P}, \mathcal{V} \rangle$  outputs the verifier's instance  $u_2$  and the prover's witness  $w_2$ .

A **reduction of knowledge** [52] is an interactive reduction,  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$ , that satisfies the following properties:

- (i) **Completeness**: For any EPT adversary  $\mathcal{A}$ , given  $\mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz})$ ,  $(\mathbf{s}, u_1, w_1) \leftarrow \mathcal{A}(\mathbf{pp})$  such that  $(\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$ , we have that the prover's output instance is equal to the verifier's output instance  $u_2$ , and that

$$(\mathbf{pp}, \mathbf{s}, \langle \mathcal{P}, \mathcal{V} \rangle((\mathbf{pk}, \mathbf{vk}), u_1, w_1)) \in \mathcal{R}_2.$$

- (ii) **Knowledge soundness**: For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , there exists an EPT extractor  $\mathcal{E}$  such that if the success probability of the adversary

$$\epsilon(\mathcal{A}, \mathcal{P}^*) := \Pr \left[ (\mathbf{pp}, \mathbf{s}, \langle \mathcal{P}^*, \mathcal{V} \rangle((\mathbf{pk}, \mathbf{vk}), u_1, \mathbf{st})) \in \mathcal{R}_2 \left| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, u_1, \mathbf{st}) \leftarrow \mathcal{A}(\mathbf{pp}) \\ (\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s}) \end{array} \right. \right]$$

$\geq 1/\text{poly}(\lambda)$ , then we have that

$$\Pr \left[ (\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \mathbf{pp} \leftarrow \mathcal{G}(1^\lambda, \mathbf{sz}) \\ (\mathbf{s}, u_1, \mathbf{st}) \leftarrow \mathcal{A}(\mathbf{pp}) \\ (\mathbf{pk}, \mathbf{vk}) \leftarrow \mathcal{K}(\mathbf{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}(\mathbf{pp}, \mathbf{s}, u_1, \mathbf{st}) \end{array} \right. \right] \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

- (iii) **Public Coin**: All of the verifier's messages are uniformly random strings of some prescribed length. Furthermore, the verifier's messages contain all of the random coins (randomness) used by the verifier.<sup>9</sup>

In this work, we are primarily interested in building folding schemes, a particular type of reduction of knowledge that reduces the task of checking a fresh instance in a relation  $\mathcal{R}$  together with a running instance in an accumulator relation  $\mathcal{R}_{\text{ACC}}$  to checking a new running instance in  $\mathcal{R}_{\text{ACC}}$ .

**Definition 10 (Folding scheme).** A folding scheme for a relation  $\mathcal{R}$  is a reduction of knowledge of type  $\mathcal{R} \times \mathcal{R}_{\text{ACC}} \rightarrow \mathcal{R}_{\text{ACC}}$  for some accumulator relation  $\mathcal{R}_{\text{ACC}}$ .

<sup>9</sup> If a reduction of knowledge is public-coin, then it trivially satisfies the property of **public reducibility** described in [53] as the execution of the verifier  $\mathcal{V}$  can be emulated using the randomness from the transcript.

**Lemma 4 (Sequential composition [51,53]).** For reductions of knowledge  $\Pi_1 = (\mathcal{G}, \mathcal{K}, \mathcal{P}_1, \mathcal{V}_1) : \mathcal{R}_1 \rightarrow \mathcal{R}_2$  and  $\Pi_2 = (\mathcal{G}, \mathcal{K}, \mathcal{P}_2, \mathcal{V}_2) : \mathcal{R}_2 \rightarrow \mathcal{R}_3$ , we have that  $\Pi_2 \circ \Pi_1 = (\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V}) : \mathcal{R}_1 \rightarrow \mathcal{R}_3$  is a reduction of knowledge where  $\mathcal{K}(\text{pp}, \mathbf{s})$  computes  $(\text{pk}, \text{vk})$  and where

$$\begin{aligned}\mathcal{P}(\text{pk}, u_1, w_1) &= \mathcal{P}_2(\text{pk}, \mathcal{P}_1(\text{pk}, u_1, w_1)) \\ \mathcal{V}(\text{vk}, u_1) &= \mathcal{V}_2(\text{vk}, \mathcal{V}_1(\text{vk}, u_1, w_1))\end{aligned}$$

**Definition 11 (The sum-check protocol [63]).** For the challenge and ambient fields  $\mathbb{K} \subseteq \mathbb{A}$  from Definition 1, the sum-check protocol  $\text{SumCheck}(\mathbf{T}; Q)$  is a classic interactive proof protocol between two PPT algorithms  $(\mathcal{P}, \mathcal{V})$ . The protocol reduces a sum-check claim  $\mathbf{T} = \sum_{x \in \{0,1\}^\ell} Q(x)$ , where  $\mathbf{T} \in \mathbb{A}$  is the claimed sum and  $Q \in \mathbb{A}^{\leq d}[X_1, \dots, X_\ell]$  is an  $\ell$ -variate polynomial of individual degree at most  $d$ , to an evaluation claim  $v \stackrel{?}{=} Q(r)$ , where  $r \xleftarrow{\$} \mathbb{K}^\ell$  is a uniformly random point and  $v \in \mathbb{A}$  is the claimed evaluation. The verifier can check this claim by querying  $Q$  at  $r$ . The protocol is public-coin, has a completeness error of 0, and has a soundness error of at most  $\ell d / |\mathbb{K}|$ . A self-contained description of the sum-check protocol can be found in this note [82].

**Definition 12 (Special sets [39]).** Let  $\mathcal{C}$  be a set and  $\ell \in \mathbb{N}$ . Consider two vectors  $x, y \in \mathcal{C}^\ell$ . We define the relation  $\equiv_i$  for  $i \in [\ell]$  as follows:

$$x \equiv_i y \iff x_i \neq y_i \wedge x_j = y_j \text{ for all } j \in [\ell] \setminus \{i\}.$$

A special set  $\text{SS}(\mathcal{C}, \ell)$  is as follows:

$$\text{SS}(\mathcal{C}, \ell) = \left\{ (\vec{c}, \vec{c}_1, \dots, \vec{c}_\ell) \in (\mathcal{C}^\ell)^{\ell+1} : \begin{array}{l} \forall i \in [\ell], \\ \vec{c} \equiv_i \vec{c}_i \end{array} \right\},$$

**Theorem 7 (Coordinate-wise extraction [39, Lemma 7.1]).** Let  $\mathcal{C}$  be a finite set,  $\ell \in \mathbb{N}$ , and  $\vec{\mathcal{C}} := \mathcal{C}^\ell$  be a challenge space. Let  $A : \vec{\mathcal{C}} \rightarrow \{0,1\}^*$  be an arbitrary (probabilistic) expected polynomial-time algorithm (adversary), and  $V : \vec{\mathcal{C}} \times \{0,1\}^* \rightarrow \{0,1\}$  be an arbitrary (probabilistic) expected polynomial-time function (verification). Define the success probability of adversary  $A$  as

$$\epsilon^V(A) := \Pr_{\vec{c} \xleftarrow{\$} \vec{\mathcal{C}}}[V(\vec{c}, A(\vec{c})) = 1]$$

Then, there exists an expected polynomial-time oracle algorithm  $E_A$  (extractor) that makes at most  $\ell+1$  queries to  $A$  in expectation and with probability at least  $\epsilon^V(A) - \frac{\ell}{|\vec{\mathcal{C}}|}$  outputs  $\ell+1$  pairs  $(\vec{c}, w), (\vec{c}_1, w_1), \dots, (\vec{c}_\ell, w_\ell)$  such that

- $V(\vec{c}, w) = 1$ ,
- for all  $i \in [\ell]$ ,  $V(\vec{c}_i, w_i) = 1$ ,
- and  $(\vec{c}, \vec{c}_1, \dots, \vec{c}_\ell) \in \text{SS}(\mathcal{C}, \ell)$ .

