# 5 Embeddings and Evaluation Homomorphism

Let  $\mathbb{B}$  be a degree- $\delta$  extension of  $\mathbb{F}$  with a fixed ordered  $\mathbb{F}$ -basis  $(\beta_1, \dots, \beta_\delta) \in \mathbb{B}^\delta$ . Later, we concretely instantiate  $\mathbb{B}$  as either  $\mathbb{F}$  or  $\mathbb{L}$  from Definition 1 for the norm and relation checks, respectively.

**Definition 13 (Coefficient Embedding).**

Element embedding: For a vector  $v = (v_1, \dots, v_d) \in \mathbb{B}^d$ , define  $\text{Emb}_{\mathbb{B}}(v) := \sum_{i=1}^d v_i X^{i-1} \in \mathbb{R}_{\mathbb{B}}$ . By Definition 2,  $\text{Emb}_{\mathbb{B}}$  is the inverse of the coefficient map  $\text{cf}(\cdot)$ ; in particular,  $\text{cf}(\text{Emb}_{\mathbb{B}}(v)) = v$ .

Vector embedding: For  $s \in \mathbb{N}_{\geq 1}$  and  $v \in \mathbb{B}^{ds}$ , partition  $v = (v_1, \dots, v_s)$  into blocks  $v_j \in \mathbb{B}^d$  and define

$$\text{Emb}_{\mathbb{B}}(v) := (\text{Emb}_{\mathbb{B}}(v_1), \dots, \text{Emb}_{\mathbb{B}}(v_s)) \in \mathbb{R}_{\mathbb{B}}^s.$$

Matrix embedding: For  $M \in \mathbb{B}^{m \times ds}$ , let  $M_i = (M_{i,1}, \dots, M_{i,s})$  denote the  $i$ -th row partitioned into blocks  $M_{i,j} \in \mathbb{B}^d$ . Define  $\text{Emb}_{\mathbb{B}}(M) \in \mathbb{R}_{\mathbb{B}}^{m \times s}$  by

$$(\text{Emb}_{\mathbb{B}}(M))_{i,j} := \text{Emb}_{\mathbb{B}}(M_{i,j}), \quad \forall i \in [m], j \in [s].$$

Since the element embedding is bijective, its vector and matrix extensions are also bijective, with inverses applied blockwise.

### **Definition 14 (Flattening and Packed Embedding).**

For  $n \in \mathbb{N}_{\geq 1}$ , consider an arbitrary vector  $z = (z_1, \dots, z_n) \in \mathbb{B}^n$ .

Flattening map: Recall that  $(\beta_1, \dots, \beta_{\delta}) \in \mathbb{B}^{\delta}$  is the fixed ordered  $\mathbb{F}$ -basis of  $\mathbb{B}$ . Uniquely write each  $z_i = \sum_{j=1}^{\delta} z_{i,j} \beta_j$  for  $z_{i,j} \in \mathbb{F}$ .

$$\text{Flat}_{\mathbb{B}}(z) := (z_{1,1}, \dots, z_{1,\delta}, \dots, z_{n,1}, \dots, z_{n,\delta}) \in \mathbb{F}^{\delta n}$$

Thus,  $\text{Flat}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{F}^{\delta n}$  is an  $\mathbb{F}$ -linear bijection.

Packed embedding: If  $d \mid \delta n$ ,

$$\text{Pack}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{R}_{\mathbb{F}}^{\delta n/d}, \quad \text{Pack}_{\mathbb{B}}(z) := \text{Emb}_{\mathbb{F}}(\text{Flat}_{\mathbb{B}}(z))$$

The packed embedding is an  $\mathbb{F}$ -linear bijection because both  $\text{Flat}_{\mathbb{B}}$  and  $\text{Emb}_{\mathbb{F}}$  are  $\mathbb{F}$ -linear bijections.

*Remark 2 (Instantiations).* For norm checks, we instantiate  $\mathbb{B} = \mathbb{F}$  (i.e.,  $\delta = 1$ ) with the trivial basis (1). Then, the map  $\text{Flat}_{\mathbb{F}}$  is the identity map.

$$\text{Flat}_{\mathbb{F}} : \mathbb{F}^{n_{\mathbb{F}}} \rightarrow \mathbb{F}^{n_{\mathbb{F}}}, \quad \text{Pack}_{\mathbb{F}} : \mathbb{F}^{n_{\mathbb{F}}} \rightarrow \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}}.$$

For relations, we instantiate  $\mathbb{B} = \mathbb{L}$  (i.e.,  $\delta = \tau$ ) with any fixed ordered  $\mathbb{F}$ -basis  $(\beta_1, \dots, \beta_{\tau}) \in \mathbb{L}^{\tau}$ . Then,

$$\text{Flat}_{\mathbb{L}} : \mathbb{L}^{n_{\mathbb{L}}} \rightarrow \mathbb{F}^{n_{\mathbb{F}}}, \quad \text{Pack}_{\mathbb{L}} : \mathbb{L}^{n_{\mathbb{L}}} \rightarrow \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}}.$$

Both packed embeddings have codomain  $\mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}}$  because Definition 1 gives  $n_{\mathbb{F}} = \tau n_{\mathbb{L}} = dn_{\mathbb{R}}$ .

For a relation vector  $z \in \mathbb{L}^{n_{\mathbb{L}}}$ , define

$$z^b := \text{Flat}_{\mathbb{L}}(z) \in \mathbb{F}^{n_{\mathbb{F}}}, \quad \mathbf{z} := \text{Pack}_{\mathbb{L}}(z) = \text{Pack}_{\mathbb{F}}(z^b) \in \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}}.$$

Thus, the norm and relation checks use the same packed ring vector  $\mathbf{z}$ , viewed as encoding  $z^b$  over  $\mathbb{F}$  and  $z$  over  $\mathbb{L}$ , respectively.

Prior works [36, 65] use an *inner-product trick* based on a linear transform  $\text{Trans}_{\mathbb{F}} : \mathbb{F}^d \rightarrow \mathbb{F}^d$ , formally defined below, to simulate, for  $a, b \in \mathbb{F}^d$ , a field inner product  $\langle a, b \rangle \in \mathbb{F}$  via a ring multiplication  $\text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(a)) \cdot \text{Emb}_{\mathbb{F}}(b) \in \mathbb{R}_{\mathbb{F}}$  satisfying  $\text{ct}(\text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(a)) \cdot \text{Emb}_{\mathbb{F}}(b)) = \langle a, b \rangle$ . We establish new results that generalize this inner-product trick to arbitrary extension fields  $\mathbb{B}$  of  $\mathbb{F}$  and their corresponding lifted rings  $\mathbb{R}_{\mathbb{B}}$ , with self-contained proofs.

**Lemma 5 (Constant-Term Matrix).** Define the symmetric matrix  $T \in \mathbb{F}^{d \times d}$  where  $T_{i,j} := \text{ct}(X^{i+j})$  for  $i, j \in \{0, \dots, d-1\}$ . Then  $T$  is invertible.

*Proof.* Write  $\mathbb{R}_{\mathbb{F}} = \mathbb{F}[X]/(X^d + f_{d-1}X^{d-1} + \dots + f_1X + f_0)$  with  $f_0 \in \{+1, -1\}$ . Since  $X^d \equiv -(f_{d-1}X^{d-1} + \dots + f_1X + f_0)$ , we have  $\text{ct}(X^d) = -f_0$ , while  $\text{ct}(X^k) = 0$  for  $1 \leq k < d$ . Thus  $T_{i,j} = 0$  for  $0 < i + j < d$ ,  $T_{i,j} = -f_0$  for  $i + j = d$ , and trivially  $T_{0,0} = \text{ct}(X^0 = 1) = 1$ . Let  $N \in \mathbb{F}^{(d-1) \times (d-1)}$  be the submatrix of  $T$  obtained by deleting row 0 and column 0. Then  $\det(T) = T_{0,0} \cdot \det(N) = \det(N)$ . Observe that  $N_{i,j} = 0$  for  $i + j < d-2$  and  $N_{i,j} = \pm 1$  for  $i + j = d-2$ . Recursively expanding along the top row yields  $\det(N) = \pm 1$ ; hence  $\det(T) = \pm 1$  and  $T$  is invertible.  $\square$

**Corollary 1.** Let  $T_{\mathbb{B}} \in \mathbb{B}^{d \times d}$  be the lift of  $T \in \mathbb{F}^{d \times d}$  under the natural inclusion  $\mathbb{F}^{d \times d} \subseteq \mathbb{B}^{d \times d}$ . Then  $T_{\mathbb{B}}$  is symmetric and invertible.

*Proof.* Symmetry is unchanged, and  $\det(T_{\mathbb{B}}) = \det(T) = \pm 1$  remains nonzero in  $\mathbb{B}$ .  $\square$

**Definition 15 (Inner Product Transform).** Recall that  $\beta := (\beta_1, \dots, \beta_{\delta}) \in \mathbb{B}^{\delta}$  is the fixed ordered basis vector for  $\mathbb{B}$  over  $\mathbb{F}$ . Let  $n \in \mathbb{N}_{\geq 1}$  satisfy  $d \mid \delta n$ .

Vector transform: Define  $\text{Trans}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{B}^{\delta n}$  as follows.

Given  $a = (a_1, \dots, a_n) \in \mathbb{B}^n$ , set  $s := \delta n/d$  and partition

$$a \otimes \beta = (\beta_1 a_1, \dots, \beta_{\delta} a_1, \dots, \beta_1 a_n, \dots, \beta_{\delta} a_n) = (u_1, \dots, u_s), \quad u_j \in \mathbb{B}^d.$$

Then set  $\text{Trans}_{\mathbb{B}}(a) := (T_{\mathbb{B}}^{-1} u_1, \dots, T_{\mathbb{B}}^{-1} u_s)$ .

Matrix transform: For  $M \in \mathbb{B}^{m \times n}$  with rows  $M_1, \dots, M_m \in \mathbb{B}^n$ , define the lifted map  $\text{Trans}_{\mathbb{B}} : \mathbb{B}^{m \times n} \rightarrow \mathbb{B}^{m \times \delta n}$  by

$$\text{Trans}_{\mathbb{B}}(M) := \begin{bmatrix} \text{Trans}_{\mathbb{B}}(M_1) \\ \vdots \\ \text{Trans}_{\mathbb{B}}(M_m) \end{bmatrix} \in \mathbb{B}^{m \times \delta n}.$$

Equivalently,  $(\text{Trans}_{\mathbb{B}}(M))_i := \text{Trans}_{\mathbb{B}}(M_i)$  for all  $i \in [m]$ .

*Remark 3 (Efficiency and Sparsity Preservation).* When  $\phi(X)$  is a power-of-two or trinomial cyclotomic, applying  $T_{\mathbb{B}}^{-1}$  to a vector in  $\mathbb{B}^d$  requires only permutations, negations, and a constant number of additions per entry, and therefore costs  $O(d)$  operations over  $\mathbb{B}$ . For  $a \in \mathbb{B}^n$ , the transform  $\text{Trans}_{\mathbb{B}}(a)$  applies  $T_{\mathbb{B}}^{-1}$  to  $\delta n/d$  blocks, requiring  $(\delta n/d) \cdot O(d) = O(\delta n)$  operations. Forming  $a \otimes \beta \in \mathbb{B}^{\delta n}$  also requires  $O(\delta n)$  operations, so the complete transform costs  $O(\delta n)$  operations over  $\mathbb{B}$ .

If  $M \in \mathbb{B}^{m \times n}$  is sparse, then its basis expansion  $M \otimes \beta \in \mathbb{B}^{m \times \delta n}$  has the same density as  $M$ . For these cyclotomics,  $T_{\mathbb{B}}^{-1}$  has constant column sparsity, so the block transform preserves sparsity up to a constant factor. Thus,  $\text{Trans}_{\mathbb{B}}(M)$  is also sparse.

**Lemma 6 (Block Inner Product Trick).** Let  $T_{\mathbb{B}}$  be the lifted constant-term matrix from Corollary 1. For all  $x, y \in \mathbb{B}^d$ ,

$$\text{ct}(\text{Emb}_{\mathbb{B}}(T_{\mathbb{B}}^{-1} x) \cdot \text{Emb}_{\mathbb{B}}(y)) = \langle x, y \rangle \in \mathbb{B}.$$

*Proof.* Let  $w := T_{\mathbb{B}}^{-1} x \in \mathbb{B}^d$ . We index the coordinates of  $w$  and  $y$ , and the rows and columns of  $T_{\mathbb{B}}$ , by  $\{0, \dots, d-1\}$ . By Definition 13,  $\text{Emb}_{\mathbb{B}}(w) \cdot \text{Emb}_{\mathbb{B}}(y) = \sum_{i,j=0}^{d-1} w_i y_j X^{i+j}$ . By Lemma 5 and Corollary 1,  $(T_{\mathbb{B}})_{i,j} = \text{ct}(X^{i+j})$  for all  $i, j \in \{0, \dots, d-1\}$ . Thus, by the linearity of the constant-term map,

$$\text{ct}(\text{Emb}_{\mathbb{B}}(w) \cdot \text{Emb}_{\mathbb{B}}(y)) = \sum_{i,j=0}^{d-1} w_i y_j \text{ct}(X^{i+j}) = w^{\top} T_{\mathbb{B}} y.$$

Since  $T_{\mathbb{B}}$  is symmetric,  $T_{\mathbb{B}}^{-1}$  is also symmetric, and hence

$$w^{\top} T_{\mathbb{B}} y = (T_{\mathbb{B}}^{-1} x)^{\top} T_{\mathbb{B}} y = x^{\top} T_{\mathbb{B}}^{-1} T_{\mathbb{B}} y = x^{\top} y = \langle x, y \rangle.$$

$\square$

**Theorem 8 (Inner Product Trick).** Let  $n \in \mathbb{N}_{\geq 1}$  satisfy  $d \mid \delta n$ , and set  $s := \delta n/d$ . Recall from Definition 13, 14, and 15 the maps

$$\text{Trans}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{B}^{ds}, \quad \text{Emb}_{\mathbb{B}} : \mathbb{B}^{ds} \rightarrow \mathbb{R}_{\mathbb{B}}^s, \quad \text{Pack}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{R}_{\mathbb{F}}^s.$$

For all  $a, b \in \mathbb{B}^n$ ,

$$\text{ct}(\langle \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(a)), \text{Pack}_{\mathbb{B}}(b) \rangle) = \langle a, b \rangle \in \mathbb{B}.$$

The ring inner product is computed in  $\mathbb{R}_{\mathbb{B}}$  using the inclusion  $\mathbb{R}_{\mathbb{F}} \subseteq \mathbb{R}_{\mathbb{B}}$ .

*Proof.* Set  $s := \delta n/d$ . Recall that  $\beta = (\beta_1, \dots, \beta_\delta)$  is the fixed ordered  $\mathbb{F}$ -basis of  $\mathbb{B}$ . For each  $i \in [n]$ , uniquely write  $b_i = \sum_{k=1}^{\delta} b_{i,k} \beta_k$  with  $b_{i,k} \in \mathbb{F}$ . Let

$$\begin{aligned} a \otimes \beta &= (\beta_1 a_1, \dots, \beta_\delta a_1, \dots, \beta_1 a_n, \dots, \beta_\delta a_n) = (u_1, \dots, u_s), \\ \text{Flat}_{\mathbb{B}}(b) &= (b_{1,1}, \dots, b_{1,\delta}, \dots, b_{n,1}, \dots, b_{n,\delta}) = (v_1, \dots, v_s), \end{aligned}$$

where  $u_j \in \mathbb{B}^d$  and  $v_j \in \mathbb{F}^d \subseteq \mathbb{B}^d$  are the corresponding  $d$ -sized blocks for every  $j \in [s]$ . By Definitions 13, 14, and 15,

$$\begin{aligned} \text{Trans}_{\mathbb{B}}(a) &= (T_{\mathbb{B}}^{-1} u_1, \dots, T_{\mathbb{B}}^{-1} u_s), \\ \text{Pack}_{\mathbb{B}}(b) &= (\text{Emb}_{\mathbb{F}}(v_1), \dots, \text{Emb}_{\mathbb{F}}(v_s)). \end{aligned}$$

Since  $v_j \in \mathbb{F}^d \subseteq \mathbb{B}^d$ , we have  $\text{Emb}_{\mathbb{F}}(v_j) = \text{Emb}_{\mathbb{B}}(v_j)$ . Therefore, by the  $\mathbb{B}$ -linearity of  $\text{ct}$  and Lemma 6 applied to each pair  $(u_j, v_j)$ ,

$$\begin{aligned} \text{ct}(\langle \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(a)), \text{Pack}_{\mathbb{B}}(b) \rangle) &= \sum_{j=1}^s \text{ct}(\text{Emb}_{\mathbb{B}}(T_{\mathbb{B}}^{-1} u_j) \cdot \text{Emb}_{\mathbb{B}}(v_j)) \\ &= \sum_{j=1}^s \langle u_j, v_j \rangle = \sum_{i=1}^n \sum_{k=1}^{\delta} (\beta_k a_i) b_{i,k} = \sum_{i=1}^n a_i \left( \sum_{k=1}^{\delta} b_{i,k} \beta_k \right) = \langle a, b \rangle. \end{aligned}$$

□

**Theorem 9 (Matrix-Vector Product Transform).** *Let  $n \in \mathbb{N}_{\geq 1}$  satisfy  $d \mid \delta n$ , and set  $s := \delta n/d$ . Recall from Definitions 13, 14, and 15 the maps*

$$\text{Trans}_{\mathbb{B}} : \mathbb{B}^{m \times n} \rightarrow \mathbb{B}^{m \times ds}, \quad \text{Emb}_{\mathbb{B}} : \mathbb{B}^{m \times ds} \rightarrow \mathbb{R}_{\mathbb{B}}^{m \times s}, \quad \text{Pack}_{\mathbb{B}} : \mathbb{B}^n \rightarrow \mathbb{R}_{\mathbb{F}}^s.$$

For every  $M \in \mathbb{B}^{m \times n}$  and  $z \in \mathbb{B}^n$ ,

$$\text{ct}(\text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \text{Pack}_{\mathbb{B}}(z)) = Mz \in \mathbb{B}^m.$$

The matrix-vector product inside  $\text{ct}$  is computed in  $\mathbb{R}_{\mathbb{B}}$  using the inclusion  $\mathbb{R}_{\mathbb{F}} \subseteq \mathbb{R}_{\mathbb{B}}$ .

*Proof.* Let  $M_1, \dots, M_m \in \mathbb{B}^n$  be the rows of  $M$ . By Definitions 13 and 15,

$$(\text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)))_i = \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M_i))$$

for every  $i \in [m]$ . Hence, by Theorem 8, the  $i$ -th coordinate of the left-hand side is

$$\text{ct}(\langle \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M_i)), \text{Pack}_{\mathbb{B}}(z) \rangle) = \langle M_i, z \rangle = (Mz)_i.$$

Since this holds for every  $i \in [m]$ ,  $\text{ct}(\text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \text{Pack}_{\mathbb{B}}(z)) = Mz$ . □

**Theorem 10 (Matrix-Vector Product Evaluation).** *Let  $\mathbb{B} \in \{\mathbb{F}, \mathbb{L}\}$  be a degree- $\delta$  extension of  $\mathbb{F}$ . Let  $m, n \in \mathbb{N}_{\geq 1}$ , with  $m$  a power of two and  $d \mid \delta n$ , and consider  $M \in \mathbb{B}^{m \times n}$ ,  $z \in \mathbb{B}^n$ , and  $r \in \mathbb{K}^{\log m}$ . Define*

$$h := \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \text{Pack}_{\mathbb{B}}(z) \in \mathbb{R}_{\mathbb{B}}^m \subseteq \mathbb{R}_{\mathbb{A}}^m, \quad y := \tilde{h}(r) \in \mathbb{R}_{\mathbb{A}}.$$

Then, for every  $\ell \in [d]$ ,  $\text{cf}(y)_{\ell} = \widetilde{\text{cf}(h)_{\ell}}(r) \in \mathbb{A}$ . In particular,  $\text{ct}(y) = \widetilde{Mz}(r) \in \mathbb{A}$ .

*Proof.* Consider an arbitrary ring element  $a := \sum_{\ell=1}^d a_{\ell} X^{\ell-1} \in \mathbb{R}_{\mathbb{B}}$  and scalar  $c \in \mathbb{K}$ . Multiplying  $a$  by  $c$  scales each of its coefficients by  $c$ :  $ca = \sum_{\ell=1}^d (ca_{\ell}) X^{\ell-1} \in \mathbb{R}_{\mathbb{A}}$ . Hence, by linearity of the coefficient map,  $\text{cf}(ca)_{\ell} = c \cdot \text{cf}(a)_{\ell} \in \mathbb{A}$  for every  $\ell \in [d]$ . By definition,  $y = \tilde{h}(r) = \sum_{\vec{x} \in \{0,1\}^{\log m}} \text{eq}(r, \vec{x}) \cdot h_{\vec{x}} \in \mathbb{R}_{\mathbb{A}}$ . Since each  $\text{eq}(r, \vec{x})$  lies in  $\mathbb{K}$ , applying the observation above termwise and using linearity of the coefficient map gives  $\text{cf}(y)_{\ell} = \widetilde{\text{cf}(h)_{\ell}}(r) \in \mathbb{A}$  for every  $\ell \in [d]$ . Taking  $\ell = 1$  gives  $\text{ct}(y) = \widetilde{\text{ct}(h)}(r) \in \mathbb{A}$ . By Theorem 9,  $\text{ct}(h) = Mz \in \mathbb{B}^m$ , and therefore  $\text{ct}(y) = \widetilde{Mz}(r) \in \mathbb{A}$ . □

**Theorem 11 (Evaluation Homomorphism).** Recall from Definition 1 the fields  $\mathbb{F}, \mathbb{L}, \mathbb{K}, \mathbb{A}$  and rings  $R_{\mathbb{F}}, R_{\mathbb{L}}, R_{\mathbb{K}}, R_{\mathbb{A}}$  satisfying

$$\mathbb{F} \subseteq \mathbb{L} \subseteq \mathbb{A}, \quad \mathbb{F} \subseteq \mathbb{K} \subseteq \mathbb{A}, \quad R_{\mathbb{F}} \subseteq R_{\mathbb{L}} \subseteq R_{\mathbb{A}}, \quad R_{\mathbb{F}} \subseteq R_{\mathbb{K}} \subseteq R_{\mathbb{A}}.$$

Let  $\mathbb{B} \in \{\mathbb{F}, \mathbb{L}\}$  be a degree- $\delta$  extension of  $\mathbb{F}$ , and let  $m, n, s, s_{\text{in}} \in \mathbb{N}_{\geq 1}$  satisfy  $\delta n = ds$ , with  $m$  a power of two. Consider  $M \in \mathbb{B}^{m \times n}$ , vectors  $z_1, \dots, z_{\ell} \in \mathbb{B}^n$ , scalars  $\rho_1, \dots, \rho_{\ell} \in R_{\mathbb{F}}$ , and  $r \in \mathbb{K}^{\log m}$ . Recall the maps

$$\text{Trans}_{\mathbb{B}} : \mathbb{B}^{m \times n} \rightarrow \mathbb{B}^{m \times ds}, \quad \text{Emb}_{\mathbb{B}} : \mathbb{B}^{m \times ds} \rightarrow R_{\mathbb{B}}^{m \times s}, \quad \text{Pack}_{\mathbb{B}} : \mathbb{B}^n \rightarrow R_{\mathbb{F}}^s.$$

Let  $\mathcal{L} : R_{\mathbb{F}}^s \rightarrow \mathbb{C}$  and  $\mathcal{L}_{\text{in}} : R_{\mathbb{F}}^s \rightarrow R_{\mathbb{F}}^{s_{\text{in}}}$  be  $R_{\mathbb{F}}$ -module homomorphisms. Define

$$\begin{aligned} \forall i \in [\ell], \quad & \mathbf{z}_i := \text{Pack}_{\mathbb{B}}(z_i) \in R_{\mathbb{F}}^s, & c_i := \mathcal{L}(\mathbf{z}_i) \in \mathbb{C}, \\ & h_i := \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \mathbf{z}_i \in R_{\mathbb{B}}^m, & y_i := \widetilde{h_i}(r) \in R_{\mathbb{A}}, \\ & \mathbf{x}_i := \mathcal{L}_{\text{in}}(\mathbf{z}_i) \in R_{\mathbb{F}}^{s_{\text{in}}}, \\ & \mathbf{z} := \sum_{i=1}^{\ell} \rho_i \mathbf{z}_i \in R_{\mathbb{F}}^s, & c := \sum_{i=1}^{\ell} \rho_i c_i \in \mathbb{C}, \\ & h := \sum_{i=1}^{\ell} \rho_i h_i \in R_{\mathbb{B}}^m, & y := \sum_{i=1}^{\ell} \rho_i y_i \in R_{\mathbb{A}}, \\ & \mathbf{x} := \sum_{i=1}^{\ell} \rho_i \mathbf{x}_i \in R_{\mathbb{F}}^{s_{\text{in}}}. \end{aligned}$$

Then, we must have

$$\begin{aligned} h &= \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \mathbf{z} \in R_{\mathbb{B}}^m & c &= \mathcal{L}(\mathbf{z}) \in \mathbb{C}, \\ \mathbf{x} &= \mathcal{L}_{\text{in}}(\mathbf{z}) \in R_{\mathbb{F}}^{s_{\text{in}}} & y &= \widetilde{h}(r) \in R_{\mathbb{A}} \\ & \quad \wedge \\ & \forall i \in [\ell], \quad \text{ct}(y_i) &= \widetilde{M z_i}(r) \in \mathbb{A}, \\ & \text{ct}(y) &= \widetilde{M z}(r) \in \mathbb{A} \end{aligned}$$

for  $z := \text{Pack}_{\mathbb{B}}^{-1}(\mathbf{z}) \in \mathbb{B}^n$ . All evaluations are computed in  $R_{\mathbb{A}}$ .

*Proof.* Since  $\mathcal{L}$  and  $\mathcal{L}_{\text{in}}$  are  $R_{\mathbb{F}}$ -module homomorphisms,

$$\begin{aligned} c &= \sum_{i=1}^{\ell} \rho_i \mathcal{L}(\mathbf{z}_i) = \mathcal{L}\left(\sum_{i=1}^{\ell} \rho_i \mathbf{z}_i\right) = \mathcal{L}(\mathbf{z}), \\ \mathbf{x} &= \sum_{i=1}^{\ell} \rho_i \mathcal{L}_{\text{in}}(\mathbf{z}_i) = \mathcal{L}_{\text{in}}\left(\sum_{i=1}^{\ell} \rho_i \mathbf{z}_i\right) = \mathcal{L}_{\text{in}}(\mathbf{z}). \end{aligned}$$

By distributivity,

$$h = \sum_{i=1}^{\ell} \rho_i h_i = \sum_{i=1}^{\ell} \rho_i (\text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \mathbf{z}_i) = \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \sum_{i=1}^{\ell} \rho_i \mathbf{z}_i = \text{Emb}_{\mathbb{B}}(\text{Trans}_{\mathbb{B}}(M)) \cdot \mathbf{z}.$$

Multilinear evaluation at  $r$  is  $R_{\mathbb{A}}$ -linear and hence  $R_{\mathbb{F}}$ -linear. Therefore,

$$y = \sum_{i=1}^{\ell} \rho_i y_i = \sum_{i=1}^{\ell} \rho_i \widetilde{h_i}(r) = \widetilde{\sum_{i=1}^{\ell} \rho_i h_i}(r) = \widetilde{h}(r).$$

By Theorem 10,  $\text{ct}(y_i) = \widetilde{M z_i}(r)$  for every  $i \in [\ell]$ . Finally, for  $z := \text{Pack}_{\mathbb{B}}^{-1}(\mathbf{z})$ , we have  $\mathbf{z} = \text{Pack}_{\mathbb{B}}(z)$ , so another application of Theorem 10 gives  $\text{ct}(y) = \widetilde{M z}(r)$ .  $\square$

*Remark 4 (Reductions).* In our reductions, we instantiate  $(\mathbb{B}, \delta, n, \beta)$  with

$$(\mathbb{F}, 1, n_{\mathbb{F}}, (1)) \quad \text{and} \quad (\mathbb{L}, \tau, n_{\mathbb{L}}, (\beta_1, \dots, \beta_{\tau})).$$

From Definition 1,  $n_{\mathbb{F}} = \tau n_{\mathbb{L}} = dn_{\mathbb{R}}$ . Remark 2 gives two variants of the witness:  $z \in \mathbb{L}^{n_{\mathbb{L}}}$  as the native relation vector for CCS, and  $z^b := \text{Flat}_{\mathbb{L}}(z) \in \mathbb{F}^{n_{\mathbb{F}}}$  as the flattened base field vector required for norm checking. These variants share the same packed representation as a ring vector, namely

$$\mathbf{z} := \text{Pack}_{\mathbb{L}}(z) = \text{Pack}_{\mathbb{F}}(z^b) \in R_{\mathbb{F}}^{n_{\mathbb{R}}}.$$

Looking ahead, we must check that  $\mathbf{z}$  has low norm and that  $\mathbf{z}$  satisfies the CCS relation over matrices  $M_j \in \mathbb{L}^{m \times n_L}$ . To do so, we will need to fold evaluations of a padded version of  $z^b$  and of  $M_j \mathbf{z}$ . Assume  $n_F \leq m$  and  $m$  is a power of two. Define

$$\text{Pad} := \begin{bmatrix} I_{n_F} \\ 0_{(m-n_F) \times n_F} \end{bmatrix} \in \mathbb{F}^{m \times n_F}, \quad z^{b, \text{pad}} := \text{Pad} z^b = \begin{bmatrix} z^b \\ 0_{m-n_F} \end{bmatrix} \in \mathbb{F}^m.$$

Since padding only appends zero coordinates,  $\|z^{b, \text{pad}}\|_\infty = \|z^b\|_\infty$ . For  $r \in \mathbb{K}^{\log m}$ , define

$$\begin{aligned} h &:= \text{Emb}_{\mathbb{F}}(\text{Trans}_{\mathbb{F}}(\text{Pad})) \cdot \mathbf{z} \in \mathbb{R}_{\mathbb{F}}^m, & y &:= \tilde{h}(r) \in \mathbb{R}_{\mathbb{K}}, \\ \forall j \in [t], \quad h_j &:= \text{Emb}_{\mathbb{L}}(\text{Trans}_{\mathbb{L}}(M_j)) \cdot \mathbf{z} \in \mathbb{R}_{\mathbb{L}}^m, & y_j &:= \tilde{h}_j(r) \in \mathbb{R}_{\mathbb{A}}. \end{aligned}$$

By Theorem 10,

$$\text{ct}(y) = \widetilde{z^{b, \text{pad}}}(r) \in \mathbb{K}, \quad \forall j \in [t], \quad \text{ct}(y_j) = \widetilde{M_j z}(r) \in \mathbb{A}.$$

Thus, Theorem 11 lets us fold both kinds of evaluation claims consistently with folds of the same committed ring vector  $\mathbf{z}$ .

