## 5 Embedding products with evaluation homomorphism

Here, we define a bijective embedding from the field  $\mathbb{F}$  into the ring  $\mathbb{R}_\mathbb{F}$ .

**Definition 7 (Coefficient Embedding).**

Element embedding: Consider a vector  $v \in \mathbb{F}^d$ . We define  $\mathbf{v} \in \mathbb{R}_\mathbb{F}$  (in bold font) to be the ring element whose coefficient vector is  $v$ , i.e.  $\mathbf{cf}(\mathbf{v}) = v$ .

Vector embedding: Recall that we define  $n_\mathbb{F} = d \cdot n_R$ . Hence, for a field vector  $z \in \mathbb{F}^{n_\mathbb{F}}$ , we have a natural partition into  $d$ -sized sub-vectors  $z = [z_1, \dots, z_{n_R}]$ . We define the ring vector  $\mathbf{z} := (z_1, \dots, z_{n_R}) \in \mathbb{R}_\mathbb{F}^{n_R}$  to be the vector of ring elements, which are the embeddings of the  $n_R = n_\mathbb{F}/d$  field sub-vectors.

Matrix embedding: For a matrix  $M \in \mathbb{F}^{m \times n_\mathbb{F}}$  with rows  $M_1, \dots, M_m \in \mathbb{F}^{n_\mathbb{F}}$ , we define  $\mathbf{M} := [\mathbf{M}_1, \dots, \mathbf{M}_m] \in \mathbb{R}_\mathbb{F}^{m \times n_R}$ , which is the vertical concatenation of all the embedded rows.

Inverse embedding: Similarly, given a ring vector  $\mathbf{v} \in \mathbb{R}_\mathbb{F}^{n_R}$  or ring matrix  $\mathbf{M} \in \mathbb{R}_\mathbb{F}^{m \times n_R}$ , we define the field vector  $v \in \mathbb{F}^{n_\mathbb{F}}$  or field matrix  $M \in \mathbb{F}^{m \times n_\mathbb{F}}$  as the inverse of previously defined coefficient embeddings.

**Theorem 3 (Inner Product Transform [36, 64]).** *There exists a linear transform  $\overline{\cdot} : \mathbb{F}^d \to \mathbb{F}^d$  such that for all  $a, b \in \mathbb{F}^d$ , we have the constant term*

$$\mathbf{ct}(\overline{\mathbf{a}} \cdot \overline{\mathbf{b}}) = \langle a, b \rangle$$

where  $\overline{\mathbf{a}}$  denotes applying the transform to  $a$  and embedding  $\overline{a}$  into the ring.

Here, we define an extension of the inner product transform  $\overline{\cdot} : \mathbb{F}^d \to \mathbb{F}^d$  (Theorem 3) to vectors and matrices.

**Definition 8 (Lifting the Transform).**

Vector Transform: Consider a vector  $v \in \mathbb{F}^{n_{\mathbb{F}}}$ , we can partition  $v$  into  $d$ -sized sub-vectors  $[v_1, \dots, v_{n_{\mathbb{R}}}]$ . We define  $\bar{\cdot} : \mathbb{F}^{n_{\mathbb{F}}} \to \mathbb{F}^{n_{\mathbb{F}}}$  to be  $\bar{v} := [\bar{v}_1, \dots, \bar{v}_{n_{\mathbb{R}}}] \in \mathbb{F}^{n_{\mathbb{F}}}$ .

Matrix Transform: Consider a matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$  with rows  $M_1, \dots, M_m \in \mathbb{F}^{n_{\mathbb{F}}}$ . We define  $\bar{\cdot} : \mathbb{F}^{m \times n_{\mathbb{F}}} \to \mathbb{F}^{m \times n_{\mathbb{F}}}$  to be  $\bar{M} := [\bar{M}_1, \dots, \bar{M}_m] \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ .

*Remark 1 (Efficiency and Sparsity Preservation).* When the cyclotomic polynomial  $\phi(X)$  is a power-of-two cyclotomic or a trinomial cyclotomic, the transform  $\bar{\cdot} : \mathbb{F}^{n_{\mathbb{F}}} \to \mathbb{F}^{n_{\mathbb{F}}}$  essentially only involves permuting and adding entries of the input vector, and hence can be computed in  $O(n_{\mathbb{F}})$  time. Since the  $\bar{\cdot}$  is linear, if the original matrix  $M$  is sparse, then the transformed matrix  $\bar{M}$  is also sparse.

**Theorem 4 (Matrix-Vector Product Transform).** Consider an arbitrary matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$  and vector  $z \in \mathbb{F}^{n_{\mathbb{F}}}$ . The matrix-vector product  $Mz \in \mathbb{F}^m$  is equal to the constant terms of the matrix-vector product  $\bar{M}z \in \mathbb{R}_{\mathbb{F}}^m$ , when viewing each ring element as a polynomial. More succinctly,  $Mz = \text{ct}(\bar{M}z)$ .

*Proof.* For brevity, we defer the proof to Appendix D.1.  $\square$

*Remark 2 (Matrix-vector Product Evaluation).* Consider an arbitrary vector  $z \in \mathbb{F}^{n_{\mathbb{F}}}$ , matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ , and multilinear evaluation point  $r \in \mathbb{K}^{\log m}$ . Define the evaluation  $y := \widehat{\bar{M}z_i}(r) = \langle \bar{M}z_i, \hat{r} \rangle \in \mathbb{R}_{\mathbb{K}}$ . Observe that multiplying a ring element in  $\mathbb{R}_{\mathbb{F}}$  with an extension field element in  $\mathbb{K}$  scales each coefficient of the ring element by the extension field element. Hence, by Definition 2, we must have that for all  $\ell \in [d]$ ,  $\text{cf}(y)_{\ell} = \widehat{\text{cf}(\bar{M}z_i)}_{\ell}(r) \in \mathbb{K}$  (i.e. the  $\ell$ -th coefficient of  $y$  is equal to the multilinear evaluation of the  $\ell$ -th coefficient vector of  $\bar{M}z_i$  at point  $r$ ). Since  $\text{ct}(y) = \text{cf}(y)_1$  and  $\text{ct}(\bar{M}z_i) = \widehat{\text{cf}(\bar{M}z_i)}_1$ , by Theorem 4, we can observe that the constant term  $\text{ct}(y) = \bar{M}z(r) \in \mathbb{K}$  is exactly the multilinear evaluation of the field vector  $Mz$  at point  $r$ .

**Theorem 5 (Evaluation Homomorphism).** Consider an arbitrary matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ , vectors  $z_1, \dots, z_{\ell} \in \mathbb{F}^{n_{\mathbb{F}}}$ , scalars  $\rho_1, \dots, \rho_{\ell} \in \mathbb{R}_{\mathbb{F}}$ , and evaluation point  $r \in \mathbb{K}^{\log m}$ . Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{C}$ ,  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}, \text{in}}}$  be arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphisms. For all  $i \in [\ell]$ , define

$$c_i := \mathcal{L}(z_i) \in \mathbb{C} \quad \mathbf{x}_i := \mathcal{L}_{\text{in}}(z_i) \in \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}, \text{in}}} \quad y_i := \widehat{\bar{M}z_i}(r) \in \mathbb{R}_{\mathbb{K}}$$

Additionally, define

$$\begin{aligned} c &:= \sum_{i \in [\ell]} \rho_i c_i \in \mathbb{C}, & \mathbf{x} &:= \sum_{i \in [\ell]} \rho_i \mathbf{x}_i \in \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}, \text{in}}}, \\ \mathbf{z} &:= \sum_{i \in [\ell]} \rho_i z_i \in \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}}, & y &:= \sum_{i \in [\ell]} \rho_i y_i \in \mathbb{R}_{\mathbb{K}} \end{aligned}$$

We must have that  $c = \mathcal{L}(\mathbf{z})$  and  $y = \widehat{\bar{M}\mathbf{z}}(r)$ . Additionally, for all  $i \in [\ell]$ ,  $\text{ct}(y_i) = \bar{M}z_i(r)$  and  $\text{ct}(y) = \bar{M}\mathbf{z}(r)$ .

*Proof.* For brevity, we defer the proof to Appendix D.2.  $\square$
