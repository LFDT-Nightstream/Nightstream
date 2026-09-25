## 5 Embedding products with evaluation homomorphism

Here, we define a bijective coefficient embedding from  $\mathbb{F}^d$  into the ring  $\mathbb{R}_\mathbb{F}$ .

**Definition 7 (Coefficient Embedding).**

Element embedding: Consider a vector  $v \in \mathbb{F}^d$ . We define  $\mathbf{v} \in \mathbb{R}_\mathbb{F}$  (in bold font) to be the ring element whose coefficient vector is  $v$ , i.e.  $\operatorname{cf}(\mathbf{v}) = v$ .

Vector embedding: Recall that we define  $n_\mathbb{F} = d \cdot n_R$ . Hence, for a field vector  $z \in \mathbb{F}^{n_\mathbb{F}}$ , we have a natural partition into  $d$ -sized sub-vectors  $z = [z_1, \dots, z_{n_R}]$ . We define the ring vector  $\mathbf{z} := (z_1, \dots, z_{n_R}) \in \mathbb{R}_\mathbb{F}^{n_R}$  to be the vector of ring elements, which are the embeddings of the  $n_R = n_\mathbb{F}/d$  field sub-vectors.

Matrix embedding: For a matrix  $M \in \mathbb{F}^{m \times n_\mathbb{F}}$  with rows  $M_1, \dots, M_m \in \mathbb{F}^{n_\mathbb{F}}$ , we define  $\mathbf{M} := [\mathbf{M}_1, \dots, \mathbf{M}_m] \in \mathbb{R}_\mathbb{F}^{m \times n_R}$ , which is the vertical concatenation of all the embedded rows.

Inverse embedding: Similarly, given a ring vector  $\mathbf{v} \in \mathbb{R}_\mathbb{F}^{n_R}$  or ring matrix  $\mathbf{M} \in \mathbb{R}_\mathbb{F}^{m \times n_R}$ , we define the field vector  $v \in \mathbb{F}^{n_\mathbb{F}}$  or field matrix  $M \in \mathbb{F}^{m \times n_\mathbb{F}}$  as the inverse of previously defined coefficient embeddings.

**Theorem 3 (Inner Product Transform [36, 64]).** *There exists a linear transform  $\overline{\cdot} : \mathbb{F}^d \to \mathbb{F}^d$  such that for all  $a, b \in \mathbb{F}^d$ , we have the constant term*

$$\operatorname{ct}(\overline{\mathbf{a}} \cdot \mathbf{b}) = \langle a, b \rangle$$

where  $\overline{\mathbf{a}}$  denotes applying the transform to  $a$  and embedding  $\overline a$  into the ring, while  $\mathbf b$  is the ordinary coefficient embedding of  $b$ . The transform is applied to exactly one factor.

Here, we define an extension of the inner product transform  $\overline{\cdot} : \mathbb{F}^d \to \mathbb{F}^d$  (Theorem 3) to vectors and matrices.

**Definition 8 (Lifting the Transform).**

Vector Transform: Consider a vector  $v \in \mathbb{F}^{n_{\mathbb{F}}}$ , which we partition into  $d$ -sized sub-vectors  $[v_1, \dots, v_{n_R}]$ . We define  $\bar{\cdot} : \mathbb{F}^{n_{\mathbb{F}}} \to \mathbb{F}^{n_{\mathbb{F}}}$  block-wise by  $\bar{v} := [\bar{v}_1, \dots, \bar{v}_{n_R}]$ . When used in a ring product, each transformed block is identified with its coefficient embedding in  $\mathbb R_{\mathbb F}$ .

Matrix Transform: Consider a matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$  with rows  $M_1, \dots, M_m \in \mathbb{F}^{n_{\mathbb{F}}}$ . We define  $\bar{\cdot} : \mathbb{F}^{m \times n_{\mathbb{F}}} \to \mathbb{F}^{m \times n_{\mathbb{F}}}$  to be  $\bar{M} := [\bar{M}_1, \dots, \bar{M}_m] \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ .

*Remark 1 (Efficiency and Sparsity).* When the cyclotomic polynomial  $\phi(X)$  is a power-of-two cyclotomic or a trinomial cyclotomic, the block transform can be implemented with a fixed pattern of permutations and additions, and hence in  $O(n_{\mathbb F})$  time. More precisely, if each column of the  $d\times d$  block transform has at most  $s$  nonzero entries, then  $\operatorname{nnz}(\bar M)\le s\operatorname{nnz}(M)$ . Thus the specific bounded-support transform preserves sparsity up to this factor; linearity alone would not imply sparsity preservation.

**Theorem 4 (Matrix-Vector Product Transform).** Consider an arbitrary matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$  and vector  $z \in \mathbb{F}^{n_{\mathbb{F}}}$ . Let  $\mathbf z\in\mathbb R_{\mathbb F}^{n_R}$  be the coefficient embedding of  $z$ , and view each transformed  $d$ -column block of  $\bar M$  as a ring element. Then

$$Mz=\operatorname{ct}(\bar M\mathbf z)\in\mathbb F^m.$$

*Proof.* For brevity, we defer the proof to Appendix D.1.  $\square$

*Remark 2 (Matrix-vector Product Evaluation).* Consider an arbitrary vector  $z \in \mathbb{F}^{n_{\mathbb{F}}}$ , its coefficient embedding  $\mathbf z\in\mathbb R_{\mathbb F}^{n_R}$ , a matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ , and a multilinear evaluation point  $r \in \mathbb{K}^{\log m}$ . Define

$$\mathbf y:=\widehat{\bar M\mathbf z}(r)\in\mathbb R_{\mathbb K}.$$

Multiplication by an extension-field scalar acts coefficient-wise. Hence, for every  $\ell\in[d]$ ,

$$\operatorname{cf}(\mathbf y)_\ell
=\widetilde{\operatorname{cf}(\bar M\mathbf z)_\ell}(r)\in\mathbb K.$$

By Theorem 4,  $\operatorname{ct}(\bar M\mathbf z)=Mz$ . Therefore,

$$\operatorname{ct}(\mathbf y)=\widetilde{Mz}(r).$$

**Theorem 5 (Evaluation Homomorphism).** Consider an arbitrary matrix  $M \in \mathbb{F}^{m \times n_{\mathbb{F}}}$ , vectors  $z_1, \dots, z_{\ell} \in \mathbb{F}^{n_{\mathbb{F}}}$  with coefficient embeddings  $\mathbf z_1,\dots,\mathbf z_\ell\in\mathbb R_{\mathbb F}^{n_R}$ , scalars  $\rho_1, \dots, \rho_{\ell} \in \mathbb{R}_{\mathbb{F}}$ , and evaluation point  $r \in \mathbb{K}^{\log m}$ . Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{C}$  and  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{R}_{\mathbb{F}}^{n_{R, \text{in}}}$  be arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphisms. For all  $i \in [\ell]$ , define

$$c_i := \mathcal{L}(\mathbf z_i) \in \mathbb{C}, \qquad \mathbf{x}_i := \mathcal{L}_{\text{in}}(\mathbf z_i) \in \mathbb{R}_{\mathbb{F}}^{n_{R, \text{in}}}, \qquad \mathbf y_i := \widehat{\bar M\mathbf z_i}(r) \in \mathbb{R}_{\mathbb{K}}.$$

Additionally, define

$$\begin{aligned} c &:= \sum_{i \in [\ell]} \rho_i c_i \in \mathbb{C}, & \mathbf{x} &:= \sum_{i \in [\ell]} \rho_i \mathbf{x}_i \in \mathbb{R}_{\mathbb{F}}^{n_{R, \text{in}}}, \\ \mathbf{z} &:= \sum_{i \in [\ell]} \rho_i \mathbf z_i \in \mathbb{R}_{\mathbb{F}}^{n_R}, & \mathbf y &:= \sum_{i \in [\ell]} \rho_i \mathbf y_i \in \mathbb{R}_{\mathbb{K}}. \end{aligned}$$

We must have

$$c=\mathcal L(\mathbf z),\qquad \mathbf x=\mathcal L_{\mathrm{in}}(\mathbf z),\qquad \mathbf y=\widehat{\bar M\mathbf z}(r).$$

Let  $z\in\mathbb F^{n_{\mathbb F}}$  be the inverse coefficient embedding of  $\mathbf z$ . Additionally, for all  $i\in[\ell]$ ,

$$\operatorname{ct}(\mathbf y_i)=\widetilde{Mz_i}(r),
\qquad
\operatorname{ct}(\mathbf y)=\widetilde{Mz}(r).$$

*Proof.* For brevity, we defer the proof to Appendix D.2.  $\square$
