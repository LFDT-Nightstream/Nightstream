## 7 Neo's folding scheme for CCS

### 7.1 Relations

**Definition 11 (Structure).** We define a **structure** as a collection of elements

$$s := \left\{ \left\{ M_j \in \mathbb{F}^{m \times n_{\mathbb{F}}} \right\}_{j \in [t]}, f \in \mathbb{F}[X_1, \dots, X_t] \right\},$$

which consists of matrices and a constraint polynomial. Define

$$D_f := \deg_{\mathrm{tot}}(f).$$

An admissible structure must satisfy

$$D_f \le u, \qquad m\text{ is a power of two}, \qquad n_{\mathbb F}\le m,$$

and its first matrix must be the canonical zero-padding injection

$$M_1:=\begin{bmatrix}I_{n_{\mathbb F}}\\0_{(m-n_{\mathbb F})\times n_{\mathbb F}}\end{bmatrix}\in\mathbb F^{m\times n_{\mathbb F}}.$$

Thus  $M_1z=(z,0^{m-n_{\mathbb F}})$  and  $\|M_1z\|_\infty=\|z\|_\infty$ . This normalization is part of the relation domain. The quantity  $D_f$ , rather than an individual-degree annotation on  $f$ , is used in the sum-check degree bound.

**Definition 12 (Norm-bounded CCS).** Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{C}$  be an arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism. For every field vector  $v$  whose length is a multiple of  $d$ , write  $\mathbf v$  for its coefficient embedding from Definition 7. Let  $s$  be a structure as defined in Definition 11. We define the **norm-bounded CCS relation**,  $\text{CCS}(b, \mathcal{L})$ , as follows:

$$\left\{ \begin{array}{l} (s; (c \in \mathbb{C}, x \in \mathbb{F}^{n_{\mathbb{F}, \text{in}}}); w \in \mathbb{F}^{n_{\mathbb{F}} - n_{\mathbb{F}, \text{in}}}) : \\ \quad \text{For } z := [x, w], \\ \quad c = \mathcal{L}(\mathbf z) \wedge \|z\|_{\infty} < b \wedge \\ \quad f(\widetilde{M_1 z}, \dots, \widetilde{M_t z}) \in \mathbb{ZS}_{\log m} \end{array} \right\}$$

**Definition 13 (Norm-bounded CCS Evaluation Relation).** Let  $s$  be a structure as defined in Definition 11. Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{C}$  be an arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism. Define  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{R}_{\mathbb{F}}^{n_{R, \text{in}}}$  to be the trivial  $\mathbb{R}_{\mathbb{F}}$ -module

homomorphism that projects the first  $n_{R,\text{in}}$  indices. We define the **norm-bounded CCS evaluation relation**,  $\text{CE}(b, \mathcal{L})$ , as follows:

$$\left\{ \left( \mathbf{s}; \left( \begin{array}{c} c \in \mathbb{C}, \\ x \in \mathbb{F}^{n_{\mathbb{F},\text{in}}}, \\ r \in \mathbb{K}^{\log m}, \\ \{y_j \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]} \end{array} \right); z \in \mathbb{F}^{n_{\mathbb{F}}} \right) : \begin{array}{l} c = \mathcal{L}(\mathbf z) \wedge \mathbf{x} = \mathcal{L}_{\text{in}}(\mathbf z) \\ \wedge \|z\|_{\infty} < b \wedge \\ \forall j \in [t], y_j = \widehat{\bar{M}_j\mathbf z}(r) \end{array} \right\}$$

**Definition 13a (Batched CCS Evaluation Relation).** For  $N\in\mathbb N_{\ge1}$ , define  $\operatorname{BatchCE}_N(b,\mathcal L)$  to have instances

$$\left(\left(c_i\in\mathbb C,\ x_i\in\mathbb F^{n_{\mathbb F,\mathrm{in}}},\ \{y_{i,j}\in\mathbb R_{\mathbb K}\}_{j\in[t]}\right)_{i\in[N]},\ r\in\mathbb K^{\log m}\right)$$

and witnesses  $(z_i\in\mathbb F^{n_{\mathbb F}})_{i\in[N]}$ , such that for every  $i\in[N]$ ,

$$\left(\mathbf s;\left(c_i,x_i,r,\{y_{i,j}\}_{j\in[t]}\right);z_i\right)\in\operatorname{CE}(b,\mathcal L).$$

The evaluation point  $r$  occurs once in the batched instance and is shared by every component. Protocol displays sometimes repeat the same symbol  $r$  inside each component tuple for readability; every such occurrence denotes this one outer point, not a separate point. Under the canonical identification of one-component tuples,  $\operatorname{BatchCE}_1(b,\mathcal L)=\operatorname{CE}(b,\mathcal L)$ . An ordinary product  $\operatorname{CE}(b,\mathcal L)^N$  continues to mean the component-wise product from Appendix C and does not impose this shared-point condition.

### 7.2 A folding scheme for CCS via interactive reductions

#### Definition 14 (Global Reduction Parameters).

Here, we define the global parameters used in our reductions:

- Define  $\mathbb{F}, \mathbb{K}, d, \mathbb{R}_{\mathbb{F}}, \mathbb{R}_{\mathbb{K}}, m, n_{\mathbb{F}}, n_R, n_{\mathbb{F},\text{in}}, n_{R,\text{in}}, u, t, k, K, b, B, B_{\mathrm{amb}}$  as in Definition 1. For a structure  $s$ , define  $D_f := \deg_{\mathrm{tot}}(f)$  as in Definition 11.
- Let  $\mathcal{C} \subseteq \mathbb{R}_{\mathbb{F}}$  be a strong sampling set (Definition 17) with expansion factor  $T$  such that  $(K + k)T(b - 1) < B$  and  $1/|\mathcal{C}| = \text{negl}(\lambda)$ .
- Let  $\text{com} := (\text{Setup}, \text{Commit})$  be a ring commitment scheme (Definition 4), which is homomorphic and  $(2B, \mathcal{C})$ -relaxed binding. For  $\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)$ , define  $\mathcal{L} := \text{Commit}(\text{pp}, \cdot) : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{C}$ , which is a  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism.
- Let  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_R} \to \mathbb{R}_{\mathbb{F}}^{n_{R,\text{in}}}$  be the trivial  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism that projects the first  $n_{R,\text{in}}$  columns.
- Let  $\mathbf{s}$  denote a structure as defined in Definition 11.

Appendix B gives fixed parameter sets and finite error estimates. It does not instantiate the asymptotic-family convention from Section 4.

### 7.3 Interactive reduction for CCS – $\Pi_{\text{CCS}}$

*Overview.* The interactive reduction  $\Pi_{\text{CCS}}$  checks that the  $K$  incoming CCS instances (Definition 12) satisfy the required CCS constraints, that the  $k$  evaluation claims from the prior folding step form a shared-point batch (Definitions 13 and 13a) at point  $r$ , and that the norms of all  $K + k$  witness vectors are less than  $b$ . To do so,  $\Pi_{\text{CCS}}$  relies on the classic sum-check protocol (Definition 6). The approach is inspired by similar reductions from [14, 55].  $\Pi_{\text{CCS}}$  defines helper polynomials that, when used in the sum-check protocol, perform the previously specified checks.  $F(\vec{X})$  encodes the CCS constraints (all  $K$  of them).  $\text{NC}(\vec{X})$  encodes the norm constraints (all  $K + k$  of them).  $\text{Eval}(\vec{X})$  encodes the evaluation claims (all  $k$  of them) from the prior step. Finally,  $Q(\vec{X})$  is defined such that if its sum over the Boolean hypercube  $\{0, 1\}^{\log(m)}$  equals the absolute carried target  $T_{\mathrm{abs}}$ , then all the respective checks hold.

#### CCS reduction $\Pi_{\text{CCS}}$

**Parameters:** Refer to Definition 14 and the admissibility conditions in Definition 11. A norm-bounded CCS structure with  $m'$  rows, matrices  $(M'_1,\dots,M'_{t'})$ , and polynomial  $f'$  can meet the first-matrix convention by setting  $t=t'+1$ , prepending the canonical  $M_1$ , and replacing  $f'$  by  $f(X_1,\dots,X_t):=f'(X_2,\dots,X_t)$ . Choose a power-of-two row length  $m\ge\max(m',n_{\mathbb F})$ . Repeating existing constraint rows is the generic safe padding rule. If  $f'(0,\dots,0)=0$ , one may instead append zero rows to every original matrix; each added row then evaluates to zero, so this specialization also preserves and reflects the constraint zero set. Nightstream uses this zero-row specialization. Both rules preserve the bounded vector  $z=[x,w]$ . This normalization does not claim a compiler from unrestricted CCS, whose public and private coordinates can have arbitrary magnitude. It increases the matrix count by one, and row padding can increase  $m$ ; all degree, soundness, and cost expressions use these normalized values. The sum-check below always uses the single  $\log m$ -variable domain. It does not require  $d\mid m$  or  $m=n_{\mathbb F}$ .

**Input**  $\in \text{CCS}(b, \mathcal{L})^K \times \operatorname{BatchCE}_k(b, \mathcal{L})$

(s;  $(c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\mathbb F,\text{in}}}); w_i \in \mathbb{F}^{n_{\mathbb F} - n_{\mathbb F,\text{in}}})_{i=1}^K$ ,
(s;  $c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\mathbb F,\text{in}}}, r \in \mathbb{K}^{\log m}, \{y_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\mathbb F}})_{i=K+1}^{K+k}$

**Output**  $\in \operatorname{BatchCE}_{K+k}(b, \mathcal{L})$

(s;  $c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\mathbb F,\text{in}}}, r' \in \mathbb{K}^{\log m}, \{y'_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\mathbb F}})_{i \in [K+k]}$

**Setup**  $\mathcal{G}(1^\lambda, n_R) \to \text{pp}$ : Output  $\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)$ .

**Encoder**  $\mathcal{K}(\text{pp}, s) \to (\text{pk}, \text{vk})$ : Output  $(\text{pp}, s), \perp$ .

**Reduction**  $\langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1) \to (u_2; w_2)$ :

1.  $\mathcal{V}$ : Send challenges  $\alpha \xleftarrow{\$} \mathbb{K}^{\log m}$  and  $\gamma \xleftarrow{\$} \mathbb{K}$  to  $\mathcal{P}$ .
2.  $\mathcal{V} \leftrightarrow \mathcal{P}$ : For all  $i \in [K]$ , define  $z_i := [x_i, w_i]$ . For every  $i\in[K+k]$ , write  $\mathbf z_i\in\mathbb R_{\mathbb F}^{n_R}$  for the coefficient embedding of  $z_i$ . Define  $\vec{X} := (X_1, \dots, X_{\log m})$  and the scalar norm polynomial

$$P_b(Z) := \prod_{a=-(b-1)}^{b-1} \left(Z-\iota_q(a)\right) \in \mathbb{F}[Z].$$

Because  $b \le B < q/2$ , its roots are distinct and, for every scalar  $a \in \mathbb{F}$ ,

$$P_b(a)=0 \quad\Longleftrightarrow\quad |\operatorname{ctr}_q(a)|<b.$$

The polynomial has degree  $\deg P_b=2b-1$ . After multiplication by the equality selector, the norm contribution to  $Q$  has individual degree at most  $2b$ . In particular,  $P_2(Z)=(Z+1)Z(Z-1)$  is cubic and its equality-gated contribution is quartic. Define

$$F(\vec{X}) := \sum_{i=1}^K \gamma^{i-1} \cdot f(\widetilde{M_1 z_i}(\vec X), \dots, \widetilde{M_t z_i}(\vec X)) \in \mathbb{K}[\vec{X}]$$

$$\text{NC}(\vec{X}) := \sum_{i=1}^{K+k} \gamma^{i-1} \cdot P_b(\widetilde{M_1z_i}(\vec{X})) \in \mathbb{K}[\vec{X}]$$

$$\text{Eval}(\vec{X}) := \text{eq}(\vec{X}, r) \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d \gamma^{I(i,j,\ell)} \cdot \widetilde{\text{cf}(\bar M_j\mathbf z_i)_\ell}(\vec{X}) \in \mathbb{K}[\vec{X}]$$

where  $I(i, j, \ell) = (i - (K + 1)) + k(j - 1) + kt(\ell - 1)$  and  $\widetilde{\text{cf}(\bar M_j\mathbf z_i)_\ell}$  is the multilinear extension of the  $\ell$ -th coefficient vector of  $\bar M_j\mathbf z_i$  (Definition 2). As  $i-(K+1)\in\{0,\ldots,k-1\}$ ,  $j-1\in\{0,\ldots,t-1\}$ , and  $\ell-1\in\{0,\ldots,d-1\}$ , the map

$$I:\{K+1,\ldots,K+k\}\times[t]\times[d]\longrightarrow\{0,\ldots,ktd-1\}$$

is the standard mixed-radix bijection.
Define

$$Q(\vec{X}) := \text{eq}(\vec{X}, \alpha) \cdot (F(\vec{X}) + \gamma^K \cdot \text{NC}(\vec{X})) + \gamma^{2K+k} \cdot \text{Eval}(\vec{X}) \in \mathbb{K}[\vec{X}].$$

Its maximum individual degree is bounded by

$$\deg_{\mathrm{ind}} Q \le D_Q := \max(D_f+1,\;2b,\;2).$$

Define the unshifted local encoding of the carried evaluation target by

$$T_{\mathrm{local}} := \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d \gamma^{I(i,j,\ell)} \cdot \text{cf}(y_{i,j})_\ell \in \mathbb{K},$$

and define the absolute target used by the joint sum-check as

$$T_{\mathrm{abs}} := \gamma^{2K+k}T_{\mathrm{local}} = \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d \gamma^{2K+k+I(i,j,\ell)} \cdot \text{cf}(y_{i,j})_\ell.$$

Thus the three coefficient blocks use the disjoint exponent ranges  $0,\dots,K-1$  for fresh constraints,  $K,\dots,2K+k-1$  for norm constraints, and  $2K+k,\dots,2K+k+ktd-1$  for carried constraints. The value  $T_{\mathrm{local}}$  is not the target of the combined sum-check.

Perform **SumCheck** ( $T_{\mathrm{abs}}$ ;  $Q$ ) (Definition 6), which reduces the claim that

$$T_{\mathrm{abs}} = \sum_{\vec{x} \in \{0,1\}^{\log m}} Q(\vec{x})$$

to a new evaluation claim  $v \stackrel{?}{=} Q(r')$  for new evaluation point  $r' \in \mathbb{K}^{\log m}$ .

3.  $\mathcal{P}$ : Send  $\forall i \in [K+k], \forall j \in [t], y'_{i,j} \leftarrow \widehat{\bar M_j\mathbf z_i}(r') \in \mathbb{R}_{\mathbb{K}}$ .
4.  $\mathcal{V}$ : Derive the claimed intermediate evaluations (Remark 2),

$$F := \sum_{i=1}^K \gamma^{i-1} \cdot f(\text{ct}(y'_{i,1}), \dots, \text{ct}(y'_{i,t})) \in \mathbb{K}$$

$$N := \sum_{i=1}^{K+k} \gamma^{i-1} \cdot P_b(\text{ct}(y'_{i,1})) \in \mathbb{K}$$

$$E := \text{eq}(r', r) \sum_{i=K+1}^{K+k} \sum_{j=1}^{t} \sum_{\ell=1}^{d} \gamma^{I(i,j,\ell)} \cdot \text{cf}(y'_{i,j})_{\ell} \in \mathbb{K}$$

Check the evaluation claim  $v \stackrel{?}{=} Q(r')$  as follows,

$$v \stackrel{?}{=} \text{eq}(r', \alpha) \cdot (F + \gamma^K \cdot N) + \gamma^{2K+k} \cdot E$$

5. Output  $(s; c_i, x_i, r', \{y'_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]}$

*Remark 3.* The padding injection  $M_1$  preserves every witness coordinate and adds only zeros. Thus  $\widetilde{M_1z}$  is the multilinear extension of the zero-padded witness, and  $\|M_1z\|_\infty=\|z\|_\infty$ . The verifier can obtain  $\widetilde{M_1z}(r')$  as  $\operatorname{ct}(\widehat{\bar M_1\mathbf z}(r'))$  by Remark 2.

**Lemma 3 (Π<sub>CCS</sub> is strong).** *The interactive reduction  $\Pi_{\text{CCS}} : \text{CCS}(b, \mathcal{L})^K \times \operatorname{BatchCE}_k(b, \mathcal{L}) \to \operatorname{BatchCE}_{K+k}(b, \mathcal{L})$  is **strong** (Definition 10), with ambient output relation  $\operatorname{BatchCE}_{K+k}(B_{\mathrm{amb}}, \mathcal{L})$ , for the function  $\phi$  that projects commitments  $(c_i)_{i \in [K+k]}$  from the instance.*

*Proof.* For brevity, we defer the proof to Appendix D.4.  $\square$

### 7.4 Random linear combination reduction – Π<sub>RLC</sub>

The interactive reduction Π<sub>RLC</sub> does exactly as the name suggests. Given  $K+k$  input CCS evaluation claims of norm  $b$ , it outputs a single CCS evaluation claim of larger norm  $B$ , which is a random linear combination of the input claims using challenges from a strong sampling set  $\mathcal{C}$  (Definition 17).

#### Random linear combination reduction Π<sub>RLC</sub>

**Parameters:** Refer to Definition 14.

**Input**  $\in \operatorname{BatchCE}_{K+k}(b, \mathcal{L})$

$(s; c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\mathbb F,\text{in}}}, r \in \mathbb{K}^{\log m}, \{y_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\mathbb F}})_{i \in [K+k]}$

**Output**  $\in \text{CE}(B, \mathcal{L})$

$(s; c \in \mathbb{C}, x \in \mathbb{F}^{n_{\mathbb F,\text{in}}}, r \in \mathbb{K}^{\log m}, \{y_j \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z \in \mathbb{F}^{n_{\mathbb F}})$

**Setup**  $\mathcal{G}(1^\lambda, n_R) \to \text{pp}$ : Output  $\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)$ .

**Encoder**  $\mathcal{K}(\text{pp}, s) \to (\text{pk}, \text{vk})$ : Output  $((\text{pp}, s), \perp)$ .

**Reduction**  $\langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1) \to (u_2; w_2)$ :

1.  $\mathcal{V}$ : Sample  $\rho_1, \dots, \rho_{K+k} \xleftarrow{\$} \mathcal{C}$ . For every  $i\in[K+k]$ , let  $\mathbf x_i\in\mathbb R_{\mathbb F}^{n_{R,\mathrm{in}}}$  be the coefficient embedding of  $x_i$ . Compute

$$c \leftarrow \sum_{i \in [K+k]} \rho_i c_i,
\qquad
\mathbf x \leftarrow \sum_{i \in [K+k]} \rho_i\mathbf x_i,
\qquad
\forall j \in [t],\quad y_j \leftarrow \sum_{i \in [K+k]} \rho_i y_{i,j},$$

and let  $x\in\mathbb F^{n_{\mathbb F,\mathrm{in}}}$  be the inverse coefficient embedding of  $\mathbf x$ .

Send  $\rho_1, \dots, \rho_{K+k}$  to  $\mathcal{P}$ .

2.  $\mathcal{P}$ : Let  $\mathbf z_i$  be the coefficient embedding of  $z_i$  for every  $i\in[K+k]$ . Compute  $\mathbf z \leftarrow \sum_{i \in [K+k]} \rho_i\mathbf z_i$ , and let  $z\in\mathbb F^{n_{\mathbb F}}$  be its inverse coefficient embedding.

3. Output  $(s; c, x, r, \{y_j\}_{j \in [t]}; z)$ .

**Lemma 4 ( $\Pi_{RLC}$  is weak).** *The interactive reduction  $\Pi_{RLC} : \operatorname{BatchCE}_{K+k}(b, \mathcal{L}) \to \text{CE}(B, \mathcal{L})$  is **weak** (Definition 9), with ambient input relation  $\operatorname{BatchCE}_{K+k}(B_{\mathrm{amb}}, \mathcal{L})$ , for the function  $\phi$  that projects commitments  $(c_i)_{i \in [K+k]}$  from the instance.*

*Proof.* For brevity, we defer the proof to Appendix D.5.  $\square$

### 7.5 Decomposition reduction – $\Pi_{DEC}$

Inspired by folklore techniques [12, 15, 71], our final reduction aims to reduce the norm of claims from  $B = b^k$  to  $b$ , which will allow us to continually fold CCS claims without increasing the norm of the openings  $(z_i)_i$  to the commitments.

#### Decomposition reduction $\Pi_{DEC}$

**Parameters:** Refer to Definition 14.

**Input**  $\in\operatorname{CE}(B,\mathcal L)$ :

$$(s;c\in\mathbb C,x\in\mathbb F^{n_{\mathbb F,\mathrm{in}}},r\in\mathbb K^{\log m},\{y_j\in\mathbb R_{\mathbb K}\}_{j\in[t]};z\in\mathbb F^{n_{\mathbb F}}).$$

**Output**  $\in\operatorname{BatchCE}_k(b,\mathcal L)$ :

$$\left(s;c_i\in\mathbb C,x_i\in\mathbb F^{n_{\mathbb F,\mathrm{in}}},r\in\mathbb K^{\log m},\{y_{i,j}\in\mathbb R_{\mathbb K}\}_{j\in[t]};z_i\in\mathbb F^{n_{\mathbb F}}\right)_{i\in[k]}.$$

**Setup**  $\mathcal G(1^\lambda,n_R)\to\mathrm{pp}$ : Output  $\mathrm{pp}\leftarrow\operatorname{Setup}(1^\lambda,n_R)$ .

**Encoder**  $\mathcal K(\mathrm{pp},s)\to(\mathrm{pk},\mathrm{vk})$ : Output  $((\mathrm{pp},s),\perp)$ .

**Reduction**  $\langle\mathcal P,\mathcal V\rangle((\mathrm{pk},\mathrm{vk}),u_1,w_1)\to(u_2;w_2)$ :

1.  $\mathcal P$ : Compute  $(z_1,\dots,z_k)\leftarrow\operatorname{split}_{b,k}(z)$ . If the result is  $\perp$ , output  $\perp$ . Otherwise, let  $\mathbf z_i$  be the coefficient embedding of  $z_i$  and compute

   $$c_i\leftarrow\mathcal L(\mathbf z_i),\qquad
   y_{i,j}\leftarrow\widehat{\bar M_j\mathbf z_i}(r)
   \quad\text{for every }i\in[k],j\in[t].$$

   Send  $(c_i,\{y_{i,j}\}_{j\in[t]})_{i\in[k]}$  to  $\mathcal V$ .
2.  $\mathcal V$ : Compute  $(x_1,\dots,x_k)\leftarrow\operatorname{split}_{b,k}(x)$ . If the result is  $\perp$ , reject. Otherwise, check

   $$c\stackrel?=\sum_{i\in[k]}b^{i-1}c_i,
   \qquad
   y_j\stackrel?=\sum_{i\in[k]}b^{i-1}y_{i,j}
   \quad\text{for every }j\in[t],$$

   where  $b$  is treated as a field element.
3. Output  $(s;c_i,x_i,r,\{y_{i,j}\}_{j\in[t]};z_i)_{i\in[k]}$ .

**Theorem 7.**  $\Pi_{DEC} : \text{CE}(B, \mathcal{L}) \to \operatorname{BatchCE}_k(b, \mathcal{L})$  is a *reduction of knowledge* (Definition 5).

*Proof.* For brevity, we defer the proof to Appendix D.6.  $\square$
