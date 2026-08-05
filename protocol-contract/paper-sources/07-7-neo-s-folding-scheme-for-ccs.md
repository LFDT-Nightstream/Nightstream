## 7 Neo's folding scheme for CCS

### 7.1 Relations

**Definition 11 (Structure).** We define a **structure** as a collection of elements

$$s := \left\{ \left\{ M_j \in \mathbb{F}^{m \times n_{\mathbb{F}}} \right\}_{j \in [t]}, f \in \mathbb{F}[X_1, \dots, X_t] \right\},$$

which consists of matrices and a constraint polynomial. Define

$$D_f := \deg_{\mathrm{tot}}(f).$$

An admissible structure must satisfy

$$D_f \le u.$$

The quantity  $D_f$ , rather than an individual-degree annotation on  $f$ , is used in the sum-check degree bound.

**Definition 12 (Norm-bounded CCS).** Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{C}$  be an arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism. Let  $s$  be a structure as defined in Definition 11. We define the **norm-bounded CCS relation**,  $\text{CCS}(b, \mathcal{L})$ , as follows:

$$\left\{ \begin{array}{l} (s; (c \in \mathbb{C}, x \in \mathbb{F}^{n_{\mathbb{F}, \text{in}}}); w \in \mathbb{F}^{n_{\mathbb{F}} - n_{\mathbb{F}, \text{in}}}) : \\ \quad \text{For } z := [x, w], \\ \quad c = \mathcal{L}(z) \wedge \|z\|_{\infty} < b \wedge \\ \quad f(\overline{M_1 z}, \dots, \overline{M_t z}) \in \mathbb{ZS}_{\log m} \end{array} \right\}$$

**Definition 13 (Norm-bounded CCS Evaluation Relation).** Let  $s$  be a structure as defined in Definition 11. Let  $\mathcal{L} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{C}$  be an arbitrary  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism. Define  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}, \text{in}}}$  to be the trivial  $\mathbb{R}_{\mathbb{F}}$ -module

homomorphism that projects the first  $n_{R,\text{in}}$  indices. We define the **norm-bounded CCS evaluation relation**,  $\text{CE}(b, \mathcal{L})$ , as follows:

$$\left\{ \left( \mathbf{s}; \left( \begin{array}{c} c \in \mathbb{C}, \\ x \in \mathbb{F}^{n_{\mathbb{F},\text{in}}}, \\ r \in \mathbb{K}^{\log m}, \\ \{y_j \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]} \end{array} \right); z \in \mathbb{F}^{n_{\mathbb{F}}} \right) : \begin{array}{l} c = \mathcal{L}(z) \wedge \mathbf{x} = \mathcal{L}_{\text{in}}(z) \\ \wedge \|z\|_{\infty} < b \wedge \\ \forall j \in [t], y_j = \widehat{\bar{M}_j z}(r) \end{array} \right\}$$

As in Definition 7, a field vector supplied to  $\mathcal L$  or  $\mathcal L_{\mathrm{in}}$  is identified with its coefficient embedding in  $\mathbb R_{\mathbb F}^{n_{\mathbb R}}$ . We use  $\widehat{\bar{M}_j z}(r)$  for the multilinear evaluation at  $r$  of the transformed matrix-vector product from Definition 8 and Remark 2.

### 7.2 A folding scheme for CCS via interactive reductions

#### Definition 14 (Global Reduction Parameters).

Here, we define the global parameters used in our reductions:

- Define  $\mathbb{F}, \mathbb{K}, d, \mathbb{R}_{\mathbb{F}}, \mathbb{R}_{\mathbb{K}}, m, n_{\mathbb{F}}, n_{\mathbb{R}}, n_{\mathbb{F},\text{in}}, n_{\mathbb{R},\text{in}}, u, t, k, K, b, B, B_{\mathrm{amb}}$  as in Definition 1. For a structure  $s$ , define  $D_f := \deg_{\mathrm{tot}}(f)$  as in Definition 11.
- Let  $\mathcal{C} \subseteq \mathbb{R}_{\mathbb{F}}$  be a strong sampling set (Definition 17) with expansion factor  $T$  such that  $(K + k)T(b - 1) < B$  and  $1/|\mathcal{C}| = \text{negl}(\lambda)$ .
- Let  $\text{com} := (\text{Setup}, \text{Commit})$  be a ring commitment scheme (Definition 4), which is homomorphic and  $(2B, \mathcal{C})$ -relaxed binding. For  $\text{pp} \leftarrow \text{Setup}(1^\lambda, m)$ , define  $\mathcal{L} := \text{Commit}(\text{pp}, \cdot) : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{C}$ , which is a  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism.
- Let  $\mathcal{L}_{\text{in}} : \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R}}} \to \mathbb{R}_{\mathbb{F}}^{n_{\mathbb{R},\text{in}}}$  be the trivial  $\mathbb{R}_{\mathbb{F}}$ -module homomorphism that projects the first  $n_{\mathbb{R},\text{in}}$  columns.
- Let  $\mathbf{s}$  denote a structure as defined in Definition 11.

In Appendix B, we instantiate these parameters with concrete values.

### 7.3 Interactive reduction for CCS – $\Pi_{\text{CCS}}$

*Overview.* The interactive reduction  $\Pi_{\text{CCS}}$  checks that the  $K$  incoming CCS instances (Definition 12) indeed satisfy the required CCS constraints, the  $k$  evaluation claims (Definition 13), from the prior folding step, hold for point  $r$ , and checks that the norms of all of the witness vectors (all  $K + k$  of them) involved are less than  $b$ . To do so,  $\Pi_{\text{CCS}}$  relies on the classic sum-check protocol (Definition 6). The approach is inspired by similar reductions from [14, 55].  $\Pi_{\text{CCS}}$  defines helper polynomials that, when used in the sum-check protocol, will perform the previously specified checks.  $F(\vec{X})$  encodes the CCS constraints (all  $K$  of them).  $\text{NC}(\vec{X})$  encodes the norm constraints (all  $K + k$  of them).  $\text{Eval}(\vec{X})$  encodes the evaluation claims (all  $k$  of them) from the prior step. Finally,  $Q(\vec{X})$  is defined such that if its sum over the boolean hypercube  $\{0, 1\}^{\log(m)}$  equals the absolute carried target  $T_{\mathrm{abs}}$ , then all the respective checks hold.

#### CCS reduction $\Pi_{\text{CCS}}$

**Parameters:** Refer to Definition 14. Without loss of generality, assume that  $m = n_{\mathbb{F}}$  and  $n_{\mathbb{F}}$  is a power of two and that  $M_1 = I_{n_{\mathbb{F}}}$  is the identity matrix.

**Input**  $\in \text{CCS}(b, \mathcal{L})^K \times \text{CE}(b, \mathcal{L})^k$

(s;  $(c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\text{F,in}}}); w_i \in \mathbb{F}^{n_{\text{F}} - n_{\text{F,in}}})_{i=1}^K$ ,  
(s;  $c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\text{F,in}}}, r \in \mathbb{K}^{\log m}, \{y_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\text{F}}})_{i=K+1}^{K+k}$

**Output**  $\in \text{CE}(b, \mathcal{L})^{K+k}$

(s;  $c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\text{F,in}}}, r' \in \mathbb{K}^{\log m}, \{y'_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\text{F}}})_{i \in [K+k]}$

**Setup**  $\mathcal{G}(1^\lambda, n_R) \to \text{pp}$ : Output  $\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)$ .

**Encoder**  $\mathcal{K}(\text{pp}, s) \to (\text{pk}, \text{vk})$ : Output  $(\text{pp}, s), \perp$ .

**Reduction**  $\langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1) \to (u_2; w_2)$ :

1.  $\mathcal{V}$ : Send challenges  $\alpha \xleftarrow{\$} \mathbb{K}^{\log m}$  and  $\gamma \xleftarrow{\$} \mathbb{K}$  to  $\mathcal{P}$ .
2.  $\mathcal{V} \leftrightarrow \mathcal{P}$ : For all  $i \in [K]$ , define  $z_i := [x_i, w_i]$ . Define  $\vec{X} := (X_1, \dots, X_{\log m})$  and the scalar norm polynomial

$$P_b(Z) := \prod_{a=-(b-1)}^{b-1} \left(Z-\iota_q(a)\right) \in \mathbb{F}[Z].$$

Because  $b \le B < q/2$ , its roots are distinct and, for every scalar  $a \in \mathbb{F}$ ,

$$P_b(a)=0 \quad\Longleftrightarrow\quad |\operatorname{ctr}_q(a)|<b.$$

The polynomial has degree  $\deg P_b=2b-1$ . After multiplication by the equality selector, the norm contribution to  $Q$  has individual degree at most  $2b$ . In particular,  $P_2(Z)=(Z+1)Z(Z-1)$  is cubic and its equality-gated contribution is quartic. Define

$$F(\vec{X}) := \sum_{i=1}^K \gamma^{i-1} \cdot f(\overline{M_1 z_i}, \dots, \overline{M_t z_i}) \in \mathbb{K}[\vec{X}]$$

$$\text{NC}(\vec{X}) := \sum_{i=1}^{K+k} \gamma^{i-1} \cdot P_b(\tilde{z}_i(\vec{X})) \in \mathbb{K}[\vec{X}]$$

$$\text{Eval}(\vec{X}) := \text{eq}(\vec{X}, r) \cdot \sum_{i=K+1}^{K+k} \sum_{j=1}^t \sum_{\ell=1}^d \gamma^{I(i,j,\ell)} \cdot \overline{\text{cf}(\overline{M_j z_i})_\ell}(\vec{X}) \in \mathbb{K}[\vec{X}]$$

where  $I(i, j, \ell) = (i - (K + 1)) + k(j - 1) + kt(\ell - 1)$  and  $\overline{\text{cf}(\overline{M_j z_i})_\ell}$  is the multi-linear extension of the  $\ell$ -th coefficient vector of  $\overline{M_j z_i}$  (Definition 2). As  $i-(K+1)\in\{0,\ldots,k-1\}$ ,  $j-1\in\{0,\ldots,t-1\}$ , and  $\ell-1\in\{0,\ldots,d-1\}$ , the map

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

3.  $\mathcal{P}$ : Send  $\forall i \in [K+k], \forall j \in [t], y'_{i,j} \leftarrow \overline{M_j z_i}(r') \in \mathbb{R}_{\mathbb{K}}$ .
4.  $\mathcal{V}$ : Derive the claimed intermediate evaluations (Remark 2),

$$F := \sum_{i=1}^K \gamma^{i-1} \cdot f(\text{ct}(y'_{i,1}), \dots, \text{ct}(y'_{i,t})) \in \mathbb{K}$$

$$N := \sum_{i=1}^{K+k} \gamma^{i-1} \cdot P_b(\text{ct}(y'_{i,1})) \in \mathbb{K}$$

$$E := \text{eq}(r', r) \sum_{i=K+1}^{K+k} \sum_{j=1}^{t} \sum_{\ell=1}^{d} \gamma^{I(i,j,\ell)} \cdot \text{cf}(y'_{i,j})_{\ell} \in \mathbb{K}$$

Check the evaluation claim  $v \stackrel{?}{=} Q(r')$  as follows,

$$v \stackrel{?}{=} \text{eq}(r', \alpha) \cdot (F + \gamma^K \cdot N) + \gamma^{2K+k} \cdot E$$

5. Output  $(s; c_i, x_i, r', \{y'_{i,j}\}_{j \in [t]}; z_i)_{i \in [K+k]}$

*Remark 3.* By choosing  $M_1 = \mathbb{I}_{n_F}$ , we simplify our notation, because folding  $\widehat{M_1 z} = \widehat{\mathbb{I}_{n_F} z}$  evaluations is equivalent to folding  $\tilde{z}$  evaluations.

**Lemma 3 (Π<sub>CCS</sub> is strong).** *The interactive reduction  $\Pi_{\text{CCS}} : \text{CCS}(b, \mathcal{L})^K \times \text{CE}(b, \mathcal{L})^k \to \text{CE}(b, \mathcal{L})^{K+k}$  ( $\text{CE}(B_{\mathrm{amb}}, \mathcal{L})^{K+k}$ ) is **strong** (Definition 10) for the function  $\phi$ , which projects commitments  $(c_i)_{i \in [K+k]}$  from the instance.*

*Proof.* For brevity, we defer the proof to Appendix D.4.  $\square$

### 7.4 Random linear combination reduction – Π<sub>RLC</sub>

The interactive reduction Π<sub>RLC</sub> does exactly as the name suggests. Given  $K+k$  input CCS evaluation claims of norm  $b$ , it outputs a single CCS evaluation claim of larger norm  $B$ , which is a random linear combination of the input claims using challenges from a strong sampling set  $\mathcal{C}$  (Definition 17).

#### Random linear combination reduction Π<sub>RLC</sub>

**Parameters:** Refer to Definition 14.

**Input**  $\in \text{CE}(b, \mathcal{L})^{K+k}$

$(s; c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{F,\text{in}}}, r \in \mathbb{K}^{\log m}, \{y_{i,j} \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_F})_{i \in [K+k]}$

**Output**  $\in \text{CE}(B, \mathcal{L})$

$(s; c \in \mathbb{C}, x \in \mathbb{F}^{n_{F,\text{in}}}, r \in \mathbb{K}^{\log m}, \{y_j \in \mathbb{R}_{\mathbb{K}}\}_{j \in [t]}; z \in \mathbb{F}^{n_F})$

**Setup**  $\mathcal{G}(1^\lambda, n_R) \to \text{pp}$ : Output  $\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)$ .

**Encoder**  $\mathcal{K}(\text{pp}, s) \to (\text{pk}, \text{vk})$ : Output  $((\text{pp}, s), \perp)$ .

**Reduction**  $\langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1) \to (u_2; w_2)$ :

1.  $\mathcal{V}$ : Sample  $\rho_1, \dots, \rho_{K+k} \xleftarrow{\$} \mathcal{C}$  and compute:

$$c \leftarrow \sum_{i \in [K+k]} \rho_i c_i, \quad x \leftarrow \sum_{i \in [K+k]} \rho_i x_i, \quad \forall j \in [t], \quad y_j \leftarrow \sum_{i \in [K+k]} \rho_i y_{i,j}$$

Send  $\rho_1, \dots, \rho_{K+k}$  to  $\mathcal{P}$ .

2.  $\mathcal{P}$ : Compute  $z \leftarrow \sum_{i \in [K+k]} \rho_i z_i$ .

3. Output  $(s; c, x, r, \{y_j\}_{j \in [t]}; z)$ .

**Lemma 4 ( $\Pi_{RLC}$  is weak).** *The interactive reduction  $\Pi_{RLC} : \text{CE}(b, \mathcal{L})^{K+k} (\text{CE}(B_{\mathrm{amb}}, \mathcal{L})^{K+k}) \to \text{CE}(B, \mathcal{L})$  is **weak** (Definition 9) for the function  $\phi$ , which projects commitments  $(c_i)_{i \in [K+k]}$  from the instance.*

*Proof.* For brevity, we defer the proof to Appendix D.5.  $\square$

### 7.5 Decomposition reduction – $\Pi_{DEC}$

Inspired by folklore techniques [12, 15, 71], our final reduction aims to reduce the norm of claims from  $B = b^k$  to  $b$ , which will allow us to continually fold CCS claims without increasing the norm of the openings  $(z_i)_i$  to the commitments.

| Decomposition reduction $\Pi_{DEC}$                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                          |
|--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------|
| <p><b>Parameters:</b> Refer to Definition 14.</p> <p><b>Input</b> <math>\in \text{CE}(B, \mathcal{L})</math><br/> <math>(s; c \in \mathbb{C}, x \in \mathbb{F}^{n_{\mathbb{F}, \text{in}}}, r \in \mathbb{K}^{\log m}, \{y_j \in R_{\mathbb{K}}\}_{j \in [t]}; z \in \mathbb{F}^{n_{\mathbb{F}}})</math></p> <p><b>Output</b> <math>\in \text{CE}(b, \mathcal{L})^k</math><br/> <math>(s; c_i \in \mathbb{C}, x_i \in \mathbb{F}^{n_{\mathbb{F}, \text{in}}}, r \in \mathbb{K}^{\log m}, \{y_{i,j} \in R_{\mathbb{K}}\}_{j \in [t]}; z_i \in \mathbb{F}^{n_{\mathbb{F}}})_{i \in [k]}</math></p> <hr/> <p><b>Setup</b> <math>\mathcal{G}(1^\lambda, n_R) \to \text{pp}</math>: Output <math>\text{pp} \leftarrow \text{Setup}(1^\lambda, n_R)</math>.</p> <p><b>Encoder</b> <math>\mathcal{K}(\text{pp}, s) \to (\text{pk}, \text{vk})</math>: Output <math>((\text{pp}, s), \perp)</math>.</p> <p><b>Reduction</b> <math>\langle \mathcal{P}, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, w_1) \to (u_2; w_2)</math>:</p> <ol style="list-style-type: none"> <li>1. <math>\mathcal{P}</math>: Compute <math>(c_i, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [k]}</math> as follows,<br/> <math display="block">(z_1, \dots, z_k) \leftarrow \text{split}_b(z), \quad c_i \leftarrow \mathcal{L}(z_i), \quad \forall j \in [t], \quad y_{i,j} \leftarrow \widehat{\bar{M}_j z_i}(r)</math> Send <math>(c_i, \{y_{i,j}\}_{j \in [t]})_{i \in [k]}</math> to <math>\mathcal{V}</math>.</li> <li>2. <math>\mathcal{V}</math>: Compute <math>(x_1, \dots, x_k) \leftarrow \text{split}_b(x)</math>. Check the following equations,<br/> <math display="block">c \stackrel{?}{=} \sum_{i \in [k]} b^{i-1} \cdot c_i \quad \text{and} \quad \forall j \in [t], \quad y_j \stackrel{?}{=} \sum_{i \in [k]} b^{i-1} \cdot y_{i,j}</math> where the norm-bound <math>b</math> is treated as a field element.</li> <li>3. Output <math>(s; c_i, x_i, r, \{y_{i,j}\}_{j \in [t]}; z_i)_{i \in [k]}</math></li> </ol> |

**Theorem 7.**  $\Pi_{DEC} : \text{CE}(B, \mathcal{L}) \to \text{CE}(b, \mathcal{L})^k$  is a *reduction of knowledge* (Definition 5).

*Proof.* For brevity, we defer the proof to Appendix D.6.  $\square$
