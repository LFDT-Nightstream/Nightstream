## C Additional Background

*Relation Products* For relations  $\mathcal{R}_1$  and  $\mathcal{R}_2$  over public parameter, structure, instance, and witness pairs we define the relation  $\mathcal{R}_1 \times \mathcal{R}_2$  such that  $(\mathbf{pp}, \mathbf{s}, (u_1, u_2), (w_1, w_2)) \in \mathcal{R}_1 \times \mathcal{R}_2$  if and only if  $(\mathbf{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}_1$ , and  $(\mathbf{pp}, \mathbf{s}, u_2, w_2) \in \mathcal{R}_2$ . We let  $\mathcal{R}^n$  denote  $\mathcal{R} \times \dots \times \mathcal{R}$  for  $n$  times.

**Lemma 5 (Schwartz-Zippel [76]).** *let  $g : \mathbb{F}^\ell \to \mathbb{F}$  be an  $\ell$ -variate polynomial of total degree at most  $d$ . Then, on any finite set  $S \subseteq \mathbb{F}$ ,*

$$\Pr_{x \leftarrow S^\ell} [g(x) = 0] \le d/|S|.$$

**Lemma 6.** *Let  $Q \in \mathbb{F}[X_1, \dots, X_\ell]$  be an arbitrary multivariate polynomial. Define multivariate polynomial  $Q'(\vec{X}, \vec{Z}) := \text{eq}(\vec{X}, \vec{Z}) \cdot Q(\vec{X})$ .*

$$0 = \sum_{\vec{x} \in \{0,1\}^{\ell}} Q'(\vec{x}, \vec{Z}) \quad \text{if and only if} \quad Q(\vec{X}) \in \mathbb{ZS}_\ell,$$

where the equality on the left is an identity in  $\mathbb F[\vec Z]$ .

**Definition 15 (Module Homomorphism).** *Modules are a generalization of vector spaces for which the field of scalars is replaced by a ring  $R$ . Suppose  $R$  is a commutative ring with identity  $1$  and  $G$  is an abelian (commutative) group. The group  $G$  is an  $R$ -module if there is an operation  $\cdot : R \times G \to G$  such that for all  $r, s \in R$  and  $x, y \in G$ ,  $r \cdot (x + y) = r \cdot x + r \cdot y$ ,  $(r + s) \cdot x = r \cdot x + s \cdot x$ ,  $(rs) \cdot x = r \cdot (s \cdot x)$ ,  $1 \cdot x = x$ . Suppose  $G_1$  and  $G_2$  are  $R$ -modules. Similarly, an  $R$ -module homomorphism is a map  $\mathcal{L} : G_1 \to G_2$  that is a generalization of a linear map of vector spaces.  $\mathcal{L}$  is an  $R$ -module homomorphism if for all  $x, y \in G_1$  and  $r \in R$ ,  $\mathcal{L}(x + y) = \mathcal{L}(x) + \mathcal{L}(y)$  and  $\mathcal{L}(r \cdot x) = r \cdot \mathcal{L}(x)$ .*

**Definition 16 (Module short integer solution [60, 63, 73]).** *Define the ring  $R_Z := \mathbb{Z}[X]/(\Phi(X))$ . The  $\operatorname{MSIS}_{n,B}^{\infty,\kappa,q}$  problem is defined as follows: Given a matrix  $M \xleftarrow{\$} \mathbb{R}_{\mathbb{F}}^{\kappa \times n}$  sampled uniformly at random, find a non-zero vector  $z \in R_Z^n$  such that  $Mz = 0 \pmod q$  and  $\|z\|_{\infty} < B$ .*

**Theorem 8 (Low norm invertibility [65, Theorem 1.1; 86, Theorem 2 and Proposition 2]).** *Write  $\eta=\prod_i p_i^{e_i}$  and let  $z=\prod_i p_i^{f_i}$  for integers  $1\le f_i\le e_i$ ; thus  $z\mid\eta$  and  $z$  and  $\eta$  have the same prime support. Suppose  $q \equiv 1 \pmod z$  and  $\text{ord}_{\eta}(q) = \eta/z$ . Define  $\mathbf{b}_{\text{inv}} := 1/\sqrt{\tau(z)} \cdot q^{1/\phi(z)}$  where  $\tau(z) := z$  if  $z$  is odd, otherwise  $\tau(z) = z/2$ . For an arbitrary  $a \in \mathbb{R}_{\mathbb{F}}$ , if  $0 < \|a\|_{\infty} < \mathbf{b}_{\text{inv}}$ , then  $a$  is invertible in  $\mathbb{R}_{\mathbb{F}}$ .*

**Definition 17 (Strong sampling sets [3]).** *Define  $\mathcal{C} \subseteq \mathbb{R}_{\mathbb{F}}$  to be any set of ring elements such that for any distinct elements  $a, b \in \mathcal{C}$ ,  $\|a - b\|_{\infty} < \mathbf{b}_{\text{inv}}$  (Theorem 8). Furthermore, we define the*

$$\text{expansion factor of } \mathcal{C} := \max_{\substack{0\ne v \in \mathbb{R}_{\mathbb{F}} \\ \rho \in \mathcal{C}}} \frac{\|\rho v\|_{\infty}}{\|v\|_{\infty}}$$

**Theorem 9 (Expansion factors [3]).** *Let  $\mathcal{C}$  be a strong sampling set over the cyclotomic ring  $\mathbb{R}_{\mathbb{F}}$  (Definition 17), We denote the Euler totient function as  $\phi$ . We must have that the expansion factor of  $\mathcal{C}$  is  $\le 2 \cdot \phi(\eta) \cdot \max_{\rho \in \mathcal{C}} \|\rho\|_{\infty}$ .*

**Definition 18 (Ajtai commitment scheme [2]).** *Let ring-message length  $n \in \mathbb{N}$ . The Ajtai commitment scheme  $\text{com} := (\text{Setup}, \text{Commit})$  consists of the following PPT algorithms:*

- **Setup** $(1^\lambda, n) \to \text{pp}$ : Choose the Module-SIS rank  $\kappa$  as a function of the security parameter, sample a random matrix  $M \xleftarrow{\$} \mathbb{R}_{\mathbb{F}}^{\kappa \times n}$ , and output  $\text{pp} \leftarrow M$ .
- **Commit** $(\text{pp}, z) \to c$ : Given parameters  $\text{pp}$  and vector  $z \in \mathbb{R}_{\mathbb{F}}^n$ , output  $Mz$ .

In this work, we are primarily interested in building folding schemes, a particular type of reduction of knowledge that reduces the task of checking instances in some relation  $\mathcal{R}_2$  into a running instance in a relation  $\mathcal{R}_1$ .

**Definition 19 (Folding scheme).** *A folding scheme for a relation  $\mathcal{R}$  is a reduction of knowledge of type  $\mathcal{R} \times \mathcal{R}_{\text{ACC}} \to \mathcal{R}_{\text{ACC}}$  for some relation  $\mathcal{R}_{\text{ACC}}$ .*

**Definition 20 (Special sets [39]).** *Let  $\mathcal{C}$  be a set and  $\ell \in \mathbb{N}$ . Consider two vectors  $x, y \in \mathcal{C}^{\ell}$ . We define the relation  $\equiv_i$  for  $i \in [\ell]$  as follows:*

$$x \equiv_i y \iff x_i \neq y_i \land x_j = y_j \text{ for all } j \in [\ell] \setminus \{i\}.$$

A special set  $\text{SS}(\mathcal{C}, \ell)$  is as follows:

$$\text{SS}(\mathcal{C}, \ell) = \left\{ (\vec{c}, \vec{c}_1, \dots, \vec{c}_\ell) \in (\mathcal{C}^\ell)^{\ell+1} : \forall i \in [\ell], \vec{c} \equiv_i \vec{c}_i \right\},$$

**Theorem 10 (Coordinate-wise extraction [39, Lemma 7.1]).** Let  $\mathcal{C}$  be a finite set,  $\ell \in \mathbb{N}$ , and  $\vec{\mathcal{C}} := \mathcal{C}^\ell$  be a challenge space. Let  $A : \vec{\mathcal{C}} \to \{0, 1\}^*$  be an arbitrary (probabilistic) expected polynomial-time algorithm (adversary), and  $V : \vec{\mathcal{C}} \times \{0, 1\}^* \to \{0, 1\}$  be an arbitrary (probabilistic) polynomial-time function (verification). Define the success probability of adversary  $A$  as

$$\epsilon^V(A) := \Pr_{\vec{c} \in \vec{\mathcal{C}}} [V(\vec{c}, A(\vec{c})) = 1]$$

Then, there exists an expected polynomial-time oracle algorithm  $E_A$  (extractor) that makes at most  $\ell + 1$  queries to  $A$  in expectation and with probability at least  $\epsilon^V(A) - \frac{\ell}{|\mathcal{C}|}$  outputs  $\ell + 1$  pairs  $(\vec{c}, w), (\vec{c}_1, w_1), \dots, (\vec{c}_\ell, w_\ell)$  such that

- $V(\vec{c}, w) = 1$ ,
- for all  $i \in [\ell]$ ,  $V(\vec{c}_i, w_i) = 1$ ,
- and  $(\vec{c}, \vec{c}_1, \dots, \vec{c}_\ell) \in \text{SS}(\mathcal{C}, \ell)$ .

In particular, the theorem immediately implies the weaker conservative bound

$$\epsilon^V(A)-\frac{\ell+1}{|\mathcal{C}|},$$

which is the form retained in Appendix D.5.
