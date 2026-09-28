# 6 Strong and weak interactive reductions

**Definition 16 (Uniqueness Adversaries).** A uniqueness adversary  $\mathcal{A}_{\text{unq}}^{\mathcal{O}}$  is a probabilistic expected polynomial-time oracle algorithm that may query its oracle  $\mathcal{O}$  adaptively. It outputs either  $\perp$  or a pair of responses returned by two different calls to  $\mathcal{O}$ . Each call to  $\mathcal{O}$  uses fresh independent randomness. The running time includes all computations performed by  $\mathcal{O}$ .

**Definition 17 (Weak Interactive Reductions).** Consider relations  $\mathcal{R}_1$ ,  $\mathcal{R}'_1$ , and  $\mathcal{R}_2$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_1 \subseteq \mathcal{R}'_1$ . Let  $\mathcal{U}_1$  be the ambient instance space of  $\mathcal{R}_1$ .

An interactive reduction  $\Pi : \mathcal{R}_1 \rightarrow \mathcal{R}_2$ , defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  (Definition 9), is **weak** if it is complete, public coin, and there exists a function  $\phi : \mathcal{U}_1 \rightarrow \mathbb{C}$  (for an arbitrary space  $\mathbb{C}$ ) such that for any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , there exists an EPT extractor  $\mathcal{E}$  such that

(i) If the success probability of the adversary  $\epsilon(\mathcal{A}, \mathcal{P}^*) \geq 1/\text{poly}(\lambda)$ , then

$$\Pr \left[ (\text{pp}, \mathbf{s}, u_1, w_1) \in \mathcal{R}'_1 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \mathbf{s}) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}) \end{array} \right. \right] \geq \epsilon(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

(ii) If  $\mathcal{A} := (\mathcal{B}, \mathcal{B}')$  such that

$$\Pr \left[ \begin{array}{c} u_1, u'_1 \neq \perp \\ \Downarrow \\ \phi(u_1) = \phi(u'_1) \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ (u_1, \text{st}) \leftarrow \mathcal{B}'(\text{st}^*) \\ (u'_1, \text{st}') \leftarrow \mathcal{B}'(\text{st}^*) \end{array} \right. \right] = 1,$$

then the following holds. Define the oracle

$$\begin{aligned} \mathcal{O}_{\text{weak}}(\text{pp}, \mathbf{s}, \text{st}^*) &\rightarrow (u_1, w_1) : \\ (a) \quad (u_1, \text{st}) &\leftarrow \mathcal{B}'(\text{st}^*). \\ (b) \quad w_1 &\leftarrow \mathcal{E}(\text{pp}, \mathbf{s}, u_1, \text{st}). \\ (c) \quad &\text{Output } (u_1, w_1). \end{aligned}$$

For any uniqueness adversary  $\mathcal{W}_{\text{unq}}^{\mathcal{O}_{\text{weak}}}$  (Definition 16), we have

$$\Pr \left[ \begin{array}{c} w_1, w'_1 \neq \perp \\ \wedge w_1 \neq w'_1 \end{array} \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\mathbf{s}, \text{st}^*) \leftarrow \mathcal{B}(\text{pp}) \\ \text{in} \leftarrow (\text{pp}, \mathbf{s}, \text{st}^*) \\ (u_1, w_1), (u'_1, w'_1) \leftarrow \mathcal{W}_{\text{unq}}^{\mathcal{O}_{\text{weak}}(\text{in})}(\text{in}) \end{array} \right. \right] \leq \text{negl}(\lambda)$$

**Definition 18 (Strong Interactive Reductions).** Consider relations  $\mathcal{R}_1$ ,  $\mathcal{R}_2$ , and  $\mathcal{R}'_2$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_2 \subseteq \mathcal{R}'_2$ . Let  $\mathcal{U}_2$  be the ambient instance space of  $\mathcal{R}_2$ .

An interactive reduction  $\Pi : \mathcal{R}_1 \rightarrow \mathcal{R}_2$ , defined by PPT algorithms  $(\mathcal{G}, \mathcal{K}, \mathcal{P}, \mathcal{V})$  (Definition 9), is **strong** if it is complete, public coin, and there exists a function  $\phi : \mathcal{U}_2 \rightarrow \mathbb{C}$  (for an arbitrary space  $\mathbb{C}$ ) such that

(i) For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ ,

$$\Pr \left[ \begin{array}{c|c} u_2, u'_2 \neq \perp & \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ \Downarrow & (\text{s}, u_1, \text{st}_1) \leftarrow \mathcal{A}(\text{pp}) \\ \phi(u_2) = \phi(u'_2) & (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ & (u_2, w_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \\ & (u'_2, w'_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st}_1) \end{array} \right] = 1$$

(ii) For any EPT adversary  $(\mathcal{A}, \mathcal{P}^*)$ , define the oracle

$\mathcal{O}_{\text{str}}(\text{pp}, \text{s}, u_1, \text{st}) \rightarrow (u_2, w_2) :$   
(a)  $(\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s})$ .  
(b)  $(u_2, w_2) \leftarrow \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st})$ .  
(c) Output  $(u_2, w_2)$ .

There exists an EPT extractor  $\mathcal{E}$  such that if

$$\epsilon'(\mathcal{A}, \mathcal{P}^*) := \Pr \left[ (\text{pp}, \text{s}, \langle \mathcal{P}^*, \mathcal{V} \rangle((\text{pk}, \text{vk}), u_1, \text{st})) \in \mathcal{R}'_2 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \end{array} \right. \right]$$

$\geq 1/\text{poly}(\lambda)$ , and, for any uniqueness adversary  $\mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}}$  (Definition 16),

$$\Pr \left[ \begin{array}{c|c} w_2, w'_2 \neq \perp & \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ \wedge & (\text{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ w_2 \neq w'_2 & \text{in} \leftarrow (\text{pp}, \text{s}, u_1, \text{st}) \\ & (u_2, w_2), (u'_2, w'_2) \leftarrow \mathcal{S}_{\text{uniq}}^{\mathcal{O}_{\text{str}}(\text{in})}(\text{in}) \end{array} \right] \leq \text{negl}(\lambda)$$

then we have that

$$\Pr \left[ (\text{pp}, \text{s}, u_1, w_1) \in \mathcal{R}_1 \left| \begin{array}{l} \text{pp} \leftarrow \mathcal{G}(1^\lambda, \text{sz}) \\ (\text{s}, u_1, \text{st}) \leftarrow \mathcal{A}(\text{pp}) \\ (\text{pk}, \text{vk}) \leftarrow \mathcal{K}(\text{pp}, \text{s}) \\ w_1 \leftarrow \mathcal{E}(\text{pp}, \text{s}, u_1, \text{st}) \end{array} \right. \right] \geq \epsilon'(\mathcal{A}, \mathcal{P}^*) - \text{negl}(\lambda).$$

**Theorem 12 (Strong-Weak Composition).** Consider relations  $\mathcal{R}_1$ ,  $\mathcal{R}_2$ ,  $\mathcal{R}'_2$  and  $\mathcal{R}_3$  over public parameters, structure, instance, and witness tuples such that  $\mathcal{R}_2 \subseteq \mathcal{R}'_2$ . Let  $\mathcal{U}_2$  be the ambient instance space of  $\mathcal{R}_2$ . Consider interactive reductions (Definition 9)  $\Pi_1 : \mathcal{R}_1 \rightarrow \mathcal{R}_2$  ( $\mathcal{R}'_2$ ),  $\Pi_2 : \mathcal{R}_2$  ( $\mathcal{R}'_2$ )  $\rightarrow \mathcal{R}_3$  such that

(i)  $\Pi_1$  is **strong** (Definition 18) with respect to a function  $\phi : \mathcal{U}_2 \rightarrow \mathbb{C}$  and  
(ii)  $\Pi_2$  is **weak** (Definition 17) with respect to the **same** function  $\phi$ ,

then the sequential composition  $\Pi_2 \circ \Pi_1 : \mathcal{R}_1 \rightarrow \mathcal{R}_3$  is a **reduction of knowledge**.

*Proof.* For brevity, we defer the proof to Appendix B.1. □

