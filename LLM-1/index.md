---
title: RLHF Algorithms — PPO, GRPO, GSPO
layout: default
permalink: /LLM-1/rlhf-ppo-grpo-gspo/
---

# RLHF Algorithms: PPO, GRPO, GSPO — Differences, Trade-offs, and Use Cases

> Over the past two years, RLHF has evolved rapidly. From PPO to GRPO to GSPO, every iteration reflects a trade-off between stability, efficiency, and scalability. Here’s how these algorithms differ — and when you should care.
> 

# **1. LLMs and RLHF: Why Reinforcement Learning Matters**

## **1.1 The Rise of LLMs**

The story of Large Language Models (LLMs) began with the **Transformer** architecture, introduced in 2017 through the landmark paper *Attention Is All You Need*. Although the design proved powerful, training Transformers at scale demanded significant GPU resources — at a time when hardware was both limited and expensive. As a result, building state-of-the-art models was less about clever ideas and more about securing compute: *“Money is all you need.”*

By 2018, models like **BERT** and the first generation **GPT** signaled the beginning of the **foundation model era**. These models showed that pretraining on massive text corpora could unlock broad generalization abilities cross both understanding and generation. Still, due to the high cost, many researchers and smaller companies could only observe rather than actively participate.

Everything changed in the end of **2022** with the release of **ChatGPT**. What had been a niche research trend suddenly became a global phenomenon. From tech giants to startups, from finance and education to automotive and consumer electronics, nearly every industry began discussing and experimenting with this new technology. The age of LLMs had truly arrived.

There are two defining characteristics in the training of OpenAI’s ChatGPT.

First, the model is trained on an exceptionally large corpus of text with a correspondingly vast parameter scale. For instance, **GPT-3** was trained on approximately **500 billion tokens** and contains up to **175 billion parameters**, making it one of the largest models of its time.

While other large-scale models such as **Google’s PaLM(540B),** **Microsoft & Nvidia’s  MT-NLG (530B)** also demonstrated the potential of massive pretraining, the combination of both extreme scale and the successful application of RLHF in ChatGPT represents a distinctive and influential milestone in the evolution of LLMs.

## **1.2 Quick recap of why supervised fine-tuning isn’t enough**

During the **pretraining stage**, the model learns from massive text corpora using the method of **next-token prediction**. Given a context, the model attempts to predict the most likely next token. If the prediction is correct, the probability assigned to that token is reinforced; if incorrect, the model adjusts its parameters.

The learning objective here is **cross-entropy loss**, expressed as:

$\mathcal{L} = - \sum_{t} \log P_\theta(x_t \mid x_{<t})$

The goal is to minimize this loss, thereby reducing uncertainty (entropy) and improving the model’s ability to predict the correct token. A well-trained model in this stage becomes proficient at continuing text when provided with an input, or **prompt**. This pretrained model is commonly referred to as the **base model**.

To make the model more useful for practical applications, the next step is **Supervised Fine-Tuning (SFT)**. In this stage, the model is trained on curated instruction–response pairs, where each dataset entry consists of a user query and a high-quality answer. This process teaches the model to produce direct, instruction-following outputs rather than unconstrained continuations. The resulting model is often called an **instruct model**.

However, SFT alone is insufficient because models at this stage may still produce incorrect answers, hallucinate facts, or generate responses that are unsafe or unsuitable. Therefore, an additional alignment mechanism—**Reinforcement Learning from Human Feedback (RLHF)**—is introduced to further refine the model’s behavior and address these limitations.

## **1.3 Role of human feedback and reward models**

The key objective of **RLHF** is to further align the model’s behavior with human preferences.

In general, the full RLHF consists of reward model training followed by reinforcement learning phase. 

During **reward model training**, the reward model serves as a **proxy for human judgment**. In the subsequent RLHF stage, the process involves an iterative loop: the **policy model** generates responses to prompts, and the **reward model** evaluates these responses to provide a preference signal. While in theory humans could directly provide feedback at each iteration, empirically this is infeasible due to the scale and cost. The reward model therefore acts as a scalable substitute for human evaluators, enabling continuous training and alignment.

Three key steps that follow the large-scale pretraining of an LLM from paper of **InstructGPT**:

step 1 is SFT , step 2 and step 3 is RLHF. 

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/0b30eb31-c952-4741-b8e0-090d7db1b412.png)

Full four stages of training GPT assistant from talk “The state of GPT” by Andrew Karpathy:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image.png)

A illustration of RLHF from  [https://huggingface.co/blog/rlhf](https://huggingface.co/blog/rlhf)

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%201.png)

In the RLHF stage, two key factors are considered when updating the **policy model** (also referred to as the *actor*). The first is the **evaluation of the model’s response** to a given prompt, typically provided by a reward model trained on human preferences. The second is the **degree of divergence** between the updated policy and the original reference model, which is controlled to prevent the policy from drifting too far from its pretrained behavior.

## **1.4 How RLHF fits into the LLM training pipeline**

| **Stage** | **Description** | **Strengths** | **Limitations** | **Model Name** |
| --- | --- | --- | --- | --- |
| **Pretraining** | Trained on massive text corpora using next-token prediction. Learns broad linguistic, factual, and reasoning knowledge. | Strong generalization, fluent text generation. | May produce unhelpful, misleading, or unsafe outputs. | **Base Model** |
| **Supervised Fine-Tuning** | Fine-tuned on curated *prompt–response* pairs created or reviewed by humans. Teaches the model to follow explicit instructions. | Improves usability and instruction-following. | Still prone to errors, hallucinations, or unsafe responses. | **Instruct Model** |
| **Reinforcement Learning** | Uses a reward model (trained on human preference data) to iteratively guide the policy model via RL (e.g., PPO, GRPO). | Aligns outputs with human values: helpful, harmless, honest. | Requires high-quality human feedback and careful RL algorithm design. | **Aligned Model** |

# **2. PPO: The Classic Workhorse**

## **2.1 Core idea: policy updates with clipping for stability**

### **Policy-Based vs. Value-Based Methods**

In reinforcement learning, two primary approaches are commonly distinguished:

- **Value-based methods**: These aim to learn a value function (e.g., Q-learning) that estimates the expected return of a state or state–action pair. The policy is then derived implicitly by acting greedily with respect to the value estimates.
- **Policy-based methods**: These optimize the policy directly by adjusting its parameters to maximize expected reward. Instead of relying on explicit value estimates, they update the policy using gradients computed with respect to the expected return.

A widely used family within policy-based methods is **policy gradient algorithms,** which alternate between:

 (1) sampling trajectories through interaction with the environment, and

 (2) optimizing a **surrogate objective function** via stochastic gradient ascent. 

**Proximal Policy Optimization (PPO)** is a classic example. It is an **actor–critic algorithm** that combines both paradigms: the actor represents the policy to be optimized, while the critic estimates values to stabilize training.

 

## 2.2 Strengths: reliable, widely adopted

**From TRPO to PPO**

Classic policy gradient methods perform only one update per data sample, often causing  learning **instability** (oversized steps) and **poor sample efficiency** (no data reuse). 

**Trust Region Policy Optimization (TRPO)** stabilizes updates by constraining the step size of policy updates, that is, constraining KL divergence between the new and old policies. 

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%202.png)

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%203.png)

However, TRPO requires solving a constrained, second-order problem, making it computationally expensive and difficult to implement in practice.

**Proximal Policy Optimization (PPO)**, proposed by Schulman et al. in 2017, simplifies this process. Instead of enforcing a hard trust-region constraint, PPO introduces a **clipped surrogate objective,** allowing multiple epochs of mini-batch updates while preventing excessively large policy shifts.

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%204.png)

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%203.png)

In other words, PPO improves both **stability** and **sample efficiency**. Intuitively, *PPO acts like a seatbelt for policy training — it allows the model to adjust itself freely, but prevents sudden, unsafe changes.*

**Mapping PPO to LLM RLHF**

In LLM RLHF, we view each decoding step as an RL step: the state is the prompt plus the previously generated tokens $s_t=(x,y_{<t})$, and the action is the next token $a_t=y_t$. Under this mapping, the length-normalized PPO objective is:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%205.png)

𝜋𝜃 and 𝜋𝜃𝑜𝑙𝑑  are the current and old policy models,

𝑞 is from the question dataset,  𝑜 is from old policy model

𝐴𝑡 is the advantage calculated from the result of reward model and value model. 

**KL regularization in RLHF** 

To curb reward model gaming and preserve pretrained behavior, a per-token KL penalty from a reference model  $\pi_{\text{ref}}$  is often added in the reward at each token.

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%206.png)

**Advantage estimation (GAE)**

PPO typically uses **Generalized Advantage Estimation** to balance bias–variance:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%207.png)

**Why PPO became the workhorse**

- **Stable:** clipping (and optional KL) curbs destructive updates
- **Sample-efficient:** supports multiple minibatch epochs per rollout
- **More general:** applies well across different tasks
- **Practical:** first-order optimization; simpler than TRPO; widely implemented

## 2.3 Limitations: computationally heavy, reward hacking issues

Despite its strengths, PPO has notable drawbacks in RLHF:

- **High compute/memory footprint.** PPO is **actor–critic**: beyond the policy (actor), a separate **value function (critic)** must be trained and stored, raising GPU demand and latency
- **Complex pipeline.** A typical RLHF setup maintains three ****modules:
    1. **Policy model** (PPO-tuned, actor), ****requiring an additional **critic** (value function)
    2. **Reward model (RM)** (scores responses)
    3. **Reference model for KL**(often frozen **SFT model**)
    
    Coordinating data, checkpoints, and updates across this stack increases engineering complexity.
    
- **Reward hacking / over-optimization.** Optimizing against a learned RM can exploit its weaknesses, yielding high reward but low real-world quality. KL helps, but tuning $\beta$ is delicate: too low → drift; too high → under-learning.
- **Token-level credit assignment.** PPO’s per-token objective can misalign with **sequence-level** rewards common in RLHF (e.g., one reward per response), contributing to instability—especially for **MoE** models and long sequences.

These limitations motivate alternatives that **reduce critic overhead** and/or **better match sequence-level rewards**, such as **GRPO** (critic-free, relative baseline) and **GSPO** (sequence-level objective and clipping).

# **3 GRPO: A Simpler Alternative**

## 3.1 Motivation: remove the critic, reduce complexity

**Group Relative Policy Optimization (GRPO)**, introduced by *DeepSeek* in 2024, aims to reduce the computational and engineering complexity of PPO in the RLHF.

PPO is an **actor–critic** algorithm: alongside the **policy model(actor)**, a **value function (critic)** must be trained to estimate state values, which increases memory usage, compute cost, and pipeline complexity. GRPO eliminates the critic while retaining stable updates, making RLHF more accessible on smaller clusters and faster to iterate.

## 3.2 Mechanism: group-based reward comparison

### Compare of PPO and GRPO

GRPO removes the critic entirely. Instead of relying on a learned value function to compute the **advantage**, it calculates advantage by comparing each response’s score against the **average score within a group of responses**. Formally, the advantage is defined relative to peer outputs rather than to an independent baseline.

- In PPO, the advantage comes from the critic’s value estimates, which serve as a **global, uniform baseline**.
- In GRPO, the advantage is based on **relative comparisons within the sampled group**, making it a **local, context-dependent baseline**.

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/de5a9985-0d9d-49aa-af1f-5d7a9d6cc1bb.png)

GRPO’s objective is:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%208.png)

**GRPO differs from PPO in two principal ways:** 

(1) advantage estimation via group-relative rewards rather than a learned value function, and

(2) an unbiased estimator for the policy gradient without a critic.

For each prompt $q$, sample $G$  responses $\{o_1,\dots,o_G\}$ from the old policy $\pi_{\theta_{\text{old}}}$. A reward model scores each response, yielding $\{r_1,\dots,r_G\}$. Normalize rewards within the group. Using **outcome supervision**, assign the normalized reward $\tilde r_i$ to every token of response $o_i$; i.e., for all time steps *t* in $o_i$, $\hat A_{i,t} \;=\; \tilde r_i = 
\frac{r_i - \mathrm{mean}(\mathbf{r})}{\mathrm{std}(\mathbf{r})}$   .

Thus, **advantages are group-relative and token-constant per response**, removing the need for a learned value function while providing a variance-reduced baseline.

GRPO estimate the KL divergence using a unbiased estimator: 

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%209.png)

## 3.3 Pros: efficient, stable in practice

Experimental results indicate that GRPO is both **more efficient** and **more stable** in practice:

1. **Efficiency**: Removing the value model significantly reduces GPU memory usage and inference costs.
2. **Stability**: Using relative judgments improves robustness, since comparisons are inherently normalized against peer responses.
3. **Good practical fit.** Works well when reward signals are **sequence-level** (one score per answer), which is common in RLHF.

## 3.4 Cons: less theoretical grounding, emerging best practices

However, GRPO also has limitations.
• **Less theoretical grounding.** GRPO is primarily **empirically motivated**; convergence properties and failure modes are less formalized than PPO.
• **Sensitivity to group design.** Performance depends on **group size $G$**, sampling temperature, and reward dispersion; too-small $G$ increases variance, too-homogeneous groups weaken the signal.

## 3.5 Summary

This table makes the trade-offs clear at a glance: 

**PPO = theoretically grounded but resource-heavy;** 

**GRPO = efficient and practical but less formalized**.

| **Aspect** | **PPO**  | **GRPO** |
| --- | --- | --- |
| **Introduced** | Schulman et al., 2017 | DeepSeek, 2024 |
| **Algorithm Type** | Actor–Critic | Policy-Only (no critic) |
| **Advantage Estimation** | From a trained **value function** (critic) → global, uniform baseline | From **relative scores** compared to group average → local, context-dependent baseline |
| **Stability Mechanism** | Clipped surrogate objective + KL penalty from reference model | +Relative judgment among sampled responses |
| **Resource Demand** | High — requires training and maintaining a separate value model (critic) | Lower — no critic, saving GPU memory and compute |
| **Practical Strengths** | Well-studied, strong theoretical grounding, widely adopted (e.g., OpenAI RLHF) | More efficient, empirically stable, reduced compute cost |
| **Limitations** | Computationally heavy; critic introduces overhead; reward hacking risks | Lacks strong theoretical foundation; primarily validated through practice |
| **Use Case Fit** | Well-resourced setups; need strong baselines | Compute-constrained settings; rapid RLHF cycles |

# **4. GSPO: Addressing Instability in Scaling**

## 4.1 Why PPO and GRPO can struggle with large-scale or MoE settings

**Mixture-of-Experts (MoE).** MoE introduces sparsity by routing each token to a small subset of expert subnetworks. This enables **very high capacity** (trillion-scale parameters) while keeping **per-token compute** manageable. In practice, MoE allows for higher model capacity without a proportional increase in inference cost.

**Stability challenges.** PPO and GRPO were originally designed with dense architectures in mind, and rewards are typically **sequence-level** (one score per response). When applied to MoE models, they often suffer from instability. PPO and GRPO optimize **token-level** objectives, which can misalign with sequence-level signals and amplify variance—especially in MoE, where routing induces **highly non-uniform activations** across tokens and samples. The result can be **unstable updates**, credit-assignment noise, and sensitivity to hyperparameters.

## 4.2 How GSPO modifies the update rule for smoother optimization

**Group Sequence Policy Optimization (GSPO)**, introduced by *Qwen* in 2025,  aligns the objective with how rewards are provided in RLHF by moving from **token-level** to **sequence-level** optimization.

- **Key difference from prior methods**:
    - PPO and GRPO compute **token-level importance ratios**.
    - GSPO instead defines the importance ratio at the **sequence level**, considering the likelihood of the entire output sequence.
- **Mechanism**:
    - Sequence-level **clipping**, **rewarding**, and **optimization** are performed.
    - This aligns more naturally with how rewards are assigned (usually at the sequence level in RLHF).
    - It also stabilizes training in large-scale MoE settings.

### **Equation and Interpretation**

In GSPO, the **sequence-level importance weight** is defined as:

$$
r(y|x) = \frac{\pi_\theta(y|x)}{\pi_{\theta_\text{old}}(y|x)}
$$

This ratio reflects how much the updated policy $\pi_\theta$ diverges from the old policy $\pi_{\theta_\text{old}}$ for generating response y given input x.

- In GRPO, the advantage is computed **relative to peer responses**, relying on group-level comparisons.
- In GSPO, the ratio has a **clear theoretical meaning**: it measures distributional shift at the **sequence level**, which naturally aligns with sequence-level rewards and provides a principled basis for clipping.

The **GSPO optimization objective** can be written as:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%2010.png)

The importance ratio $s_i(θ)$ is based on sequence likelihood:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%2011.png)

 $\hat{A_i}$  denotes the group-based advantage estimation:

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%2012.png)

## 4.3 When to consider GSPO: high-variance or unstable training regimes

### **When to Use GSPO**

- **PPO**: good general-purpose algorithm, widely adopted.
- **GRPO**: efficient alternative when compute is limited, but lacks strong theory.
- **GSPO**: particularly valuable for **unstable training scenarios**, such as large-scale **MoE-based LLMs**, where token-level objectives fail to ensure stability.

By aligning optimization at the sequence level, GSPO provides a more robust approach for the next generation of massive, sparse LLM architectures.

# 5. Kimi K2’s RL

## 5.1 what’s kimi’s RL

### **Kimi K2 and RL for Large MoE Models**

Kimi also trains **large-scale Mixture-of-Experts (MoE) models**, which makes its reinforcement learning strategy particularly interesting to compare with approaches from DeepSeek (GRPO) and Qwen (GSPO).

![image.png](RLHF%20Algorithms%20PPO,%20GRPO,%20GSPO%20%E2%80%94%20Differences,%20Tra%20274b2305df2180abbe2bd1be8005746b/image%2013.png)

**Kimi K2’s RL algorithm** combines elements of both worlds: it adopts a **relative baseline** similar to GRPO, while introducing an **explicit KL regularization term** reminiscent of PPO/GSPO. This hybrid formulation is designed to balance efficiency with stability at MoE scale.

- **Difference vs. DeepSeek (GRPO):** GRPO eliminates the critic and relies solely on relative scoring within response groups. Kimi, however, augments relative scoring with a KL penalty, making the optimization more controlled.
- **Difference vs. Qwen (GSPO):** GSPO emphasizes sequence-level clipping to ensure stability in MoE training. Kimi instead uses a squared loss with KL regularization, achieving stability in a more heuristic, empirically grounded way.

**Commonalities across RL approaches:**

- All three methods (GRPO, GSPO, and now Kimi’s variant) **remove reliance on heavy critic training**, reducing computational burden.
- Each incorporates, either directly or indirectly, a mechanism to **control policy divergence**—through KL penalties (PPO, Kimi), relative scoring (GRPO), or sequence-level clipping (GSPO).
- The shared goal is to improve **efficiency and stability** in large-scale RLHF for LLMs, particularly when scaling to massive or sparse architectures such as MoE.

# **6. Putting It Together: When to Use Which**

1. **PPO** – A reliable **default choice** when sufficient computational resources are available. It is theoretically grounded, widely tested, and remains the standard baseline for RLHF.
2. **GRPO** – Well-suited for **efficiency-focused settings** or **smaller GPU clusters**, where the removal of the critic significantly reduces memory and compute costs.
3. **GSPO** – Designed for **scaling to large or heterogeneous architectures**, especially **MoE-based LLMs**, where token-level objectives struggle to ensure stability.
4. **Future Directions** – Likely to involve **hybrid or adaptive strategies**, combining the theoretical guarantees of PPO/GSPO with the efficiency of GRPO, or dynamically switching algorithms depending on training conditions.

As models grow in size and sparsity, we may see hybrids that combine the efficiency of GRPO, the theoretical grounding of GSPO, and the pragmatic engineering of Kimi’s RL. The field is still evolving, and the choice of algorithm will depend increasingly on resource budgets, architecture, and alignment goals.

## **📚 Related Papers**

### **🔹 Foundational Works**

- **Vaswani et al., 2017 — Attention Is All You Need**
    
    Introduced the Transformer architecture, the foundation for all modern LLMs and RLHF pipelines.
    
    [arXiv:1706.03762](https://arxiv.org/abs/1706.03762)
    
- **Schulman et al., 2017 — Proximal Policy Optimization Algorithms (PPO)**
    
    The core algorithm behind early RLHF implementations (e.g., InstructGPT). Introduced clipping for stability and simplicity over TRPO.
    
    [arXiv:1707.06347](https://arxiv.org/abs/1707.06347)
    
- **Ouyang et al., 2022 — Training Language Models to Follow Instructions with Human Feedback (InstructGPT)**
    
    The seminal paper establishing the RLHF framework for LLM alignment, using PPO with a reward model trained on human preferences.
    
    [arXiv:2203.02155](https://arxiv.org/abs/2203.02155)
    

---

### **🔹 Algorithmic Evolution: GRPO, GSPO, and Beyond**

- **DeepSeek AI, 2024 — DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)**
    
    Introduced a critic-free variant of PPO that computes advantages via group-relative comparisons. Improves efficiency and stability for large-scale LLM RLHF.
    
    [arXiv:2402.03300](https://arxiv.org/abs/2402.03300)
    
- **Alibaba Qwen Team, 2025 — Group Sequence Policy Optimization (GSPO)**
    
    Advances GRPO by moving optimization from token-level to sequence-level, improving stability in Mixture-of-Experts (MoE) and large models.
    
    [arXiv:2507.18071](https://arxiv.org/abs/2507.18071)
    
- **Kimi K2 Technical Report, 2025**
    
    Describes Kimi’s approach combining GRPO’s relative baseline with PPO/GSPO-style KL regularization to balance efficiency and stability.
    
    [arXiv:2507.20534](https://arxiv.org/abs/2507.20534)
