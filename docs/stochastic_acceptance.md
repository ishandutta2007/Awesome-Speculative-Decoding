# Stochastic Acceptance (Modified Nucleus Sampling)

Stochastic Acceptance allows speculative decoding to be used during creative generation (Temperature $T > 0$) while mathematically guaranteeing that the final output distribution matches the target model's original distribution.

## 💡 Core Concept
If we reject draft tokens that don't match the target model's sample, we bias the output. The stochastic acceptance algorithm uses a modified rejection sampling scheme to correct the output probability distribution.

## ⚙️ Rejection Sampling Rule
For a draft token $x$ proposed by the draft model $q(x)$ and evaluated by the target model $p(x)$:
1. Accept $x$ with probability:
   $$\min\left(1, \frac{p(x)}{q(x)}\right)$$
2. If accepted, proceed to the next token.
3. If rejected, sample a replacement token from the adjusted distribution:
   $$p'(x) = \max\left(0, p(x) - q(x)\right)$$ normalized.

## 📊 Decision Tree

```mermaid
flowchart TD
    Start[Draft Token x] --> Prob{Roll probability: p_accept = min(1, p(x)/q(x))}
    Prob -- Roll <= p_accept --> Accept[Accept Token x]
    Prob -- Roll > p_accept --> Reject[Reject Token x] --> Resample[Sample from normalized max(0, p(x)-q(x))]
```

## 🔗 References
* [Leviathan et al. (2022) - Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192)
* [Chen et al. (2023) - Accelerating Large Language Model Decoding with Speculative Decoding](https://arxiv.org/abs/2302.01318)
