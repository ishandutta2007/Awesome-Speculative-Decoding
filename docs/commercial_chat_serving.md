# High-Throughput Commercial Chat Serving (vLLM / TensorRT-LLM)

In production, speculative decoding must be integrated into high-performance serving frameworks to deliver real-world latency reductions.

## 💡 Integration Challenges
To be useful in production serving engines (like vLLM or TensorRT-LLM), speculative decoding must:
1. Support complex scheduling (continuous batching).
2. Efficiently manage KV caches for both the draft and target models.
3. Be compatible with paging mechanisms (PagedAttention).

## ⚙️ Architecture in Serving Engines
Serving engines implement a decoupled proposer-verifier loop. The proposer (e.g., draft model, n-gram lookup, or EAGLE drafter) runs asynchronously or synchronously with the scheduler to supply candidates to the verifier kernel.

## 📊 Serving Pipeline

```mermaid
flowchart LR
    Queue[Request Queue] --> Scheduler[vLLM Scheduler]
    Scheduler --> Proposer[Draft Model / Proposer]
    Proposer -->|Candidate Tokens| Verifier[Target Model / Verifier]
    Verifier -->|KV Cache updates| PagedCache[PagedAttention Cache]
```

## 🔗 References
* [vLLM Speculative Decoding Documentation](https://docs.vllm.ai/en/latest/models/spec_decode.html)
* [TensorRT-LLM Repository](https://github.com/NVIDIA/TensorRT-LLM)
