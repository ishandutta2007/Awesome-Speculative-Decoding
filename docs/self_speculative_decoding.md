# Self-Speculative Decoding (Layer Skip / Draft Heads)

Self-Speculative Decoding consolidates the drafting and target architectures into a single neural network model, optimizing VRAM and deployment complexity.

## 💡 Core Concept
Instead of maintaining a separate draft model, the main LLM itself performs drafting. This is done by either:
1. **Layer Skipping:** Routing draft tokens through only a subset of the model's layers (e.g., first 5 out of 32 layers) to get quick predictions.
2. **Draft Heads:** Predicting future tokens using auxiliary linear heads attached to intermediate or final layers.

## 📊 Layer Skip vs. Draft Heads

```mermaid
flowchart TD
    subgraph Layer Skipping
        LStart[Input] --> L1[Layers 1-5]
        L1 -->|Draft Prediction| LExit[Early Exit / Draft]
        LExit -->|Verification| LFull[Layers 6-32]
    end
    
    subgraph Draft Heads
        HStart[Input] --> HFull[Full Base LLM]
        HFull --> HeadA[Head A: n+1]
        HFull --> HeadB[Head B: n+2]
    end
```

## 🔗 References
* [Draft & Verify: Lossless Large Language Model Acceleration via Self-Speculative Decoding](https://arxiv.org/abs/2309.08168)
* [LayerSkip: Enabling Early Exit Inference and Self-Speculative Decoding](https://arxiv.org/abs/2404.16710)
