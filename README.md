# 🔋 PEFT for Abstractive Summarization
### Comparative Study of Parameter-Efficient Fine-Tuning Methods for ruT5-base

📌 **Project Description**

In this study, I compared classical full fine-tuning with parameter-efficient fine-tuning (PEFT) methods for the task of abstractive news summarization in Russian, aiming to understand:
- How much the number of trainable parameters and training time are reduced when using adapters
- Whether generation quality (measured by ROUGE metrics) is preserved when significantly reducing parameters
- How different adapter injection strategies (which modules to inject into) affect the results

**Dataset**: `gazeta` — news articles and their short summaries from Gazeta.Ru portal, ~5000 examples for training, 500 for validation. Source: [IlyaGusev/gazeta](https://huggingface.co/datasets/IlyaGusev/gazeta).

**Base Model**: `ai-forever/ruT5-base` — pre-trained seq2seq model for Russian, T5 architecture, ~300M parameters.

---

**Methods**:
  - **Baseline**: Full fine-tuning (all parameters trainable)
  - **LoRA** (Low-Rank Adaptation): approximation of weight updates using low-rank matrices $W + \Delta W = W + \frac{\alpha}{r}BA^T$
    - `r = 4`, `α = 16`, inject into `[q, v]` (library implementation)
    - `r = 16`, `α = 16`, inject into `[q, v]` (library implementation)
    - Custom implementation: `r = 4`, inject into `[q, v]` and separately into `[q, k, v]`
  - **(IA)³** (Infused Adapter by Inhibiting and Amplifying): trainable gate vectors for activation rescaling
    - Inject into `[k, v]` and separately into `[k, v, wo]`

---

🎯 **Research Goals**

1. **Implement LoRA from scratch**: write a custom `LoRALayerWrapper`, verify forward/backward pass against `PEFT`
2. **Compare training efficiency**: measure training time and number of trainable parameters for each method
3. **Evaluate summarization quality**: use ROUGE-1/2/L/Lsum metrics on the validation set
4. **Compare with a pre-trained model**: test on real news articles against `IlyaGusev/rut5_base_sum_gazeta`

---

📊 **Key Results**

| Approach | Config | Trainable Parameters | Time (2 epochs) | val loss | ROUGE-1 | ROUGE-L |
|----------|-------------|---------------------|-----------------|----------|---------|---------|
| **FFT (baseline)** | Full fine-tuning | ~297M | ~20 min | **4.18** | **0.235** | **0.188** |
| **LoRA** | r=4, α=16, [q,v] | ~442K | ~17.5 min | 4.19 | 0.229 | 0.182 |
| **LoRA** | r=16, α=16, [q,v] | ~1.77M | ~17.6 min | 4.19 | 0.233 | 0.186 |
| **Custom LoRA** | r=4, α=16, [q,v] | ~2.21M | ~18.3 min | 4.18 | 0.225 | 0.180 |
| **Custom LoRA** | r=4, α=16, [q,k,v] | ~664K | ~17.8 min | 4.76 | 0.187 | 0.137 |
| **(IA)³** | [k,v] | ~55K | ~17.0 min | 11.58 | 0.041 | 0.037 |
| **(IA)³** | [k,v,wo] | ~129K | ~17.3 min | 11.02 | 0.049 | 0.034 |

---

📈 **Potential Improvements**

- [ ] Train on the full dataset for more epochs to enable a fairer quality evaluation

---

🔗 **References**

- [LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685)
- [(IA)^3: Infused Adapter by Inhibiting and Amplifying](https://arxiv.org/abs/2205.05638)
- [ruT5-base on HuggingFace](https://huggingface.co/ai-forever/ruT5-base)
- [Gazeta Summaries Dataset](https://huggingface.co/datasets/IlyaGusev/gazeta)
