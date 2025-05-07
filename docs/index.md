---
layout: default

title: > 
  TokenSwift: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens
venue: ICML 2025
authors:
    - name: Tong Wu
      tag: <i class="fas fa-star" style='font-size:11px'></i> 1
      url: https://wutong4012.github.io/
    - name: Junzhe Shen
      url: https://junzheshen.github.io
      tag: <i class="fas fa-star" style='font-size:11px'></i> 1, 2
    - name: Zixia Jia
      tag: 1
      url: https://openreview.net/profile?id=~Zixia_Jia1
    - name: Yuxuan Wang
      url: https://patrick-tssn.github.io/
      tag: 1
    - name: Zilong Zheng
      url: https://zilongzheng.github.io
      tag: 1, <i class="fa fa-envelope"></i>
affiliations:
    - name: NLCo Lab, BIGAI
      tag: 1
    - name: LUMIA Lab, Shanghai Jiao Tong University
      tag: 2
misc: > 
  <sup><i class="fas fa-star" style='font-size:11px'></i></sup> Equal Contribution.
  <sup><i class="fa fa-envelope"></i></sup> Corresponding authors.

arxiv: https://arxiv.org/abs/2502.18890
code: https://github.com/bigai-nlco/TokenSwift
---


<div class="container is-max-desktop">
<div class="hero-body">
<figure class="image" id="framework">
    <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/TokenSwift.png' | relative_url }}" />
    <figcaption><span class="dnerf">Figure 1.</span> <b>An overview of TokenSwift.</b> First, target model (LLM) with partial KV cache and three linear layers outputs 4 logits in a single forward pass. Tree-based attention is then applied to construct candidate tokens. Secondly, top-<math><mi>k</mi></math> candidate 4-grams are retrieved accordingly. These candidates compose draft tokens, which are fed into the LLM with full KV cache to generate target tokens. The verification is performed by checking if draft tokens match exactly with target tokens. Finally, we randomly select one of the longest valid draft tokens, and update 4-gram table and KV cache accordingly.</figcaption>
</figure>
</div>
</div>


<section class="section">
    <div class="container is-max-desktop" markdown="1"> 
<h2 style="font-size: 2em; font-weight: bold;">📦 Demo</h2>
<div align="center">
<video width="960" height="540" controls>
    <source src="https://github.com/user-attachments/assets/5094fca7-0b12-470c-a7b6-456d254855d1" type="video/mp4">
    Your browser does not support the video tag.
</video>
</div>
<br/>
<br/>
Recent advances in large language models (LLMs), amplified by their long context capacities, have demonstrated remarkable proficiency in intricate reasoning ([OpenAI-o1](https://arxiv.org/abs/2412.16720); [DeepSeek-R1](https://arxiv.org/abs/2501.12948)), agentic thinking ([Reflexion](https://proceedings.neurips.cc/paper_files/paper/2023/file/1b44b878bb782e6954cd888628510e90-Paper-Conference.pdf); [ReAct](https://arxiv.org/pdf/2210.03629); [RAM](https://arxiv.org/pdf/2404.12045)), and creative writing ([Wang et al., 2023](https://arxiv.org/pdf/2311.04459); [Mikhaylovskiy, 2023](https://aclanthology.org/2023.inlg-genchal.2.pdf)), etc. These advancements **necessitate the ability to generate lengthy sequences**, *e.g.*, o1-like reasoning tends to generate protracted chain-of-thought trajectories before reaching final conclusions.
<br/>
<br/>
However, generating ultra-long sequences (up tp 100K tokens) is painfully slow. For example, generating 100K tokens with LLaMA3.1-8B can take approximately five hours (<a href="#speed">Figure 2</a>), hindering real-world applications.
<br/>
<br/>
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="speed">
  <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/speed.png' | relative_url }}" style="width: 80%; max-width: 600px; height: auto"/>
  <figcaption><span class="dnerf">Figure 2.</span> Comparison of the time taken to generate 100K tokens using autoregressive (AR) and TokenSwift with prefix length of 4096 on Llama3.1-8b. As seen, TokenSwift accelerates the AR process from nearly 5 hours to just 90 minutes.</figcaption>
</figure>
<h2 style="font-size: 2em; font-weight: bold;">Is Speculative Decoding Enough?</h2>
A straightforward solution is to take advantage of recent success in speculative decoding (SD). However, existing methods are generally tailored for generating short sequences, *e.g.*, [TriForce](https://arxiv.org/pdf/2404.11912) and [MagicDec](https://arxiv.org/pdf/2408.11049) are limited to generating 256 and 64 tokens, respectively. Directly extending their generation length to 100K tokens would inevitably encounter failures due to KV cache budget constraints. Furthermore, when applied to optimized KV cache architectures such as Group Query Attention (GQA), these methods yield only marginal acceleration gains for short-sequence generation (<a href="#sd_speedup">Figure 3</a>). This observation leads to a pivotal research question:
<br/>
<br/>
*Is it possible to achieve model-agnostic **lossless** accelerations, akin to those seen in short-sequence SDs, for generating **ultra-long** sequences, with **minimal** training overhead?*
<br/>
<br/>
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="sd_speedup">
  <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/sd_speedup.png' | relative_url }}" style="width: 40%; max-width: 600px; height: auto"/>
  <figcaption><span class="dnerf">Figure 3</span></figcaption>
</figure>
<h2 style="font-size: 2em; font-weight: bold;">Why Ultra-Long Sequences Are a Headache</h2>
Generating ultra-long sequences exposes three critical bottlenecks:

1. **Frequent Model Reloading:** When generating  ultra-long sequence, such as 100K tokens, the GPU must reload the model weights over 100,000 times. This repetitive process poses the challenge: How can we reduce the frequency of model reloading?
2. **Prolonged Growing of KV Cache:** [TriForce](https://arxiv.org/pdf/2404.11912) and [MagicDec](https://arxiv.org/pdf/2408.11049) have demonstrated that a small KV cache budget can be used during the drafting phase. While their one-time compression strategy at the prefill stage can handle scenarios with long prefixes and short outputs, it fails to address cases involving ultra-long outputs. The challenge lies in determining when and how to dynamically update the KV cache within limited budget.
3. **Repetitive Content Generation:** When generating sequences of considerable length, *e.g.*, 100K, the model tends to produce repetitive sentences. While eliminating this issue is not our focus, it is still essential and challenging to mitigate repetition patterns in ultra-long sequences.

</div>
</section>

<section class="section" style="background-color:#efeff081" >
    <div class="container is-max-desktop" markdown="1">


<h2 style="font-size: 2em; font-weight: bold;"> TokenSwift: Tailored Solutions for Each Challenge </h2>
### **1. Multi-Token Generation & Token Reutilization**
Instead of generating one token at a time, TokenSwift predicts multiple tokens in a single forward pass. Inspired by Medusa, it adds lightweight linear layers to the base model, and utilizes tree attention to enable *Multi-Token Generation*. To further boost efficiency, it reuses frequent n-grams (phrases) from earlier text, reducing redundant computations.
### **2. Dynamic KV Cache Management**
TokenSwift intelligently prunes less important KV pairs while preserving critical context. It keeps the initial prompt’s KV cache intact and dynamically updates the rest based on importance scores derived from attention patterns.
### **3. Contextual Penalty and Random N-gram Selection**
To combat repetition, TokenSwift penalizes recently generated tokens within a sliding window, nudging the model toward diverse outputs. This works alongside sampling strategies like [Nucleus Sampling](https://arxiv.org/pdf/1904.09751), [min-<math><mi>p</mi></math>](https://arxiv.org/pdf/2407.01082), and [<math xmlns="http://www.w3.org/1998/Math/MathML"><mi>η</mi></math>-sampling](https://arxiv.org/pdf/2210.15191)

<h2 style="font-size: 2em; font-weight: bold;"> Results: 3x Faster, Scalable, and Robust</h2>
<a href="#table1">Table 1</a> and <a href="#table2">Table 2</a> are the main results, showing TokenSwift can consistenly achieve over <math xmlns="http://www.w3.org/1998/Math/MathML"><mn>3</mn>  <mo>×</mo></math> acceleration across various model scales and architecture.
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table1">
  <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/table1.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Table 1.</span> Experimental results for LLaMA2 and LLaMA3.1 under varying prefix lengths, generating sequences from 20K to 100K tokens.</figcaption>
</figure>
<br/>
<br/>
<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="table2">
  <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/table2.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Table 2.</span> Experimental results of TokenSwift for Qwen2.5 across different scales under prefix length 4096, generating sequences from 20K to 100K tokens. The time is measured in minutes. </figcaption>
</figure>
<br/>
<br/>

<div style="display: flex; justify-content: center; align-items: flex-start;">
  <figure class="image" style="display: flex; flex-direction: column; margin-right: 20px; text-align: center">
    <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/abl_ngram.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
    <figcaption>Ablation on Token Reutilization: Enabling Token Reutilization (<math><mi>k</mi></math>=20) significantly improves the overall acceptance rate and speedup throughout the generation process. </figcaption>
  </figure>
  <figure class="image" style="display: flex; flex-direction: column; text-align: center">
    <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/abl_penalty.png' | relative_url }}" style="width: 100%; max-width: 1000px; height: auto"/>
    <figcaption>Ablation on Contextual Penalty: Applying Contextual Penalty significantly improves the diversity of generation despite the sampling method. </figcaption>
  </figure>
</div>


<figure class="image" style="display: flex; justify-content: center; align-items: center; flex-direction: column;" id="case">
  <img src="{{ 'https://bigai-nlco.github.io/TokenSwift/assets/img/case.png' | relative_url }}" style="width: 50%; max-width: 1000px; height: auto"/>
  <figcaption><span class="dnerf">Case Study on Llama3.1-8b</span> Left: fragments of generated text without Contextual Penalty. Right: fragments of generated text with Contextual Penalty. The <span style="color: blue">blue</span> text is the repetition part. </figcaption>
</figure>

## BibTex
{:.title}

```bibtex
@misc{wu2025hoursminuteslosslessacceleration,
      title={From Hours to Minutes: Lossless Acceleration of Ultra Long Sequence Generation up to 100K Tokens}, 
      author={Tong Wu and Junzhe Shen and Zixia Jia and Yuxuan Wang and Zilong Zheng},
      year={2025},
      eprint={2502.18890},
      archivePrefix={arXiv},
      primaryClass={cs.CL},
      url={https://arxiv.org/abs/2502.18890}, 
}
```

</div>
</section>
