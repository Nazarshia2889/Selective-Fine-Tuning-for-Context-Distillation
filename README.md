# Selective-Fine-Tuning-for-Context-Distillation


\title{182 Proposal: The Role of Weight Sparsity and Granularity in Selective Fine-Tuning for Context Distillation}
\author{Michael Liu, Evan Luo, Arshia Nayebnazar, Yash Saxena}

\begin{document}

\maketitle

\section{Abstract}
%In-context learning (ICL) is a widely used method of leveraging prompt tokens to enhance the capabilities of large language models (LLMs), including factual Q\&A and reasoning; however, LLMs lose these capabilities when the context is removed. Context distillation (CD) is the concept of ``internalizing'' capabilities gained through ICL by fine-tuning the language model's weights, such that the relevant context is no longer necessary in the prompt. Various approaches exist for CD, usually involving a ``student'' model (without context) learning from an identical ``teacher'' model (with context). 
Context distillation (CD) is the concept of ``internalizing'' capabilities gained through in-context learning (ICL) by fine-tuning the language model's weights, such that the relevant context is no longer necessary in the prompt. CD typically involves full-parameter fine-tuning, which is costly to run for every new piece of context, or LoRA, which typically underperforms full fine-tuning. Recent selective fine-tuning methods, which update only a subset of the original weights, report performance close to full fine-tuning. In this work, we compare supervised fine-tuning (SFT) for CD across levels of weight sparsity (updating 4-17\% of weights) and units of granularity (individual weights, rows of linear layers, or whole transformer blocks), selecting the units with the largest gradient magnitude of the distillation loss, to determine whether specific subsets of weights are most relevant for targeted fine-tuning of ICL-induced capabilities.

\section{Introduction and Background}

Learning by Distilling Context \citep{snell2022learning} introduces context distillation, where a model learns to reproduce behavior induced by instructions, reasoning steps, or examples without requiring that context at inference time. Towards Evolving Context Parameterization \citep{shi2026towards} highlights the computational cost of context distillation, reporting approximately 452 seconds per document and 49 GiB of additional peak memory for context distillation on SQuAD.

Cartridges \citep{eyuboglu2026cartridges} compares approaches to internalizing long-document context, finding that learned KV prefixes outperform LoRA at equal memory budgets, but does not investigate which pretrained weights should be updated.

The Fisher update-sparsity study \citep{zhang2026fisher} shows that updating just 0.1\% of carefully selected weights recovers approximately 85\% of full training's improvement in an RL setting, while random selection provides almost no improvement. Salience-Aware Sparse Fine-Tuning \citep{liu2025salience} compares eight weight-selection criteria and finds that simple gradient-magnitude selection performs competitively with more sophisticated approaches.

S$^2$FT \citep{yang2024s2ft} selectively fine-tunes attention heads and MLP channels, demonstrating improvements in efficiency, generalization, and forgetting compared to conventional fine-tuning methods. LISA \citep{pan2024lisa} reduces training costs by updating a small, rotating subset of transformer layers rather than the entire model. StructTune \citep{liu2026structtune} compares structured row- and column-level updates against individual-weight selection, finding that structured sparsity often performs better at comparable parameter budgets.

\section{Key Questions}

%CD methods are costly to run in practice, since repeated updates are needed for every new piece of context [Evolving Context Parameterization], and they typically use full fine-tuning or LoRA. Meanwhile, recent work outside CD shows that selective weight updates can match full fine-tuning at the weight [Liu et al.], block [S²FT], and layer [LISA] levels, and that distillation updates are naturally sparse [Fisher]. The choice of granularity has been compared for general fine-tuning [StructTune], and update types have been compared for CD [Cartridges], but no work compares selection granularity for CD specifically. Therefore, we are interested in better understanding the performance differences of fine-tuned models for CD across different levels of selectivity. More precisely, we are looking to answer the following questions:
CD methods are costly to run in practice due to repeated updates being necesary for every new piece of context [Evolving Context Parameterization]; however, recent work has shown that selective weight updates can match full fine-tuning at the weight [Liu et al.], block [S²FT], and layer [LISA] levels. While there is research about the choice of granularity for full fine-tuning, no work compares selection granularity for CD specifically. We are interested in understanding the performance differences of distilling context through fine-tuning across different levels of parameter selectivity. In particular:

\begin{enumerate}
    \item Are there specific subsets of weights that are most relevant towards updating a particular ICL-induced capability, instead of updating all of the parameters?
    \item What is the tradeoff between efficiency, primarily in terms of memory and latency, and performance across different levels of selectivity?
    \item How do selectivity and granularity requirements change across different ICL tasks? For example, do reasoning-tasks enhanced by ICL require more weights to change than knowledge retrieval Q\&A?
\end{enumerate}

\section{Hypotheses}
\begin{enumerate}
    \item We generally expect that a subset of weights will be sufficient to match full fine-tuning approaches for CD across different tasks. We expect the lowest granularity level (individual weights) will be most effective due to its flexibility, even though the selected weights will likely be broadly distributed across the model.

    \item Furthermore, we believe that the minimum subset of weights will be small enough across tasks that there will be a notable improvement in efficiency in terms of memory and latency during fine-tuning.

    \item We expect that the minimum subset of weights will vary between tasks. Basic Q\&A will likely require fewer weight updates than more complex few-shot reasoning tasks.
    
\end{enumerate}


\section{Methods}
%Methods: how you will test your hypothesis, which experiments you will run, and the expected compute.
%\\\\
We use Qwen3.5-0.8B in non-thinking mode, where a frozen teacher receives the context, and a student initialized from the same weights is trained without it via SFT on the teacher's responses. We evaluate three task families, each with three contexts, 500 training queries, and 200 test queries. In label mapping, the context shows 16 sentiment or topic examples with arbitrary labels (e.g., positive reviews labeled "apple"), and queries are new sentences to label. In knowledge retrieval, the context describes 10 fictional people with 5 facts each, and queries ask about these facts; test questions use wordings not seen in training. In multi-step reasoning, the context shows 8 worked step-by-step solutions to addition in an unfamiliar base (e.g., base 7), and queries are new problems. We retain contexts where the teacher outperforms the zero-shot model by at least 20 percentage points on a validation set.

We compare selective fine-tuning at three granularities: individual weights, rows of linear layers, and transformer blocks, excluding embeddings and normalization parameters. Units are ranked globally by their average absolute gradient of the distillation loss over eight batches, and the top units are selected under matched budgets of approximately 4\%, 8\%, and 17\% of eligible weights (1, 2, and 4 blocks). Selection is fixed during training. We compare against random selection at each granularity and budget, and against full fine-tuning. Learning rates are tuned per granularity on one pilot context, and all other hyperparameters are held constant.

We measure the fraction of ICL improvement recovered without the context, forgetting on an MMLU subset, trainable parameters, peak memory, training time, and storage per context, and we examine selection overlap across tasks. Experiments total 183 runs on a Colab A100 GPU (162 selective, 9 full fine-tuning, 12 learning-rate tuning), estimated at 15--31 GPU-hours.

\clearpage

\bibliographystyle{apalike}
\bibliography{references}

\end{document}
