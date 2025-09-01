Plan for Lazy-Evaluated DSL Code Generation (DevStarL + LoRA + Tools)
Background & Rationale

Our project initially considered using BERT for generating code in a lazy-evaluated DSL. However, we hypothesize that vanilla BERT will not deliver optimal performance for this task. BERT is an encoder-only Transformer model and lacks a decoder mechanism to generate sequences, meaning it isn’t inherently designed for producing code step-by-step
medium.com
. It is technically possible to force BERT to generate text, but that architecture was not intended for generative tasks – the quality falls short of decoder-based language models
medium.com
. Moreover, BERT was pre-trained on natural language, not on programming languages or domain-specific languages, so it may struggle to capture the syntax and semantics of a DSL. These factors motivate moving away from BERT to a more suitable approach for code generation.

Another challenge is the nature of our target domain: lazy-evaluated DSLs. In a lazy DSL like Ibis, operations are not executed immediately; instead, they build an expression tree and only execute (often by compiling to another engine) when needed
medium.com
. This means the model must generate code that is not only syntactically correct but also semantically correct when eventually evaluated. Ensuring correctness is harder because errors might only surface at “execution” time (e.g. type mismatches or invalid operations). We need a solution that addresses these issues by leveraging both advanced language modeling and the DSL’s own tools (compiler/type-checker) to validate correctness.

Proposed Approach Overview

To tackle the above challenges, we propose a new approach, which we refer to as DevStarL, combining several state-of-the-art strategies: a large code-oriented LLM, Low-Rank Adapter fine-tuning (LoRA), and integration of compiler and type-checking tools. The plan is to fine-tune a pre-trained code language model (instead of BERT) on our DSL tasks, using LoRA for efficiency, and to incorporate the DSL’s compiler or type system both during and after generation for validation. By doing so, we aim to significantly improve the correctness and performance of generated DSL code.

Key components of the approach include:

Large Code Model (DevStarL Base): Utilize a Transformer model that has been pre-trained on source code (or otherwise suited to code generation) as our base. Such models have learned programming language syntax and patterns, giving them a better starting point than a text-only model. Prior work shows that models like CodeGPT, CodeT5, etc., outperform general models on code tasks
aclanthology.org
. This base will form the backbone of “DevStarL,” ensuring the model understands code structure and DSL syntax from the start.

LoRA Fine-Tuning: Apply Low-Rank Adaptation (LoRA) during fine-tuning to efficiently adapt the large model to our specific DSL domain. LoRA freezes the original model weights and introduces small trainable weight matrices, greatly reducing the number of parameters we need to train
web.stanford.edu
. This makes fine-tuning feasible on our hardware (a single H100 GPU) even for a 20B-parameter model. Using LoRA will significantly cut memory usage and training time while preserving model performance
web.stanford.edu
, allowing us to experiment with a 20B model on one GPU as we have successfully done before (per our experience).

Tool-Assisted Generation (Compiler & Type Checker): Integrate the DSL’s compiler and type checker into the model’s generation loop. The idea is to have the model’s output checked (either dynamically during generation or after generation) by the actual DSL implementation to catch errors. For example, we can attempt to compile or type-check the generated code: if the code fails to compile or has type errors, that feedback can be used to guide the model to correct it. This approach is inspired by recent research showing that using compiler feedback dramatically improves code generation success rates
aclanthology.org
. Likewise, type-constrained decoding has been shown to cut compilation errors by over 50% and improve functional correctness in code synthesis across various model sizes
openreview.net
. By leveraging these tools, our model won’t be guessing blindly – it can verify and refine its outputs to ensure they are valid and executable in the DSL environment.

In summary, DevStarL marries a strong code-focused LLM with efficient fine-tuning and a “human-in-the-loop” style feedback via compilers. This should address BERT’s shortcomings by using a model actually built for generation, and ensure higher quality outputs by systematically weeding out mistakes through the DSL’s own formal checks.

Domain Focus: Ibis as an Example DSL

We will demonstrate this approach using Ibis as the example lazy-evaluated DSL. Ibis is a Python library that lets users write Pandas-like code which is then translated into SQL or other backends for execution
medium.com
. In Ibis, operations (like filters, aggregations, joins) are deferred – they build an intermediate representation (IR) that only executes when you trigger an action (like fetching results). This lazy evaluation model is representative of the kind of DSL we target, where correctness of the IR is crucial since execution happens downstream (e.g., in a SQL engine).

Using Ibis has a twofold benefit for our project plan:

Representative Challenge: It provides realistic semantics (e.g., type rules, supported operations) that our model must learn. If our model can generate correct Ibis expressions from, say, a natural language description or another input, it demonstrates the effectiveness of our approach. Any mistakes (like using an unsupported operation or mismatched types) can be caught by Ibis’s compiler when translating to SQL or by its type system.

Tool Access: We can directly use the Ibis API as our “compiler/type checker.” For instance, after the model generates an Ibis expression, we can attempt to compile it to a SQL query or compute a schema. If Ibis throws an error (type error, syntax issue, etc.), we know the output is invalid. This can feed back into refining the model’s output (either by an automated loop or during evaluation).

It’s important to note that while we focus on Ibis for concreteness, the approach generalizes to other lazy-evaluation DSLs (we could apply the same method to similar frameworks). In future, one could replace Ibis with another DSL – the only changes needed would be swapping in the corresponding compiler/type-checker. For example, if there were a DSL “Ybis” with analogous lazy semantics, our approach would remain applicable with minimal adjustments. This highlights that our plan isn’t tied to Ibis alone, but to the class of problems involving generating code for DSLs that have formal evaluation mechanisms.

Modeling Strategy

Model Architecture: We will start with a large pre-trained language model that is well-suited for code generation. Candidates include open-source code models like StarCoder, CodeLlama, or a GPT-style model known to perform well on programming tasks. The model size can be in the order of tens of billions of parameters (around 20B, as we have the capacity to fine-tune such a model on a single H100 GPU). Using a model of this scale and type is crucial because it provides a strong prior knowledge of programming syntax and possibly even some knowledge of analytical query patterns. Unlike BERT, these models are autoregressive decoders, meaning they generate code token-by-token in a left-to-right manner – exactly what we need for synthesizing valid code.

We will refer to our fine-tuned model as DevStarL (to denote a Developer/DSL Star model adapted with LoRA). The base model (DevStarL base) will be chosen for its performance on code; for instance, if we pick CodeLlama-13B or StarCoder-15B and possibly augment it, that will serve as the foundation. We may also consider an ensemble or multi-step architecture: for example, one component to generate initial code and another to refine errors, but initially we’ll focus on a single-model solution augmented by tool feedback.

Hypothesis on Modeling: By using a model pre-trained on code, we expect it to handle the DSL syntax more gracefully than BERT. Many DSL constructs (especially Ibis, which largely mimics pandas/SQL syntax) will be somewhat familiar to a model that has seen Python and SQL code. This reduces the learning burden – the model won’t start from scratch in understanding what a table.join(other_table) means, for example. The large parameter count also gives it capacity to learn the nuances of lazy evaluation semantics that might be absent from smaller models. In short, the modeling strategy is to leverage a powerful code-oriented language model to compensate for BERT’s weaknesses in this domain.

Training Plan

Data Collection & Preparation: We will gather a training dataset for the model that pairs problem descriptions with correct DSL code (Ibis expressions) or otherwise tasks the model to generate Ibis code. Possible sources for this data include:

Ibis documentation and examples (to get pairs of English descriptions and Ibis code).

Hand-crafted examples or conversions of existing SQL query datasets (e.g., take an NL2SQL dataset and translate the SQL queries to Ibis equivalents as training data).

Synthetic data generation: for example, define some analytical tasks and produce the Ibis API calls that solve them. This might require manual effort but ensures we cover various edge cases (group by, window functions, etc.).

We will ensure the dataset is comprehensive enough to cover the DSL’s functionality and that each example’s code has been verified to be correct (passes the compiler/type checker). If available, we might use a portion of data for fine-tuning and hold out some for validation (and possibly a separate test set for final evaluation).

Fine-Tuning with LoRA: Fine-tuning the chosen model on our DSL data will be done using the LoRA approach. As mentioned, LoRA drastically cuts down training cost by introducing only a few new trainable weights
web.stanford.edu
. All the original model weights remain frozen, so we avoid overfitting and can train with limited compute. With LoRA, fine-tuning even a 20B parameter model on a single GPU becomes practical. This method has the added benefit of speed and lower memory usage
web.stanford.edu
, which means we can iterate faster and even try multiple fine-tuning runs (e.g., to tune hyperparameters or try different data augmentations) within our compute budget.

The training process will involve:

Setting up the model in a training framework (such as PyTorch with HuggingFace Transformers, which has support for LoRA through libraries like peft).

Injecting LoRA adapters into key layers of the model (likely the query/key/value projection matrices in Transformers, as done in the LoRA paper).

Training on our DSL dataset for a number of epochs or until performance converges. We will monitor metrics like the loss on training and validation sets, and perhaps functional metrics like percentage of outputs that compile or match reference solutions in the validation set.

Given that LoRA reduces trainable parameters, we might be able to experiment with different hyperparameters (learning rates, LoRA rank, etc.) to maximize performance without lengthy retraining. We’ll also be cautious about not under-training (since LoRA adds limited capacity, underfitting is possible if the rank is too low, for example).

Throughout training, Weights & Biases (W&B) will be used for experiment tracking and management. We will log training runs, hyperparameters, and performance metrics to W&B for transparency and reproducibility. Using W&B allows us to efficiently compare experiments and ensures no results are lost – reproducibility is critical for research
wandb.ai
. In practice, each training run will automatically record metrics and model checkpoints to the W&B dashboard, so we can analyze them later and even share results with the team easily. (For citation: W&B is a popular tool in academia for tracking machine learning experiments
wandb.ai
, helping make research more reproducible and collaborative.)

Validation During Training: We will set aside a portion of data as a validation set to periodically evaluate the model during training. This validation will involve generating DSL code from the model (given some input prompt) and checking:

Does the generated code compile/type-check in Ibis without errors?

If we have an expected correct answer, how often does the model’s output match it or produce an equivalent result?

By monitoring these, we can detect overfitting or identify if the model is improving in practical correctness, not just in minimizing loss. If the model frequently produces invalid code in validation, that indicates we need to incorporate more guidance (possibly more training data examples of correct syntax or even adjust training to penalize invalid outputs).

Additionally, if resources allow, we might perform a small human evaluation of the quality of outputs (since sometimes code can be logically incorrect even if it compiles). But primarily, compiler/type-checker feedback will be our gauge of validity.

Inference and Validation Plan

Once the model (DevStarL fine-tuned with LoRA) is trained, we will deploy the approach to generate DSL code for new problems and thoroughly evaluate its performance. This phase includes how the model interacts with the compiler tool and how we measure success on test tasks.

Inference Procedure: For a given input (for example, a description of a data analysis task or a question that should be answered by a query), the model will generate an Ibis code snippet. We will employ a decoding strategy suitable for code: likely beam search or temperature-controlled sampling to balance correctness and diversity. We may generate multiple candidate solutions in one go (e.g. a beam of size 5) because one of the strengths of our approach is we can verify each candidate using the DSL’s tools.

After generation, we perform the following tool-assisted validation:

Compile/Type-Check Candidates: Each candidate Ibis expression is passed to the Ibis compiler (translating it to SQL or executing .execute() on a small sample data if possible) or a type-check routine. Any candidate that raises an error (syntax error, type error, etc.) is marked as invalid and discarded or sent for correction.

Iterative Refinement (if applicable): If none of the candidates pass the compiler’s checks, we can employ an iterative refinement loop. This might involve feeding the error messages back to the model in a prompt (for example: "The previous attempt failed with error X, please correct the code") and letting it try again. This interactive generation process can continue for a few iterations until a valid solution is found or a max number of attempts is reached. Incorporating the compiler feedback in this way effectively creates a feedback loop where the model learns to fix its mistakes on the fly, akin to how a developer debugs code. Such feedback-driven generation is inspired by methods like CompCoder which showed dramatic improvements in compilation success by reinforcing based on compiler results
aclanthology.org
aclanthology.org
.

Select Final Output: If one or more candidates compile successfully, we then may have the model rank them or simply choose the first valid one. Optionally, if multiple valid solutions exist, we could choose the one that seems simplest or most efficient (though this likely requires additional heuristics or metrics – as a first step, any valid solution is acceptable).

Testing & Evaluation: We will evaluate the system on a test set of tasks (distinct from training and validation data). For each task, the goal is to produce a correct DSL program. Our evaluation criteria will include:

Accuracy/Success Rate: The percentage of test tasks for which the model produces a correct and working solution. “Correct” here means it yields the expected result when executed. If our test tasks come with ground-truth outputs or known answers, we will run the generated Ibis code on test databases and compare results to the expected output. If the tasks are specified as “generate code that does X,” we will manually or programmatically verify that the code indeed accomplishes X.

Compilation/Type-Check Success Rate: We expect a high fraction of the model’s outputs to be immediately free of syntax or type errors, thanks to our approach. We’ll measure what proportion of initial outputs pass the Ibis compiler. This can be compared to a baseline (for instance, how often would a baseline model like BERT or a naive approach produce valid code). A significant improvement here would validate the efficacy of integrating the type checker. Earlier research suggests our approach should greatly reduce errors – e.g., using type constraints cut TypeScript compilation errors by over half in a similar setting
openreview.net
, so we have a benchmark to surpass.

Efficiency (optional): How many iterations on average does the model need with the feedback loop to get a valid solution? Fewer is better for usability. We will track, for each test problem, if the first attempt was valid or if it needed multiple refinement rounds. Ideally, most cases succeed on the first try, showing the model inherently learned to produce correct code. If many require iteration, that’s still fine (the system ultimately succeeds) but could indicate room for improving training (so the model internalizes the type rules more strongly).

We will also compare our results against the initial BERT-based approach (if we have benchmarks from it or from literature). We anticipate DevStarL to vastly outperform a BERT-based model in both validity and accuracy, given the architectural and training advantages. Any comparisons will be documented to quantify the gain from our new approach.

Throughout inference and testing, we continue to use Weights & Biases for logging outcomes. We will log each run’s results, such as which candidates were generated and which passed or failed compilation, as well as final accuracy metrics. This logging will help in analyzing failure cases after the fact (we can inspect what kinds of errors slipped through, etc.). It also provides a record for reproducibility – anyone can see which model version produced which result on which task via the W&B run logs.

Finally, we must consider edge cases during inference:

If the DSL or Ibis has certain functions that are hard to use, the model might still struggle. We will identify if there are any systematic errors (e.g., always getting join syntax wrong) and could address those in a post-hoc manner (by adding more training examples or adjusting the prompt format).

Performance-wise, generating code with a 20B model might be slow (a few seconds per query). But since this is an offline setting for now, that is acceptable. If needed, we can prune the model or use half-precision during inference to speed it up.

Experiment Management and Logging

As mentioned, a crucial aspect of our plan is using Weights & Biases (W&B) for experiment management. Every training experiment, model version, and evaluation run will be tracked. W&B will log hyperparameters (model size, LoRA rank, learning rates, etc.), metrics (loss curves, accuracy, success rates), and even example outputs. This practice ensures that our research is reproducible and well-organized – no result exists only on one machine; everything is versioned in the cloud. This kind of systematic tracking is considered a best practice in modern ML research, as it makes it easy to reproduce and compare results across different runs
wandb.ai
. We can cite W&B’s own documentation which emphasizes experiment reproducibility and collaboration
wandb.ai
.

In addition, W&B allows us to create reports and dashboards. We will set up a dashboard showing key comparisons (e.g., a chart of compilation success rate over time as we add the tool-feedback technique, or a table of final results vs baseline). This will help in communicating progress to stakeholders and quickly identifying what changes improved the model’s performance.

We will also maintain version control for our code (likely on Git), including the scripts for data preprocessing, training, and the inference pipeline with the compiler loop. Coupled with W&B logs, this will make it straightforward for any team member (or reviewer) to trace a given result back to the exact code and configuration that produced it.

Considerations and Next Steps

In executing this plan, we have several considerations to keep in mind:

Quality of Training Data: The model is only as good as the examples it sees. We must ensure our training dataset covers a wide range of DSL scenarios (from simple filters to complex aggregations) and is correct. Any systematic gaps or errors in training data could lead to the model learning the wrong behavior. If Ibis or the chosen DSL has certain quirks, we should include those patterns in training data (for example, if certain operations require casting types, show that in examples).

Lazy Evaluation Semantics: Because the DSL is lazy, some errors won’t manifest until execution. For instance, Ibis might allow constructing an expression that only fails when executed on data (like dividing by a column of incompatible type). Our compile/type-check step might not catch logic errors (only syntactic and type errors). Thus, some incorrect outputs might slip through if they are logically flawed but type-correct. To mitigate this, our validation on the test set uses actual execution on data where possible, to truly ensure correctness. In a deployed system, one might integrate runtime tests as well (for example, run the query on a small sample to verify it yields plausible results).

Tool Integration Overhead: Each compiler or type check call adds overhead. We need to ensure this doesn’t make inference prohibitively slow. Compiling an Ibis expression to SQL is usually fast (milliseconds), so it should be fine. If we ended up with a slower DSL compiler, one could parallelize checking of candidates or cache some results. For our case, this isn’t likely a big issue, but it’s a general consideration for tool-assisted generation.

Generalizability: While we claim the approach generalizes, in practice each new DSL would require its own integration. We should abstract our implementation so that swapping Ibis for another tool is straightforward. For example, our code could define an interface for “DSLCompiler” with methods compile(code) and check_types(code) so that plugging in a different backend just means writing a wrapper for that interface.

Model Limitations: Even a 20B model has limitations. If the DSL uses very domain-specific concepts (like specialized functions or keywords the model didn’t see in pre-training), we may need to teach it via fine-tuning or prompt examples. We should be prepared to extend the training data or use prompt engineering (giving the model some preamble about DSL usage) if we observe it hallucinating or misusing certain DSL functions. The tool feedback loop will catch many errors, but preventing errors at the source (via better modeling) is more efficient.

After implementing and evaluating this plan with Ibis, the next steps would include: scaling up the approach (more data, possibly trying a larger model if available), testing on genuinely novel tasks or bigger real datasets to ensure it works in practice, and potentially integrating a reinforcement learning step. For example, once we have the pipeline, we could use reinforcement learning from compiler feedback: allow the model to generate code and use a reward signal for code that passes all tests vs code that doesn’t, and fine-tune the model further with that signal. This is a more advanced iteration (similar in spirit to how OpenAI’s Codex was refined with unit tests), but it could further cement the model’s ability to produce correct programs consistently.

In conclusion, this plan lays out a comprehensive strategy to move away from a BERT-based solution to a more potent LoRA-fine-tuned code LLM with integrated compiler checks. By addressing modeling, training, and inference in tandem and leveraging modern tools (both algorithmic like LoRA and practical like W&B), we expect to achieve significantly better performance on lazy-evaluated DSL code generation. This approach is grounded in recent research insights and tailored to the specifics of our problem domain, giving us confidence in its potential to succeed. All experiments and results will be carefully tracked, and we will iteratively refine the plan as we gather empirical evidence of what works best for our DSL tasks. With this, we aim to validate our core hypothesis: that a specialized, tool-informed LLM can indeed outperform a generic model like BERT by a wide margin on generating correct, executable DSL code.

Sources:

BERT’s generative limitations
medium.com
medium.com

Ibis and lazy evaluation background
medium.com

Compiler feedback improving code generation
aclanthology.org
aclanthology.org

Type-aware decoding reducing errors
openreview.net

LoRA fine-tuning benefits
web.stanford.edu
web.stanford.edu

Experiment tracking with Weights & Biases
wandb.ai
