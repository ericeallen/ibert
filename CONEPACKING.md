Right — “cone-packer” is a term I use for a surgical context assembler: instead of dumping the entire repo into a long-context model, you pack just the dependency cone around the thing you care about. Think of it like computing a “slice” of the project’s graph.

How it works

        [ Target symbol / query ]
                 /\
                /  \
               /    \
              / Direct deps \
             /--------------\
            / Indirect deps  \
           /------------------\
          / Docs/ Summaries/  \
         /  Broader context    \
        /-----------------------\


Pick a target symbol/task
Example: “Why does orders.join(customers) fail?” → target = orders.join.

Compute the dependency cone

Upstream: everything that defines or influences the symbol (imports, schema definitions, column names).

Downstream: everything that uses the symbol (tests, call sites).

Lateral: directly related helpers/utilities (e.g. column type definitions, operator cards).

This is essentially walking the symbol/reference graph (which Serena/LSP gives you) forward and backward.

Assemble a context pack

Files/snippets = target + its cone (defs, refs, examples, nearby tests).

Add semantic summaries (operator card, type rules).

Order it: start with operator spec, then the target function, then examples/tests.

Instead of 1M tokens, you often get 5k–30k tokens that are laser-focused.

Feed to the model
The LLM sees everything it needs to answer the question or propose a repair — but not a haystack of irrelevant files.

Why it matters

Efficiency: Keeps prompt size 10–50× smaller than “paste the repo.”

Accuracy: Reduces “lost in the middle” degradation by cutting noise.

Determinism: You can explain why those files are in context (graph-based), which is impossible if you just shovel in a repo dump.

Extensibility: Works on huge repos where even 1M tokens isn’t enough.

In your stack

Serena gives you symbol/def/ref graph → compute cones easily.

Semgrep tags help identify semantic cones (e.g. all windows + partition ops).

Retriever fetches examples/tests outside the cone if needed.

Cone-packer outputs a structured bundle (think JSON or ordered sections) that you pass to Devstral or Gemini.

So: instead of “mega-context,” build a cone-packer: a small service that, given a query or symbol, packs just the relevant slice of the code + docs into the prompt. It gives you the clarity of retrieval with the expressive power of long-context models. Would you like me to sketch a quick architecture diagram / pseudo-pipeline for a cone-packer in your whybis project?
