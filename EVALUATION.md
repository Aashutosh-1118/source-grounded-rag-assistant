# Evaluating this RAG system

## What evaluation means here

Evaluation is a repeatable test that tells you how good your system is with numbers instead of impressions. You write a set of questions where you already know the right answer, run the system on them, and score the results. When you change something (chunk size, retrieval method, threshold, model), you run it again and see whether the numbers went up or down.

Without this, "I improved the retrieval" is just a feeling. With it, you can say "hit rate went from 67% to 89%".

## What gets measured

| Metric | Question it answers | How it is computed |
|---|---|---|
| Hit rate@k | Did the right passage make it into the top k results? | A retrieved chunk counts as a hit if it comes from the expected file and contains the key fact |
| MRR | How high was the right passage ranked? | Average of 1 / rank of the first hit. 1.0 means always first |
| Answer accuracy | Is the final answer correct? | The answer must contain the expected keywords |
| Faithfulness | Is every claim in the answer supported by the passages? | An LLM judge says YES or NO |
| Correct refusal | Does it refuse questions the documents cannot answer? | Share of unanswerable questions that were refused |
| Wrong refusal | Does it refuse questions it should answer? | Share of answerable questions that were refused |
| Latency | How long does an answer take? | Average and 95th percentile |

Retrieval metrics matter most. If the right passage never reaches the model, no prompt can fix the answer.

## Build your own question set

The included set in `evaluation/` uses a small made-up corpus so the script runs out of the box. For your resume, use real documents.

1. Put 3 to 5 real documents in `evaluation/corpus/` (articles saved as `.md` or `.txt`, or PDFs). Pick a topic you can explain, such as mortgage articles.
2. Write 20 to 30 questions in `evaluation/eval_set.json`. Roughly 70 percent should be answerable and 30 percent not answerable from your documents.
3. For each answerable question, fill in:
   - `source`: the file name that contains the answer
   - `evidence`: one or more short phrases that appear in the right passage, copied exactly from the file
   - `answer_keywords`: words that a correct answer must contain. Use `|` for alternatives, for example `"5 percent|5%"`
4. For unanswerable questions, only `id`, `question` and `"answerable": false` are needed. Include some that are on a related topic but not covered (these are the hard ones), and a couple that are completely off topic.
5. Mix question styles: exact numbers, paraphrased questions, and questions that use different words than the document.

Format of one entry:

```json
{
  "id": "q01",
  "question": "What was the average 30-year fixed rate in March?",
  "answerable": true,
  "source": "home_loan_primer.md",
  "evidence": ["6.8 percent"],
  "answer_keywords": ["6.8"]
}
```

## Run it

```bash
# Retrieval only. Needs no API key, just the embedding model.
python -m evaluation.run_eval --no-generation --strategies vector hybrid hybrid_rerank

# Full run with answers and the faithfulness judge. Needs GROQ_API_KEY.
python -m evaluation.run_eval --strategies vector hybrid hybrid_rerank --gen-strategy hybrid

# Find a good MIN_RELEVANCE value
python -m evaluation.run_eval --no-generation --tune-threshold
```

Output is printed, saved to `evaluation/results/latest.md`, and a timestamped JSON file is saved next to it.

## How to read the results

- **Hit rate low for every strategy:** the problem is upstream. Try a different chunk size, check that the loader extracted clean text, or check that your `evidence` phrases match the file.
- **Hybrid beats vector:** your questions contain exact terms (numbers, names) and keyword search is helping. This is the usual result.
- **Reranker beats hybrid:** the right passage is in the candidates but ranked too low. The reranker fixes the order, at the cost of extra latency.
- **High hit rate but low answer accuracy:** retrieval works but the prompt or model is the problem.
- **Low faithfulness:** the model is adding claims that are not in the passages. Tighten the prompt or lower the temperature.
- **High wrong refusal:** `MIN_RELEVANCE` is too high. **Low correct refusal:** it is too low. The sweep table shows the value that balances both. Copy it into `.env`.

## Rules for honest numbers

- Do not tune on the same questions you report. If you adjust the threshold with `--tune-threshold`, keep a few questions aside, or say clearly that the threshold was tuned on the full set.
- Report the size of the set (for example "30 questions"). Small sets are fine if you say so.
- The LLM judge is a rough check. Spot check a few of its verdicts by hand.
- Run the baseline (`vector`) first and keep its numbers. Improvements only mean something next to a baseline.

