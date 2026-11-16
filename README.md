# Paper Summarizer Bot

Command-line tool that reads a research paper (PDF), extracts the key sections
(abstract, introduction, method, results, conclusion), and produces a
plain-language summary with key takeaways.

## How it works
1. Extracts text from the PDF (`pypdf`).
2. Detects section headings with regular expressions.
3. Summarizes each section with **extractive summarization**: sentences are
   ranked by TF-IDF weight (`scikit-learn`) and the top ones are kept in order.
4. Builds a markdown report with a "Key takeaways" list.

## Install & run
```bash
pip install -r requirements.txt
python summarizer.py paper.pdf
python summarizer.py paper.pdf --sentences 4 --out summary.md
```

## Limitations / future work
- Scanned PDFs need OCR first.
- Extractive only; an abstractive model (e.g. BART or T5 via Hugging Face)
  could produce more fluent summaries.
