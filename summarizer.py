"""Paper Summarizer Bot
Extracts key sections (abstract, method, results, conclusion) from a research
paper PDF and generates a plain-language summary with key takeaways.

Usage:
    python summarizer.py paper.pdf
    python summarizer.py paper.pdf --sentences 4 --out summary.md
"""
import argparse
import re
from pypdf import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np

# Heading name -> regex variants found in papers
SECTION_PATTERNS = {
    "abstract": r"abstract",
    "introduction": r"introduction",
    "method": r"(?:materials\s+and\s+)?methods?|methodology|approach|proposed\s+method",
    "results": r"results?(?:\s+and\s+discussion)?|experiments?|evaluation",
    "conclusion": r"conclusions?|concluding\s+remarks|discussion",
}


def read_pdf(path: str) -> str:
    reader = PdfReader(path)
    return "\n".join((page.extract_text() or "") for page in reader.pages)


def clean(text: str) -> str:
    text = re.sub(r"-\n(\w)", r"\1", text)      # fix hyphenated line breaks
    text = re.sub(r"[ \t]+", " ", text)
    return text


def find_headings(text: str) -> list:
    """Return [(section_name, start_index_of_heading, end_index_of_heading)]."""
    found = []
    for name, pat in SECTION_PATTERNS.items():
        # heading on its own line, optional numbering like "3." or "III."
        rx = re.compile(
            rf"^\s*(?:\d+\.?|[IVX]+\.?)?\s*(?:{pat})\s*:?\s*$", re.I | re.M
        )
        m = rx.search(text)
        if m:
            found.append((name, m.start(), m.end()))
    return sorted(found, key=lambda x: x[1])


def extract_sections(text: str) -> dict:
    text = clean(text)
    heads = find_headings(text)
    sections = {}
    for i, (name, _, end) in enumerate(heads):
        stop = heads[i + 1][1] if i + 1 < len(heads) else len(text)
        body = text[end:stop].strip()
        sections[name] = re.sub(r"\s*\n\s*", " ", body)
    # Fallback: abstract is often written inline as "Abstract— ..."
    if "abstract" not in sections:
        m = re.search(r"abstract[\s:.\-—]+(.{200,2000}?)(?:\n\n|introduction)",
                      text, re.I | re.S)
        if m:
            sections["abstract"] = re.sub(r"\s+", " ", m.group(1)).strip()
    return sections


def split_sentences(text: str) -> list:
    sents = re.split(r"(?<=[.!?])\s+(?=[A-Z])", text)
    return [s.strip() for s in sents if 40 <= len(s.strip()) <= 400]


def summarize(text: str, n: int = 3) -> list:
    """Extractive summary: rank sentences by mean TF-IDF weight, keep original order."""
    sents = split_sentences(text)
    if len(sents) <= n:
        return sents
    tfidf = TfidfVectorizer(stop_words="english").fit_transform(sents)
    scores = np.asarray(tfidf.mean(axis=1)).ravel()
    top = sorted(np.argsort(scores)[-n:])
    return [sents[i] for i in top]


def build_report(sections: dict, n: int) -> str:
    titles = {
        "abstract": "What the paper is about",
        "introduction": "Background",
        "method": "How they did it",
        "results": "What they found",
        "conclusion": "Conclusion",
    }
    lines = ["# Paper Summary\n"]
    takeaways = []
    for key, title in titles.items():
        if key in sections and sections[key]:
            pts = summarize(sections[key], n if key != "abstract" else n + 1)
            if not pts:
                continue
            lines.append(f"## {title}\n" + " ".join(pts) + "\n")
            if key in ("results", "conclusion"):
                takeaways.append(pts[0])
    if takeaways:
        lines.append("## Key takeaways")
        lines += [f"- {t}" for t in takeaways]
    if len(lines) == 1:
        lines.append("Could not detect sections. The PDF may be scanned or use "
                     "unusual headings.")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser(description="Summarize a research paper PDF.")
    ap.add_argument("pdf", help="path to the paper (PDF)")
    ap.add_argument("--sentences", type=int, default=3,
                    help="sentences per section (default 3)")
    ap.add_argument("--out", help="save summary to this file (markdown)")
    args = ap.parse_args()

    text = read_pdf(args.pdf)
    report = build_report(extract_sections(text), args.sentences)
    print(report)
    if args.out:
        with open(args.out, "w", encoding="utf-8") as f:
            f.write(report)
        print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
