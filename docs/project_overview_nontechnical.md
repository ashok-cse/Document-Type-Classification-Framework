# German Document Type Classification — Project Overview

*A plain-language summary for non-technical readers*

**Project:** DTCF — German Document Type Classification Framework
**Author:** Ashok Kumar Meena
**Course:** Pattern Recognition (Phase 2)

---

## What problem does this solve?

Organisations deal with huge volumes of scanned paper documents — financial
statements, legal texts, government paperwork, instruction manuals, research
papers, patents, and more. Before anyone can use these documents, they usually
need to be **sorted by type**. Doing this by hand is slow, repetitive, and easy
to get wrong.

This project builds a system that looks at a scanned page of a **German document**
and automatically decides **what kind of document it is** — without anyone having
to read it.

## What does it actually do?

You give the system a picture of a document page. It returns one of six answers:

| Category | Plain meaning |
|---|---|
| **Financial reports** | Company accounts, balance sheets, earnings statements |
| **Scientific articles** | Academic and research papers |
| **Laws and regulations** | Legal texts, statutes, official rules |
| **Government tenders** | Public-sector contracts and procurement notices |
| **Manuals** | Instruction and how-to guides |
| **Patents** | Invention filings and patent documents |

Crucially, it makes this decision by looking at the **visual layout of the page** —
the arrangement of text, headings, tables, columns, and white space — **not** by
reading the words. A financial report simply *looks* different from a manual or a
patent, and the system learns to recognise those visual "fingerprints".

## How does it learn?

The system learns the same way a person might learn to recognise document types
at a glance: **by seeing many examples.**

We use a public collection of real document pages called **DocLayNet** and focus
on the German-language pages within it. The computer is shown thousands of these
pages along with the correct answer for each. Over time, it picks up on the
patterns that distinguish one category from another. After this "training", it can
make predictions on pages it has never seen before.

We actually built **two versions** and compared them:

1. **A model built entirely from scratch** — a simple baseline to set a reference point.
2. **A model that reuses knowledge from a much larger, pre-trained system** (a
   well-known image-recognition model called ResNet50). This is like hiring
   someone who already understands images in general and only needs to be taught
   the specifics of document types — it tends to learn faster and perform better.

## How do we know it works?

We don't just trust the system — we measure it. After training, we test it on
documents it has never seen and check:

- **How often it's correct overall.**
- **Whether it's reliable for every category**, not just the easy ones.
- **Where it gets confused** — for example, if it sometimes mixes up two
  similar-looking categories.
- **How well it holds up on imperfect inputs** — we simulate messy real-world
  conditions like a blurry or skewed phone-camera photo of a page, to see if it
  still copes.

All of these checks are turned into easy-to-read charts and tables.

## Where can it be used?

The trained system is wrapped in a small **web service**: you upload a document
image in your browser, and it instantly shows the predicted category along with
how confident it is. This makes it practical to plug into a real document-handling
workflow — for example, automatically routing incoming scans to the right
department.

## Limitations (in plain terms)

- It only recognises the **six categories** above. A document outside these types
  will still be forced into the closest-looking one.
- It was trained on **German** documents specifically.
- It judges by **appearance only** — two very different documents that happen to
  share a similar layout can occasionally be confused.

## The takeaway

This project shows that a computer can learn to **sort German documents by type
just by looking at how each page is laid out**, quickly and at scale — turning a
tedious manual task into an instant, automatic one.
