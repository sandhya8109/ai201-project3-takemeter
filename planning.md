# TakeMeter – planning.md
## Project 3: Fine-Tuned Discourse Classifier

---

## 1. Community

**Chosen community:** r/TrueFilm (reddit.com/r/TrueFilm)

r/TrueFilm is a subreddit dedicated to serious discussion of cinema. Unlike general film communities, it explicitly discourages low-effort posts and encourages substantive engagement with film as an art form. This makes it a strong fit for a classification task: the community has real, shared norms about what good discourse looks like, which means "quality" isn't purely subjective here — it's grounded in whether a post actually engages with craft, narrative, or themes.

The discourse is varied enough to be interesting: some posts are genuine critical essays, others are confident opinions stated without support, and others are pure viewer reactions. All three types appear regularly, which means label diversity won't be a collection problem.

---

## 2. Label Taxonomy

### `analysis`
**Definition:** The post makes a structured argument about a film's craft, narrative, themes, or technique. It references specific scenes, cinematographic choices, performances, or comparisons to other works, and reasons toward a conclusion rather than simply asserting one.

**Key signal:** "because," citations of specific scenes or shots, craft vocabulary (mise-en-scène, pacing, framing), comparisons to other films or directors, a thesis that is argued rather than declared.

**Example 1:**
> "Barry Lyndon's treatment of male honour is paradoxical — the opening duel is played for laughs to mock masculine pride, but Lord Bullington's arc reframes honour as something that can be both destructive and necessary. Kubrick isn't simply satirising these values; he's showing how they become tragic when misapplied."

**Example 2:**
> "Kubrick's one-point perspective in The Shining isn't just a stylistic choice — it creates a power dynamic between viewer and subject that shifts as Jack's sanity deteriorates. Early scenes use it to make the Overlook feel ordered; later it makes Jack feel like a predator in a trap."

---

### `hot_take`
**Definition:** A bold, confident opinion about a film or filmmaker stated without supporting reasoning. The post asserts a position — often contrarian or superlative — rather than arguing for it. The claim might be defensible, but the post doesn't defend it.

**Key signal:** Superlatives ("best," "worst," "most overrated"), declarative sentences with no evidence, contrarian framing ("unpopular opinion," "fight me," "and I'm tired of pretending otherwise"), enthusiasm without specifics.

**Example 1:**
> "Rosemary's Baby is the most perfectly constructed film ever made. Zero fat. Every scene works exactly as it should. It's a clockwork. Total masterpiece."

**Example 2:**
> "Scorsese hasn't made a truly great film since Goodfellas and everyone knows it. The Irishman is three and a half hours of self-indulgence. Unpopular opinion but someone had to say it."

---

### `reaction`
**Definition:** An immediate emotional response to something watched. The post is primarily about the viewer's feelings or experience rather than a judgment about the work's quality or craft. Little to no argument — the post expresses a feeling in the moment.

**Key signal:** "just finished," "I can't believe," exclamation marks, plot summary with emotional commentary, "WOW," "MUST-SEE," first-person emotional language without claims about the film itself.

**Example 1:**
> "Just finished watching Obsession and WOW. I literally jumped out of my seat three times. The lead performance is insane — I cannot stop thinking about it. Absolute must-see, go in blind!"

**Example 2:**
> "Finally watched Mulholland Drive last night and I have no idea what I just watched but I am OBSESSED. My brain is completely broken. How has it taken me this long to see this film??"

---

## 3. Hard Edge Cases

### The enthusiastic-but-vague post (analysis vs. hot_take)

**Example:**
> "I'm not sure I've seen a more perfectly constructed film than Rosemary's Baby. Every scene, every shot, every performance tone and note seem to work in a completely tireless movie that spends the right amount of time and emphasis required, beat by beat. The film is like a clockwork."

This *feels* like analysis because it's calm and specific-sounding. But the observations are impressionistic ("zero fat," "clockwork," "ease") rather than technical — it doesn't explain *why* any specific scene works or *how* the construction achieves its effect. It asserts quality rather than arguing for it.

**Decision rule:** If a post uses craft vocabulary but makes no falsifiable or specific claim about technique — no scene cited, no mechanism explained — label it `hot_take`. The test is: could someone disagree with this post and point to a specific counter-example? If the post gives them nothing to disagree with, it's a hot_take.

### The plot-summary-with-opinion post (reaction vs. hot_take)

**Example:**
> "Just watched The Godfather for the first time. The transformation of Michael Corleone is the best character arc in cinema history. Absolutely floored."

This has a bold claim ("best character arc in cinema history") embedded in a reaction post.

**Decision rule:** If the dominant register of the post is emotional/experiential ("just watched," "absolutely floored") and the claim is a single throwaway superlative rather than the point of the post, label it `reaction`. If the post leads with the claim and the emotion is secondary, label it `hot_take`.

### The short post problem

Posts under ~30 words are genuinely hard because there's not enough signal. 

**Decision rule:** For very short posts, lean toward `hot_take` if there's any evaluative claim, and `reaction` if there's no claim at all — just feeling.

---

## 4. Data Collection Plan

**Source:** r/TrueFilm — top posts and comments from the past year, collected manually by browsing and copy-pasting into a CSV.

**Target distribution:**
- `analysis`: ~80 examples (40%)
- `hot_take`: ~70 examples (35%)
- `reaction`: ~50 examples (25%)

r/TrueFilm skews toward analysis, so hot_takes and reactions may require more active hunting — specifically in comment sections under popular posts, where shorter and more emotional responses appear more often.

**If a label is underrepresented after 150 examples:** Specifically search for posts containing "unpopular opinion," "just finished," "change my mind," or "overrated" to surface more hot_takes and reactions.

**CSV columns:**
- `text` — the post or comment text
- `label` — one of: analysis, hot_take, reaction
- `notes` — free-text notes on difficult cases (optional, will not be used in training)

**Balance check:** After labeling, if any single label exceeds 70% of the dataset, collect more examples from the underrepresented labels before training.

---

## 5. Evaluation Metrics

**Primary metric: per-class F1 score**

Accuracy alone is misleading here because the dataset may be somewhat imbalanced (analysis will likely be the majority class on r/TrueFilm). A model that predicts `analysis` for everything would get inflated accuracy. F1 accounts for both precision and recall per class, which tells us whether the model is actually learning each distinction.

**Secondary metrics:**
- **Confusion matrix** — to identify which label pairs are being confused and in which direction
- **Overall accuracy** — for direct comparison against the zero-shot baseline

**Why not just accuracy:** If 50% of the test set is `analysis`, a model that always predicts `analysis` gets 50% accuracy. Per-class F1 exposes this failure mode immediately.

---

## 6. Definition of Success

**Minimum bar for "it works":**
- Fine-tuned model accuracy exceeds zero-shot baseline by at least 10 percentage points
- No single class has F1 below 0.50 — the model should be learning all three distinctions, not just the easy ones
- The confusion matrix shows no single off-diagonal cell dominating (i.e., no single label pair accounts for more than 40% of all errors)

**Bar for "good enough to deploy":**
- Overall accuracy ≥ 0.75 on the test set
- All per-class F1 scores ≥ 0.65
- The model correctly handles the hard edge case (enthusiastic-but-vague posts) more than 60% of the time

If the fine-tuned model doesn't beat the baseline by at least 10 points, I'll treat that as a signal to investigate label consistency before calling the project done.

---

## 7. AI Tool Plan

### Label stress-testing
I will paste my label definitions and edge case descriptions into Claude and ask it to generate 10 posts that sit at the boundary between `analysis` and `hot_take` — the hardest boundary in this taxonomy. If it produces posts I can't cleanly classify, I'll tighten the definitions before annotating 200 examples. This step happens before data collection.

### Annotation assistance
I may use an LLM to pre-label batches of 20–30 examples at a time by providing my label definitions and asking for one label per post. If I do this, I will review and correct every pre-assigned label individually — no bulk acceptance. I will track which examples were pre-labeled in the `notes` column of my CSV and disclose this in the README's AI usage section.

### Failure analysis
After getting my wrong predictions from the notebook, I will paste the full list of misclassified examples into Claude and ask it to identify common patterns — similar length, sarcasm, specific label pairs, low-information posts, etc. I will then verify each suggested pattern by re-reading the examples myself before including it in the evaluation report. I will note which patterns I confirmed and which I discarded.

---

## Hard Annotation Decisions (updated during Milestone 3)

*This section will be filled in during data collection. At least 3 specific difficult examples and my labeling decisions will be documented here.*

---

## Stretch Feature Plans

*This section will be updated before starting any stretch features.*
