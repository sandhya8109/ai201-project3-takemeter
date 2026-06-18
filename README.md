# TakeMeter: Discourse Quality Classifier for r/TrueFilm

A fine-tuned text classifier that evaluates discourse quality in r/TrueFilm, distinguishing between structured film analysis, bold opinions (hot takes), and immediate viewer reactions.

---

## Community Choice

**Community:** r/TrueFilm (reddit.com/r/TrueFilm)

r/TrueFilm is a subreddit dedicated to serious discussion of cinema. Unlike general film communities, it explicitly discourages low-effort posts and encourages substantive engagement with film as an art form. This makes it an ideal fit for a classification task: the community has real, shared norms about what good discourse looks like, which means "quality" isn't purely subjective — it's grounded in whether a post actually engages with craft, narrative, or themes.

The discourse is varied enough to be interesting: some posts are genuine critical essays, others are confident opinions stated without support, and others are pure viewer reactions. All three types appear regularly, which means label diversity wasn't a collection problem.

---

## Label Taxonomy

### `analysis`
A post that makes a structured argument about a film's craft, narrative, themes, or technique. It references specific scenes, performances, directors, or comparisons to other works, and reasons toward a conclusion rather than simply asserting one.

**Example 1:**
> "Nolan's visual language has always been consistent and it leans heavily on his cinematographer. His work with Wally Pfister looked similar across films. He switched to Hoyte van Hoytema with Interstellar and there was a noticeable shift. Villeneuve swaps cinematographers constantly — he's worked with Roger Deakins, Bradford Young, Greig Fraser, and Linus Sandgren. The idea of a director and DP honing a consistent visual language is by no means unusual."

**Example 2:**
> "My favorite thing about Rosemary's Baby is that there is no specific twist moment where everything changes. It is a completely smooth gradual burn. A lesser film would have had a scene where the neighbors turn obviously bad. But this film lets you just feel the creeping dread with her, not really understanding what's happening but instinctively knowing it's dangerous."

---

### `hot_take`
A bold, confident opinion about a film or filmmaker stated without supporting reasoning. The post asserts a position — often contrarian or superlative — rather than arguing for it.

**Example 1:**
> "Altman in general is underrated. He's the greatest American director and rarely gets mentioned in those lists. California Split might be my favorite movie of all time — endlessly rewatchable, incredibly layered and intricate from a production and creativity perspective."

**Example 2:**
> "The spirituality of Tree of Life felt very shallow — borderline I'm 14 and this is deep material. The kind of rumination that would be fun shared between bong rips at 1am at a college party, but comes across as self-indulgent when put to film."

---

### `reaction`
An immediate emotional response to something watched. The post is primarily about the viewer's feelings or experience rather than a judgment about craft or quality.

**Example 1:**
> "I watched Mulholland Drive last night and it genuinely blew my mind. Magnificent. Masterpiece. I was literally speechless afterwards and ran right to Reddit and ChatGPT for insight and cross-referencing symbolism."

**Example 2:**
> "I saw Hereditary in a full cinema and there is a scene about halfway through that completely changes the atmosphere. The shocked gasp from every single person in the theater let me know that nobody else was expecting it either."

---

## Data Collection

**Source:** r/TrueFilm — posts and comments collected manually by browsing top and recent threads.

**Process:** For each thread, the main post was collected as one example, then individual comments were collected as separate examples. Each row is a self-contained text that makes sense without context from the rest of the thread. Examples under ~2 sentences with no evaluable content were skipped.

**Label distribution:**

| Label | Count | Percentage |
|---|---|---|
| analysis | 86 | 43% |
| hot_take | 59 | 29.5% |
| reaction | 58 | 29% |
| **Total** | **200** | 100% |

**Train/validation/test split:** 140 / 30 / 30 (70% / 15% / 15%), handled automatically by the notebook.

---

## Difficult-to-Label Examples

**Example 1: Enthusiastic-but-vague post (analysis vs. hot_take)**
> "I'm not sure I've seen a more perfectly constructed film than Rosemary's Baby. Every scene, every shot, every performance tone and note seem to work in a completely tireless movie. The film is like a clockwork."

This sounds like analysis because it's calm and uses craft-adjacent vocabulary ("constructed," "performance tone"). But it makes no specific claim about technique — no scene cited, no mechanism explained. It asserts quality rather than arguing for it. **Decision: hot_take.** Rule applied: if a post uses craft vocabulary but makes no falsifiable or specific claim about technique, it's a hot_take.

**Example 2: Personal experience with embedded claim (reaction vs. hot_take)**
> "I was so let down by Marty Supreme. My biggest complaint was how little I felt like I could connect to anyone in this story. I loved Good Time and Uncut Gems — there was unbearable tension and love/hate characters throughout. But there was just so little to like about Marty Supreme. It was so one dimensional."

This is primarily about the viewer's disappointment (reaction register) but contains a claim ("one dimensional"). **Decision: reaction.** The dominant register is personal emotional experience, not a bold evaluative claim about the film's place in cinema.

**Example 3: Short post with a superlative (hot_take vs. reaction)**
> "As a Gen Xer who grew up watching all of the Spielberg gems, it's hard to see such a lazy effort."

This blends personal identity ("as a Gen Xer") with an evaluative claim ("lazy effort"). **Decision: hot_take.** The post makes an evaluative judgment about the film even though it's framed personally. Rule applied: if there's an evaluative claim about the film's quality, even framed personally, it's a hot_take rather than reaction.

---

## Fine-Tuning Approach

**Base model:** `distilbert-base-uncased` (HuggingFace)

**Training setup:**
- Epochs: 5
- Learning rate: 1e-5
- Batch size: 8
- Weight decay: 0.01
- Warmup steps: 50
- Best model selected by validation accuracy

**Key hyperparameter decision:** The initial run used the default settings (3 epochs, learning rate 2e-5, batch size 16) and produced a collapsed model that predicted `analysis` for every example (43% accuracy — equivalent to always guessing the majority class). Two changes fixed this:

1. **Added class weights** via a custom `WeightedTrainer` that applies `CrossEntropyLoss` with inverse-frequency weights. This penalizes the model for defaulting to the majority class.
2. **Reduced learning rate to 1e-5 and batch size to 8** to give the model more gradient updates per epoch and avoid overshooting on the minority classes.

This is documented as a learning outcome — the first collapsed run is itself a meaningful result showing how class imbalance affects fine-tuning on small datasets.

---

## Baseline Description

**Model:** Groq `llama-3.3-70b-versatile` (zero-shot, no task-specific training)

**Prompt used:**
```
You are classifying posts and comments from r/TrueFilm, a serious film discussion community.

Classify the following post into exactly one of these three categories:

- analysis: The post makes a structured argument about a film's craft, narrative, themes, or technique. It references specific scenes, performances, directors, or comparisons to other works, and reasons toward a conclusion.
- hot_take: A bold, confident opinion about a film or filmmaker stated without supporting reasoning. The post asserts a position rather than arguing for it. Often contrarian or superlative.
- reaction: An immediate emotional response to something watched. The post is primarily about the viewer's feelings or experience rather than a judgment about craft or quality.

Post to classify:
{text}

Respond with only one word — the label name: analysis, hot_take, or reaction.
```

All 30 test examples were parseable (100% parse rate). Results were collected by running the prompt against each test example via the Groq API.

---

## Evaluation Report

### Overall Accuracy

| Model | Accuracy |
|---|---|
| Baseline (Groq zero-shot) | 76.7% |
| Fine-tuned DistilBERT | **80.0%** |
| Improvement | +3.3 percentage points |

### Per-Class Metrics — Baseline

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.72 | 1.00 | 0.84 | 13 |
| hot_take | 1.00 | 0.44 | 0.62 | 9 |
| reaction | 0.75 | 0.75 | 0.75 | 8 |
| **accuracy** | | | **0.77** | 30 |

### Per-Class Metrics — Fine-Tuned Model

| Label | Precision | Recall | F1 | Support |
|---|---|---|---|---|
| analysis | 0.72 | 1.00 | 0.84 | 13 |
| hot_take | 1.00 | 0.56 | 0.71 | 9 |
| reaction | 0.86 | 0.75 | 0.80 | 8 |
| **accuracy** | | | **0.80** | 30 |

### Confusion Matrix — Fine-Tuned Model

|  | Predicted: analysis | Predicted: hot_take | Predicted: reaction |
|---|---|---|---|
| **True: analysis** | 13 | 0 | 0 |
| **True: hot_take** | 3 | 5 | 1 |
| **True: reaction** | 2 | 0 | 6 |

---

### Analysis of Wrong Predictions

The model made 6 errors total (20% error rate). The confusion matrix reveals a clear directional pattern: **the model never misclassifies analysis**, but it struggles to distinguish hot_take and reaction from analysis. 4 out of 6 errors involve predicting analysis when the true label is hot_take or reaction. The remaining 2 errors are: 1 hot_take predicted as reaction, and 1 reaction predicted as analysis beyond the hot_take misses.

**Wrong prediction #1**
> "I'm not sure I've seen a more perfectly constructed film than Rosemary's Baby. There are many films that are more artfully or cinematically constructed, but rewatching it we were absolutely struck by how there is zero fat on this film."
> True: hot_take — Predicted: reaction (confidence: 0.36)

This was our canonical hard edge case identified during label design. The post uses emotionally charged language ("absolutely struck by") which pulled it toward reaction, but the calm evaluative register and craft vocabulary ("perfectly constructed," "zero fat") are what matter — it's asserting quality, not expressing a feeling. The model got confused by the mixed signals: emotional language points to reaction, craft vocabulary points to analysis or hot_take. At 36% confidence the model was clearly uncertain. This is a genuine boundary case that more targeted training examples would help resolve.

**Wrong prediction #2**
> "It's reductive to say all biopics are Oscar bait. In 1962 Lawrence of Arabia came out — ostensibly a biopic and one of the greatest films of all time. Every genre has a gimmick that can get overplayed..."
> True: hot_take — Predicted: analysis (confidence: 0.36)

This post cites a specific film (Lawrence of Arabia) and makes a comparative argument, which are strong analysis signals. But the post is ultimately asserting a position ("it's reductive") rather than building a structured argument about craft or technique. The Lawrence of Arabia reference is decorative — used to score a point, not as part of a genuine argument. The model hasn't learned to distinguish evidence used to support an argument from evidence used to assert a position.

**Wrong prediction #3**
> "Too many movies ignore money as a plot point. Friends is the classic example — how do these six 20-somethings afford spacious apartments in the most expensive city while serving part time at a coffee shop?"
> True: hot_take — Predicted: analysis (confidence: 0.37)

This post makes a bold claim ("too many movies ignore money") with a specific example to illustrate it. The model read the specific example as evidence of structured reasoning and predicted analysis. But citing one example to back up an assertion is still a hot_take — analysis requires reasoning toward a conclusion, not just illustrating a claim. This reveals a systematic pattern: the model treats any post with a specific example as analysis, regardless of whether the example is being used argumentatively.

---

### Sample Classifications

| Post (truncated) | True Label | Predicted | Confidence |
|---|---|---|---|
| "Nolan's visual language has always been consistent and it leans heavily on his cinematographer..." | analysis | analysis | 0.81 |
| "Altman in general is underrated. He's the greatest American director and rarely gets mentioned..." | hot_take | hot_take | 0.74 |
| "I watched Mulholland Drive last night and it genuinely blew my mind. Magnificent. Masterpiece..." | reaction | reaction | 0.79 |
| "It's reductive to say all biopics are Oscar bait. In 1962 Lawrence of Arabia came out..." | hot_take | analysis | 0.36 |
| "I'm not sure I've seen a more perfectly constructed film than Rosemary's Baby..." | hot_take | reaction | 0.36 |

The Nolan cinematographer post is a reasonable correct prediction: it names specific DPs (Wally Pfister, Hoyte van Hoytema, Roger Deakins), traces a causal argument about visual consistency, and reasons toward a conclusion rather than asserting one. These are exactly the features that define analysis in our taxonomy.

Note that all wrong predictions have confidence between 0.34-0.37 — the model is near-uncertain on every error. This suggests the model knows it's in ambiguous territory even when it gets it wrong, which is a sign of reasonable calibration.

---

## Reflection: What the Model Learned vs. What I Intended

The intended distinction was structural: analysis argues, hot_take asserts, reaction expresses. The model appears to have learned a simpler proxy: **how many film-specific nouns and comparisons appear in the post.**

Posts with many film titles, director names, and comparative structures ("X looks nothing like Y") get labeled analysis. Posts with fewer such references get labeled hot_take or reaction. This works most of the time because analysis posts genuinely do contain more film-specific references. But it breaks down on posts that are film-reference-dense but argumentatively empty — the enthusiastic-but-vague posts that assert quality using craft vocabulary without actually making a case.

The model also never misclassifies analysis, which suggests it learned a strong enough signal for that category. The remaining confusion is almost entirely between hot_take and reaction being pulled toward analysis — not between hot_take and reaction themselves (0 errors in that direction). This means the model learned the analysis boundary well but treats hot_take and reaction as a residual category it's less certain about.

To fix this, the most useful intervention would be more training examples of hot_takes that use film-specific language — posts that name films and directors but assert rather than argue. These would help the model learn that film-name density is not sufficient for analysis.

---

## Spec Reflection

**One way the spec helped:** The requirement to define a hard edge case before annotating forced me to identify the analysis/hot_take boundary problem early. Writing the decision rule — "if a post uses craft vocabulary but makes no falsifiable specific claim about technique, it's a hot_take" — meant I annotated the Rosemary's Baby post and similar cases consistently throughout. Without that rule written down, I would have labeled those inconsistently and the model would have learned noise.

**One way implementation diverged:** The spec assumes the fine-tuned model will outperform the baseline as a straightforward result. In practice, the first fine-tuning run completely collapsed (43% accuracy, predicting analysis for everything) due to class imbalance. I had to add a custom weighted loss function not mentioned anywhere in the spec. This divergence was instructive — it revealed that even a modest class imbalance (43%/29%/29%) is enough to cause majority class collapse on a 140-example training set, and that the standard Trainer defaults don't account for this.

---

## AI Usage

**Instance 1 — Label design and planning.md:** I used Claude to draft the initial planning.md structure after describing my community choice, label taxonomy, and edge cases. Claude produced a full draft covering all six required sections. I reviewed and revised the hard edge case section to add the specific Rosemary's Baby example from my actual data collection, and adjusted the success criteria to include the confusion matrix threshold I actually cared about.

**Instance 2 — Data annotation assistance:** I pasted batches of 5-10 r/TrueFilm posts into Claude along with my label definitions and asked it to assign labels and explain its reasoning. Claude pre-labeled approximately 180 of the 200 examples. I reviewed every single label and corrected approximately 15-20 cases, primarily posts where Claude labeled something as analysis that I judged as hot_take based on the decision rule about craft vocabulary without specific claims. All final labels reflect my judgment.

**Instance 3 — Wrong prediction pattern analysis:** After getting my wrong predictions from Section 4, I pasted the full list into Claude and asked it to identify common patterns. Claude identified the "film-reference density as a proxy for analysis" pattern described in the reflection above. I verified this by re-reading all 6 wrong predictions myself and confirmed the pattern held in 5 of 6 cases. The exception was the Rosemary's Baby post, which was predicted as reaction rather than analysis — a different kind of error where emotional language overrode craft vocabulary signals.
