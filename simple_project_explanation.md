# GlossWeaver in plain English

GlossWeaver takes text that looks like `ME GO STORE YESTERDAY` and tries to
produce fluent English such as `I went to the store yesterday.` It starts after
sign recognition, so it never reads video or tracks hands.

The baseline fine-tunes a small T5 text model. The proposed version also asks
the shared encoder to predict which kinds of English grammar need to be
recovered—articles, prepositions, auxiliaries, pronouns, tense/aspect, and
inflection. That extra task is used only while training. At inference, the
system receives the gloss text alone.

A second technique creates reproducible telegraphic fragments from training
sentences. These fragments are explicitly synthetic and are never described as
real ASL. The experiment tests the two techniques separately and together.

The main experimental dataset is ASLG-PC12. It is publicly accessible and
contains aligned gloss and English text, but the gloss side was generated with
rules rather than collected as fully natural human-produced ASL. The project
uses it for controlled model comparisons and states that limitation clearly.
