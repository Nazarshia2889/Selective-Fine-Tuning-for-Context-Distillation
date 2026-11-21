"""Knowledge retrieval: context describes 10 fictional people with 5 facts each.

Train questions use one set of phrasings/frames, val+test use disjoint phrasings/frames, so test
wordings are never seen in training. 50 facts x 2 phrasings x 5 frames = 500 train questions;
50 x 2 x 3 = 300 held-out questions -> 200 test + 100 val (defaults).
"""
from __future__ import annotations

import itertools
import re

from sfcd.config import DataCfg
from sfcd.tasks.base import Context, Example, Splits, Task, split_unique

_SYL = ["ka", "lo", "mir", "ten", "vas", "dra", "quo", "bel", "fen", "zar", "ori", "pex",
        "lun", "sho", "tri", "gav", "nel", "rho", "dun", "yel", "cor", "vim", "hal", "ost"]
_TOWN_SUFFIX = ["holt", "mere", "brook", "vale", "stead", "ford", "wick", "crest"]
PROFESSIONS = ["cartographer", "beekeeper", "glassblower", "astronomer", "locksmith", "violinist",
               "botanist", "stonemason", "archivist", "perfumer", "falconer", "clockmaker",
               "translator", "puppeteer", "shipwright", "cheesemaker"]
COLORS = ["crimson", "teal", "amber", "lavender", "olive", "magenta", "cobalt", "ochre",
          "turquoise", "maroon", "saffron", "indigo", "coral", "charcoal", "jade", "mauve"]
SPECIES = ["parrot", "tortoise", "ferret", "goldfish", "hedgehog", "rabbit", "iguana", "cat", "dog"]

ATTR_QUESTIONS = {  # attribute -> (train phrasings, held-out phrasings)
    "hometown": (["In which town was {n} born?", "What is {n}'s hometown?"],
                 ["Where does {n} come from originally?", "Name the town where {n} was born."]),
    "profession": (["What does {n} do for a living?", "What is {n}'s job?"],
                   ["What is {n}'s occupation?", "Which profession does {n} have?"]),
    "color": (["What is {n}'s favorite color?", "Which color does {n} like most?"],
              ["Which colour is {n} most fond of?", "Name {n}'s preferred color."]),
    "pet": (["What is the name of {n}'s pet?", "What is {n}'s pet called?"],
            ["How is {n}'s animal companion named?", "Give the name of the pet that {n} owns."]),
    "year": (["In what year was {n} born?", "What is {n}'s birth year?"],
             ["When was {n} born? Give the year.", "Which year marks {n}'s birth?"]),
}
FRAMES_TRAIN = ["{q}", "Question: {q}", "{q} Answer briefly.", "Quick question: {q}", "I'd like to know: {q}"]
FRAMES_HELDOUT = ["Please tell me: {q}", "{q} Reply concisely.", "Here is a question for you. {q}"]


def _word(rng, n_syl: int) -> str:
    return "".join(rng.choice(_SYL) for _ in range(n_syl)).capitalize()


class KnowledgeTask(Task):
    name = "knowledge"
    system_prompt = "You are a helpful assistant. Answer the question with a short phrase."
    max_new_tokens = 24
    n_people = 10

    def build(self, context_id: int, cfg: DataCfg) -> tuple[Context, Splits]:
        rng = self.rng(context_id, cfg)
        names: set[str] = set()
        while len(names) < self.n_people:
            names.add(f"{_word(rng, 2)} {_word(rng, 3)}")
        people = []
        used_pets: set[str] = set()
        for n in sorted(names):
            pet = _word(rng, 2)
            while pet in used_pets:
                pet = _word(rng, 2)
            used_pets.add(pet)
            people.append({
                "name": n,
                "hometown": _word(rng, 2) + rng.choice(_TOWN_SUFFIX),
                "profession": rng.choice(PROFESSIONS),
                "color": rng.choice(COLORS),
                "species": rng.choice(SPECIES),
                "pet": pet,
                "year": str(rng.randint(1890, 2005)),
            })
        rng.shuffle(people)
        lines = ["Here are facts about some people."]
        for p in people:
            lines.append(
                f"{p['name']} was born in {p['year']} in the town of {p['hometown']}. "
                f"{p['name']} works as a {p['profession']}. {p['name']}'s favorite color is {p['color']}. "
                f"{p['name']} has a {p['species']} named {p['pet']}."
            )
        context = Context(self.name, context_id, "\n".join(lines), meta={"people": people})

        def make(phrase_idx: int, frames: list[str]) -> list[Example]:
            out = []
            for p, (attr, phr) in itertools.product(people, ATTR_QUESTIONS.items()):
                for q, frame in itertools.product(phr[phrase_idx], frames):
                    out.append(Example(frame.format(q=q.format(n=p["name"])), p[attr],
                                       meta={"person": p["name"], "attr": attr}))
            return out

        train_pool = make(0, FRAMES_TRAIN)
        held_pool = make(1, FRAMES_HELDOUT)
        (train,) = split_unique(train_pool, [cfg.n_train], rng)
        test, val = split_unique(held_pool, [cfg.n_test, cfg.n_val], rng)
        return context, {"train": train, "val": val, "test": test}

    def is_correct(self, prediction: str, example: Example) -> bool:
        norm = lambda s: re.sub(r"\s+", " ", s.lower()).strip()
        return norm(example.answer) in norm(prediction)
