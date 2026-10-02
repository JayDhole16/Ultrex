"""The English layer: a small local language model that can take a seat at the negotiating table.

The trained agents bargain in numbers. This module lets a visitor name a topic — who gets the bigger
office, how to split the prize money — and puts a language model in one seat so the exchange reads as an
argument while it is still the same game: the model's words are turned into a real offer, and the RL agent
on the other side answers it with a real counter-offer.

Everything runs on this machine. The default model is about 1 GB and takes a few seconds a turn on a CPU;
the first use downloads it to the usual Hugging Face cache. No key, no network after that.

Honesty matters more here than polish, so these are built in rather than bolted on:

  * The price is a choice, not a parse. A small model asked politely for "KEEP: 60" obeys about half the
    time, and the number it does produce is close to noise. So after it has argued its corner it is handed a
    short menu — demand 40, 50, 60, 70, 80 or 90 of the points — and every option is scored by the model's
    own logits. The best-scoring one is its price for this round. Nothing can fail to parse, the number is
    always a real number, and it is still the model's own preference rather than something invented for it.
  * The price is scored after, and conditioned on, the model's own sentence, and the price alone decides
    the deal: it accepts anything already at or above what it just asked for, and counters with that number
    otherwise. A model this small can still sound warmer or harsher than the number it picks, so the page
    shows both, and the scores behind the number.
  * A final offer is a different question. When there is nothing left to counter with, the choice is between
    "accept" and "refuse", scored the same way. In testing it accepts whatever is on the table, which is
    also the rational answer when refusing pays nothing.

What to expect from the 0.5B default, measured over 30 turns: with nothing or a poor offer on the table it
mostly asks for half; offered 60 it mostly asks 70, offered 80 mostly 90. It anchors on the offer and pushes
a notch higher. A turn takes 3-5 seconds on a laptop CPU.
"""

from __future__ import annotations

import contextlib
import os
import re
import threading

DEFAULT_MODEL = "Qwen/Qwen2.5-0.5B-Instruct"
DEMANDS = (40, 50, 60, 70, 80, 90)  # the menu it chooses from, as a percentage of the pool
DEFAULT_THREADS = max(1, min(8, (os.cpu_count() or 2) // 2))  # 8.1 s a turn on one thread, 2.9 s on eight
SYSTEM = """You are a person on one side of a dispute about {topic}, not an assistant. There are {pool} points to share.
Speak in character: one short, persuasive sentence to the other side that gives a concrete reason why you deserve more.
Talk about {topic} itself, never about the rules, the rounds or the numbers.
Reply with exactly one line: SAY: <your sentence>"""
TURN = """{position}
Rounds left: {rounds_left}.
What do you say to them?"""


def _import_transformers():
    try:
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer
    except ImportError as error:
        return None, f"{error.name} is not installed (pip install transformers)"
    return (torch, AutoModelForCausalLM, AutoTokenizer), ""


def status(model=DEFAULT_MODEL):
    """Whether the English layer can run here, and what it would use."""
    parts, reason = _import_transformers()
    if parts is None:
        return {"available": False, "reason": reason}
    cached = False
    try:
        from huggingface_hub import try_to_load_from_cache

        cached = bool(try_to_load_from_cache(model, "config.json"))
    except Exception:
        pass
    return {
        "available": True,
        "model": model,
        "cached": cached,
        "description": f"{model}, running on this machine"
        + ("" if cached else " (about 1 GB downloads on first use)"),
    }


class Debater:
    """A local instruct model in one seat: it argues in English, and its words become a real offer."""

    def __init__(self, model=DEFAULT_MODEL, threads=DEFAULT_THREADS, max_sentence_tokens=42):
        self.model_name = model
        self.threads = threads
        self.max_sentence_tokens = max_sentence_tokens
        self.lock = threading.Lock()
        self._loading = threading.Lock()  # the page warms the model up while a turn may already want it
        self.tokenizer = None
        self.model = None
        self.error = None

    # ---- loading ----------------------------------------------------------------------------------

    @property
    def ready(self):
        return self.model is not None

    def load(self):
        """Load the model on first use; a demo should not pay for this until someone asks for it."""
        with self._loading:
            return self._load()

    def _load(self):
        if self.model is not None or self.error:
            return self.model is not None
        parts, reason = _import_transformers()
        if parts is None:
            self.error = reason
            return False
        torch, AutoModelForCausalLM, AutoTokenizer = parts
        try:
            self.torch = torch
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            try:
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, dtype=torch.bfloat16)
            except TypeError:  # transformers before the dtype rename
                self.model = AutoModelForCausalLM.from_pretrained(self.model_name, torch_dtype=torch.bfloat16)
            self.model.eval()
        except Exception as error:
            self.error = f"{type(error).__name__}: {error}"
            self.model = None
            return False
        return True

    # ---- speaking ---------------------------------------------------------------------------------

    def speak(self, topic, pool, position, rounds_left, keep_hint=0.5, final=False):
        """One turn: a sentence about the topic, and the model's move.

        Returns {"say": str, "keep": float in (0, 1), "accept": bool, "considered": [...]}. On a final offer
        (``final=True``) the move is accept-or-refuse and ``keep`` is not used.
        """
        if not self.load():
            return {"say": "", "keep": keep_hint, "accept": False, "considered": [], "error": self.error}
        with self.lock, self._threads():
            messages = [
                {"role": "system", "content": SYSTEM.format(topic=topic, pool=pool)},
                {"role": "user", "content": TURN.format(position=position, rounds_left=rounds_left)},
            ]
            prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
            sentence, context = self._sentence(prompt)
            choice = self._final(context) if final else self._choose(context)
        return {"say": sentence, **choice}

    @contextlib.contextmanager
    def _threads(self):
        """Use more cores for the language model only. Thread count is process-wide in torch, and the policy
        networks next to it are faster on one thread, so it goes back to what it was afterwards."""
        previous = self.torch.get_num_threads()
        if self.threads and self.threads != previous:
            self.torch.set_num_threads(self.threads)
        try:
            yield
        finally:
            if self.torch.get_num_threads() != previous:
                self.torch.set_num_threads(previous)

    def _sentence(self, prompt):
        torch = self.torch
        inputs = self.tokenizer(prompt, return_tensors="pt")
        with torch.no_grad():
            settings = dict(
                max_new_tokens=self.max_sentence_tokens,
                do_sample=True,
                temperature=0.8,
                top_p=0.9,
                pad_token_id=self.tokenizer.eos_token_id,
            )
            try:  # stop at the end of the first sentence rather than paying for a paragraph
                out = self.model.generate(**inputs, **settings, stop_strings=[".", "!", "?"], tokenizer=self.tokenizer)
            except (TypeError, ValueError):  # transformers without stop_strings
                out = self.model.generate(**inputs, **settings)
        text = self.tokenizer.decode(out[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True)
        sentence = text.strip().splitlines()[0] if text.strip() else ""
        sentence = re.sub(r"^\s*(SAY|Say|say)\s*:\s*", "", sentence).strip().strip('"')
        sentence = re.split(r"(?<=[.!?])\s", sentence, maxsplit=1)[0]
        return sentence[:200], prompt + f"SAY: {sentence}\nDECISION: I"

    def _choose(self, context):
        """Score every price on the menu and take the one the model likes best.

        The options share a shape and a length, so what is being compared is the numbers themselves rather
        than one phrasing against another.
        """
        scores = self._logprobs(context, [f" demand {level} of the points." for level in DEMANDS])
        best = max(range(len(DEMANDS)), key=scores.__getitem__)
        return {
            "keep": DEMANDS[best] / 100.0,
            "accept": False,
            "considered": [{"keep": level / 100.0, "score": round(score, 2)} for level, score in zip(DEMANDS, scores)],
        }

    def _final(self, context):
        """Take it or leave it: the two answers share a shape, so neither is favoured by its phrasing."""
        accept, refuse = self._logprobs(context, [" accept their final offer.", " refuse their final offer."])
        return {
            "keep": 0.5,
            "accept": accept > refuse,
            "considered": [{"option": "accept", "score": round(accept, 2)}, {"option": "refuse", "score": round(refuse, 2)}],
        }

    def _logprobs(self, context, continuations):
        """Mean log-probability of each continuation after the context: how much the model wants to say it.

        Continuations of equal token length go through in a single batched forward pass.
        """
        torch = self.torch
        start = self.tokenizer(context, return_tensors="pt").input_ids.shape[1]
        rows = [self.tokenizer(context + text, return_tensors="pt").input_ids[0] for text in continuations]
        if len({len(row) for row in rows}) == 1:
            batches = [torch.stack(rows)]
        else:
            batches = [row.unsqueeze(0) for row in rows]
        scores = []
        with torch.no_grad():
            for ids in batches:
                logits = self.model(ids).logits.float()
                logprobs = torch.log_softmax(logits[:, start - 1 : -1], dim=-1)
                picked = logprobs.gather(2, ids[:, start:].unsqueeze(2)).squeeze(2)
                scores.extend(float(row.mean()) if picked.shape[1] else float("-inf") for row in picked)
        return scores

def describe_position(pool, offered_tokens, can_accept, round_index, opponent_line, final=False):
    """The state of the negotiation, in the words the model is asked to respond to.

    The round number is left out on purpose: a small model repeats whatever bookkeeping it is shown, and the
    rounds left are already in the prompt.
    """
    lines = []
    if opponent_line:
        lines.append(f"They just said: {opponent_line}")
    if final:
        lines.append(
            f"Their final offer leaves you {offered_tokens} of {pool} points. There will be no more rounds: "
            "if you refuse, you both get nothing."
        )
    elif can_accept:
        lines.append(f"Their offer leaves you {offered_tokens} of {pool} points.")
    else:
        lines.append("Nothing is on the table yet; you speak first.")
    return " ".join(lines)
