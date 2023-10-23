from typing import List, NamedTuple
from collections import defaultdict
import torch
from pyctcdecode import build_ctcdecoder
from string import ascii_lowercase
import kenlm

from .char_text_encoder import CharTextEncoder


class Hypothesis(NamedTuple):
    text: str
    prob: float


class CTCCharTextEncoder(CharTextEncoder):
    EMPTY_TOK = "^"

    def __init__(self, alphabet: List[str] = None):
        super().__init__(alphabet)
        vocab = [self.EMPTY_TOK] + list(self.alphabet)
        self.ind2char = dict(enumerate(vocab))
        self.char2ind = {v: k for k, v in self.ind2char.items()}

        vocab_lm = ascii_lowercase + ' '
        self.kenlm = 'lowercase_3-gram.pruned.1e-7.arpa'
        self.beam_search = build_ctcdecoder([""] + [i for i in vocab_lm], kenlm_model_path=self.kenlm, alpha=0.5, beta=0.1)

    def ctc_decode(self, inds: List[int]) -> str:

        last_char = None
        result = []

        for ind in inds:
            ind = ind.item() if torch.is_tensor(ind) else ind
            if ind == last_char:
                continue
            if ind != self.char2ind[self.EMPTY_TOK]:
                result.append(self.ind2char[ind])
            last_char = ind

        return ''.join(result)

    def ctc_beam_search(self, probs: torch.tensor,
                        beam_size: int = 100) -> List[Hypothesis]:
        """
        Performs beam search and returns a list of pairs (hypothesis, hypothesis probability).
        """
        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        state = {('', self.EMPTY_TOK): 1.0}
        for frame in probs:
            state = self.extend_and_merge(state, frame)
            state = self.truncate(state, beam_size)

        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])

        return state_list[0][0][0]

    def extend_and_merge(self, state, frame):
        new_state = defaultdict(float)
        for ind_proba, proba in enumerate(frame):
            prob = self.ind2char[ind_proba]
            for (pref, last_char), state_prob in state.items():
                prefix = pref if prob == last_char else (pref + prob).replace(self.EMPTY_TOK, "")
                new_state[(prefix, prob)] += proba * state_prob

        return new_state

    def truncate(self, state, beam_size):
        state_list = list(state.items())
        state_list.sort(key=lambda x: -x[1])
        return dict(state_list[:beam_size])

    def ctc_beam_search_from_liba(self, probs: torch.tensor, probs_length, beam_size: int = 100) -> List[Hypothesis]:

        assert len(probs.shape) == 2
        char_length, voc_size = probs.shape
        assert voc_size == len(self.ind2char)

        hypos: List[Hypothesis] = []
        decoders = self.beam_search.decode_beams(probs[:probs_length, :], beam_size)
        for item in decoders:
            hypos.append(Hypothesis(item[0], item[-1]))

        return sorted(hypos, key=lambda x: x.prob, reverse=True)
