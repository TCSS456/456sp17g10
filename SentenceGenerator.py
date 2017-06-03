import markovify
import nltk
import re

BEGIN = "___BEGIN__"
END = "___END__"

class SentenceGenerator(markovify.Text):
  def word_split(self, sentence):
    words = re.split(self.word_split_pattern, sentence)
    words = [w for w in words if len(w) > 0]
    words = [ "::".join(tag) for tag in nltk.pos_tag(words)]
    return words

  def word_join(self, words):
    sentence = " ".join(word.split("::")[0] for word in words)
    return sentence

class CustomChain(markovify.Chain):
  def precompute_begin_state(self):
    """
    Caches the summation calculation and available choices for BEGIN * state_size.
    Significantly speeds up chain generation on large corpuses. Thanks, @schollz!
    """
    begin_state = tuple([ BEGIN ] * self.state_size)
    choices, weights = zip(*self.model[begin_state].items())
    cumdist = list(accumulate(weights))
    self.begin_cumdist = cumdist
    self.begin_choices = choices

  def move(self, state):
    """
    Given a state, choose the next item at random.
    """
    if state == tuple([ BEGIN ] * self.state_size):
        choices = self.begin_choices
        cumdist = self.begin_cumdist
    else:
        choices, weights = zip(*self.model[state].items())
        cumdist = list(accumulate(weights))
    print(weights);
    r = random.random() * cumdist[-1]
    selection = choices[bisect.bisect(cumdist, r)]
    return selection