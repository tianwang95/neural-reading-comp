"""
Attentive LSTM Reader Model
---------------------------

At a high level, this model reads both the story and the question forwards and backwards, and represents the document as a weighted sum of its token where each individual token weight is decided by an attention mechanism that reads the question.

At an implementation layer, we have two LSTM's that read the story (one forwards and one backwards) which then get merged by concatenation. These LSTM's return sequences so that the output of each timestep of this layer is an input to the next attention layer.

Similarly,  the question is read by two LSTMs that get concatenated, but this time we don't return sequences, we use the final output of each one.


