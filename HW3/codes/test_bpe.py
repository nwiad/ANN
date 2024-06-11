from tokenizer import get_tokenizer

sentences = [
    "The quick brown fox jumps over the lazy dog.",
    "She sells seashells by the seashore.",
    "Peter Piper picked a peck of pickled peppers.",
    "How much wood would a woodchuck chuck if a woodchuck could chuck wood?"
]

for sentence in sentences:
    print(sentence)
    print("BPE tokenizer:", get_tokenizer("./tokenizer").tokenize(sentence))
    # spplit by space
    print("Split by space:", sentence.split())
    print()