import numpy as np

def example_decoder(seq):
    vocab_size = 5  # 假设词汇表大小为5
    return np.random.dirichlet(np.ones(vocab_size))

def beam_search(decoder, input, beam_size=4, max_length=10):
    data = [(input, 0)]
    for _ in range(max_length):
        candidate = []
        for seq, score in data:
            props = decoder(seq)
            for i, prop in enumerate(props):
                new_seq = np.append(seq, i)
                new_score = score + np.log(prop+1e-5)
                candidate.append((new_seq, new_score))
        candidate.sort(key=lambda x: x[-1], reverse=True)
        data = candidate[:beam_size]
    return data[0][0]

np.random.seed(42)
initial_input = np.array([0])
best_sequence = beam_search(example_decoder, initial_input, beam_size=3, max_length=10)
print(best_sequence)