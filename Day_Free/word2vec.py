import sys
sys.path.append("./deep-learning-from-scratch-2")
import numpy as np
from common.layers import MatMul
from common.util import preprocess, create_contexts_target, convert_one_hot

c0 = np.array([[1, 0, 0, 0, 0, 0, 0]])
c1 = np.array([[0, 0, 1, 0, 0, 0, 0]])

# 가중치 초기화 
W_in = np.random.randn(7, 3)
W_out = np.random.randn(3, 7)
# 계층생성 
in_layer0 = MatMul(W_in)
in_layer1 = MatMul(W_in)
out_layer = MatMul(W_out)

# 순전파 
h0 = in_layer0.forward(c0)
h1 = in_layer1.forward(c1)
h = 0.5 * (h0 + h1)
s = out_layer.forward(h)

text = 'You say goodbye and I say hello'
corpus, word_to_id, id_to_word = preprocess(text)
contexts, target = create_contexts_target(corpus, window_size=1)
vocab_size=len(word_to_id)
# def create_contexts_target(corpus, window_size=1):
#   target = corpus[window_size: - window_size]
#   contexts = list()
#   for idx in range(window_size, len(corpus) - window_size):
#     cs = list()
#     for t in range(-window_size, window_size + 1):
#       if t == 0:
#         continue
#       cs.append(corpus[idx + t])
#     contexts.append(cs)

#   return np.array(contexts), np.array(target)
# contexts, target = create_contexts_target(corpus, window_size=1)
# print(contexts)
# print(target)
