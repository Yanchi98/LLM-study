from FlagEmbedding import FlagModel
from config import model_path

sentences_1 = ["我想买苹果手机", "我想吃苹果"]
sentences_2 = ["苹果是一种水果", "iphone深受果粉喜爱"]
model = FlagModel(model_path,
                  query_instruction_for_retrieval="为这个句子生成表示以用于检索相关文章：",
                  use_fp16=True) # Setting use_fp16 to True speeds up computation with a slight performance degradation
# q-q匹配，不用instruction
embeddings_1 = model.encode(sentences_1)
embeddings_2 = model.encode(sentences_2)
similarity = embeddings_1 @ embeddings_2.T
print(similarity)


# q-p匹配，需要instruction, encode_queries会自动添加上instruction
queries = ["我想买苹果手机", "我想吃苹果"]
passages = ["苹果是一种水果", "iphone深受果粉喜爱"]
q_embeddings = model.encode_queries(queries)
p_embeddings = model.encode(passages)
scores = q_embeddings @ p_embeddings.T
print(scores)