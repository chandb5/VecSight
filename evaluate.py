from utils import Reader
from vectordb import Memory
from time import time

memory = Memory(load_vec_files=True)
memory.load_vector_files("./datasets/siftsmall/siftsmall_base.fvecs")
query_reader = Reader('./datasets/siftsmall/siftsmall_query.fvecs')
query_reader.read()

query_vector_idx = 0

start_time = time()
query_results = memory.search_vector(query_reader.data[query_vector_idx], top_k=100)
end_time = time()

print("Search took {:.4f} ms".format((end_time - start_time) * 1000))
ground_truths = Reader('./datasets/siftsmall/siftsmall_groundtruth.ivecs')
ground_truths.read()

count = 0
for idx, res in enumerate(query_results):
    if str(ground_truths.data[query_vector_idx][idx]) in res["chunk"]:
        count += 1
    print(res, ground_truths.data[query_vector_idx][idx])

print(count)