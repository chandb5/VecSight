from vectordb import Memory
from time import time

# Create Memory with HNSW as the preferred search algorithm
memory = Memory(
    hnsw_preference=True,
    hnsw_params={
        "M": 5,
        "ef_construction": 150,
        "space": "cosine",
    }
)
   
memory.save(
    [
        "apples are green",
        "oranges are orange",
        "I lost my phone and need help tracking it.",
        "The laptop screen is flickering randomly.",
        "How do I reset my email password?",
        "My order hasnt arrived yet. What should I do?",
        "The coffee maker stopped working after a week.",
        "Looking for a budget-friendly smartphone with a good camera.",
        "The weather today is perfect for a beach trip.",
        "Best way to cook pasta without it sticking together?",
        "My dog is not eating properly. Should I be worried?",
        "The movie was amazing, full of unexpected twists!",
    ],
)

query = (
    "help me troubleshoot"
)

# Using HNSW (default based on hnsw_preference=True)
# HNSW is expected to be slower in case of very small datasets or very low dimensions
# but faster in case of larger datasets or higher dimensions
start_time = time()
results = memory.search(query, top_n=4, unique=True)
end_time = time()
print("HNSW Search took {:.4f} ms".format((end_time - start_time) * 1000))

print("HNSW Results:")
for result in results:
    print(result)
    
# Compare with non-HNSW search
start_time = time()
results = memory.search(query, top_n=4, unique=True, use_hnsw=False)
end_time = time()
print("\nStandard Search took {:.4f} ms".format((end_time - start_time) * 1000))

print("Standard Results:")
for result in results:
    print(result)