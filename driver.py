from vectordb import Memory
from time import time

memory = Memory()

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

start_time = time()
results = memory.search(query, top_n=4, unique=True)
end_time = time()
print("Search took {:.4f} ms".format((end_time - start_time) * 1000))

for result in results:
    print(result)