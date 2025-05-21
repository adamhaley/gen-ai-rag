from sentence_transformers import SentenceTransformer, util

model = SentenceTransformer('all-MiniLM-L6-v2')  # fast and works offline

sentences = [
    "The meeting was moved to Friday afternoon.",
    "We rescheduled the team discussion for later this week.",
    "I made pasta with mushrooms and garlic last night."
]

embeddings = model.encode(sentences, convert_to_tensor=True)

# Calculate cosine similarities
similarity_matrix = util.cos_sim(embeddings, embeddings)

# Print the similarity results
for i in range(len(sentences)):
    for j in range(i + 1, len(sentences)):
        print(f"Similarity between sentence {i+1} and {j+1}: {similarity_matrix[i][j]:.4f}")

