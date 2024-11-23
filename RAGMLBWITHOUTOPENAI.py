import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Load the text data
with open("/Users/shubhamgupta/Downloads/messi.txt", "r") as file:
    data = file.read()

# Split the data into chunks
chunk_size = 100
chunks = [data[i:i + chunk_size] for i in range(0, len(data), chunk_size)]

# Define the question
question = "What is Messi's record in World Cups?"

# Combine chunks and the question for TF-IDF
all_texts = chunks + [question]

# Compute TF-IDF embeddings
vectorizer = TfidfVectorizer()
tfidf_matrix = vectorizer.fit_transform(all_texts)

# Get the embedding of the question (last vector)
question_embedding = tfidf_matrix[-1]
chunk_embeddings = tfidf_matrix[:-1]

# Compute cosine similarities
similarities = cosine_similarity(question_embedding, chunk_embeddings).flatten()

# Get the top chunks
top_indices = similarities.argsort()[-5:][::-1]
top_chunks = [chunks[i] for i in top_indices]

# Combine top chunks into context
context = " ".join(top_chunks)

# Generate the final response (rudimentary answer generation)
print("Context used for answering the question:")
print(context)
print("\nAnswer:")

# Here, we're simply combining the context for simplicity.
# You can integrate further NLP techniques to refine the response.
print(f"Based on the context: {context}, the answer is likely about Messi's World Cup performances.")
