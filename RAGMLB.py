import numpy as np

##INSERT MESSI.TXT file##
with open("messi.txt", "r") as file:
    data = file.read()

chunks = [data[i:i+50] for i in range(0, len(data), 50)]

import openai
openai.api_key = "<INSERT API KEY>"

def get_embedding(text):
    response = openai.Embedding.create(
        input=text,
        model="text-embedding-ada-002"
    )
    return response["data"][0]["embedding"]

chunk_embeddings = {chunk: get_embedding(chunk) for chunk in chunks}


question = "What is Messi's record in World Cups?"
question_embedding = get_embedding(question)



def cosine_similarity(vec1, vec2):
    return np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))

similarities = [
    (chunk, cosine_similarity(question_embedding, chunk_emb))
    for chunk, chunk_emb in chunk_embeddings.items()
]
similarities = sorted(similarities, key=lambda x: x[1], reverse=True)

top_chunks = [chunk for chunk, _ in similarities[:5]]
context = " ".join(top_chunks)

full_prompt = f"Using the following context:\n{context}\n\nAnswer the question:\n{question}"

response = openai.Completion.create(
    model="text-davinci-003",
    prompt=full_prompt,
    max_tokens=150
)
print(response["choices"][0]["text"].strip())
