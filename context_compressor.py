from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

from config import EMBEDDING_MODEL


class ContextCompressor:

    def __init__(self):

        self.model = SentenceTransformer(
            EMBEDDING_MODEL
        )

    def compress(
        self,
        documents,
        similarity_threshold=0.85
    ):

        compressed_docs = []

        embeddings = []

        for doc in documents:

            current_embedding = self.model.encode(
                doc.page_content
            )

            is_duplicate = False

            for existing_embedding in embeddings:

                similarity = cosine_similarity(
                    [current_embedding],
                    [existing_embedding]
                )[0][0]

                if similarity > similarity_threshold:

                    is_duplicate = True

                    break

            if not is_duplicate:

                compressed_docs.append(doc)

                embeddings.append(
                    current_embedding
                )

        return compressed_docs
    
if __name__ == "__main__":

    print(
        "Context Compressor Ready"
    )