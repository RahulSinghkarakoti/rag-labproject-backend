from sentence_transformers import SentenceTransformer

from app.rag.embeddings.base import BaseEmbedder
from app.rag.model_registry import ModelRegistry


def _load_model(model_name: str) -> SentenceTransformer:
    return ModelRegistry.get_embedding_model(model_name, loader=SentenceTransformer)


_load_model.cache_clear = ModelRegistry.clear_embedding_models


class SentenceTransformerEmbedder(BaseEmbedder):
    """
    Sentence Transformer based embedding provider.
    """

    def __init__(
        self,
        model_name: str = "all-MiniLM-L6-v2",
    ):
        self.model_name = model_name

        self.model = _load_model(model_name)

    def embed_documents(
        self,
        texts: list[str],
    ) -> list[list[float]]:

        if not texts:
            return []

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embeddings.tolist()

    def embed_query(
        self,
        text: str,
    ) -> list[float]:

        embedding = self.model.encode(
            text,
            convert_to_numpy=True,
            normalize_embeddings=True,
        )

        return embedding.tolist()

    def dimension(self) -> int:
        return self.model.get_sentence_embedding_dimension()
