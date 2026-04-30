from qdrant_client import AsyncQdrantClient

_client: AsyncQdrantClient | None = None


def set_qdrant(client: AsyncQdrantClient) -> None:
    global _client
    _client = client


def get_qdrant() -> AsyncQdrantClient:
    assert _client is not None, "Qdrant client not initialised"
    return _client
