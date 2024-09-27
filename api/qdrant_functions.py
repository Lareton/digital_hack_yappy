from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams

HOST = "localhost"
VECTOR_SIZE = 128
VECTOR_DISTANCE = "Cosine"
VECTOR_CONFIG = VectorParams(size=VECTOR_SIZE, distance=VECTOR_DISTANCE)

# Connect to a local Qdrant instance (running on localhost:6333)
client = QdrantClient(host=HOST, port=6333)

if not client.collection_exists("licensed_videos"):
    client.create_collection(
        collection_name="licensed_videos",
        vectors_config=VECTOR_CONFIG)

if __name__ == '__main__':
    print(client.collection_exists("licensed_videos"))
