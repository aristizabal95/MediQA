from pathlib import Path
import embed_anything
from embed_anything import EmbedData, EmbeddingModel, TextEmbedConfig, WhichModel
from mediqa.rag.adapters.chromadb_adapter import ChromaAdapter

from mediqa.config.core import RAGConfig, config, PACKAGE_ROOT, DATASET_DIR

class VectorDBManager:
    def __init__(self, config: RAGConfig):
        self.config = config

        self.model = EmbeddingModel.from_pretrained_hf(
            WhichModel.Jina, model_id=config.encoder_name
        )

        # with semantic encoder
        self.embedder = EmbeddingModel.from_pretrained_hf(
            WhichModel.Jina, model_id=config.encoder_name
        )
        self.embed_config = TextEmbedConfig(
            chunk_size=config.chunk_size,
            batch_size=config.batch_size,
            splitting_strategy=config.encoding_strategy,
            semantic_encoder=self.embedder,
        )
        db_location = str(Path(PACKAGE_ROOT) / self.config.db_location)
        self.vdb_adapter = ChromaAdapter(db_location, config.embedding_dimension)
        
        # Automatically get or create index
        self.vdb_adapter.create_index(config.index_name)

    def populate(self, knowledge_path: Path):
        data = embed_anything.embed_directory(knowledge_path, self.embedder, config=self.embed_config, adapter=self.vdb_adapter)
        return data

    def retrieve(self, query: str):
        query_embedding = embed_anything.embed_query([query], embedder=self.model)[0].embedding
        results = self.vdb_adapter.collection.query(
            query_embeddings=[query_embedding],
            n_results=self.config.n_results
        )

        return results

if __name__ == "__main__":
    rag = VectorDBManager(config.rag_config)
    knowledge_path = Path(DATASET_DIR) / config.rag_config.knowledge_path
    rag.populate(knowledge_path)