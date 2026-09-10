"""Compatibility entry point for refreshing the document index."""

from src.retrieval.ingest import ingest


class KnowledgeManager:
    def rebuild_knowledge_base(self):
        return ingest()


if __name__ == "__main__":
    print(KnowledgeManager().rebuild_knowledge_base())
