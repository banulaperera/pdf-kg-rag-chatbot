"""
Knowledge Graph Utilities
Load and interact with the knowledge graph created by the GitHub Actions workflow.
"""

import json
from pathlib import Path
from typing import List, Optional, Set
import networkx as nx
from collections import defaultdict
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


class KnowledgeGraphIntegration:
    """Integration with the knowledge graph"""

    def __init__(self, graph_path: Optional[str] = None):
        self.graph = None
        self.doc_topics = defaultdict(list)
        self.entity_docs = defaultdict(set)
        if graph_path is None:
            graph_path = Path(__file__).parent.parent / "knowledge_graph.json"
        self.load_graph(graph_path)

    def load_graph(self, graph_path: str):
        """Load the knowledge graph from JSON (supports 'edges' or 'links' formats)"""
        gp = Path(graph_path)
        if not gp.exists():
            logger.warning(f"Knowledge graph not found at {gp}")
            return

        try:
            with open(gp) as f:
                data = json.load(f)

            # Determine format
            if 'links' in data and 'nodes' in data:
                # Standard node-link format
                self.graph = nx.node_link_graph(data)
            else:
                # Fallback: expect 'nodes' + 'edges'
                self.graph = nx.Graph()
                nodes = data.get('nodes', [])
                for n in nodes:
                    node_id = n.get('id') or n.get('name') or n.get('label')
                    if node_id is None:
                        continue
                    attrs = {k: v for k, v in n.items() if k not in ('id', 'name')}
                    self.graph.add_node(node_id, **attrs)

                for e in data.get('edges', []):
                    src = e.get('source')
                    tgt = e.get('target')
                    if src is None or tgt is None:
                        continue
                    attrs = {k: v for k, v in e.items() if k not in ('source', 'target')}
                    self.graph.add_edge(src, tgt, **attrs)

            # Build lookup indices (works with either format)
            self.doc_topics.clear()
            self.entity_docs.clear()
            edge_iter = data.get('edges') or data.get('links') or []
            for edge in edge_iter:
                source = edge.get('source')
                target = edge.get('target')
                relation = edge.get('relation', '')
                if relation == 'discusses' and source and target:
                    self.doc_topics[source].append(target)
                elif relation == 'mentions' and source and target:
                    self.entity_docs[target].add(source)

            logger.info(f"Loaded knowledge graph with {self.graph.number_of_nodes()} nodes and {self.graph.number_of_edges()} edges")
        except Exception as e:
            logger.error(f"Error loading knowledge graph: {e}")

    def get_related_topics(self, document: str) -> List[str]:
        """Get topics related to a document"""
        return self.doc_topics.get(document, [])

    def get_related_documents(self, topic: str) -> Set[str]:
        """Get documents that discuss a topic"""
        related = set()
        if self.graph and topic in self.graph.nodes():
            for neighbor in self.graph.neighbors(topic):
                if self.graph.nodes[neighbor].get('type') == 'document':
                    related.add(neighbor)
        return related

    def expand_query(self, query: str) -> List[str]:
        """Expand query with related topics from the graph"""
        expanded = [query]

        # Extract potential entities from query
        words = query.lower().split()

        for word in words:
            # Look for matching topics
            if self.graph:
                for node in self.graph.nodes():
                    node_label = self.graph.nodes[node].get('label', '').lower()
                    if node_label and word in node_label and node_label not in query:
                        expanded.append(node_label)

        return expanded[:5]  # Limit to 5 expanded terms