#!/usr/bin/env python3
"""
Run Local Workflow

This script simulates the GitHub Actions workflow locally so candidates
can test their knowledge graph building without pushing to GitHub.
"""

import os
import json
import re
from pathlib import Path
from collections import defaultdict

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    HAS_DEPS = True
except ImportError:
    HAS_DEPS = False
    print("Warning: Some dependencies not available. Install with:")
    print("  pip install networkx matplotlib")


def extract_text_from_file(file_path):
    """Extract text from text files"""
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_entities_and_topics(text, doc_name):
    """Extract entities, topics, and key concepts"""
    topics = set()
    
    # Extract topics from headers (lines starting with #)
    header_pattern = r'^#+\s+(.+)$'
    for match in re.finditer(header_pattern, text, re.MULTILINE):
        topics.add(match.group(1).strip().lower())
    
    # Extract key technical terms
    technical_terms = set(re.findall(
        r'\b(?:[A-Z][a-z]+(?:[A-Z][a-z]+)+|'
        r'[A-Z]{2,}|'
        r'(?:machine|deep|neural|natural|artificial)\s+(?:learning|network|intelligence|processing))\b',
        text
    ))
    topics.update(t.lower() for t in technical_terms)
    
    return list(topics)


def build_knowledge_graph(pdfs_dir):
    """Build knowledge graph from PDFs"""
    if not HAS_DEPS:
        print("Cannot build graph without dependencies")
        return None, {}
    
    G = nx.Graph()
    document_data = {}
    all_topics = defaultdict(list)
    
    pdf_files = list(Path(pdfs_dir).glob('*.*'))
    print(f"Found {len(pdf_files)} files to process")
    
    for file_path in pdf_files:
        if file_path.suffix.lower() not in ['.pdf', '.txt']:
            continue
        
        print(f"Processing {file_path.name}...")
        doc_name = file_path.stem
        text = extract_text_from_file(file_path)
        
        if not text:
            continue
        
        # Extract topics
        topics = extract_entities_and_topics(text, doc_name)
        
        # Add document node
        G.add_node(doc_name, type='document', file=file_path.name)
        
        # Store document data
        document_data[doc_name] = {
            'file': file_path.name,
            'topics': topics[:30],
            'text_length': len(text),
            'word_count': len(text.split())
        }
        
        # Add topic nodes and edges
        for topic in topics[:30]:
            topic_id = f"topic_{topic.replace(' ', '_')}"
            G.add_node(topic_id, type='topic', label=topic)
            G.add_edge(doc_name, topic_id, relation='discusses')
            all_topics[topic].append(doc_name)
    
    # Create relationships between documents
    docs = [node for node, data in G.nodes(data=True) if data.get('type') == 'document']
    for i, doc1 in enumerate(docs):
        for doc2 in docs[i+1:]:
            doc1_topics = set(document_data[doc1]['topics'])
            doc2_topics = set(document_data[doc2]['topics'])
            common_topics = doc1_topics & doc2_topics
            
            if len(common_topics) >= 3:
                G.add_edge(doc1, doc2,
                         relation='related',
                         common_topics=list(common_topics)[:5])
    
    return G, document_data


def save_knowledge_graph(G, document_data, output_file='knowledge_graph.json'):
    """Save knowledge graph as JSON"""
    graph_data = {
        'nodes': [],
        'edges': [],
        'documents': document_data,
        'statistics': {
            'total_nodes': G.number_of_nodes(),
            'total_edges': G.number_of_edges(),
            'total_documents': len(document_data)
        }
    }
    
    for node, data in G.nodes(data=True):
        graph_data['nodes'].append({
            'id': node,
            **data
        })
    
    for u, v, data in G.edges(data=True):
        graph_data['edges'].append({
            'source': u,
            'target': v,
            **data
        })
    
    with open(output_file, 'w') as f:
        json.dump(graph_data, f, indent=2)
    
    print(f"\n✅ Knowledge graph saved to {output_file}")
    return graph_data


def visualize_knowledge_graph(G, output_file='knowledge_graph_visualization.png'):
    """Create visualization of the knowledge graph"""
    plt.figure(figsize=(20, 16))
    
    # Separate nodes by type
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'document']
    topic_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'topic']
    
    # Use spring layout
    pos = nx.spring_layout(G, k=2, iterations=50, seed=42)
    
    # Draw nodes
    nx.draw_networkx_nodes(G, pos, nodelist=doc_nodes,
                          node_color='lightblue', node_size=3000,
                          alpha=0.9, label='Documents')
    nx.draw_networkx_nodes(G, pos, nodelist=topic_nodes,
                          node_color='lightgreen', node_size=1000,
                          alpha=0.7, label='Topics')
    
    # Draw edges
    nx.draw_networkx_edges(G, pos, alpha=0.3, width=1)
    
    # Draw labels for document nodes
    doc_labels = {n: n for n in doc_nodes}
    nx.draw_networkx_labels(G, pos, labels=doc_labels, font_size=10, font_weight='bold')
    
    plt.title('PDF Knowledge Graph', fontsize=20, fontweight='bold')
    plt.legend(scatterpoints=1, loc='upper left', fontsize=12)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Visualization saved to {output_file}")
    plt.close()


def main():
    """Main function"""
    print("="*60)
    print("Local Knowledge Graph Workflow")
    print("="*60)
    
    pdfs_dir = 'pdfs'
    
    if not Path(pdfs_dir).exists():
        print(f"Error: {pdfs_dir}/ directory not found")
        return 1
    
    print("\n📊 Building knowledge graph from documents...")
    G, document_data = build_knowledge_graph(pdfs_dir)
    
    if G is None:
        return 1
    
    print(f"\n📈 Knowledge Graph Statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Total documents: {len(document_data)}")
    
    # Save knowledge graph
    save_knowledge_graph(G, document_data)
    
    # Create visualization
    visualize_knowledge_graph(G)
    
    # Print summary
    print("\n📄 Document Summary:")
    for doc_name, data in document_data.items():
        print(f"  {doc_name}:")
        print(f"    Words: {data['word_count']}")
        print(f"    Topics: {len(data['topics'])}")
    
    print("\n" + "="*60)
    print("✅ Workflow completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - knowledge_graph.json")
    print("  - knowledge_graph_visualization.png")
    print("\nYou can now test the chatbot with:")
    print("  python -m chatbot.rag_chatbot")
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
