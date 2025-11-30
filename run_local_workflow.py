#!/usr/bin/env python3
"""
Knowledge Graph Builder for PDF Documents

This script extracts detailed information from PDFs, identifies entities,
topics, methodologies, and key findings to build comprehensive knowledge graphs.
"""

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
    """Extract text from PDF files"""
    
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix == '.pdf':
        try:
            from PyPDF2 import PdfReader
        except ImportError:
            print("Warning: PyPDF2 not installed. Install with: pip3 install PyPDF2")
            return ""
        try:
            reader = PdfReader(str(file_path))
            text_chunks = []
            for page_num, page in enumerate(reader.pages):
                try:
                    page_text = page.extract_text()
                    if page_text:
                        text_chunks.append(page_text)
                except Exception as e:
                    print(f"Warning: Could not extract text from page {page_num}")
                    continue
            return "\n".join(text_chunks)
        except Exception as e:
            print(f"Error extracting text from PDF {file_path}: {e}")
            return ""

    # Fallback for text files
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            return f.read()
    except Exception as e:
        print(f"Error reading {file_path}: {e}")
        return ""


def extract_metadata(text, doc_name):
    """Extract metadata like authors, dates, and institutions"""
    metadata = {
        'authors': [],
        'institutions': [],
        'date': None,
        'keywords': [],
        'abstract': None
    }

    # Extract author names (common patterns)
    author_pattern = r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)[,\s*]'
    authors = re.findall(author_pattern, text[:1000])
    metadata['authors'] = list(set(authors))[:10]

    # Extract year
    year_pattern = r'\b(20\d{2}|19\d{2})\b'
    years = re.findall(year_pattern, text[:2000])
    if years:
        metadata['date'] = years[0]

    # Extract institutions (common patterns)
    inst_pattern = r'(University|Institute|Laboratory|Center|School|Department)\s+(?:of|for)?\s+([A-Za-z\s&]+?)(?:,|;|$)'
    institutions = re.findall(inst_pattern, text[:2000])
    metadata['institutions'] = [' '.join(inst) for inst in institutions][:5]

    # Extract abstract (usually first meaningful paragraph after title)
    abstract_pattern = r'(?:Abstract|Summary)\s*[:]?\s*([^.]+\.[^.]+\.[^.]+\.)'
    abstract_match = re.search(abstract_pattern, text, re.IGNORECASE)
    if abstract_match:
        metadata['abstract'] = abstract_match.group(1).strip()[:200]

    return metadata


def extract_entities_and_topics(text, doc_name):
    """Extract comprehensive entities, topics, and concepts"""
    topics = set()
    entities = defaultdict(list)

    # Extract headers and section titles
    header_pattern = r'^(#{1,6})\s+(.+?)$'
    for match in re.finditer(header_pattern, text, re.MULTILINE):
        header = match.group(2).strip().lower()
        if header and len(header) > 2:
            topics.add(header)

    # Extract key technical and domain-specific terms
    # Chemical compounds
    chemical_pattern = r'\b(cannabidiol|CBD|THC|cannabinoid|terpene|myrcene|CBDa|CBG|CBC|anthocyanin|chlorophyll)\b'
    chemicals = set(re.findall(chemical_pattern, text, re.IGNORECASE))
    topics.update(t.lower() for t in chemicals)
    entities['chemicals'].extend(list(chemicals))

    # Biological/Medical terms
    bio_pattern = r'\b(pain|arthritis|epilepsy|neuropathic|anxiety|inflammation|receptor|nociception|antinociceptive)\b'
    biological = set(re.findall(bio_pattern, text, re.IGNORECASE))
    topics.update(t.lower() for t in biological)
    entities['biological_terms'].extend(list(biological))

    # Methods and techniques
    method_pattern = r'\b(spectroscopy|chromatography|HPLC|GC-MS|von Frey|behavioral|clinical trial|study|analysis|measurement)\b'
    methods = set(re.findall(method_pattern, text, re.IGNORECASE))
    topics.update(t.lower() for t in methods)
    entities['methods'].extend(list(methods))

    # Organisms/Models
    organism_pattern = r'\b(mice|mouse|dogs|cats|horses|birds|parrots|humans|plants|lettuce|rats)\b'
    organisms = set(re.findall(organism_pattern, text, re.IGNORECASE))
    topics.update(t.lower() for t in organisms)
    entities['organisms'].extend(list(organisms))

    # Extract findings and results (sentences with key words)
    finding_keywords = ['showed', 'demonstrated', 'revealed', 'indicated', 'found', 'observed', 'suggests', 'improved', 'increased', 'decreased']
    findings = []
    for keyword in finding_keywords:
        pattern = rf'[^.]*\b{keyword}\b[^.]*\.'
        matches = re.findall(pattern, text, re.IGNORECASE)
        findings.extend(matches[:2])
    entities['findings'] = findings[:5]

    # Extract numerical values and measurements
    measurement_pattern = r'(\d+(?:\.\d+)?\s*(?:mg|kg|nm|μmol|hours|days|weeks|%|correlation|coefficient))'
    measurements = re.findall(measurement_pattern, text, re.IGNORECASE)
    entities['measurements'].extend(measurements[:10])

    return list(topics), dict(entities)


def build_knowledge_graph(pdfs_dir):
    """Build comprehensive knowledge graph from PDFs"""
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

        print(f"\nProcessing {file_path.name}...")
        doc_name = file_path.stem
        text = extract_text_from_file(file_path)

        if not text:
            print(f"  ⚠️  Could not extract text from {file_path.name}")
            continue

        print(f"   Extracted {len(text)} characters")

        # Extract metadata
        metadata = extract_metadata(text, doc_name)
        print(f"   Found {len(metadata['authors'])} authors, date: {metadata['date']}")

        # Extract topics and entities
        topics, entities = extract_entities_and_topics(text, doc_name)
        print(f"   Identified {len(topics)} topics and {len(entities)} entity categories")

        # Add document node with detailed attributes
        G.add_node(doc_name, 
                  type='document',
                  file=file_path.name,
                  authors=metadata['authors'],
                  date=metadata['date'],
                  institutions=metadata['institutions'],
                  entities=len(entities))

        # Store comprehensive document data
        document_data[doc_name] = {
            'file': file_path.name,
            'metadata': metadata,
            'topics': topics[:40],
            'entities': entities,
            'text_length': len(text),
            'word_count': len(text.split()),
            'char_count': len(text)
        }

        # Add topic nodes and edges
        for topic in topics[:40]:
            topic_id = f"topic_{topic.replace(' ', '_').replace('-', '_')}"
            G.add_node(topic_id, type='topic', label=topic)
            G.add_edge(doc_name, topic_id, relation='discusses', weight=1.0)
            all_topics[topic].append(doc_name)

        # Add entity type nodes
        for entity_type, entity_list in entities.items():
            if entity_list:
                entity_type_id = f"entity_{entity_type}"
                G.add_node(entity_type_id, type='entity_category', label=entity_type)
                G.add_edge(doc_name, entity_type_id, relation='contains_entities', count=len(entity_list))

                # Add individual entities
                for entity in set(entity_list):
                    entity_id = f"{entity_type}_{entity.lower().replace(' ', '_')}"
                    G.add_node(entity_id, type='entity', label=entity, category=entity_type)
                    G.add_edge(doc_name, entity_id, relation='mentions', weight=1.0)
                    G.add_edge(entity_type_id, entity_id, relation='includes')

        # Add author nodes
        for author in metadata['authors']:
            author_id = f"author_{author.lower().replace(' ', '_')}"
            G.add_node(author_id, type='author', label=author)
            G.add_edge(doc_name, author_id, relation='authored_by')

    # Create relationships between documents based on shared topics
    docs = [node for node, data in G.nodes(data=True) if data.get('type') == 'document']
    print(f"\n Creating document relationships...")
    for i, doc1 in enumerate(docs):
        for doc2 in docs[i+1:]:
            doc1_topics = set(document_data[doc1]['topics'])
            doc2_topics = set(document_data[doc2]['topics'])
            common_topics = doc1_topics & doc2_topics

            if len(common_topics) >= 2:
                G.add_edge(doc1, doc2,
                         relation='related',
                         common_topics=list(common_topics)[:5],
                         similarity=len(common_topics) / min(len(doc1_topics), len(doc2_topics)))

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
            'total_documents': len(document_data),
            'node_types': {}
        }
    }

    # Count node types
    node_type_counts = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_type = data.get('type', 'unknown')
        node_type_counts[node_type] += 1
        graph_data['nodes'].append({
            'id': node,
            'label': data.get('label', node),
            **data
        })

    graph_data['statistics']['node_types'] = dict(node_type_counts)

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


def visualize_knowledge_graph(G, output_file='knowledge_graph_visualization.png', max_nodes=50):
    """Create visualization of the knowledge graph"""
    plt.figure(figsize=(24, 20))

    # Filter nodes for better visualization
    doc_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'document']
    topic_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'topic'][:max_nodes-len(doc_nodes)]
    entity_nodes = [n for n, d in G.nodes(data=True) if d.get('type') == 'entity'][:5]

    # Create subgraph for visualization
    nodes_to_keep = doc_nodes + topic_nodes + entity_nodes
    G_vis = G.subgraph(nodes_to_keep).copy()

    # Use spring layout with better parameters
    pos = nx.spring_layout(G_vis, k=3, iterations=100, seed=42, weight=None)

    # Draw nodes by type
    nx.draw_networkx_nodes(G_vis, pos, nodelist=doc_nodes,
                          node_color="#00FFA2", node_size=5000,
                          alpha=0.9, label='Documents', node_shape='s')
    nx.draw_networkx_nodes(G_vis, pos, nodelist=topic_nodes,
                          node_color="#0D00FC", node_size=2000,
                          alpha=0.7, label='Topics')
    nx.draw_networkx_nodes(G_vis, pos, nodelist=entity_nodes,
                          node_color="#FF0000", node_size=1500,
                          alpha=0.6, label='Entities')

    # Draw edges with varying thickness
    edges = G_vis.edges(data=True)
    for u, v, data in edges:
        weight = data.get('weight', 1.0)
        nx.draw_networkx_edges(G_vis, pos, [(u, v)], alpha=0.2 + (0.3 * weight), 
                              width=0.5 + (2 * weight))

    # Draw labels for document nodes
    doc_labels = {n: n.replace('_', ' ').title()[:15] for n in doc_nodes}
    nx.draw_networkx_labels(G_vis, pos, labels=doc_labels, 
                           font_size=10, font_weight='bold', font_color='black')

    plt.title('PDF Knowledge Graph', fontsize=24, fontweight='bold', pad=20)
    plt.legend(scatterpoints=1, loc='upper left', fontsize=14, framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Visualization saved to {output_file}")
    plt.close()

def visualize_knowledge_graph_enhanced(
    G,
    output_prefix='knowledge_graph',
    max_topics=60,
    max_entities=150,
    layout='spring'
):
    if not HAS_DEPS:
        print("Visualization skipped (missing deps)")
        return

    # Derive subsets
    doc_nodes = [n for n,d in G.nodes(data=True) if d.get('type') == 'document']
    topic_nodes = [n for n,d in G.nodes(data=True) if d.get('type') == 'topic'][:max_topics]
    entity_nodes = [n for n,d in G.nodes(data=True) if d.get('type') == 'entity'][:max_entities]

    keep = set(doc_nodes + topic_nodes + entity_nodes)
    H = G.subgraph(keep).copy()

    # Compute degrees for scaling
    deg = H.degree()
    max_deg = max((v for _,v in deg), default=1)

    # Optional simple community detection (might be expensive on large graphs)
    try:
        from networkx.algorithms.community import greedy_modularity_communities
        comms = list(greedy_modularity_communities(H))
        community_map = {}
        for i, c in enumerate(comms):
            for n in c:
                community_map[n] = i
    except Exception:
        community_map = {}

    # Color palettes
    base_colors = {
        'document': "#1b998b",
        'topic': "#2d7dd2",
        'entity': "#e84855",
        'entity_category': "#ffb400",
        'author': "#6a4c93"
    }
    community_colors = [
        "#264653","#2a9d8f","#8ab17d","#e9c46a","#f4a261",
        "#e76f51","#6d597a","#b56576","#355070","#0081a7"
    ]

    def node_color(n, data):
        if community_map:
            idx = community_map.get(n, 0) % len(community_colors)
            return community_colors[idx]
        return base_colors.get(data.get('type'), "#888888")

    # Choose layout
    if layout == 'kamada':
        pos = nx.kamada_kawai_layout(H)
    elif layout == 'circular':
        pos = nx.circular_layout(H)
    else:
        pos = nx.spring_layout(H, k=2, iterations=150, seed=42)

    # Figure 1: Mixed graph
    plt.figure(figsize=(26,22))
    node_sizes = []
    node_colors = []
    for n, data in H.nodes(data=True):
        d = deg[n]
        scale = 400 + (2200 * (d / max_deg))
        if data.get('type') == 'document':
            scale *= 1.4
        node_sizes.append(scale)
        node_colors.append(node_color(n, data))

    nx.draw_networkx_nodes(
        H, pos,
        node_size=node_sizes,
        node_color=node_colors,
        linewidths=0.6,
        edgecolors='white',
        alpha=0.92
    )

    # Edge styling
    edge_colors = []
    edge_widths = []
    for u,v,data in H.edges(data=True):
        rel = data.get('relation')
        if rel == 'discusses':
            edge_colors.append("#2d7dd2")
            edge_widths.append(1.4)
        elif rel == 'mentions':
            edge_colors.append("#e84855")
            edge_widths.append(1.0)
        elif rel == 'authored_by':
            edge_colors.append("#6a4c93")
            edge_widths.append(1.2)
        elif rel == 'related':
            edge_colors.append("#1b998b")
            edge_widths.append(2.0)
        else:
            edge_colors.append("#999999")
            edge_widths.append(0.6)

    nx.draw_networkx_edges(
        H, pos,
        edge_color=edge_colors,
        width=edge_widths,
        alpha=0.35
    )

    # Labels only for documents + high degree topics
    label_nodes = {}
    for n,data in H.nodes(data=True):
        if data.get('type') == 'document':
            label_nodes[n] = data.get('label', n)[:22]
        elif data.get('type') == 'topic' and deg[n] > 1:
            label_nodes[n] = data.get('label', n)[:18]
    nx.draw_networkx_labels(
        H, pos,
        labels=label_nodes,
        font_size=11,
        font_weight='semibold',
        font_color='black'
    )

    plt.title("Enhanced PDF Knowledge Graph", fontsize=28, fontweight='bold', pad=24)
    plt.axis('off')
    plt.tight_layout()
    out_file_main = f"{output_prefix}_enhanced.png"
    plt.savefig(out_file_main, dpi=170, bbox_inches='tight', facecolor='white')
    plt.close()
    print(f"✅ Enhanced visualization saved: {out_file_main}")

    # Figure 2: Document–Topic heat map (adjacency)
    doc_topic_edges = [(u,v) for u,v,d in G.edges(data=True) if d.get('relation') == 'discusses']
    if doc_topic_edges:
        docs = sorted({u for u,v in doc_topic_edges if G.nodes[u].get('type') == 'document'})
        topics = sorted({v for u,v in doc_topic_edges if G.nodes[v].get('type') == 'topic'})
        # Limit
        topics = topics[:max_topics]
        import numpy as np
        mat = np.zeros((len(docs), len(topics)), dtype=int)
        topic_index = {t:i for i,t in enumerate(topics)}
        doc_index = {d:i for i,d in enumerate(docs)}
        for u,v in doc_topic_edges:
            if u in doc_index and v in topic_index:
                mat[doc_index[u], topic_index[v]] += 1

        plt.figure(figsize=(1.4+0.35*len(topics), 1.4+0.6*len(docs)))
        plt.imshow(mat, aspect='auto', cmap='viridis')
        plt.colorbar(label='Mentions / weight')
        plt.yticks(range(len(docs)), [d[:20] for d in docs], fontsize=9)
        plt.xticks(range(len(topics)), [G.nodes[t].get('label', t)[:14] for t in topics], rotation=90, fontsize=9)
        plt.title("Document–Topic Matrix", fontsize=20, pad=16)
        plt.tight_layout()
        out_file_matrix = f"{output_prefix}_doc_topic_matrix.png"
        plt.savefig(out_file_matrix, dpi=160, bbox_inches='tight')
        plt.close()
        print(f"✅ Matrix visualization saved: {out_file_matrix}")

    # Figure 3: Degree distribution
    deg_values = [v for _,v in G.degree()]
    if deg_values:
        import numpy as np
        plt.figure(figsize=(10,6))
        plt.hist(deg_values, bins=min(30, max(5, int(np.sqrt(len(deg_values))))), color="#2d7dd2", alpha=0.85)
        plt.title("Node Degree Distribution", fontsize=20)
        plt.xlabel("Degree")
        plt.ylabel("Count")
        plt.grid(alpha=0.25)
        out_file_deg = f"{output_prefix}_degree_hist.png"
        plt.tight_layout()
        plt.savefig(out_file_deg, dpi=140)
        plt.close()
        print(f"✅ Degree histogram saved: {out_file_deg}")

def main():
    """Main function"""
    print("="*60)
    print("PDF Knowledge Graph Builder")
    print("="*60)

    pdfs_dir = 'pdfs'

    if not Path(pdfs_dir).exists():
        print(f"Error: {pdfs_dir}/ directory not found")
        print(f"Creating {pdfs_dir}/ directory...")
        Path(pdfs_dir).mkdir(parents=True, exist_ok=True)
        print(f"Please place your PDF files in {pdfs_dir}/ and run again")
        return 1

    print("\n Building enhanced knowledge graph from documents...")
    G, document_data = build_knowledge_graph(pdfs_dir)

    if G is None:
        return 1

    print(f"\n Knowledge Graph Statistics:")
    print(f"  Total nodes: {G.number_of_nodes()}")
    print(f"  Total edges: {G.number_of_edges()}")
    print(f"  Total documents: {len(document_data)}")

    # Calculate additional statistics
    node_types = defaultdict(int)
    for node, data in G.nodes(data=True):
        node_types[data.get('type', 'unknown')] += 1

    print(f"\n  Node Types:")
    for node_type, count in sorted(node_types.items()):
        print(f"    {node_type}: {count}")

    # Save knowledge graph
    graph_data = save_knowledge_graph(G, document_data)

    # Create visualization
    if HAS_DEPS:
        visualize_knowledge_graph(G)
        visualize_knowledge_graph_enhanced(G, layout='spring')

    # Print document summary
    print("\n Document Summary:")
    for doc_name, data in sorted(document_data.items()):
        print(f"  {doc_name}:")
        print(f"    Words: {data['word_count']:,}")
        print(f"    Topics: {len(data['topics'])}")
        print(f"    Entity Categories: {len(data['entities'])}")

    print("\n" + "="*60)
    print("✅ Workflow completed successfully!")
    print("="*60)
    print("\nGenerated files:")
    print("  - knowledge_graph.json")
    if HAS_DEPS:
        print("  - knowledge_graph_visualization.png")
        print("  - knowledge_graph_enhanced.png")
        print("  - knowledge_graph_doc_topic_matrix.png")
        print("  - knowledge_graph_degree_hist.png")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
