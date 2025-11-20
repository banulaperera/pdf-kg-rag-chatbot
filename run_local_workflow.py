#!/usr/bin/env python3
"""
Enhanced Knowledge Graph Builder for PDF Documents

This script extracts detailed information from PDFs, identifies entities,
topics, methodologies, and key findings to build comprehensive knowledge graphs.
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
    """Extract text from PDF files"""
    from pathlib import Path
    p = Path(file_path)
    suffix = p.suffix.lower()

    if suffix == '.pdf':
        try:
            from PyPDF2 import PdfReader
        except Exception:
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

        print(f"  ✓ Extracted {len(text)} characters")

        # Extract metadata
        metadata = extract_metadata(text, doc_name)
        print(f"  ✓ Found {len(metadata['authors'])} authors, date: {metadata['date']}")

        # Extract topics and entities
        topics, entities = extract_entities_and_topics(text, doc_name)
        print(f"  ✓ Identified {len(topics)} topics and {len(entities)} entity categories")

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
                          node_color='#FF6B6B', node_size=5000,
                          alpha=0.9, label='Documents', node_shape='s')
    nx.draw_networkx_nodes(G_vis, pos, nodelist=topic_nodes,
                          node_color='#4ECDC4', node_size=2000,
                          alpha=0.7, label='Topics')
    nx.draw_networkx_nodes(G_vis, pos, nodelist=entity_nodes,
                          node_color='#95E1D3', node_size=1500,
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
                           font_size=10, font_weight='bold', font_color='white')

    plt.title('PDF Knowledge Graph', fontsize=24, fontweight='bold', pad=20)
    plt.legend(scatterpoints=1, loc='upper left', fontsize=14, framealpha=0.9)
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(output_file, dpi=150, bbox_inches='tight',
               facecolor='white', edgecolor='none')
    print(f"✅ Visualization saved to {output_file}")
    plt.close()


# def generate_summary_report(document_data, output_file='knowledge_graph_summary.txt'):
#     """Generate a detailed summary report"""
#     with open(output_file, 'w') as f:
#         f.write("="*80 + "\n")
#         f.write("PDF KNOWLEDGE GRAPH SUMMARY REPORT\n")
#         f.write("="*80 + "\n\n")

#         f.write(f"Total Documents Processed: {len(document_data)}\n\n")

#         for doc_name, data in sorted(document_data.items()):
#             f.write(f"\n{'─'*80}\n")
#             f.write(f"DOCUMENT: {doc_name}\n")
#             f.write(f"{'─'*80}\n")
#             f.write(f"File: {data['file']}\n")
#             f.write(f"Size: {data['text_length']:,} characters, {data['word_count']:,} words\n\n")

#             metadata = data.get('metadata', {})
#             if metadata.get('authors'):
#                 f.write(f"Authors: {', '.join(metadata['authors'])}\n")
#             if metadata.get('date'):
#                 f.write(f"Date: {metadata['date']}\n")
#             if metadata.get('institutions'):
#                 f.write(f"Institutions: {'; '.join(metadata['institutions'])}\n")
#             if metadata.get('abstract'):
#                 f.write(f"\nAbstract: {metadata['abstract']}...\n")

#             f.write(f"\nTop Topics ({len(data['topics'])} total):\n")
#             for topic in data['topics'][:10]:
#                 f.write(f"  • {topic}\n")

#             entities = data.get('entities', {})
#             if entities:
#                 f.write(f"\nEntity Categories:\n")
#                 for cat, items in entities.items():
#                     if items:
#                         f.write(f"  {cat}: {', '.join(set(items)[:5])}\n")

#             if entities.get('findings'):
#                 f.write(f"\nKey Findings (sample):\n")
#                 for finding in entities.get('findings', [])[:3]:
#                     f.write(f"  • {finding[:100]}...\n")

#     print(f"✅ Summary report saved to {output_file}")


def main():
    """Main function"""
    print("="*60)
    print("Enhanced PDF Knowledge Graph Builder")
    print("="*60)

    pdfs_dir = 'pdfs'

    if not Path(pdfs_dir).exists():
        print(f"Error: {pdfs_dir}/ directory not found")
        print(f"Creating {pdfs_dir}/ directory...")
        Path(pdfs_dir).mkdir(parents=True, exist_ok=True)
        print(f"Please place your PDF files in {pdfs_dir}/ and run again")
        return 1

    print("\n📊 Building enhanced knowledge graph from documents...")
    G, document_data = build_knowledge_graph(pdfs_dir)

    if G is None:
        return 1

    print(f"\n📈 Knowledge Graph Statistics:")
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

    # Generate summary report
    # generate_summary_report(document_data)

    # Print document summary
    print("\n📄 Document Summary:")
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
    print("  - knowledge_graph_summary.txt")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
