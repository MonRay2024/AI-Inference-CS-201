import networkx as nx
import matplotlib.pyplot as plt
from typing import Dict
from merkle_analysis import CodeBlock

def visualize_merkle_tree(root: CodeBlock, impact_scores: Dict[str, float] = None):
    """Visualize Merkle tree with optional impact scores"""
    G = nx.DiGraph()
    
    def add_nodes(block: CodeBlock):
        # Add node with impact score if available
        score = impact_scores.get(block.hash, 0) if impact_scores else 0
        G.add_node(block.hash[:8], 
                  impact=score,
                  lines=f"{block.start_line}-{block.end_line}")
        
        if block.children:
            for child in block.children:
                G.add_node(child.hash[:8],
                          impact=impact_scores.get(child.hash, 0) if impact_scores else 0,
                          lines=f"{child.start_line}-{child.end_line}")
                G.add_edge(block.hash[:8], child.hash[:8])
                add_nodes(child)
    
    add_nodes(root)
    
    # Draw the graph
    pos = nx.spring_layout(G)
    plt.figure(figsize=(12, 8))
    
    # Draw nodes with colors based on impact
    node_colors = [G.nodes[node]['impact'] for node in G.nodes()]
    nx.draw(G, pos, node_color=node_colors, 
            with_labels=True, node_size=1000,
            cmap=plt.cm.YlOrRd)
    
    plt.title("Code Merkle Tree with Impact Scores")
    plt.show() 