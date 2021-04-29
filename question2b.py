import networkx as nx
from igraph import *


def read_graph():
    fb_graph = nx.read_edgelist("Wiki-Vote.txt",  create_using=nx.DiGraph, nodetype=int)
    return fb_graph

def plot_graph(graph): 
    ig = Graph.TupleList(graph.edges, directed=True)
    plot(ig)

def compute_pagerank(fb_graph):
    pagerank_scores = nx.pagerank(fb_graph,alpha=0.85,tol=0.0001)
    print("top 5 nodes with pagerank scores are:",sorted([(b, a) for a, b in pagerank_scores.items()], reverse=True)[:5])

def compute_hits(fb_graph):
    hits_scores = nx.hits(fb_graph,tol=0.0001)
    authorities_scores = hits_scores[1]
    hub_scores = hits_scores[0]
    print("top 5 nodes according to authorities scores are:",sorted([(b, a) for a, b in authorities_scores.items()], reverse=True)[:5])
    print("top 5 nodes according to hub scores are:",sorted([(b, a) for a, b in hub_scores.items()], reverse=True)[:5])

if __name__ == "__main__":
    fb_graph = read_graph()
    print("Summary information of the graph is:", nx.info(fb_graph))
    compute_pagerank(fb_graph)
    compute_hits(fb_graph)
    plot_graph(fb_graph)