# Graph Construction

This directory contains the logic for constructing a graph from the processed keywords.

## `graph_builder.py`

This script contains the `GraphBuilder` class, which uses the `networkx` library to create a graph of keywords. The nodes of the graph are the keywords, and the edges represent the relationships between them.

### Attributes

-   **Nodes**: Each node represents a keyword and has the following attributes:
    -   `sentiment`: The sentiment score of the keyword.
    -   `count`: The number of times the keyword appears.
-   **Edges**: The edges between nodes represent the co-occurrence of keywords in the same text chunk. The weight of the edge can be used to represent the strength of the relationship.

### Usage

To use the `GraphBuilder`, first install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Then, you can import the `GraphBuilder` class and use it to build a graph from your processed keywords:

```python
from graph_builder import GraphBuilder

keywords = [
    {'keyword': 'API', 'sentiment': 0.8, 'count': 5},
    {'keyword': 'customer service', 'sentiment': -0.9, 'count': 3},
    {'keyword': 'powerful', 'sentiment': 0.9, 'count': 2},
]

graph_builder = GraphBuilder()
graph_builder.add_keywords(keywords)
graph_builder.add_edge('API', 'powerful', weight=2)

graph = graph_builder.get_graph()

# You can now use the graph object for further analysis or visualization
print(graph.nodes(data=True))
print(graph.edges(data=True))
```
