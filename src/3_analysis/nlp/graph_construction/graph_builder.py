import networkx as nx

class GraphBuilder:
    """
    A class to build a graph of keywords from processed text data.
    """

    def __init__(self):
        """
        Initializes the GraphBuilder with an empty graph.
        """
        self.graph = nx.Graph()

    def add_keywords(self, keywords):
        """
        Adds a list of keywords as nodes to the graph.

        Args:
            keywords: A list of dictionaries, where each dictionary contains a keyword,
                      its sentiment score, and its count.
        """
        for keyword_data in keywords:
            self.graph.add_node(
                keyword_data['keyword'],
                sentiment=keyword_data['sentiment'],
                count=keyword_data['count']
            )

    def add_edge(self, keyword1, keyword2, weight=1):
        """
        Adds an edge between two keywords in the graph.

        Args:
            keyword1: The first keyword.
            keyword2: The second keyword.
            weight: The weight of the edge (default is 1).
        """
        self.graph.add_edge(keyword1, keyword2, weight=weight)

    def get_graph(self):
        """
        Returns the graph.

        Returns:
            The networkx graph object.
        """
        return self.graph
