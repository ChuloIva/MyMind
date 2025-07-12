import numpy as np
import umap
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import logging
from typing import List, Dict, Any, Optional, Tuple
from openai import OpenAI
import json
from src.common.config import settings

logger = logging.getLogger(__name__)

class TherapeuticGraphBuilder:
    def __init__(self, api_key: str = settings.openai_api_key):
        """Initialize graph builder with embedding capabilities"""
        self.client = OpenAI(api_key=api_key) if api_key else OpenAI()
        self.vectorizer = TfidfVectorizer(
            max_features=1000,
            stop_words='english',
            ngram_range=(1, 2)
        )
        
    def build_session_graph(self, session_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Build visualization graph from session data
        
        Args:
            session_data: Processed session data with keywords and segments
            
        Returns:
            Graph data with nodes, edges, and coordinates
        """
        try:
            # Extract concepts and keywords
            concepts = self._extract_concepts(session_data)
            
            if not concepts:
                logger.warning("No concepts found in session data")
                return self._empty_graph()
            
            # Create embeddings
            embeddings = self._create_embeddings(concepts)
            
            # Generate 2D coordinates using UMAP
            coordinates_2d = self._generate_coordinates(embeddings, method="umap")
            
            # Build graph structure
            nodes = self._create_nodes(concepts, coordinates_2d)
            edges = self._create_edges(concepts, embeddings)
            
            # Add clustering and analysis
            clusters = self._identify_clusters(coordinates_2d, concepts)
            
            return {
                'nodes': nodes,
                'edges': edges,
                'clusters': clusters,
                'metadata': {
                    'total_concepts': len(concepts),
                    'total_nodes': len(nodes),
                    'total_edges': len(edges),
                    'embedding_method': 'openai',
                    'reduction_method': 'umap'
                }
            }
            
        except Exception as e:
            logger.error(f"Failed to build session graph: {e}")
            return self._empty_graph()
    
    def _extract_concepts(self, session_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Extract therapeutic concepts from session data"""
        concepts = []
        
        # Get processed segments
        segments = session_data.get('processed_segments', [])
        if not segments:
            segments = session_data.get('combined_segments', [])
        
        for segment in segments:
            # Extract keywords
            keywords = segment.get('keywords', [])
            for keyword in keywords:
                concept = {
                    'term': keyword.get('term', ''),
                    'relevance': keyword.get('relevance', 0.5),
                    'category': keyword.get('category', 'general'),
                    'sentiment': keyword.get('sentiment', 0.0),
                    'context': segment.get('text', ''),
                    'speaker': segment.get('speaker', 'UNKNOWN'),
                    'timestamp': segment.get('start', 0),
                    'therapeutic_themes': segment.get('therapeutic_themes', [])
                }
                concepts.append(concept)
            
            # Extract emotional indicators
            emotional_indicators = segment.get('emotional_indicators', [])
            for indicator in emotional_indicators:
                concept = {
                    'term': indicator,
                    'relevance': 0.7,
                    'category': 'emotion',
                    'sentiment': segment.get('sentiment_scores', {}).get('compound', 0),
                    'context': segment.get('text', ''),
                    'speaker': segment.get('speaker', 'UNKNOWN'),
                    'timestamp': segment.get('start', 0),
                    'therapeutic_themes': segment.get('therapeutic_themes', [])
                }
                concepts.append(concept)
        
        # Remove duplicates and filter by relevance
        unique_concepts = {}
        for concept in concepts:
            term = concept['term'].lower()
            if term not in unique_concepts or concept['relevance'] > unique_concepts[term]['relevance']:
                unique_concepts[term] = concept
        
        return list(unique_concepts.values())
    
    def _create_embeddings(self, concepts: List[Dict[str, Any]]) -> np.ndarray:
        """Create embeddings for concepts using OpenAI"""
        texts = []
        for concept in concepts:
            # Create rich text representation
            text = f"{concept['term']} {concept['category']} {' '.join(concept['therapeutic_themes'])}"
            texts.append(text)
        
        try:
            # Get embeddings from OpenAI
            response = self.client.embeddings.create(
                model="text-embedding-3-small",
                input=texts
            )
            
            embeddings = np.array([item.embedding for item in response.data])
            logger.info(f"Created embeddings for {len(concepts)} concepts")
            return embeddings
            
        except Exception as e:
            logger.error(f"Failed to create OpenAI embeddings: {e}")
            # Fallback to TF-IDF
            return self._create_tfidf_embeddings(texts)
    
    def _create_tfidf_embeddings(self, texts: List[str]) -> np.ndarray:
        """Fallback TF-IDF embeddings"""
        try:
            embeddings = self.vectorizer.fit_transform(texts).toarray()
            logger.info(f"Created TF-IDF embeddings for {len(texts)} concepts")
            return embeddings
        except Exception as e:
            logger.error(f"Failed to create TF-IDF embeddings: {e}")
            # Return random embeddings as last resort
            return np.random.rand(len(texts), 100)
    
    def _generate_coordinates(
        self, 
        embeddings: np.ndarray, 
        method: str = "umap"
    ) -> np.ndarray:
        """Generate 2D coordinates for visualization"""
        try:
            if method == "umap":
                reducer = umap.UMAP(
                    n_components=2,
                    n_neighbors=min(15, len(embeddings) - 1),
                    min_dist=0.1,
                    metric='cosine',
                    random_state=42
                )
            else:  # t-SNE
                reducer = TSNE(
                    n_components=2,
                    perplexity=min(30, len(embeddings) - 1),
                    random_state=42
                )
            
            coordinates = reducer.fit_transform(embeddings)
            logger.info(f"Generated 2D coordinates using {method}")
            return coordinates
            
        except Exception as e:
            logger.error(f"Failed to generate coordinates with {method}: {e}")
            # Return random coordinates
            return np.random.rand(len(embeddings), 2)
    
    def _create_nodes(
        self, 
        concepts: List[Dict[str, Any]], 
        coordinates: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Create node data for visualization"""
        nodes = []
        
        for i, concept in enumerate(concepts):
            node = {
                'id': f"node_{i}",
                'label': concept['term'],
                'x': float(coordinates[i, 0]),
                'y': float(coordinates[i, 1]),
                'size': max(10, concept['relevance'] * 30),
                'color': self._get_node_color(concept),
                'category': concept['category'],
                'sentiment': concept['sentiment'],
                'relevance': concept['relevance'],
                'speaker': concept['speaker'],
                'timestamp': concept['timestamp'],
                'therapeutic_themes': concept['therapeutic_themes']
            }
            nodes.append(node)
        
        return nodes
    
    def _create_edges(
        self, 
        concepts: List[Dict[str, Any]], 
        embeddings: np.ndarray
    ) -> List[Dict[str, Any]]:
        """Create edges based on concept similarity"""
        edges = []
        
        # Calculate similarity matrix
        similarity_matrix = cosine_similarity(embeddings)
        
        # Create edges for highly similar concepts
        threshold = 0.3  # Similarity threshold
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = similarity_matrix[i, j]
                
                if similarity > threshold:
                    edge = {
                        'source': f"node_{i}",
                        'target': f"node_{j}",
                        'weight': float(similarity),
                        'color': 'rgba(128, 128, 128, 0.5)'
                    }
                    edges.append(edge)
        
        return edges
    
    def _identify_clusters(
        self, 
        coordinates: np.ndarray, 
        concepts: List[Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        """Identify concept clusters"""
        from sklearn.cluster import KMeans
        
        try:
            # Determine optimal number of clusters
            n_clusters = min(5, max(2, len(concepts) // 3))
            
            kmeans = KMeans(n_clusters=n_clusters, random_state=42)
            cluster_labels = kmeans.fit_predict(coordinates)
            
            clusters = []
            for i in range(n_clusters):
                cluster_concepts = [
                    concepts[j]['term'] for j, label in enumerate(cluster_labels) if label == i
                ]
                
                if cluster_concepts:
                    clusters.append({
                        'id': f"cluster_{i}",
                        'concepts': cluster_concepts,
                        'center': {
                            'x': float(kmeans.cluster_centers_[i, 0]),
                            'y': float(kmeans.cluster_centers_[i, 1])
                        },
                        'theme': self._identify_cluster_theme(cluster_concepts)
                    })
            
            return clusters
            
        except Exception as e:
            logger.error(f"Failed to identify clusters: {e}")
            return []
    
    def _get_node_color(self, concept: Dict[str, Any]) -> str:
        """Get color for node based on category and sentiment"""
        category_colors = {
            'symptom': '#ff6b6b',
            'emotion': '#4ecdc4',
            'cognitive': '#45b7d1',
            'behavioral': '#96ceb4',
            'social': '#feca57',
            'general': '#a8a8a8'
        }
        
        base_color = category_colors.get(concept['category'], '#a8a8a8')
        
        # Adjust opacity based on sentiment
        sentiment = concept.get('sentiment', 0)
        opacity = 0.5 + abs(sentiment) * 0.5
        
        return base_color + f"{int(opacity * 255):02x}"
    
    def _identify_cluster_theme(self, concepts: List[str]) -> str:
        """Identify the main theme of a concept cluster"""
        # Simple theme identification based on keywords
        themes = {
            'emotional': ['anxiety', 'stress', 'fear', 'anger', 'sadness', 'joy'],
            'cognitive': ['thinking', 'belief', 'thought', 'memory', 'attention'],
            'behavioral': ['action', 'behavior', 'habit', 'activity', 'response'],
            'social': ['relationship', 'family', 'friend', 'social', 'communication'],
            'therapeutic': ['therapy', 'treatment', 'goal', 'progress', 'intervention']
        }
        
        concept_text = ' '.join(concepts).lower()
        
        theme_scores = {}
        for theme, keywords in themes.items():
            score = sum(1 for keyword in keywords if keyword in concept_text)
            theme_scores[theme] = score
        
        return max(theme_scores.items(), key=lambda x: x[1])[0] if theme_scores else 'general'
    
    def _empty_graph(self) -> Dict[str, Any]:
        """Return empty graph structure"""
        return {
            'nodes': [],
            'edges': [],
            'clusters': [],
            'metadata': {
                'total_concepts': 0,
                'total_nodes': 0,
                'total_edges': 0,
                'embedding_method': 'none',
                'reduction_method': 'none'
            }
        }

def build(nodes: List[str]) -> List[Dict[str, Any]]:
    """
    Simple function to build graph from list of concept strings
    
    Args:
        nodes: List of concept strings
        
    Returns:
        List of node coordinates
    """
    if not nodes:
        return []
    
    # Create fake session data from nodes
    session_data = {
        'processed_segments': [
            {
                'keywords': [{'term': node, 'relevance': 0.8, 'category': 'general', 'sentiment': 0.0}],
                'text': node,
                'speaker': 'UNKNOWN',
                'start': i,
                'therapeutic_themes': []
            }
            for i, node in enumerate(nodes)
        ]
    }
    
    builder = TherapeuticGraphBuilder()
    graph = builder.build_session_graph(session_data)
    
    return graph.get('nodes', [])