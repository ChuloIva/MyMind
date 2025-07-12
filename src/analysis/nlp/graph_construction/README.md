# Graph Construction & Embedding Visualization

This module provides UMAP-based dimensionality reduction for therapeutic concept visualization, transforming high-dimensional text embeddings into interactive 2D representations.

## Core Implementation

### `graph_builder.py`

The current implementation uses UMAP for dimensionality reduction of text embeddings:

```python
from common.embeddings import embed_batch
import umap
import numpy as np

def build(nodes: list[str]):
    vecs = embed_batch(nodes)
    xy = umap.UMAP(n_components=2).fit_transform(np.array(vecs))
    return [{"id": n, "x": float(x), "y": float(y)} for n,(x,y) in zip(nodes, xy)]
```

## Technical Architecture

### Embedding Pipeline
1. **Text Input**: Therapeutic terms and concepts
2. **Embedding Generation**: OpenAI `text-embedding-3-small` model
3. **Dimensionality Reduction**: UMAP projection to 2D coordinates
4. **Coordinate Mapping**: JSON output for visualization

### Core Features
- **High-Quality Embeddings**: 1536-dimensional vectors from OpenAI
- **Semantic Preservation**: UMAP maintains semantic relationships
- **Interactive Visualization**: 2D coordinates for web-based plotting
- **Scalable Processing**: Efficient batch processing of concepts

## Usage Examples

### Basic Usage
```python
from graph_builder import build

# Therapeutic concepts
concepts = [
    "anxiety",
    "work stress",
    "relationship issues",
    "coping mechanisms",
    "therapy goals"
]

# Generate visualization coordinates
coordinates = build(concepts)

# Output structure
# [
#   {"id": "anxiety", "x": 1.2, "y": -0.8},
#   {"id": "work stress", "x": 0.8, "y": -1.2},
#   {"id": "relationship issues", "x": -0.5, "y": 0.9},
#   {"id": "coping mechanisms", "x": -1.1, "y": 0.3},
#   {"id": "therapy goals", "x": 0.2, "y": 1.5}
# ]
```

### Session-Specific Analysis
```python
# Build visualization for specific therapy session
def build_session_graph(session_id: str) -> dict:
    """Create concept graph for therapy session"""
    
    # Extract keywords from session
    keywords = get_session_keywords(session_id)
    terms = [kw["term"] for kw in keywords]
    
    # Generate coordinates
    coordinates = build(terms)
    
    # Enrich with session-specific data
    enriched_coords = []
    for coord in coordinates:
        term_data = next(kw for kw in keywords if kw["term"] == coord["id"])
        enriched_coords.append({
            **coord,
            "sentiment": term_data["sentiment"],
            "confidence": term_data["confidence"],
            "frequency": term_data.get("frequency", 1),
            "category": term_data.get("category", "unknown")
        })
    
    return {
        "session_id": session_id,
        "graph": enriched_coords,
        "generated_at": datetime.now().isoformat()
    }
```

### Multi-Session Comparison
```python
# Compare concept evolution across sessions
def build_evolution_graph(session_ids: list[str]) -> dict:
    """Track concept evolution across multiple sessions"""
    
    evolution_data = {
        "sessions": [],
        "concept_trajectories": {},
        "emerging_concepts": [],
        "declining_concepts": []
    }
    
    all_concepts = set()
    
    # Process each session
    for session_id in session_ids:
        session_graph = build_session_graph(session_id)
        evolution_data["sessions"].append(session_graph)
        
        # Track all concepts
        session_concepts = {coord["id"] for coord in session_graph["graph"]}
        all_concepts.update(session_concepts)
    
    # Analyze concept trajectories
    for concept in all_concepts:
        trajectory = []
        for session_data in evolution_data["sessions"]:
            concept_data = next(
                (coord for coord in session_data["graph"] if coord["id"] == concept),
                None
            )
            trajectory.append(concept_data)
        
        evolution_data["concept_trajectories"][concept] = trajectory
    
    # Identify emerging and declining concepts
    evolution_data["emerging_concepts"] = find_emerging_concepts(evolution_data)
    evolution_data["declining_concepts"] = find_declining_concepts(evolution_data)
    
    return evolution_data
```

## UMAP Configuration

### Standard Configuration
```python
# Default UMAP parameters for therapeutic analysis
umap_config = {
    "n_components": 2,          # 2D visualization
    "n_neighbors": 15,          # Local neighborhood size
    "min_dist": 0.1,           # Minimum distance between points
    "metric": "cosine",         # Distance metric for embeddings
    "random_state": 42          # Reproducible results
}

reducer = umap.UMAP(**umap_config)
```

### Advanced Configuration
```python
# Optimized for therapeutic concept clustering
def create_therapeutic_umap(n_concepts: int) -> umap.UMAP:
    """Create UMAP reducer optimized for therapeutic concepts"""
    
    # Adjust parameters based on concept count
    if n_concepts < 20:
        n_neighbors = max(5, n_concepts // 3)
        min_dist = 0.2
    elif n_concepts < 50:
        n_neighbors = 15
        min_dist = 0.1
    else:
        n_neighbors = 30
        min_dist = 0.05
    
    return umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric="cosine",
        random_state=42,
        transform_seed=42  # Consistent transforms
    )
```

## Performance Optimization

### Batch Processing
```python
# Efficient batch processing for large concept sets
def build_batch(concept_batches: list[list[str]], max_batch_size: int = 100):
    """Process multiple concept sets efficiently"""
    
    results = []
    
    for batch in concept_batches:
        # Limit batch size for memory efficiency
        if len(batch) > max_batch_size:
            sub_batches = [batch[i:i+max_batch_size] for i in range(0, len(batch), max_batch_size)]
            batch_results = []
            
            for sub_batch in sub_batches:
                coords = build(sub_batch)
                batch_results.extend(coords)
            
            results.append(batch_results)
        else:
            coords = build(batch)
            results.append(coords)
    
    return results
```

### Caching Strategy
```python
# Cache embeddings for repeated analysis
import pickle
import hashlib

class EmbeddingCache:
    def __init__(self, cache_dir: str = "cache/embeddings"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cache_key(self, terms: list[str]) -> str:
        """Generate cache key for term list"""
        terms_str = "|".join(sorted(terms))
        return hashlib.md5(terms_str.encode()).hexdigest()
    
    def get_cached_embeddings(self, terms: list[str]) -> Optional[list[list[float]]]:
        """Retrieve cached embeddings if available"""
        cache_key = self.get_cache_key(terms)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_embeddings(self, terms: list[str], embeddings: list[list[float]]):
        """Cache embeddings for future use"""
        cache_key = self.get_cache_key(terms)
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        with open(cache_file, 'wb') as f:
            pickle.dump(embeddings, f)

# Usage with caching
embedding_cache = EmbeddingCache()

def build_with_cache(nodes: list[str]):
    """Build graph with embedding caching"""
    
    # Check cache first
    cached_embeddings = embedding_cache.get_cached_embeddings(nodes)
    
    if cached_embeddings is not None:
        vecs = cached_embeddings
    else:
        vecs = embed_batch(nodes)
        embedding_cache.cache_embeddings(nodes, vecs)
    
    # Apply UMAP
    xy = umap.UMAP(n_components=2).fit_transform(np.array(vecs))
    return [{"id": n, "x": float(x), "y": float(y)} for n,(x,y) in zip(nodes, xy)]
```

## Quality Metrics

### Embedding Quality Assessment
```python
# Validate embedding and projection quality
def assess_visualization_quality(terms: list[str], coordinates: list[dict]) -> dict:
    """Assess quality of generated visualization"""
    
    quality_metrics = {
        "semantic_preservation": 0.0,
        "cluster_separation": 0.0,
        "therapeutic_relevance": 0.0,
        "visualization_clarity": 0.0
    }
    
    # Semantic preservation (compare original vs projected distances)
    original_embeddings = embed_batch(terms)
    projected_coords = [(coord["x"], coord["y"]) for coord in coordinates]
    
    # Calculate correlation between original and projected distances
    from scipy.spatial.distance import pdist
    from scipy.stats import pearsonr
    
    original_distances = pdist(original_embeddings, metric='cosine')
    projected_distances = pdist(projected_coords, metric='euclidean')
    
    correlation, _ = pearsonr(original_distances, projected_distances)
    quality_metrics["semantic_preservation"] = max(0, correlation)
    
    # Cluster separation (silhouette score)
    if len(coordinates) > 2:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        n_clusters = min(3, len(coordinates) // 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        cluster_labels = kmeans.fit_predict(projected_coords)
        
        if len(set(cluster_labels)) > 1:
            quality_metrics["cluster_separation"] = silhouette_score(projected_coords, cluster_labels)
    
    # Therapeutic relevance
    therapeutic_terms = load_therapeutic_vocabulary()
    relevant_count = sum(1 for term in terms if term in therapeutic_terms)
    quality_metrics["therapeutic_relevance"] = relevant_count / len(terms)
    
    # Visualization clarity (avoid overlapping points)
    min_distance = min(
        np.sqrt((c1["x"] - c2["x"])**2 + (c1["y"] - c2["y"])**2)
        for i, c1 in enumerate(coordinates)
        for j, c2 in enumerate(coordinates)
        if i != j
    )
    quality_metrics["visualization_clarity"] = min(1.0, min_distance / 0.1)
    
    return quality_metrics
```

## Integration Examples

### Frontend Integration
```python
# API endpoint for visualization data
@app.get("/api/sessions/{session_id}/visualization")
def get_session_visualization(session_id: str):
    """Get visualization data for frontend"""
    
    # Build session graph
    session_graph = build_session_graph(session_id)
    
    # Assess quality
    terms = [coord["id"] for coord in session_graph["graph"]]
    quality = assess_visualization_quality(terms, session_graph["graph"])
    
    return {
        "session_id": session_id,
        "visualization": session_graph["graph"],
        "quality_metrics": quality,
        "generated_at": session_graph["generated_at"]
    }
```

### Database Integration
```python
# Store visualization data in database
from src.database.models import SessionVisualization

def save_visualization(session_id: str, coordinates: list[dict]):
    """Save visualization data to database"""
    
    visualization = SessionVisualization(
        session_id=session_id,
        coordinates=coordinates,
        generated_at=datetime.now()
    )
    
    db.add(visualization)
    db.commit()
    
    return visualization.id
```

## Troubleshooting

### Common Issues

**1. Memory Issues with Large Concept Sets**
```python
# Solution: Process in batches
def build_large_set(nodes: list[str], batch_size: int = 50):
    if len(nodes) <= batch_size:
        return build(nodes)
    
    # Process in batches and combine
    all_coords = []
    for i in range(0, len(nodes), batch_size):
        batch = nodes[i:i+batch_size]
        coords = build(batch)
        all_coords.extend(coords)
    
    return all_coords
```

**2. Poor Clustering Quality**
```python
# Solution: Adjust UMAP parameters
def build_with_custom_params(nodes: list[str], n_neighbors: int = None, min_dist: float = None):
    # Auto-adjust parameters based on data size
    if n_neighbors is None:
        n_neighbors = max(5, min(15, len(nodes) // 3))
    if min_dist is None:
        min_dist = 0.1 if len(nodes) > 20 else 0.2
    
    vecs = embed_batch(nodes)
    reducer = umap.UMAP(n_components=2, n_neighbors=n_neighbors, min_dist=min_dist)
    xy = reducer.fit_transform(np.array(vecs))
    
    return [{"id": n, "x": float(x), "y": float(y)} for n,(x,y) in zip(nodes, xy)]
```

This implementation provides a robust foundation for therapeutic concept visualization, enabling clinicians to understand complex relationships between therapeutic themes and track their evolution over time.
