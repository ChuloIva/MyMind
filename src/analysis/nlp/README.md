# 3.1. Natural Language Processing (NLP)

This module provides advanced NLP capabilities for therapeutic session analysis, focusing on semantic understanding, concept relationships, and visual representation of therapeutic themes.

## Core Components

### Graph Construction
- **Embedding Generation**: Convert text to vector representations
- **Dimensionality Reduction**: UMAP/t-SNE for 2D visualization
- **Concept Mapping**: Semantic relationship visualization
- **Temporal Analysis**: Track concept evolution over time

### Keyword Analysis
- **Semantic Extraction**: Meaningful term identification
- **Relationship Mapping**: Inter-concept connections
- **Sentiment Integration**: Emotional context preservation
- **Confidence Scoring**: Reliability metrics

## Key Functions

### Embedding Visualization
- **UMAP Projection**: High-dimensional embedding reduction to 2D coordinates
- **Interactive Mapping**: Real-time concept relationship visualization
- **Cluster Analysis**: Automatic thematic grouping
- **Temporal Evolution**: Session-to-session concept tracking

### Semantic Analysis
- **Keyword Generation**: Therapeutically relevant term extraction
- **Sentiment Scoring**: Emotional valence association with concepts
- **Entity Recognition**: Life domain identification (work, relationships, health)
- **Context Preservation**: Maintain semantic meaning across transformations

## Implementation Architecture

### Core Processing Pipeline
```python
# Input: Keywords from preprocessing module
keywords = [
    {"term": "anxiety", "sentiment": -0.7, "confidence": 0.92},
    {"term": "work stress", "sentiment": -0.6, "confidence": 0.88},
    {"term": "relationship", "sentiment": 0.3, "confidence": 0.85}
]

# Process through NLP pipeline
from nlp.graph_construction.graph_builder import build

# Generate embeddings and 2D coordinates
terms = [kw["term"] for kw in keywords]
coordinates = build(terms)

# Output: Visualization-ready data
# [
#   {"id": "anxiety", "x": 1.2, "y": -0.8},
#   {"id": "work stress", "x": 0.8, "y": -1.2},
#   {"id": "relationship", "x": -0.5, "y": 0.9}
# ]
```

### Advanced Features
- **Multi-Session Analysis**: Track concept evolution across therapy sessions
- **Cluster Detection**: Identify thematic groups automatically
- **Semantic Similarity**: Calculate relationship strength between concepts
- **Temporal Alignment**: Maintain temporal context in visualizations

## Technical Specifications

### Embedding Model
- **Model**: `text-embedding-3-small` (OpenAI)
- **Dimensions**: 1536 → 2 (UMAP reduction)
- **Context Window**: 8,192 tokens
- **Languages**: English (primary), multilingual support

### Performance Metrics
- **Processing Speed**: 100+ terms per second
- **Memory Usage**: <1GB for typical sessions
- **Accuracy**: 85%+ semantic similarity preservation
- **Latency**: <500ms for visualization generation

### Quality Assurance
- **Embedding Quality**: Cosine similarity validation
- **Cluster Coherence**: Silhouette score optimization
- **Temporal Consistency**: Cross-session alignment verification
- **Clinical Relevance**: Therapeutic significance scoring

## Integration Points

### Database Integration
```python
# Retrieve processed keywords from database
from src.database.models import SessionSentence

def get_session_keywords(session_id: str) -> list[dict]:
    """Extract keywords from database for NLP processing"""
    sentences = db.query(SessionSentence).filter(
        SessionSentence.session_id == session_id
    ).all()
    
    all_keywords = []
    for sentence in sentences:
        if sentence.keywords:
            all_keywords.extend(sentence.keywords)
    
    return all_keywords
```

### Visualization Pipeline
```python
# Generate complete visualization data
def create_session_visualization(session_id: str) -> dict:
    """Create comprehensive session visualization"""
    
    # Get processed keywords
    keywords = get_session_keywords(session_id)
    terms = [kw["term"] for kw in keywords]
    
    # Generate embeddings and coordinates
    coordinates = build(terms)
    
    # Enrich with sentiment and confidence data
    for coord in coordinates:
        term_data = next(kw for kw in keywords if kw["term"] == coord["id"])
        coord.update({
            "sentiment": term_data["sentiment"],
            "confidence": term_data["confidence"],
            "category": term_data.get("category", "unknown"),
            "frequency": calculate_term_frequency(term_data["term"], keywords)
        })
    
    return {
        "session_id": session_id,
        "visualization": coordinates,
        "clusters": identify_clusters(coordinates),
        "themes": extract_themes(coordinates),
        "generated_at": datetime.now().isoformat()
    }
```

### Multi-Session Analysis
```python
# Track concept evolution across sessions
def analyze_concept_evolution(client_id: str, session_ids: list[str]) -> dict:
    """Analyze how concepts change across therapy sessions"""
    
    evolution_data = {
        "client_id": client_id,
        "sessions": len(session_ids),
        "concept_timeline": [],
        "emerging_themes": [],
        "declining_themes": [],
        "persistent_themes": []
    }
    
    for session_id in session_ids:
        session_viz = create_session_visualization(session_id)
        evolution_data["concept_timeline"].append({
            "session_id": session_id,
            "concepts": session_viz["visualization"],
            "themes": session_viz["themes"]
        })
    
    # Analyze patterns
    evolution_data["emerging_themes"] = find_emerging_themes(evolution_data["concept_timeline"])
    evolution_data["declining_themes"] = find_declining_themes(evolution_data["concept_timeline"])
    evolution_data["persistent_themes"] = find_persistent_themes(evolution_data["concept_timeline"])
    
    return evolution_data
```

## Advanced Analytics

### Cluster Analysis
```python
# Automatic thematic clustering
def identify_clusters(coordinates: list[dict], min_cluster_size: int = 3) -> list[dict]:
    """Identify semantic clusters in concept space"""
    
    # Extract coordinates for clustering
    points = np.array([[coord["x"], coord["y"]] for coord in coordinates])
    
    # Apply DBSCAN clustering
    from sklearn.cluster import DBSCAN
    clustering = DBSCAN(eps=0.5, min_samples=min_cluster_size).fit(points)
    
    # Group concepts by cluster
    clusters = {}
    for i, label in enumerate(clustering.labels_):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(coordinates[i])
    
    # Format cluster data
    cluster_data = []
    for cluster_id, concepts in clusters.items():
        if cluster_id != -1:  # Exclude noise points
            cluster_data.append({
                "cluster_id": cluster_id,
                "concepts": concepts,
                "theme": extract_cluster_theme(concepts),
                "sentiment": calculate_cluster_sentiment(concepts),
                "size": len(concepts)
            })
    
    return cluster_data
```

### Temporal Analysis
```python
# Track concept changes over time
def analyze_temporal_patterns(evolution_data: dict) -> dict:
    """Analyze how concepts evolve over therapy sessions"""
    
    patterns = {
        "concept_stability": {},
        "sentiment_trends": {},
        "theme_progression": [],
        "intervention_effectiveness": {}
    }
    
    # Analyze concept stability
    for concept in get_all_concepts(evolution_data):
        appearances = count_concept_appearances(concept, evolution_data)
        stability = calculate_stability_score(appearances)
        patterns["concept_stability"][concept] = stability
    
    # Analyze sentiment trends
    for concept in patterns["concept_stability"]:
        sentiment_series = extract_sentiment_series(concept, evolution_data)
        trend = calculate_trend(sentiment_series)
        patterns["sentiment_trends"][concept] = trend
    
    # Identify theme progression
    patterns["theme_progression"] = identify_theme_progression(evolution_data)
    
    return patterns
```

## Quality Metrics

### Embedding Quality
```python
# Validate embedding quality
def validate_embedding_quality(terms: list[str], embeddings: list[list[float]]) -> dict:
    """Assess quality of generated embeddings"""
    
    quality_metrics = {
        "semantic_coherence": 0.0,
        "clustering_quality": 0.0,
        "therapeutic_relevance": 0.0,
        "temporal_consistency": 0.0
    }
    
    # Semantic coherence (cosine similarity between related terms)
    related_pairs = identify_related_pairs(terms)
    similarities = []
    for term1, term2 in related_pairs:
        idx1, idx2 = terms.index(term1), terms.index(term2)
        similarity = cosine_similarity(embeddings[idx1], embeddings[idx2])
        similarities.append(similarity)
    
    quality_metrics["semantic_coherence"] = np.mean(similarities)
    
    # Clustering quality (silhouette score)
    if len(embeddings) > 2:
        from sklearn.metrics import silhouette_score
        from sklearn.cluster import KMeans
        
        kmeans = KMeans(n_clusters=min(3, len(embeddings)//2))
        cluster_labels = kmeans.fit_predict(embeddings)
        quality_metrics["clustering_quality"] = silhouette_score(embeddings, cluster_labels)
    
    # Therapeutic relevance (based on clinical vocabulary)
    therapeutic_terms = load_therapeutic_vocabulary()
    relevant_count = sum(1 for term in terms if term in therapeutic_terms)
    quality_metrics["therapeutic_relevance"] = relevant_count / len(terms)
    
    return quality_metrics
```

### Performance Monitoring
```python
# Monitor NLP processing performance
class NLPMonitor:
    def __init__(self):
        self.processing_times = []
        self.quality_scores = []
        self.error_rates = []
    
    def track_processing(self, func):
        def wrapper(*args, **kwargs):
            start_time = time.time()
            try:
                result = func(*args, **kwargs)
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                
                # Validate quality if possible
                if "validation" in kwargs:
                    quality = validate_embedding_quality(args[0], result)
                    self.quality_scores.append(quality)
                
                return result
            except Exception as e:
                self.error_rates.append(1)
                logging.error(f"NLP processing error: {e}")
                raise
        return wrapper
```

## Future Enhancements

### Planned Features
- **Multi-Language Support**: Expand beyond English
- **Dynamic Embeddings**: Real-time embedding updates
- **Custom Models**: Therapy-specific embedding models
- **3D Visualization**: Enhanced spatial representation

### Research Areas
- **Therapeutic Embedding Spaces**: Domain-specific vector representations
- **Intervention Effectiveness**: Measure concept change post-intervention
- **Predictive Modeling**: Early warning systems based on concept patterns
- **Cultural Adaptation**: Culture-specific concept relationships

This NLP module provides the foundation for understanding and visualizing therapeutic concepts, enabling clinicians to see patterns and relationships that might not be immediately apparent in text-based analysis.
