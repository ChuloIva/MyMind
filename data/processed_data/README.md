# Processed Data

This directory contains the results of AI analysis performed on transcripts, including keyword extraction, sentiment analysis, and therapeutic insights.

## Data Categories

### 1. Keyword Analysis
- **Keywords**: Extracted terms with sentiment scores
- **Temporal mapping**: Start/end timestamps
- **Clustering**: Thematic groupings
- **Frequency analysis**: Usage patterns

### 2. Embedding Visualizations
- **Vector representations**: Text embeddings
- **UMAP projections**: 2D visualization coordinates
- **Cluster analysis**: Related concept groups
- **Similarity metrics**: Distance calculations

### 3. Therapeutic Insights
- **Cognitive distortions**: CBT-based pattern detection
- **Schema identification**: Maladaptive patterns
- **Emotional patterns**: Mood trajectory analysis
- **Progress indicators**: Session-to-session changes

### 4. Generated Reports
- **Session summaries**: Key themes and insights
- **Progress reports**: Client trajectory analysis
- **Recommendations**: Therapeutic intervention suggestions
- **Visualizations**: Charts and graphs

## File Structure

```
processed_data/
├── keywords/              # Keyword extraction results
│   └── {session_id}_keywords.json
├── embeddings/            # Vector representations
│   └── {session_id}_embeddings.json
├── insights/              # Therapeutic analysis
│   └── {session_id}_insights.json
├── reports/               # Generated summaries
│   └── {session_id}_report.md
└── visualizations/        # Charts and graphs
    └── {session_id}_viz.json
```

## Data Formats

### Keywords (`keywords/`)
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "processed_at": "2023-12-01T14:30:00Z",
  "model": "gpt-4o-mini",
  "keywords": [
    {
      "sentence_id": 0,
      "keywords": [
        {
          "term": "anxiety",
          "sentiment": -0.7,
          "start_ms": 1500,
          "end_ms": 2200,
          "confidence": 0.92
        }
      ]
    }
  ]
}
```

### Embeddings (`embeddings/`)
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "model": "text-embedding-3-small",
  "embeddings": [
    {
      "id": "segment_0",
      "text": "I'm feeling anxious today",
      "vector": [0.1, -0.2, 0.3, ...],
      "umap_coords": {"x": 1.2, "y": -0.8}
    }
  ]
}
```

### Insights (`insights/`)
```json
{
  "session_id": "550e8400-e29b-41d4-a716-446655440000",
  "analysis_date": "2023-12-01T14:30:00Z",
  "therapeutic_methods": {
    "cbt": {
      "distortions": [
        {
          "type": "catastrophizing",
          "confidence": 0.85,
          "examples": ["Everything will go wrong"],
          "timestamp": 1500
        }
      ]
    },
    "schema_therapy": {
      "schemas": [
        {
          "name": "abandonment",
          "strength": 0.7,
          "evidence": ["I'm worried they'll leave me"]
        }
      ]
    }
  },
  "sentiment_trajectory": {
    "overall": -0.3,
    "by_minute": [-0.2, -0.4, -0.1, 0.2],
    "volatility": 0.6
  }
}
```

## Processing Pipeline

### 1. Keyword Extraction
- **Input**: Session transcripts
- **Model**: GPT-4o-mini with structured output
- **Output**: Temporal keyword mapping
- **Storage**: PostgreSQL with JSONB indexing

### 2. Embedding Generation
- **Input**: Transcript segments
- **Model**: text-embedding-3-small
- **Processing**: UMAP dimensionality reduction
- **Output**: 2D visualization coordinates

### 3. Therapeutic Analysis
- **CBT Analysis**: Cognitive distortion detection
- **Schema Therapy**: Maladaptive pattern identification
- **Sentiment Analysis**: Emotional trajectory tracking
- **Progress Metrics**: Session-to-session comparison

### 4. Report Generation
- **Template-based**: Markdown report structure
- **Streaming**: Real-time insight generation
- **Personalization**: Client-specific recommendations
- **Visualization**: Interactive charts and graphs

## Quality Assurance

### Validation Checks
- **Confidence thresholds**: Minimum accuracy requirements
- **Consistency validation**: Cross-session pattern checking
- **Therapeutic validity**: Clinical accuracy verification
- **Data integrity**: Format and structure validation

### Performance Metrics
- **Processing speed**: Analysis completion time
- **Accuracy scores**: Therapeutic insight precision
- **Coverage metrics**: Content analysis completeness
- **User feedback**: Clinical validation scores

## Data Lifecycle

### Retention Policy
- **Keywords**: 2 years
- **Embeddings**: 1 year
- **Insights**: 5 years (clinical record)
- **Reports**: Client-controlled retention

### Archival Process
- **Compression**: Gzip for long-term storage
- **Encryption**: AES-256 for archived data
- **Indexing**: Metadata for retrieval
- **Backup**: Multi-region replication

## Integration & Access

### API Endpoints
- **Keyword search**: Query by term or theme
- **Insight retrieval**: Get analysis by session
- **Report generation**: Create summaries on demand
- **Visualization**: Export charts and graphs

### Database Integration
- **Real-time updates**: Stream processing results
- **Query optimization**: Indexed search capabilities
- **Relationship mapping**: Cross-session correlations
- **Audit logging**: Processing history tracking
