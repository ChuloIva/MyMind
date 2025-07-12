# RAG Implementation - Retrieval-Augmented Generation

This module provides a sophisticated question-answering system using LangChain's RetrievalQA framework, enabling contextual information retrieval from therapy session data and therapeutic literature.

## Core Implementation

### `rag.py`

The current implementation uses LangChain with FAISS vector store for efficient retrieval:

```python
from langchain.chains import RetrievalQA
from langchain.llms import OpenAI
from langchain.vectorstores import FAISS
from langchain.embeddings import OpenAIEmbeddings
from langchain.text_splitter import CharacterTextSplitter
from uuid import UUID

def get_qa_chain(session_id: UUID):
    """
    Create a RetrievalQA chain for session-specific queries.
    Loads session data and creates searchable knowledge base.
    """
    # Load session documents
    documents = load_session_documents(session_id)
    
    # Split documents into chunks
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    
    # Create embeddings and vector store
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(texts, embeddings)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever(),
    )
    return qa_chain
```

## Architecture Overview

### Components
1. **Document Loading**: Session transcripts and therapeutic literature
2. **Text Splitting**: Optimal chunk size for retrieval
3. **Embedding Generation**: OpenAI embeddings for semantic search
4. **Vector Store**: FAISS for efficient similarity search
5. **QA Chain**: LangChain RetrievalQA for answer generation

### Data Flow
```
Session Data → Text Chunks → Embeddings → Vector Store → Retrieval → Answer Generation
```

## Usage Examples

### Basic Session Query
```python
from rag import get_qa_chain

# Create QA chain for specific session
session_id = "550e8400-e29b-41d4-a716-446655440000"
qa_chain = get_qa_chain(session_id)

# Ask questions about the session
answer = qa_chain.run("What are the main themes discussed in this session?")
print(answer)

# Example output:
# "The main themes in this session include work-related anxiety, 
# relationship conflicts, and coping strategies. The client expressed 
# concerns about job security and mentioned using avoidance behaviors."
```

### Advanced Query Examples
```python
# Therapeutic assessment questions
therapeutic_questions = [
    "What cognitive distortions were identified?",
    "What coping mechanisms does the client use?",
    "What are the client's primary concerns?",
    "What progress has been made since the last session?",
    "What therapeutic interventions were suggested?"
]

for question in therapeutic_questions:
    answer = qa_chain.run(question)
    print(f"Q: {question}")
    print(f"A: {answer}\n")
```

### Multi-Session Analysis
```python
def analyze_multiple_sessions(session_ids: list[UUID]) -> dict:
    """Analyze patterns across multiple therapy sessions"""
    
    results = {
        "session_summaries": {},
        "common_themes": [],
        "progress_indicators": [],
        "therapeutic_recommendations": []
    }
    
    for session_id in session_ids:
        qa_chain = get_qa_chain(session_id)
        
        # Get session summary
        summary = qa_chain.run("Summarize the key points from this session")
        results["session_summaries"][str(session_id)] = summary
        
        # Extract themes
        themes = qa_chain.run("What are the main therapeutic themes?")
        results["common_themes"].append(themes)
    
    # Analyze common patterns
    all_themes = " ".join(results["common_themes"])
    pattern_qa = create_pattern_analysis_chain(all_themes)
    
    results["progress_indicators"] = pattern_qa.run("What progress patterns are evident?")
    results["therapeutic_recommendations"] = pattern_qa.run("What therapeutic interventions are recommended?")
    
    return results
```

## Enhanced Features

### Custom Document Loading
```python
def load_session_documents(session_id: UUID) -> list[Document]:
    """Load comprehensive session data for RAG"""
    
    documents = []
    
    # Load session transcript
    transcript = get_session_transcript(session_id)
    documents.append(Document(
        page_content=transcript,
        metadata={"type": "transcript", "session_id": str(session_id)}
    ))
    
    # Load processed keywords
    keywords = get_session_keywords(session_id)
    keyword_text = format_keywords_for_rag(keywords)
    documents.append(Document(
        page_content=keyword_text,
        metadata={"type": "keywords", "session_id": str(session_id)}
    ))
    
    # Load therapeutic insights
    insights = get_session_insights(session_id)
    insight_text = format_insights_for_rag(insights)
    documents.append(Document(
        page_content=insight_text,
        metadata={"type": "insights", "session_id": str(session_id)}
    ))
    
    # Load client history (if available)
    client_history = get_client_history(session_id)
    if client_history:
        documents.append(Document(
            page_content=client_history,
            metadata={"type": "history", "session_id": str(session_id)}
        ))
    
    return documents
```

### Therapeutic Literature Integration
```python
def create_therapeutic_knowledge_base() -> FAISS:
    """Create knowledge base with therapeutic literature"""
    
    # Load therapeutic reference materials
    therapeutic_docs = [
        load_cbt_guidelines(),
        load_schema_therapy_manual(),
        load_diagnostic_criteria(),
        load_intervention_protocols()
    ]
    
    # Combine with session data
    all_documents = []
    for doc_set in therapeutic_docs:
        all_documents.extend(doc_set)
    
    # Create comprehensive vector store
    embeddings = OpenAIEmbeddings()
    knowledge_base = FAISS.from_documents(all_documents, embeddings)
    
    return knowledge_base
```

### Contextual QA Chain
```python
def create_contextual_qa_chain(session_id: UUID) -> RetrievalQA:
    """Create QA chain with therapeutic context"""
    
    # Load session-specific documents
    session_docs = load_session_documents(session_id)
    
    # Load therapeutic knowledge base
    therapeutic_kb = create_therapeutic_knowledge_base()
    
    # Combine session data with therapeutic literature
    embeddings = OpenAIEmbeddings()
    session_vectorstore = FAISS.from_documents(session_docs, embeddings)
    
    # Merge vector stores
    combined_vectorstore = merge_vector_stores(session_vectorstore, therapeutic_kb)
    
    # Create enhanced QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(temperature=0.1),  # More focused responses
        chain_type="stuff",
        retriever=combined_vectorstore.as_retriever(
            search_kwargs={"k": 5}  # Retrieve top 5 relevant chunks
        ),
        return_source_documents=True  # Include source information
    )
    
    return qa_chain
```

## Advanced Query Patterns

### Therapeutic Assessment
```python
def conduct_therapeutic_assessment(session_id: UUID) -> dict:
    """Comprehensive therapeutic assessment using RAG"""
    
    qa_chain = create_contextual_qa_chain(session_id)
    
    assessment = {
        "cognitive_patterns": {},
        "emotional_patterns": {},
        "behavioral_patterns": {},
        "therapeutic_goals": {},
        "intervention_suggestions": {}
    }
    
    # Cognitive assessment
    assessment["cognitive_patterns"] = {
        "distortions": qa_chain.run("What cognitive distortions are present?"),
        "thinking_patterns": qa_chain.run("What thinking patterns are evident?"),
        "schemas": qa_chain.run("What underlying schemas are activated?")
    }
    
    # Emotional assessment
    assessment["emotional_patterns"] = {
        "primary_emotions": qa_chain.run("What are the primary emotions expressed?"),
        "emotional_regulation": qa_chain.run("How does the client regulate emotions?"),
        "triggers": qa_chain.run("What emotional triggers are identified?")
    }
    
    # Behavioral assessment
    assessment["behavioral_patterns"] = {
        "coping_strategies": qa_chain.run("What coping strategies are used?"),
        "avoidance_patterns": qa_chain.run("What avoidance behaviors are present?"),
        "adaptive_behaviors": qa_chain.run("What adaptive behaviors are noted?")
    }
    
    # Therapeutic planning
    assessment["therapeutic_goals"] = qa_chain.run("What therapeutic goals should be prioritized?")
    assessment["intervention_suggestions"] = qa_chain.run("What interventions are recommended?")
    
    return assessment
```

### Progress Tracking
```python
def track_therapeutic_progress(session_ids: list[UUID]) -> dict:
    """Track progress across multiple sessions"""
    
    progress_data = {
        "session_count": len(session_ids),
        "timeline": [],
        "improvement_areas": [],
        "persistent_challenges": [],
        "therapeutic_gains": []
    }
    
    for i, session_id in enumerate(session_ids):
        qa_chain = get_qa_chain(session_id)
        
        session_progress = {
            "session_number": i + 1,
            "session_id": str(session_id),
            "mood_assessment": qa_chain.run("How would you rate the client's mood?"),
            "progress_indicators": qa_chain.run("What progress indicators are evident?"),
            "challenges": qa_chain.run("What challenges are still present?"),
            "insights": qa_chain.run("What new insights were gained?")
        }
        
        progress_data["timeline"].append(session_progress)
    
    # Analyze overall progress
    combined_data = combine_session_data(progress_data["timeline"])
    overall_qa = create_pattern_analysis_chain(combined_data)
    
    progress_data["improvement_areas"] = overall_qa.run("What areas show improvement?")
    progress_data["persistent_challenges"] = overall_qa.run("What challenges persist?")
    progress_data["therapeutic_gains"] = overall_qa.run("What therapeutic gains are evident?")
    
    return progress_data
```

## Performance Optimization

### Caching Strategy
```python
from functools import lru_cache
import pickle

class RAGCache:
    def __init__(self, cache_dir: str = "cache/rag"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)
    
    def get_cached_vectorstore(self, session_id: UUID) -> Optional[FAISS]:
        """Retrieve cached vector store"""
        cache_file = self.cache_dir / f"{session_id}_vectorstore.pkl"
        if cache_file.exists():
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        return None
    
    def cache_vectorstore(self, session_id: UUID, vectorstore: FAISS):
        """Cache vector store for future use"""
        cache_file = self.cache_dir / f"{session_id}_vectorstore.pkl"
        with open(cache_file, 'wb') as f:
            pickle.dump(vectorstore, f)

# Usage with caching
rag_cache = RAGCache()

@lru_cache(maxsize=100)
def get_cached_qa_chain(session_id: UUID) -> RetrievalQA:
    """Get QA chain with caching"""
    
    # Check for cached vector store
    cached_vectorstore = rag_cache.get_cached_vectorstore(session_id)
    
    if cached_vectorstore:
        vectorstore = cached_vectorstore
    else:
        # Create new vector store
        documents = load_session_documents(session_id)
        text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
        texts = text_splitter.split_documents(documents)
        embeddings = OpenAIEmbeddings()
        vectorstore = FAISS.from_documents(texts, embeddings)
        
        # Cache for future use
        rag_cache.cache_vectorstore(session_id, vectorstore)
    
    # Create QA chain
    qa_chain = RetrievalQA.from_chain_type(
        llm=OpenAI(),
        chain_type="stuff",
        retriever=vectorstore.as_retriever()
    )
    
    return qa_chain
```

## Quality Assurance

### Answer Validation
```python
def validate_rag_answer(question: str, answer: str, source_docs: list) -> dict:
    """Validate quality of RAG-generated answers"""
    
    quality_metrics = {
        "relevance": 0.0,
        "accuracy": 0.0,
        "completeness": 0.0,
        "therapeutic_appropriateness": 0.0
    }
    
    # Check relevance (question-answer alignment)
    relevance_score = calculate_semantic_similarity(question, answer)
    quality_metrics["relevance"] = relevance_score
    
    # Check accuracy (answer supported by sources)
    accuracy_score = validate_answer_sources(answer, source_docs)
    quality_metrics["accuracy"] = accuracy_score
    
    # Check completeness (comprehensive answer)
    completeness_score = assess_answer_completeness(question, answer)
    quality_metrics["completeness"] = completeness_score
    
    # Check therapeutic appropriateness
    therapeutic_score = assess_therapeutic_appropriateness(answer)
    quality_metrics["therapeutic_appropriateness"] = therapeutic_score
    
    return quality_metrics
```

## Integration Examples

### API Integration
```python
@app.post("/api/sessions/{session_id}/query")
def query_session(session_id: UUID, query: str):
    """Query session data using RAG"""
    
    try:
        qa_chain = get_cached_qa_chain(session_id)
        result = qa_chain({"query": query})
        
        # Validate answer quality
        quality = validate_rag_answer(
            query, 
            result["result"], 
            result.get("source_documents", [])
        )
        
        return {
            "session_id": str(session_id),
            "query": query,
            "answer": result["result"],
            "sources": [doc.metadata for doc in result.get("source_documents", [])],
            "quality_metrics": quality
        }
    except Exception as e:
        return {"error": str(e)}
```

### Therapeutic Dashboard Integration
```python
def generate_session_insights(session_id: UUID) -> dict:
    """Generate comprehensive session insights for dashboard"""
    
    qa_chain = get_cached_qa_chain(session_id)
    
    insights = {
        "session_summary": qa_chain.run("Provide a brief session summary"),
        "key_themes": qa_chain.run("What are the 3 main themes?"),
        "emotional_state": qa_chain.run("What is the client's emotional state?"),
        "progress_indicators": qa_chain.run("What progress is evident?"),
        "next_steps": qa_chain.run("What are the recommended next steps?"),
        "therapeutic_focus": qa_chain.run("What should be the therapeutic focus?")
    }
    
    return insights
```

This RAG implementation provides powerful question-answering capabilities for therapeutic analysis, enabling clinicians to quickly extract insights and track progress across therapy sessions.