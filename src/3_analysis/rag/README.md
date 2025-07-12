# RAG Implementation

This directory contains a basic implementation of a Retrieval-Augmented Generation (RAG) system.

## `rag.py`

This script contains a `SimpleRAG` class that uses a pre-trained model from the `transformers` library to perform question answering. The class can be initialized with a list of documents, which are then indexed for retrieval.

### Usage

To use the `SimpleRAG` class, first install the necessary dependencies:

```bash
pip install -r requirements.txt
```

Then, you can import the `SimpleRAG` class and use it to answer questions based on your documents:

```python
from rag import SimpleRAG

documents = [
    "The DSM-5 is the fifth edition of the Diagnostic and Statistical Manual of Mental Disorders.",
    "Cognitive Behavioral Therapy (CBT) is a type of psychotherapy.",
]

rag = SimpleRAG(documents)

question = "What is CBT?"
answer = rag.answer(question)

print(answer)
```