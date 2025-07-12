from transformers import pipeline

class SimpleRAG:
    """
    A simple Retrieval-Augmented Generation (RAG) class for question answering.
    """

    def __init__(self, documents):
        """
        Initializes the SimpleRAG with a list of documents.

        Args:
            documents: A list of strings, where each string is a document.
        """
        self.documents = documents
        self.qa_pipeline = pipeline("question-answering")

    def answer(self, question):
        """
        Answers a question based on the documents.

        Args:
            question: The question to answer.

        Returns:
            The answer to the question.
        """
        context = " ".join(self.documents)
        result = self.qa_pipeline(question=question, context=context)
        return result['answer']
