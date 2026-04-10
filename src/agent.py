from typing import Callable

from .store import EmbeddingStore


class KnowledgeBaseAgent:
    """
    An agent that answers questions using a vector knowledge base.

    Retrieval-augmented generation (RAG) pattern:
        1. Retrieve top-k relevant chunks from the store.
        2. Build a prompt with the chunks as context.
        3. Call the LLM to generate an answer.
    """

    def __init__(self, store: EmbeddingStore, llm_fn: Callable[[str], str]) -> None:
        """Store references to the vector store and the LLM function."""
        self.store = store
        self.llm_fn = llm_fn

    def _build_prompt(self, question: str, context_chunks: list[dict]) -> str:
        """
        Xây dựng prompt rõ ràng, chuyên nghiệp cho LLM.
        """
        context_text = "\n\n".join([
            f"Context {i+1}:\n{chunk['content']}"
            for i, chunk in enumerate(context_chunks)
        ])

        prompt = f"""Bạn là một trợ lý thông minh và trung thực. Hãy trả lời câu hỏi của người dùng dựa trên thông tin được cung cấp trong Context.

### Context (thông tin tham khảo):
{context_text}

### Câu hỏi:
{question}

### Hướng dẫn:
- Chỉ sử dụng thông tin từ Context để trả lời.
- Nếu Context không đủ thông tin để trả lời, hãy nói rõ rằng bạn không có đủ thông tin.
- Trả lời bằng tiếng Việt, ngắn gọn, rõ ràng và lịch sự.
- Không bịa thông tin.

Trả lời:"""

        return prompt

    def answer(self, question: str, top_k: int = 3) -> str:
        """
        Trả lời câu hỏi bằng cách sử dụng Retrieval-Augmented Generation (RAG).
        """
        if not question or not question.strip():
            return "Câu hỏi không hợp lệ. Vui lòng đặt câu hỏi cụ thể."

        # Bước 1: Retrieve relevant chunks từ vector store
        retrieved_chunks = self.store.search(query=question, top_k=top_k)

        if not retrieved_chunks:
            return "Tôi không tìm thấy thông tin liên quan để trả lời câu hỏi này."

        # Bước 2: Xây dựng prompt với context
        prompt = self._build_prompt(question, retrieved_chunks)

        # Bước 3: Gọi LLM để sinh câu trả lời
        try:
            answer = self.llm_fn(prompt)
            return answer.strip()
        except Exception as e:
            return f"Lỗi khi gọi LLM: {str(e)}. Vui lòng thử lại sau."

    def get_context(self, question: str, top_k: int = 3) -> list[dict]:
        """Trả về các chunk context được retrieve (dùng để debug hoặc hiển thị)."""
        return self.store.search(query=question, top_k=top_k)