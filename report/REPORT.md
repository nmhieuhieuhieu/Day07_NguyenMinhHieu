# Báo Cáo Lab 7: Embedding & Vector Store

**Họ tên:** [Tên sinh viên]
**Nhóm:** [Tên nhóm]
**Ngày:** [Ngày nộp]

---

# 1. Warm-up (5 điểm)

## Cosine Similarity (Ex 1.1)

### High cosine similarity nghĩa là gì?
High cosine similarity (giá trị gần tiến về 1) nghĩa là hai vector có hướng rất gần nhau trong không gian vector, thể hiện sự tương đồng cao về mặt ngữ nghĩa giữa hai đoạn văn bản.

### Ví dụ HIGH similarity:
- Sentence A: "Tôi rất thích học lập trình Python."
- Sentence B: "Lập trình bằng ngôn ngữ Python là niềm đam mê của tôi."

**Tại sao tương đồng:**  
Cả hai câu đều chia sẻ cùng một chủ ngữ, hành động và đối tượng (Python), chỉ khác nhau về cách diễn đạt từ ngữ nhưng giữ nguyên ý nghĩa cốt lõi.

### Ví dụ LOW similarity:
- Sentence A: "Hôm nay trời nắng đẹp."
- Sentence B: "Hệ thức lượng trong tam giác vuông rất quan trọng."

**Tại sao khác:**  
Hai câu thuộc hai lĩnh vực hoàn toàn khác nhau (thời tiết và toán học), không có sự liên quan về từ vựng hay ngữ nghĩa.

### Tại sao cosine similarity được ưu tiên hơn Euclidean distance cho text embeddings?
Cosine similarity tập trung vào hướng của vector thay vì độ dài, giúp so sánh nội dung văn bản hiệu quả mà không bị ảnh hưởng bởi việc đoạn văn dài hay ngắn (vốn làm tăng khoảng cách Euclidean).

---

## Chunking Math (Ex 1.2)

### Bài toán:
Document 10,000 ký tự, `chunk_size = 500`, `overlap = 50`. Bao nhiêu chunks?

### Trình bày phép tính:
Sử dụng công thức cho chiến lược Fixed Size Chunker:

- Step:
  
  $$
  Step = Chunk\_Size - Overlap = 500 - 50 = 450
  $$

- Số lượng chunks:

  $$
  Number\ of\ chunks = \left\lceil \frac{Total\ Length - Overlap}{Step} \right\rceil
  $$

  $$
  = \left\lceil \frac{10000 - 50}{450} \right\rceil
  = \left\lceil \frac{9950}{450} \right\rceil
  \approx \left\lceil 22.11 \right\rceil
  $$

### Đáp án:
**23 chunks**

---

### Nếu overlap tăng lên 100 thì sao?

- Step mới:

  $$
  500 - 100 = 400
  $$

- Số lượng chunks sẽ **tăng lên** vì bước nhảy nhỏ hơn.

### Tại sao muốn overlap nhiều hơn?

Overlap lớn giúp:
- Giữ lại ngữ cảnh giữa các chunk
- Tránh việc câu bị cắt ngang gây mất nghĩa
- Giúp LLM hiểu tốt hơn mối liên hệ giữa các đoạn văn liền kề
---

## 2. Document Selection — Nhóm (10 điểm)

### Domain & Lý Do Chọn

**Domain:** Bộ luật Hình sự 2015

**Tại sao nhóm chọn domain này?**
> *Trong lĩnh vực pháp luật, hệ thống không chỉ cần trả lời đúng mà còn phải truy xuất chính xác và cung cấp nguồn rõ ràng (điều, khoản, văn bản). LLM thuần có thể trả lời “nghe hợp lý” nhưng dễ sai hoặc hallucination, trong khi RAG buộc mô hình dựa trên các điều luật được truy xuất thực tế. Nhờ đó, RAG đảm bảo câu trả lời luôn có căn cứ pháp lý cụ thể, chính xác*

### Data Inventory

| # | Tên tài liệu | Nguồn | Số ký tự | Metadata đã gán |
|---|--------------|-------|----------|-----------------|
| 1 | Bộ luật hình sự Việt Nam 2015 | Cổng thông tin chính phủ | 562,623 | source, chuong, dieu, tieu_de, lang |

### Metadata Schema

| Trường metadata | Kiểu | Ví dụ giá trị | Tại sao hữu ích cho retrieval? |
|----------------|------|---------------|-------------------------------|
| `source` | str | `"Bộ luật hình sự 2015"` | Phân biệt tài liệu khi store có nhiều bộ luật |
| `chuong` | int | `14` | Filter theo chương — ví dụ "Tội phạm về kinh tế" |
| `dieu` | int | `174` | Retrieve chính xác một điều luật cụ thể |
| `tieu_de` | str | `"Tội lừa đảo chiếm đoạt tài sản"` | Cung cấp context cho chunk khi hiển thị kết quả |
| `lang` | str | `"vi"` | Dự phòng nếu sau này thêm bản dịch tiếng Anh |

---
---

## 3. Chunking Strategy — Cá nhân chọn, nhóm so sánh (15 điểm)

### Baseline Analysis

Chạy `ChunkingStrategyComparator().compare()` trên 1 tài liệu:


| Tài liệu  | Strategy              | Chunk Count | Avg Length | Preserves Context? |
|----------|----------------------|------------|------------|--------------------|
| blhs.txt | FixedSizeChunker     | 1147       | 499.85     | Thấp               |
| blhs.txt | SentenceChunker      | 1108       | 495.39     | Trung bình         |
| blhs.txt | RecursiveChunker     | 1538       | 356.57     | Cao                |
### Strategy Của Tôi

**Loại:** RecursiveChunker

**Mô tả cách hoạt động:**

**Cơ chế:**  
  Strategy này thực hiện chia văn bản theo phương pháp đệ quy, ưu tiên các dấu phân cách có ý nghĩa cấu trúc lớn nhất trước (như `\n\n` - đoạn văn, sau đó đến `\n` - dòng, rồi mới đến `.` - câu).

 **Nguyên lý:**  
  Nó cố gắng giữ trọn vẹn các phần nội dung trong một "đoạn" hoặc "khoản" của luật. Chỉ khi độ dài của một phần vượt quá `chunk_size` quy định, nó mới tiến hành chia nhỏ tiếp bằng dấu phân cách cấp thấp hơn.

 **Dấu hiệu:**  
  Dựa vào các dấu hiệu phân cách tự nhiên của văn bản pháp luật, chiến lược này giảm thiểu việc cắt ngang giữa một điều luật hoặc khoản luật đang trình bày dở dang.

**Tại sao tôi chọn strategy này cho domain nhóm?**
> *Mặc dù số lượng chunk lớn hơn (1538) và độ dài trung bình nhỏ hơn (356.57), nhưng đây là phương pháp tối ưu nhất cho BLHS vì nó bảo toàn được tính logic và ngữ cảnh của các quy định pháp luật trong từng chunk, giúp máy học hiểu rõ hơn về mối quan hệ giữa các mệnh đề.*


### So Sánh: Strategy của tôi vs Baseline

| Tài liệu | Strategy | Chunk Count | Avg Length | Retrieval Quality? |
|-----------|----------|-------------|------------|--------------------|
| | best baseline | SentenceChunker|  671|835 | Giu dươc context|
| | **của tôi** |  RecursiveChunker (`recursive`) |1538 |356.57 | Giu duoc context|

### So Sánh Với Thành Viên Khác

| Thành viên | Strategy | Retrieval Score (/10) | Điểm mạnh | Điểm yếu |
|-----------|----------|----------------------|-----------|----------|
| Tôi | RecursiveChunker | 4.5| Chia đúng theo câu nên ngữ nghĩa rõ ràng, dễ hiểu với embedding model. Chunk thường sạch và ít bị cắt giữa ý.|Có thể mất context nếu thông tin nằm ở nhiều câu liên tiếp. Một số câu quá ngắn hoặc quá dài gây mất cân bằng chunk size. | 
| Nam | SentenceChunker| 4.2| Giữ ngữ cảnh tốt vì chia theo cấu trúc văn bản (paragraph → sentence → word). Phù hợp với tài liệu có cấu trúc như luật. Giảm việc cắt giữa câu.| Phức tạp hơn khi triển khai, đôi khi chunk không đều kích thước, có thể tạo nhiều chunk hơn baseline.|
|Phúc| FixedSizeChunker| 3.5| Rất đơn giản, dễ implement, tốc độ xử lý nhanh. Chunk size ổn định nên dễ kiểm soát token.| Không giữ ngữ cảnh tốt, dễ cắt giữa câu hoặc điều luật, làm giảm chất lượng embedding và retrieval.|

**Strategy nào tốt nhất cho domain này? Tại sao?**
> Với domain văn bản pháp luật như Bộ luật Hình sự Việt Nam 2015, Recursive Chunking thường là strategy tốt nhất trong ba phương pháp (Recursive, Sentence, Fixed).

**Strategy nào tốt nhất cho domain này? Tại sao?**
> *Viết 2-3 câu:*

---

## 4. My Approach — Cá nhân (10 điểm)

### Chunking Functions

#### `SentenceChunker.chunk` — approach:
> Tôi sử dụng biểu thức chính quy (Regex) `(?<=[\.\!\?])\s+|(?<=\.)\n` để phân tách văn bản dựa trên các dấu kết thúc câu và khoảng trắng hoặc dòng mới. Cách tiếp cận này giúp xử lý các trường hợp văn bản bị xuống dòng giữa chừng hoặc có nhiều khoảng trắng thừa, sau đó các câu được gom nhóm lại dựa trên tham số `max_sentences_per_chunk` để tạo thành các đoạn có ngữ cảnh toàn vẹn.

#### `RecursiveChunker.chunk` / `_split` — approach:
> Thuật toán hoạt động theo nguyên tắc chia để trị, ưu tiên phân tách văn bản theo danh sách các ký tự ngăn cách từ lớn đến nhỏ như `\n\n`, `\n`, rồi đến dấu cách. Base case (điều kiện dừng) xảy ra khi độ dài đoạn văn bản đã nhỏ hơn hoặc bằng `chunk_size`, hoặc khi không còn ký tự ngăn cách nào khả dụng thì sẽ thực hiện cắt cứng theo số ký tự để đảm bảo giới hạn kích thước.

---

### EmbeddingStore

#### `add_documents` + `search` — approach:
> Tài liệu được chuyển đổi thành vector thông qua một hàm embedding, sau đó được lưu trữ trong danh sách Python hoặc bộ nhớ của ChromaDB kèm theo metadata. Khi tìm kiếm, hệ thống tính toán điểm tương đồng giữa vector truy vấn và các vector đã lưu (sử dụng tích vô hướng `_dot` hoặc khoảng cách cosine), sau đó sắp xếp giảm dần theo điểm số để trả về kết quả phù hợp nhất.

#### `search_with_filter` + `delete_document` — approach:
> Trong `search_with_filter`, hệ thống thực hiện lọc (filter) các tài liệu dựa trên metadata trước khi thực hiện tính toán độ tương đồng hoặc truyền điều kiện lọc trực tiếp vào truy vấn của ChromaDB. Việc xóa tài liệu được thực hiện bằng cách lọc bỏ các record có `doc_id` tương ứng trong danh sách hoặc gọi phương thức `delete` của collection nếu sử dụng cơ sở dữ liệu vector.

---

### KnowledgeBaseAgent

#### `answer` — approach:
> Hàm `answer` nhận câu hỏi từ người dùng, thực hiện tìm kiếm các đoạn văn bản liên quan nhất từ `EmbeddingStore` để làm ngữ cảnh (context). Sau đó, context này được inject (nhúng) vào một System Prompt có cấu trúc chặt chẽ, yêu cầu mô hình ngôn ngữ chỉ được trả lời dựa trên thông tin đã cung cấp để giảm thiểu hiện tượng ảo giác (hallucination).

### Test Results

```
tests/test_solution.py::TestProjectStructure::test_root_main_entrypoint_exists PASSED                                                                       [  2%] 
tests/test_solution.py::TestProjectStructure::test_src_package_exists PASSED                                                                                [  4%] 
tests/test_solution.py::TestClassBasedInterfaces::test_chunker_classes_exist PASSED                                                                         [  7%] 
tests/test_solution.py::TestClassBasedInterfaces::test_mock_embedder_exists PASSED                                                                          [  9%] 
tests/test_solution.py::TestFixedSizeChunker::test_chunks_respect_size PASSED                                                                               [ 11%] 
tests/test_solution.py::TestFixedSizeChunker::test_correct_number_of_chunks_no_overlap PASSED                                                               [ 14%] 
tests/test_solution.py::TestFixedSizeChunker::test_empty_text_returns_empty_list PASSED                                                                     [ 16%] 
tests/test_solution.py::TestFixedSizeChunker::test_no_overlap_no_shared_content PASSED                                                                      [ 19%]
tests/test_solution.py::TestFixedSizeChunker::test_overlap_creates_shared_content PASSED                                                                    [ 21%] 
tests/test_solution.py::TestFixedSizeChunker::test_returns_list PASSED                                                                                      [ 23%] 
tests/test_solution.py::TestFixedSizeChunker::test_single_chunk_if_text_shorter PASSED                                                                      [ 26%] 
tests/test_solution.py::TestSentenceChunker::test_chunks_are_strings PASSED                                                                                 [ 28%] 
tests/test_solution.py::TestSentenceChunker::test_respects_max_sentences PASSED                                                                             [ 30%] 
tests/test_solution.py::TestSentenceChunker::test_returns_list PASSED                                                                                       [ 33%] 
tests/test_solution.py::TestSentenceChunker::test_single_sentence_max_gives_many_chunks PASSED                                                              [ 35%] 
tests/test_solution.py::TestRecursiveChunker::test_chunks_within_size_when_possible PASSED                                                                  [ 38%] 
tests/test_solution.py::TestRecursiveChunker::test_empty_separators_falls_back_gracefully PASSED                                                            [ 40%]
tests/test_solution.py::TestRecursiveChunker::test_handles_double_newline_separator PASSED                                                                  [ 42%] 
tests/test_solution.py::TestRecursiveChunker::test_returns_list PASSED                                                                                      [ 45%] 
tests/test_solution.py::TestEmbeddingStore::test_add_documents_increases_size PASSED                                                                        [ 47%]
tests/test_solution.py::TestEmbeddingStore::test_add_more_increases_further PASSED                                                                          [ 50%]
tests/test_solution.py::TestEmbeddingStore::test_initial_size_is_zero PASSED                                                                                  [ 52%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_content_key PASSED                                                                     [ 54%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_have_score_key PASSED                                                                       [ 57%]
tests/test_solution.py::TestEmbeddingStore::test_search_results_sorted_by_score_descending PASSED                                                           [ 59%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_at_most_top_k PASSED                                                                        [ 61%]
tests/test_solution.py::TestEmbeddingStore::test_search_returns_list PASSED                                                                                 [ 64%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_non_empty PASSED                                                                                [ 66%]
tests/test_solution.py::TestKnowledgeBaseAgent::test_answer_returns_string PASSED                                                                           [ 69%]
tests/test_solution.py::TestComputeSimilarity::test_identical_vectors_return_1 PASSED                                                                       [ 71%] 
tests/test_solution.py::TestComputeSimilarity::test_opposite_vectors_return_minus_1 PASSED                                                                  [ 73%] 
tests/test_solution.py::TestComputeSimilarity::test_orthogonal_vectors_return_0 PASSED                                                                      [ 76%] 
tests/test_solution.py::TestComputeSimilarity::test_zero_vector_returns_0 PASSED                                                                            [ 78%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_counts_are_positive PASSED                                                                      [ 80%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_each_strategy_has_count_and_avg_length PASSED                                                   [ 83%] 
tests/test_solution.py::TestCompareChunkingStrategies::test_returns_three_strategies PASSED                                                                 [ 85%] 
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_filter_by_department PASSED                                                                [ 88%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_no_filter_returns_all_candidates PASSED                                                    [ 90%]
tests/test_solution.py::TestEmbeddingStoreSearchWithFilter::test_returns_at_most_top_k PASSED                                                               [ 92%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                        [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_reduces_collection_size PASSED                                                        [ 95%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                              [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_false_for_nonexistent_doc PASSED                                              [ 97%] 
tests/test_solution.py::TestEmbeddingStoreDeleteDocument::test_delete_returns_true_for_existing_doc PASSED   
```

**Số tests pass:** _42_ / _42_

---

## 5. Similarity Predictions — Cá nhân (5 điểm)

| Pair | Sentence A | Sentence B | Dự đoán | Actual Score | Đúng? |
|------|-----------|-----------|---------|--------------|-------|
| 1 | Trộm cắp tài sản trị giá lớn bị xử lý như thế nào? | Lấy cắp tài sản có giá trị cao sẽ bị phạt ra sao? | high | 0.89 | ✓ |
| 2 | Người phạm tội lần đầu có được giảm nhẹ không? | Người phạm tội nhiều lần có bị tăng nặng không? | low | 0.42 | ✓ |
| 3 | Cướp tài sản bị phạt bao nhiêu năm tù? | Hành vi chiếm đoạt tài sản bằng vũ lực bị xử lý thế nào? | high | 0.85 | ✓ |
| 4 | Hôm nay trời rất đẹp | Mức phạt tù tối đa cho tội giết người là bao nhiêu? | low | 0.05 | ✓ |
| 5 | Gây tai nạn rồi bỏ chạy bị xử lý ra sao? | Không cứu giúp người bị nạn sau tai nạn có bị phạt không? | high | 0.67 | ✓ |

---

**Kết quả nào bất ngờ nhất? Điều này nói gì về cách embeddings biểu diễn nghĩa?**  
> Pair 5 là bất ngờ nhất vì hai câu không dùng từ giống nhau nhiều nhưng vẫn có similarity khá cao. Điều này cho thấy embeddings có khả năng nắm bắt ngữ nghĩa sâu (semantic meaning), không chỉ dựa vào từ vựng bề mặt mà còn hiểu mối quan hệ hành vi tương tự trong cùng ngữ cảnh.

---

## 6. Results — Cá nhân (10 điểm)

Chạy 5 benchmark queries của nhóm trên implementation cá nhân của bạn trong package `src`. 

### 5 Benchmark Queries + Gold Answers

| # | Query | Gold Answer | Chunk chứa thông tin | Loại query |
|---|------|------------|----------------------|------------|
| 1 | Người từ đủ 14 tuổi đến dưới 16 tuổi phải chịu trách nhiệm hình sự về những tội nào? | Tội giết người, cố ý gây thương tích, hiếp dâm, hiếp dâm người dưới 16 tuổi, cưỡng dâm người từ đủ 13 đến dưới 16 tuổi, cướp tài sản, bắt cóc nhằm chiếm đoạt tài sản, và các tội phạm rất nghiêm trọng, đặc biệt nghiêm trọng trong các điều 143, 150, 151, 170, 171, 173, 178, 248-252, 265, 266, 285-287, 289, 290, 299, 303, 304 (Điều 12) | Điều 12 | Factual — tra cứu 1 điều cụ thể |
| 2 | Trộm cắp tài sản trị giá 500 triệu đồng trở lên thì bị phạt tù bao nhiêu năm? | Phạt tù từ 12 năm đến 20 năm (Điều 173, khoản 4) | Điều 173 | Factual — cần trích mức phạt chính xác |
| 3 | Các tình tiết giảm nhẹ trách nhiệm hình sự bao gồm những gì? | 22 tình tiết tại Điều 51 khoản 1, gồm: ngăn chặn/giảm bớt tác hại; tự nguyện bồi thường; vượt quá phòng vệ chính đáng; tình thế cấp thiết; bị kích động; hoàn cảnh khó khăn; chưa gây thiệt hại; phạm tội lần đầu ít nghiêm trọng; bị đe dọa/cưỡng bức; phụ nữ có thai; người đủ 70 tuổi; khuyết tật nặng; tự thú, thành khẩn khai báo; lập công chuộc tội... Ngoài ra Tòa án có thể coi các tình tiết khác là giảm nhẹ (khoản 2) | Điều 51 | Multi-chunk — câu trả lời dài, nhiều mục |
| 4 | Lái xe gây tai nạn chết người rồi bỏ chạy thì bị xử lý thế nào? | Phạt tù từ 3 năm đến 10 năm theo Điều 260 khoản 2 điểm c: “Gây tai nạn rồi bỏ chạy để trốn tránh trách nhiệm hoặc cố ý không cứu giúp người bị nạn” | Điều 260 | Suy luận — cần kết hợp khoản 1 (chết người) + khoản 2 điểm c (bỏ chạy) |
| 5 | Tội phạm được phân thành mấy loại và mức phạt tối đa của mỗi loại là bao nhiêu? | 4 loại theo Điều 9: (1) Ít nghiêm trọng — đến 3 năm tù; (2) Nghiêm trọng — trên 3 đến 7 năm; (3) Rất nghiêm trọng — trên 7 đến 15 năm; (4) Đặc biệt nghiêm trọng — trên 15 năm đến 20 năm, chung thân hoặc tử hình | Điều 9 | Tổng hợp — cần liệt kê đầy đủ 4 loại |

### Kết Quả Của Tôi

| # | Query | Top-1 Retrieved Chunk (tóm tắt) | Score | Relevant? | Agent Answer (tóm tắt) |
|---|-------|--------------------------------|-------|-----------|------------------------|
| 1 | Người từ đủ 14 đến dưới 16 tuổi chịu TNHS về tội nào? | Điều 9: phân loại tội phạm theo mức độ nghiêm trọng | 0.62 | ✗ | Trả lời nhầm sang phân loại tội phạm, không liệt kê đúng các tội cụ thể |
| 2 | Trộm cắp tài sản ≥500 triệu bị phạt bao nhiêu? | Điều 173 khoản 1: trộm cắp tài sản cơ bản | 0.71 | ✗ | Trả lời mức phạt 6 tháng đến 3 năm (sai mức nghiêm trọng) |
| 3 | Các tình tiết giảm nhẹ gồm những gì? | Điều 52: các tình tiết tăng nặng | 0.65 | ✗ | Nhầm sang tình tiết tăng nặng, trả lời sai hoàn toàn |
| 4 | Gây tai nạn chết người rồi bỏ chạy bị xử lý? | Điều 260 khoản 1: vi phạm giao thông gây hậu quả | 0.74 | △ | Chỉ nói gây tai nạn chết người (1–5 năm), thiếu yếu tố bỏ chạy |
| 5 | Tội phạm phân thành mấy loại? | Điều 9: phân loại tội phạm | 0.88 | ✓ | Trả lời đúng 4 loại nhưng thiếu chi tiết mức hình phạt tối đa |

---

**Số queries trả về chunk relevant trong top-3: ** **2 / 5**

---

## 7. What I Learned (5 điểm — Demo)

**Điều hay nhất tôi học được từ thành viên khác trong nhóm:**
> Tôi học được cách tối ưu chunking bằng cách kết hợp metadata (điều, khoản) để giúp truy xuất chính xác hơn thay vì chỉ dựa vào nội dung text. Ngoài ra, việc debug retrieval bằng cách in top-k chunks cũng giúp hiểu rõ hệ thống đang sai ở đâu.

**Điều hay nhất tôi học được từ nhóm khác (qua demo):**
> Một số nhóm sử dụng hybrid search (kết hợp keyword + embedding) cho kết quả tốt hơn rõ rệt trong domain luật. Điều này cho thấy chỉ dùng semantic search đôi khi chưa đủ khi cần truy xuất chính xác điều luật cụ thể.

**Nếu làm lại, tôi sẽ thay đổi gì trong data strategy?**
> Tôi sẽ gắn thêm metadata chi tiết hơn (điều, khoản, loại tội) và thử nghiệm chunk overlap lớn hơn để giữ ngữ cảnh tốt hơn. Ngoài ra, tôi sẽ bổ sung thêm các câu hỏi thực tế để cải thiện khả năng retrieval trong các tình huống mơ hồ.

---

## Tự Đánh Giá

| Tiêu chí | Loại | Điểm tự đánh giá |
|----------|------|-------------------|
| Warm-up | Cá nhân | 5 / 5 |
| Document selection | Nhóm | 9 / 10 |
| Chunking strategy | Nhóm | 13 / 15 |
| My approach | Cá nhân | 9 / 10 |
| Similarity predictions | Cá nhân | 4 / 5 |
| Results | Cá nhân | 8 / 10 |
| Core implementation (tests) | Cá nhân | 30 / 30 |
| Demo | Nhóm | 5 / 5 |
| **Tổng** | | **83 / 90** |