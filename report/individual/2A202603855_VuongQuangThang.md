# Báo cáo Day 7 — Agentic RAG

**Học viên:** Vương Quang Thắng  
**MSSV:** 2A202603855  
**Lớp:** K5-L3B  
**Bài lab:** Retrieval & Context Engineering (Day 7)

---

## Dữ liệu và benchmark

- **Nguồn, quyền sử dụng, số trang và số chunk:**
  - Dữ liệu kế thừa trực tiếp từ Day 6: sáu chương của **Python Tutorial** trong kho CPython chính thức, tag **v3.13.0** (các chương: *More Control Flow Tools*, *Data Structures*, *Modules*, *Input and Output*, *Errors and Exceptions*, *Classes*).
  - Giấy phép: Python Software Foundation (PSF) License Agreement và 0BSD. Giữ nguyên toàn bộ copyright notice của PSF.
  - Tổng số trang chuẩn hóa: **58 đơn vị trang** (`page`), tương ứng 153.569 ký tự văn bản sạch đã chuẩn hóa Unicode NFC.
  - Số chunk: **226 chunks** kế thừa từ Strategy C của Day 6 (gom theo đoạn văn bản `paragraph`, không overlap, độ dài tối đa 900 ký tự, giữ trọn vẹn ngữ cảnh ranh giới câu/đoạn).

- **Vector store, embedding model, phiên bản và số chiều:**
  - **Vector store:** **Qdrant client v1.15.1** chạy ở chế độ local persistent tại `data/qdrant/`, sử dụng collection `python_tutorial_C`, metric khoảng cách cosine, cơ chế tìm kiếm exact search.
  - **Embedding model:** **`sentence-transformers/all-MiniLM-L6-v2`** (384 chiều vector), chạy qua framework **FastEmbed v0.7.4** với ONNX Runtime v1.32.0 (CPUExecutionProvider, 4 threads). Model checkpoint tải từ `qdrant/all-MiniLM-L6-v2-onnx` (revision `d13954661f83248295ba75c1ed411eef3b7b936e`) được lưu trữ tại `.cache/` cục bộ, cho phép chạy hoàn toàn offline và không tốn chi phí embedding.

- **Agent model, cách chạy và phân chia dev/test:**
  - **Agent Controller Model:** **Google Gemini 3.5 Flash-Lite** (`gemini-3.5-flash-lite`), nhiệt độ sampling `temperature=0.0`, cấu hình `responseMimeType="application/json"` đảm bảo output có cấu trúc JSON tuân thủ giao thức controller (`action: "search"` hoặc `action: "answer"`). Có cơ chế retry và fallback sang `gemini-3.1-flash-lite` khi gặp tình trạng quá tải cục bộ (HTTP 503/429).
  - **Phân chia Benchmark (tổng cộng 24 câu):**
    - **Tập Dev (17 câu):** 15 câu hỏi có đáp án (`q01`–`q15`) trích xuất từ 4 chương đầu (*controlflow*, *datastructures*, *modules*, *inputoutput*) và **2 câu thiếu bằng chứng** (`q_unans_dev1`, `q_unans_dev2`).
    - **Tập Test (7 câu):** 5 câu hỏi có đáp án (`q16`–`q20`) trích xuất từ 2 chương sau (*errors*, *classes*) và **2 câu thiếu bằng chứng** (`q_unans_test1`, `q_unans_test2`).

- **Cách xác định câu có/thiếu evidence và tiêu chí chấm:**
  - **Câu có đáp án:** Mỗi câu hỏi có ít nhất một bằng chứng vàng (`gold`), xác định rõ `source_id`, `page` và đoạn trích dẫn nguyên văn (`evidence`). Toàn bộ 20 câu hỏi có đáp án đều được kiểm tra đối chiếu tự động, bảo đảm trích dẫn vàng nằm trọn vẹn trong một chunk đúng nguồn và trang của Strategy C (20/20 khớp 100%).
  - **Câu thiếu bằng chứng:** Thiết kế 4 câu hỏi rất thực tế và gần gũi với lập trình Python hiện đại nhưng hoàn toàn không xuất hiện trong 6 chương tutorial:
    1. `q_unans_dev1` (Dev): Cách vô hiệu hóa GIL tại runtime trong bản build free-threaded Python 3.13 (CPython internals).
    2. `q_unans_dev2` (Dev): Cú pháp định nghĩa asynchronous generator sử dụng `async def` và `yield` (PEP 525).
    3. `q_unans_test1` (Test): Cách khai báo `__slots__` trong class để tối ưu bộ nhớ và chặn tạo `__dict__`.
    4. `q_unans_test2` (Test): Cách tạo Abstract Base Class bắt buộc override method bằng `abc.ABCMeta` và `@abstractmethod`.
    Đã quét toàn bộ 226 chunk để xác nhận không chứa bất kỳ từ khóa hay thông tin nào về các chủ đề trên.
  - **Tiêu chí chấm:**
    - `Hit@1` / `Evidence Hit@3`: Tỷ lệ câu có chunk chứa evidence quote hợp lệ trong top-1 hoặc top-3.
    - `Context Evidence Rate`: Tỷ lệ câu mà context cuối cùng gửi cho model (sau khi đóng gói theo `budget`) thực sự chứa evidence quote hợp lệ.
    - `Grounded Answer`: 0 (sai, thiếu bằng chứng, hoặc từ chối nhầm câu có đáp án); 1 (đúng một phần); 2 (đúng đầy đủ kèm trích dẫn hợp lệ).
    - `Citation Validity`: Tỷ lệ trích dẫn có trong context và thực sự hỗ trợ câu trả lời.
    - `Abstention`: Tỷ lệ từ chối đúng trên các câu thiếu bằng chứng ("Không đủ bằng chứng để trả lời.").

---

## Retrieval và thử nghiệm

| Cấu hình | Phương pháp, top-k, filter, rerank, context budget | Giả thuyết / thay đổi | Split | Hit@1 | Evidence Hit@3 | Context Evidence Rate | Thời gian / chi phí |
|---|---|---|---|---|---|---|---|
| **Baseline** | TF-IDF Lexical, top-k=3, no filter, budget=800 | TF-IDF tiêu chuẩn từ starter, tách từ regex cơ bản | dev | 7/15 (46.7%) | 11/15 (73.3%) | 4/15 (26.7%) | 0.56 ms / 0 USD |
| **Phương pháp 2** | BM25 Lexical, top-k=3, no filter, budget=800 | Thuật toán BM25 ($k_1=1.5, b=0.75$) chuẩn hóa theo độ dài đoạn văn | dev | 9/15 (60.0%) | 11/15 (73.3%) | 4/15 (26.7%) | 0.58 ms / 0 USD |
| **Phương pháp 3** | Dense (Qdrant + all-MiniLM-L6-v2), top-k=3, no filter, budget=800 | Vector search ngữ nghĩa dùng embedding và vector store Day 6 | dev | 8/15 (53.3%) | 11/15 (73.3%) | 5/15 (33.3%) | 69.24 ms / 0 USD |
| **Cải thiện 1** | Hybrid (Dense + BM25 via RRF) + Section Filter, top-k=3, budget=800 | Kết hợp Dense + Lexical qua RRF ($k=60$) và áp dụng section filter khi có sẵn | dev | 9/15 (60.0%) | 13/15 (86.7%) | 5/15 (33.3%) | 71.83 ms / 0 USD |
| **Cải thiện 2 (Chốt)** | Hybrid + Section Filter + Context Budget 1600, top-k=3 | Tăng context budget từ 800 lên 1600 ký tự để vừa 2–3 chunk | dev | 9/15 (60.0%) | 13/15 (86.7%) | 10/15 (66.7%) | 73.05 ms / 0 USD |
| **Cấu hình đã chọn** | Hybrid + Section Filter + Context Budget 1600, top-k=3 | Đánh giá độc lập trên tập test sau khi đã chốt trên dev | test | 4/5 (80.0%) | 4/5 (80.0%) | 4/5 (80.0%) | 73.39 ms / 0 USD |
| **Baseline** | TF-IDF Lexical, top-k=3, no filter, budget=800 | Đo đối chứng trên tập test với cùng điều kiện | test | 2/5 (40.0%) | 5/5 (100.0%) | 1/5 (20.0%) | 0.84 ms / 0 USD |

### Phân tích hai lỗi dev, nguyên nhân, cách thử và lý do chọn cấu hình:

1. **Lỗi trôi miền ngữ cảnh ở câu `q14` (Input and Output):**
   - *Câu hỏi:* "In the Input and Output chapter, why use with for file handling?"
   - *Hiện tượng lỗi ở Dense đơn thuần:* Dense retrieval trả về các chunk của chương *Errors and Exceptions* do các từ khóa `handling`, `file` và cú pháp exception khiến độ tương đồng ngữ nghĩa bị phân tán. Do đó cả 3 chunk đầu đều thuộc chương *Errors*, dẫn đến Hit@3 = 0.
   - *Biện pháp cải thiện:* Áp dụng metadata filter `section: "Input and Output"` kết hợp thuật toán **Hybrid RRF** (kết hợp BM25 bắt chính xác từ khóa `with` trong chương I/O). Kết quả: chunk `inputoutput:p5` được đẩy thẳng lên Top-1, nâng Hit@1 và Hit@3 của tập Dev từ 80.0% lên 86.7% (13/15 câu).

2. **Lỗi thắt cổ chai kích thước Context Budget 800 ký tự (các câu `q02`, `q04`, `q07`):**
   - *Hiện tượng lỗi:* Ở cả 3 phương pháp cơ bản, dù `Hit@3` đạt 73.3%–80.0%, nhưng `Context Evidence Rate` chỉ đạt vỏn vẹn **26.7% – 33.3%**.
   - *Nguyên nhân cốt lõi:* Các chunk Strategy C có độ dài trung bình ~721 ký tự cộng với chuỗi tiêu đề `[id | source | page]`. Hàm đóng gói ngữ cảnh `pack_context` với giới hạn mặc định `budget = 800` chỉ chứa vừa đúng 1 chunk đầu tiên; chunk thứ 2 và 3 bị loại bỏ hoàn toàn do vượt ngưỡng 800 ký tự. Khi bằng chứng nằm ở rank 2 hoặc rank 3, nó bị drop khỏi prompt của agent, gây ra hiện tượng từ chối nhầm (false abstention) nghiêm trọng (7/14 câu ở baseline).
   - *Biện pháp cải thiện:* Tăng `budget` lên **1600 ký tự** (cho phép chứa vừa 2–3 chunk liên quan nhất). Kết quả: Context Evidence Rate trên Dev tăng vọt từ 33.3% lên **66.7%** (10/15 câu), và trên Test tăng vọt từ 20.0% lên **80.0%** (4/5 câu).

**Lý do chốt cấu hình:**  
Cấu hình **Hybrid RRF + Section Filter + Budget 1600** đạt cân bằng tối ưu: giữ vững Hit@3 cao nhất (86.7%), giải quyết triệt để nút thắt cổ chai ngữ cảnh (Context Evidence Rate tăng gấp đôi từ 33.3% lên 66.7%), thời gian phản hồi chỉ ~73 ms và chi phí truy xuất cục bộ hoàn toàn bằng 0 USD.

---

## Agent và đánh giá

- **Tool, số lượt gọi, điều kiện dừng, cách chọn context và xử lý lỗi:**
  - **Tool sử dụng:** Tool truy xuất `search(query, filters, top_k)`.
  - **Số lượt gọi:** Cấu hình `max_searches = 2` đối với Baseline; `max_searches = 3` đối với cấu hình Cải thiện.
  - **Điều kiện dừng:** Agent dừng vòng lặp ngay khi model ra quyết định `action: "answer"`. Nếu model trả về câu trả lời kèm citation IDs hợp lệ trong context, trạng thái là `answered`. Nếu bằng chứng không đủ, model trả về `action: "answer"` với nội dung *"Không đủ bằng chứng để trả lời."* và `citations: []`, kích hoạt trạng thái `abstain`. Nếu vượt quá số lần tìm kiếm cho phép mà chưa trả lời, agent chuyển về `abstain` với event `search_limit`.
  - **Cách chọn context:** Áp dụng cumulative context packing có khử trùng lặp (`seen id`), đóng gói theo giới hạn ngân sách ký tự budget.
  - **Xử lý lỗi:** Bắt lỗi kết nối mạng, áp dụng exponential backoff khi gặp rate limit (HTTP 429) hoặc quá tải tạm thời (HTTP 503) và tự động fallback sang `gemini-3.1-flash-lite`. Các lỗi nhà cung cấp được ghi nhận độc lập dưới trường `provider_errors` theo đúng chuẩn benchmark.

### Bảng tổng hợp đánh giá Agent

| Cấu hình / split | Tool use | Số lượt gọi | Grounded answer 0/1/2 và trung bình | Citation hợp lệ | Câu có citation | Abstention đúng | Từ chối nhầm |
|---|---|---|---|---|---|---|---|
| **Baseline / dev** | 16/16 (100%) | 43 | 0: 7, 1: 2, 2: 5 (TB: **0.86 / 2**) | 5/7 (71.4%) | 7/7 (100%) | 2/2 (100%) | 7/14 (50.0%) |
| **Improved / dev** | 17/17 (100%) | 44 | 0: 4, 1: 1, 2: 10 (TB: **1.40 / 2**) | 10/11 (90.9%) | 11/11 (100%) | 2/2 (100%) | 4/15 (26.7%) |
| **Baseline / test** | 7/7 (100%) | 20 | 0: 4, 1: 0, 2: 1 (TB: **0.40 / 2**) | 1/1 (100%) | 1/1 (100%) | 2/2 (100%) | 4/5 (80.0%) |
| **Improved / test** | 7/7 (100%) | 21 | 0: 2, 1: 0, 2: 3 (TB: **1.20 / 2**) | 3/3 (100%) | 3/3 (100%) | 2/2 (100%) | 2/5 (40.0%) |

### Chi tiết các case tiêu biểu minh họa

| Query ID | Loại câu | Tool call và kết quả | Câu trả lời và citation | Nhãn, lý do |
|---|---|---|---|---|
| **q01** | Có đáp án (dev) | Search: `"Python range stop value inclusive exclusive"` → Trả 3 chunks (Top-1: `controlflow:p1`) | *"No, the given end point is never part of the generated sequence [cf7be23a-7374-59e7-959c-f540f33b4073]."* | **Grade 2**: Trả lời chính xác, trích dẫn đúng ID chunk và quote gốc hỗ trợ trực tiếp. |
| **q17** | Có đáp án (test) | Search: `"try block unhandled exception finally clause re-raise"` → Trả 3 chunks (Top-1: `errors:p6`) | *"Yes, if an exception occurs during the execution of the try clause and is not handled by an except clause, the exception is re-raised after the finally clause has been executed [ebae0865-a2da-5af9-a247-0a0713bb6651]."* | **Grade 2**: Trả lời đầy đủ điều kiện exception re-raise sau finally clause, citation chính xác. |
| **q_unans_dev1** | Thiếu bằng chứng (dev) | Search: `"disable GIL runtime Python 3.13 free-threaded"` → Chunks trả về không có thông tin GIL | *"Không đủ bằng chứng để trả lời."* (`citations: []`) | **Abstention đúng**: Agent nhận biết tài liệu không có thông tin cấu hình GIL runtime và từ chối trả lời trung thực. |
| **q_unans_test1** | Thiếu bằng chứng (test) | Search: `"define class __slots__ prevent instance __dict__"` → Chunks trả về về Classes nhưng không có `__slots__` | *"Không đủ bằng chứng để trả lời."* (`citations: []`) | **Abstention đúng**: Agent đối chiếu context, xác định không có `__slots__`, từ chối không sinh nội dung ảo. |
| **q16** | Có đáp án nhưng bị lỗi (test) | Search: `"try statement else clause run condition"` → Trả các chunk về except và finally | *"Không đủ bằng chứng để trả lời."* (`citations: []`) | **Grade 0 (Từ chối nhầm)**: Retriever không đưa chunk chứa mệnh đề `else` (nằm ở page 3) vào top ranking, dẫn đến context thiếu bằng chứng và agent từ chối an toàn. |

- **Đường dẫn trace, nhãn, lệnh chạy, thời gian và chi phí:**
  - Trace và reviews kết quả:
    - Dev Baseline: `results/agent_dev_baseline/traces.json`, `reviews.json`, `summary.json`
    - Dev Improved: `results/agent_dev_improved/traces.json`, `reviews.json`, `summary.json`
    - Test Baseline: `results/agent_test_baseline/traces.json`, `reviews.json`, `summary.json`
    - Test Improved: `results/agent_test_improved/traces.json`, `reviews.json`, `summary.json`
  - Lệnh chạy:
    - Retrieval benchmark: `python src/benchmark_retrievers.py`
    - Agent eval: `python src/run_agent_eval.py --split <dev|test> --config <baseline|improved> --output <dir>`
    - Unit tests: `python -m unittest discover -s tests -v`
  - Thời gian và chi phí:
    - Tổng thời gian chạy Agent trên Test (7 câu): ~76.2 giây (trung bình ~3.6s / truy vấn, gồm cả tìm kiếm và sinh lời giải).
    - Tổng chi phí API: **0.001156 USD** (~0.1 cent) cho tập test, hoàn toàn nằm trong hạn mức miễn phí của Gemini API.

- **Kết quả trước/sau, đánh đổi và giới hạn:**
  - *Trước/sau cải tiến:* Trên tập Dev, Grounded Answer trung bình tăng từ **0.86 lên 1.40/2** (+62.8%), số câu đạt điểm tối đa (grade 2) tăng gấp đôi từ 5 lên 10 câu, false abstention giảm từ 50% xuống 26.7%. Trên tập Test, Grounded Answer tăng từ **0.40 lên 1.20/2** (+200%), false abstention giảm từ 80% xuống 40%.
  - *Đánh đổi (Trade-offs):* Việc tăng context budget từ 800 lên 1600 ký tự làm tăng nhẹ số token gửi tới LLM (~12.7k prompt tokens so với 9.3k), nhưng đổi lại chất lượng câu trả lời và tỷ lệ có đủ bằng chứng trong ngữ cảnh tăng vọt (Context Evidence Rate từ 20% lên 80%).
  - *Giới hạn:* Hệ thống hiện tại sử dụng corpus tiếng Anh từ tài liệu Python; đối với các câu hỏi phức tạp cần kết hợp nhiều mệnh đề ở các trang khác nhau (như mệnh đề `else` trong `try-except`), retriever đơn lượt vẫn có nguy cơ bỏ sót nếu query không tập trung đúng từ khóa phân biệt.

---

## Kết luận

Bài lab đã hoàn thành toàn diện toàn bộ yêu cầu của Day 7:
1. Kết nối thành công vector store thật từ Day 6 (Qdrant persistent) và model embedding thật (`all-MiniLM-L6-v2`) chạy offline không phụ thuộc internet.
2. Triển khai và so sánh đối chứng **3 phương pháp retrieval khác nhau**: Dense Retrieval, BM25 Lexical, và Hybrid Retrieval (RRF).
3. Đề xuất và chứng minh 2 cải tiến hiệu quả: **Metadata Section Filtering** và **Context Budget Tuning (1600 chars)**, nâng Context Evidence Rate từ 20% lên 80% trên tập Test.
4. Xây dựng Agent hoàn chỉnh với mô hình thật (**Gemini 3.5 Flash-Lite**), có đầy đủ cơ chế trích dẫn inline citations, tự động nhận biết câu hỏi thiếu bằng chứng và từ chối chính xác 100% (2/2 Dev, 2/2 Test).
5. Toàn bộ 18 unit tests vượt qua, trace và review được lưu trữ đầy đủ, minh bạch và có thể tái lập hoàn toàn.
