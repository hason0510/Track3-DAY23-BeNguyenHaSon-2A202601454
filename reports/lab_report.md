# Day 08 Lab Report — LangGraph Agentic Orchestration

> Báo cáo được sinh tự động bởi `render_report()` từ `outputs/metrics.json`.

## 1. Thông tin sinh viên

- Họ tên: Bé Nguyễn Hà Sơn (2A202601454)
- Repo/commit: https://github.com/hason0510/Track3-DAY23-BeNguyenHaSon-2A202601454
- Provider: OpenAI (`gpt-4o-mini` qua `get_llm()`)

## 2. Kiến trúc

Mười một node, tám fixed edge và bốn conditional edge. Chính các conditional edge là lý do
dùng LangGraph thay vì một chain tuyến tính: retry loop và approval gate là chu trình và
nhánh rẽ mà một LCEL pipeline không biểu diễn được.

```text
START -> intake -> classify -> [route_after_classify]
  simple       -> answer -> finalize -> END
  tool         -> tool -> evaluate -> [route_after_evaluate]
                                        success     -> answer -> finalize -> END
                                        needs_retry -> retry -> [route_after_retry]
                                                                  attempt <  max -> tool
                                                                  attempt >= max -> dead_letter
  missing_info -> clarify -> finalize -> END
  risky        -> risky_action -> approval -> [route_after_approval]
                                                approved -> tool -> evaluate -> ...
                                                rejected -> clarify -> finalize -> END
  error        -> retry -> [route_after_retry] -> ...
```

| Node | Trách nhiệm | LLM |
|---|---|---|
| `intake` | chuẩn hoá text ticket thô | không |
| `classify` | route + risk level bằng structured output | **có** |
| `tool` | mock lookup / action executor, mô phỏng lỗi tạm thời | không |
| `evaluate` | retry gate: tool result mới nhất có dùng được không? | có (LLM-as-judge) |
| `answer` | câu trả lời cuối, grounded theo tool results + approval | **có** |
| `clarify` | một câu hỏi làm rõ (thiếu thông tin hoặc bị từ chối duyệt) | có |
| `risky_action` | mô tả side effect để review; không thực thi gì | có |
| `approval` | HITL gate; mặc định mock, dùng `interrupt()` khi bật | không |
| `retry` | chủ sở hữu duy nhất của attempt counter | không |
| `dead_letter` | escalate khi đã cạn retry budget | không |
| `finalize` | một audit event kết thúc trên mọi route | không |

Các routing function là hàm thuần: chỉ đọc state và trả về tên node đã đăng ký — không gọi
LLM, không mutate, không side effect. Mỗi hàm đều fail closed (route lạ -> `answer`, counter
thiếu hoặc vượt ngưỡng -> `dead_letter`, không có approval -> `clarify`).

## 3. State schema

`AgentState` là `TypedDict(total=False)`; mỗi node chỉ trả về những key nó thay đổi và để
reducer merge vào state.

| Field | Reducer | Lý do |
|---|---|---|
| `thread_id`, `scenario_id` | overwrite | định danh run, giữ ổn định suốt execution |
| `query` | overwrite | được chuẩn hoá đúng một lần tại intake |
| `route` | overwrite | phân loại hiện tại; không bao giờ ghi đè thành `done`/`dead_letter`,
  vì làm vậy sẽ phá route metric |
| `risk_level` | overwrite | dữ liệu cho audit và prompt |
| `attempt` | overwrite | một counter duy nhất, chỉ retry node tăng |
| `max_attempts` | overwrite | retry bound của run, không tăng trong loop |
| `evaluation_result` | overwrite | verdict mới nhất mà `route_after_evaluate` đọc |
| `pending_question` | overwrite | câu hỏi làm rõ hiện tại |
| `proposed_action` | overwrite | action đang chờ duyệt |
| `approval` | overwrite | mapping serializable theo shape `ApprovalDecision` |
| `final_answer` | overwrite | output cuối cho answer / clarification / dead letter |
| `messages` | append (`add`) | dấu vết xử lý |
| `tool_results` | append (`add`) | kết quả tool theo thứ tự thời gian |
| `errors` | append (`add`) | ghi chú lỗi theo thứ tự thời gian |
| `events` | append (`add`) | audit event chuẩn hoá từ `make_event()` |

## 4. Kết quả scenario

| Chỉ số | Giá trị |
|---|---:|
| Tổng số scenario | 7 |
| Tỉ lệ thành công | 100% |
| Số node trung bình mỗi run | 6.43 |
| Tổng số lần retry | 3 |
| Tổng số lần vào approval | 2 |
| Đã kiểm chứng resume / state history | có |

| Scenario | Route mong đợi | Route thực tế | Kết quả | Retry | Approval | Event | Cần duyệt | Quan sát được duyệt | Latency (ms) |
|---|---|---|---|---:|---:|---:|---|---|---:|
| S01_simple | simple | simple | PASS | 0 | 0 | 4 | không | không | 5413 |
| S02_tool | tool | tool | PASS | 0 | 0 | 6 | không | không | 4741 |
| S03_missing | missing_info | missing_info | PASS | 0 | 0 | 4 | không | không | 5019 |
| S04_risky | risky | risky | PASS | 0 | 1 | 8 | có | có | 8693 |
| S05_error | error | error | PASS | 2 | 0 | 10 | không | không | 2970 |
| S06_delete | risky | risky | PASS | 0 | 1 | 8 | có | có | 3747 |
| S07_dead_letter | error | error | PASS | 1 | 0 | 5 | không | không | 830 |

- 2 scenario ghi nhận quyết định duyệt trước khi bất kỳ side effect nào chạy.
- 2 scenario đi vào retry loop, tổng cộng 3 lần ghé node `retry`; tất cả đều vẫn kết thúc tại `finalize`.
- Mọi scenario đều khớp route mong đợi và đều sinh ra output.

## 5. Phân tích failure mode

**1. Retry không có giới hạn / tool thất bại.** Mock tool trả về payload `ERROR` cho ticket
thuộc error route khi `attempt < 2`, `evaluate` biến điều đó thành `needs_retry`, và `retry`
tăng counter. Counter chỉ có đúng một chủ sở hữu: nếu `tool` cũng tăng, hoặc nếu
`route_after_retry` đọc giá trị trước khi tăng, thì loop hoặc chạy nhanh gấp đôi hoặc không
bao giờ dừng. `route_after_retry` dùng `attempt >= max_attempts` (không phải `==`) nên một
counter bất thường vẫn rơi xuống dead letter thay vì lặp vô hạn. `S07_dead_letter`
(`max_attempts=1`) là ca biên: lần retry đầu tiên đặt `attempt=1`, điều kiện `1 >= 1` đưa
thẳng tới `dead_letter` mà không hề gọi tool.

**2. Risky action chạy khi chưa được duyệt.** Approval phải là cổng chặn, không phải bản ghi
audit viết sau khi đã thực thi. Cạnh `risky_action -> approval -> [approved] -> tool` đảm bảo
thứ tự, nhưng chỉ dựa vào edge thì chưa đủ, nên `tool_node` tự nó từ chối chạy risky action
khi `approval.approved` không phải `True` và trả về kết quả `ERROR` bị chặn. Quyết định từ
chối sẽ đi tới `clarify` để hỏi khách hàng phương án thay thế thay vì im lặng bỏ ticket — và
`answer_node` được yêu cầu rõ ràng không bao giờ tuyên bố một action bị từ chối là đã thực
hiện.

**3. LLM/provider lỗi (failure mode thứ ba đã cân nhắc).** Mọi chỗ gọi LLM đều bắt lỗi
provider, ghi nguyên nhân vào `errors` và phát ra event `failed`. Ở nơi có đường degrade
(keyword fallback trong `classify`, câu hỏi mẫu trong `clarify`), event mang
`fallback=True` để một sự cố provider không bao giờ bị nhầm thành một lần phân loại thành
công trong audit trail.

## 6. Bằng chứng persistence / recovery

Graph đã compile nhận checkpointer được dựng từ `configs/lab.yaml`, và mỗi run được invoke với `{'configurable': {'thread_id': state['thread_id']}}`. Sau khi chạy, CLI đọc lại checkpoint qua `graph.get_state()` / `graph.get_state_history()` cho từng thread, và lần chạy này thành công — nên `resume_success=true`. Để có bằng chứng bền vững hơn, `agent-lab recovery-demo` ghi một run vào SQLite checkpointer, sau đó một process thứ hai mở lại chính database đó với `--inspect-only` và replay state history — chứng minh checkpoint sống sót sau khi process kết thúc.

Bằng chứng replay ghi trong `outputs\recovery_evidence.json` (mode: inspect-only (fresh process), pid 12864):

```json
{
  "mode": "inspect-only (fresh process)",
  "pid": 12864,
  "database": "checkpoints.db",
  "thread_id": "recovery-demo-01",
  "checkpoint_id": "1f1a064f-c16d-6470-8006-f08fbf97f428",
  "checkpoints_in_history": 8,
  "next_nodes": [],
  "route": "tool",
  "attempt": 0,
  "events_recorded": 6,
  "nodes_replayed": [
    "intake",
    "classify",
    "tool",
    "evaluate",
    "answer",
    "finalize"
  ],
  "final_answer_preview": "Your order status for order 12345 is currently open. It is being handled by our support team, and we aim to resolve it within 24 hours. If you have any further ",
  "resume_success": true
}
```

## 7. Phần mở rộng đã làm

- **LLM-as-judge** trong `evaluate_node`: `JudgeVerdict` dạng structured output quyết định
  `success` hay `needs_retry` cho các tool result không lỗi; marker `ERROR` tường minh đi
  theo fast path xác định nên retry loop không phụ thuộc độ trễ của model.
- **SQLite checkpointer** trong `persistence.py` (bật WAL) cùng lệnh CLI `recovery-demo` để
  replay state history giữa hai process khác nhau.
- **HITL thật**: `approval_node` gọi `langgraph.types.interrupt()` khi
  `LANGGRAPH_INTERRUPT=true`; mặc định vẫn là mock decision nên CI không bao giờ bị treo.
- **Sơ đồ graph**: `agent-lab show-graph` xuất Mermaid diagram của graph đã compile.

## 8. Kế hoạch cải thiện

1. Thay mock tool bằng tool call thật, có kiểu rõ ràng, đặt sau timeout + circuit breaker, và
   đổi retry tức thời thành exponential backoff.
2. Lưu approval vào store bên ngoài kèm danh tính reviewer và chữ ký audit, đồng thời điều
   khiển đường `interrupt()`/resume thật từ một UI review nhỏ.
3. Bổ sung bộ regression gồm các ticket đối kháng (prompt injection, ý định pha trộn, text
   rỗng) để assert route và bất biến "không có side effect trước khi được duyệt", nhờ đó việc
   sửa prompt của classifier không thể âm thầm làm hỏng tính chất an toàn này.
