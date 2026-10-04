# STEP8 – Phân tích kết quả benchmark

> Họ tên: Phùng Quốc Việt | MSSV: 2A202602456

---

## 1. Vì sao Advanced có recall tốt hơn Baseline?

**Baseline** chỉ giữ lịch sử hội thoại trong cùng một `thread_id`. Khi người dùng mở thread mới, toàn bộ context bị xóa — agent không còn biết tên, nghề nghiệp hay sở thích của người dùng. Đây là thiết kế cố ý: baseline là mốc so sánh, không phải mục tiêu.

**Advanced** thêm lớp persistent memory qua `User.md`. Mỗi lượt hội thoại:

1. `extract_profile_updates()` trích các fact ổn định từ tin nhắn người dùng (tên, nơi ở, nghề nghiệp, đồ uống yêu thích, …).
2. `upsert_fact()` ghi vào `state/profiles/<user_id>/User.md` trên đĩa.
3. Bất kỳ thread mới nào cho cùng `user_id` đều đọc lại `User.md` → recall cross-session hoạt động ngay cả khi agent chưa từng "gặp" thread_id đó.

**Kết quả đo được:**

| Benchmark | Baseline recall | Advanced recall |
|---|---|---|
| Standard (10 hội thoại) | 0.071 | **1.000** |
| Stress (16 turns dài) | 0.000 | **1.000** |

---

## 2. Vì sao Advanced có thể tốn token hơn ở hội thoại ngắn?

Mỗi lượt của Advanced phải mang theo ba thành phần vào prompt:

```
prompt_size = tokens(User.md) + tokens(compact_summary) + tokens(recent_messages)
```

Ở hội thoại ngắn (1–5 turns), chưa có gì để compact, trong khi `User.md` đã có nội dung từ các phiên trước. Kết quả là **overhead từ User.md và summary cộng dồn vào mỗi lượt**, khiến tổng prompt lớn hơn Baseline vốn chỉ giữ messages hiện tại.

**Số liệu Standard Benchmark (~10 turns/conv):**

| Agent | Prompt tokens processed |
|---|---|
| Baseline | 21,222 |
| Advanced | 26,149 (+23%) |

Đây là **trade-off có chủ đích**: chi thêm ~23% prompt token để đổi lấy recall tăng từ 0.071 lên 1.000. Ở hội thoại ngắn, chi phí này không được bù đắp bởi compact — nhưng giá trị recall hoàn toàn xứng đáng cho use case production.

---

## 3. Vì sao compact giúp Advanced có lợi thế ở hội thoại dài?

Khi thread dài vượt `compact_threshold_tokens`, `CompactMemoryManager` nén các message cũ thành summary và chỉ giữ `keep_messages` message gần nhất. Kết quả là **prompt context bị bounded** thay vì tăng tuyến tính như Baseline.

Cơ chế:

```
Trước compact:  [msg1, msg2, msg3, msg4, msg5]  → tổng lớn
Sau compact:    [summary(msg1-3), msg4, msg5]   → tổng nhỏ hơn đáng kể
```

**Số liệu Stress Benchmark (16 turns, nội dung rất dài):**

| Agent | Prompt tokens | Compactions |
|---|---|---|
| Baseline | 24,229 | 0 |
| Advanced | 13,060 | 18 |
| Tiết kiệm | **−46.1%** | — |

Baseline phải kéo toàn bộ 16 turns vào mỗi lượt (không compact được). Advanced nén dần qua 18 lần compaction, chỉ giữ 2–4 turns gần nhất + summary của phần cũ → prompt cost không phình.

**Lưu ý quan trọng:** compact chủ yếu tối ưu `prompt tokens processed` (context input), không nhất thiết giảm `agent tokens only` (output generation) vì agent vẫn cần sinh ra câu trả lời có độ dài tương đương.

---

## 4. File memory tăng trưởng ra sao và rủi ro gì đi kèm?

### Tăng trưởng thực tế

`User.md` tăng theo số **key mới** được extract, không theo số turns. Sau 10 hội thoại chuẩn: **254 bytes**. Sau 1 stress conversation 16 turns: **190 bytes**. Đây là mức rất nhỏ — nhưng theo thời gian sẽ tích lũy.

### Rủi ro 1 – File phình to

Nếu người dùng thường xuyên cung cấp facts mới hoặc nhiều loại preference, `User.md` có thể chứa hàng chục đến hàng trăm key → overhead prompt mỗi lượt tăng dần. Giải pháp: **memory decay** (giảm ưu tiên hoặc xóa facts cũ ít được nhắc lại).

### Rủi ro 2 – Lưu sai fact (false positive)

Regex `extract_profile_updates()` có thể bắt nhầm context. Ví dụ:

- `"Hà Nội chỉ là nơi mình đi họp"` → nếu không có noise guard, location bị lưu thành "Hà Nội"
- `"hay là chuyển sang product manager (câu đùa)"` → nếu không có joke guard, profession bị lưu sai

Đã xử lý bằng `_NOISE_LOCATION`, `_NOISE_JOB` pattern và **confidence threshold** (Bonus): fact có score < 0.5 bị discard.

### Rủi ro 3 – Correction bị bỏ sót

Nếu người dùng đổi thông tin bằng câu không rõ ràng (`"thật ra bây giờ ở chỗ khác rồi"`), `extract_profile_updates()` có thể không nhận ra. Đã giảm thiểu bằng `_CORRECTION_MARKERS` regex, nhưng vẫn có false negative với cách diễn đạt bất thường.

### Rủi ro 4 – Conflict giữa fact cũ và mới

Nếu không có `upsert_fact()` update-in-place, `User.md` sẽ chứa cả fact cũ và mới song song, gây confusion. Implementation hiện tại dùng regex để **overwrite in-place** → `User.md` luôn chứa giá trị mới nhất cho mỗi key.
