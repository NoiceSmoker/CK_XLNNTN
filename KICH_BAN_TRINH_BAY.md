# Kịch bản trình bày (13 slide, khoảng 7 phút 30 giây)

Slide: https://claude.ai/artifact/MSpxFCuiRjTzybaxUjCMu9

**Mạch chính:** bài toán khó → tái lập CroCoAlign → thấy kém → tìm nguyên nhân → sửa đúng chỗ đó → chứng minh bằng
kết quả và phân tích lỗi. Mỗi slide kết thúc bằng một câu dẫn trả lời "vì sao sang slide sau".

---

## Slide 1 · Bìa (15 giây)

> Em xin trình bày đề tài 05: dóng hàng câu Hán – Việt cho Đại Việt Sử Ký Toàn Thư. Đề tài có ba yêu cầu: tiền xử lý
> dữ liệu, tái lập mô hình CroCoAlign theo bài báo EACL 2024, và đánh giá, cải tiến. Nhóm đã thực hiện đầy đủ cả ba.

**Dẫn:** Trước hết, em xin nói rõ bài toán.

## Slide 2 · Bài toán (45 giây)

> Dóng hàng câu là cho văn bản gốc và bản dịch, tìm các nhóm câu tương ứng. Một câu Hán có thể ứng với không, một hoặc
> nhiều câu Việt; ví dụ zh_6 được dịch thành ba câu, zh_1 và zh_2 được gộp thành một. Ta không thể nối câu thứ i với câu
> thứ i vì số câu hai bên khác nhau: mục Sĩ Vương có 75 câu Hán nhưng 91 câu Việt.
>
> Dữ liệu này khó vì ba lý do: Hán cổ không có dấu câu, văn ngôn thế kỷ XV nằm ngoài dữ liệu mà các mô hình đã học, và
> bản dịch chèn nhiều chú giải. Nhưng có một thuận lợi: bản dịch giữ đúng thứ tự câu của bản gốc. Đặc điểm này sẽ là
> chìa khoá cho cải tiến của nhóm.

**Dẫn:** Vậy dữ liệu được xây dựng thế nào, và cái khó đầu tiên là tách câu Hán, nhóm giải quyết ra sao?

## Slide 3 · Ngữ liệu (45 giây)

> Nhóm thu thập toàn bộ 14 mục Ngoại kỷ từ nomfoundation: 178 trang, 1.703 câu Hán và 1.833 câu Việt. Mỗi đoạn chữ Hán
> trên trang nguồn đều đi kèm phiên âm Hán-Việt đã có dấu câu, như Hình 1.

**Lý do cách làm:**

> Vì vậy nhóm không cần mô hình tách câu cho văn ngôn. Mô hình như thế không có sẵn, và nếu dùng thì lỗi tách câu sẽ
> lan sang bước dóng hàng. Thay vào đó, mỗi đoạn được xem là một câu, dựa vào chính dấu câu của người biên soạn. Phiên
> âm còn là cầu nối từ vựng: bản dịch giữ nguyên tên người, địa danh, nên nhóm tận dụng nó cho cả việc gán đáp án lẫn
> một baseline.

**Dẫn:** Dữ liệu đã đúng định dạng đầu vào của CroCoAlign. Tiếp theo là mô hình.

## Slide 4 · Kiến trúc CroCoAlign (60 giây)

> CroCoAlign gồm ba bước. Một, LaBSE, được giữ cố định, biến mỗi câu thành vector 768 chiều. Hai, Context Encoder, một
> Transformer 6 lớp dùng chung cho cả hai phía, đưa ngữ cảnh của các câu lân cận vào từng câu; đây là đóng góp chính của
> bài báo. Ba, với mọi cặp câu, mô hình nhân hai vector rồi đưa qua bộ phân loại; lớn hơn 0,5 thì nối.

**Lý do cách làm:**

> Nhóm dùng checkpoint chính thức và chạy zero-shot, vì đề yêu cầu tái lập mô hình gốc, và cũng không có dữ liệu
> Hán – Việt đã dóng sẵn để huấn luyện lại. Mã gốc dùng thư viện cũ, nên nhóm chuyển sang torch 2.x và viết lại phần nạp
> mô hình, không sửa mã của tác giả.

**Nhấn mạnh (chỉ vào dòng đỏ):**

> Khi suy luận, mô hình xử lý theo lô và mỗi cặp được quyết định độc lập, không có ràng buộc thứ tự. Em xin ghi nhớ
> điểm này.

**Dẫn:** Để biết mô hình chạy tốt hay không, cần có đáp án để chấm.

## Slide 5 · Thiết kế đánh giá (45 giây)

> ĐVSKTT không có sẵn đáp án dóng hàng như Opus Books trong bài báo, nên nhóm phải tự xây dựng.

**Lý do cách làm:**

> Đáp án phải độc lập với mô hình được chấm. Nếu sinh đáp án bằng LaBSE, các hệ dùng LaBSE sẽ được lợi, giống như tự
> chấm bài mình. Vì vậy nhóm lập phiếu cho ba mục 7, 9, 10, gán nhãn bằng Claude Code theo năm quy tắc do nhóm đặt ra,
> ghi lý do cho mọi nhóm không phải 1-1, và dùng script kiểm tra mỗi câu chỉ xuất hiện một lần. Kết quả là 185 nhóm.
>
> Vì tập nhỏ, nhóm chọn tham số bằng kiểm định chéo bỏ-một-mục: chấm mục nào thì chọn tham số trên hai mục còn lại, để
> không chọn tham số trên chính tập được chấm. Thước đo là F1 strict, nghĩa là nhóm phải khớp chính xác.

**Dẫn:** Chấm CroCoAlign trên bộ đáp án này, nhóm thấy kết quả thấp. Khi xem kỹ thì lỗi đến từ chính điểm em vừa nhắc:
quyết định từng cặp độc lập. Đó là lý do cho phương pháp đề xuất.

## Slide 6 · Phương pháp đề xuất (45 giây)

**Lý do cách làm (nói trước):**

> Vì bản dịch giữ đúng thứ tự câu, ràng buộc thứ tự gần như luôn đúng; trong 185 nhóm đáp án không có nhóm nào đảo thứ
> tự. Cách chuẩn để áp ràng buộc này là quy hoạch động, như Gale–Church và Vecalign đã dùng, và nó chỉ tốn O(nm).

> Quy hoạch động tìm một đường đi trên ma trận điểm, chỉ đi xuống hoặc sang phải, sao cho tổng điểm lớn nhất, như
> Hình 3. Các liên kết đảo thứ tự, ô đỏ, bị loại.
>
> Nhóm xây dựng bốn hệ dùng chung bộ giải mã này, chỉ khác hàm điểm, như Bảng 2. Hai baseline không dùng mạng nơ-ron làm
> mốc tham chiếu: Gale–Church theo độ dài, và Hán-Việt theo âm tiết trùng với phiên âm. Cải tiến 1 giữ nguyên LaBSE của
> chính checkpoint CroCoAlign. Đây là thí nghiệm có kiểm soát: cùng biểu diễn câu, chỉ đổi cách ra quyết định, để biết
> lỗi nằm ở đâu. Cải tiến 2 kết hợp LaBSE với Hán-Việt.

**Dẫn:** Em tóm hai cách ra quyết định bằng công thức.

## Slide 7 · Hàm mục tiêu (25 giây, nói lướt)

> Bên trái, CroCoAlign học bằng cross-entropy trên toàn ma trận, phương trình 1, và khi suy luận thì nối từng cặp có xác
> suất lớn hơn 0,5, phương trình 2. Bên phải, quy hoạch động cực đại tổng điểm trên một đường đi đơn điệu, phương
> trình 3. Bốn hệ chỉ khác hàm điểm S, từ phương trình 4 đến 7. Riêng nhóm 1-2 và 2-1, điểm được tính trên văn bản đã
> ghép, để câu ngắn không kéo điểm xuống.

**Dẫn:** Và đây là kết quả.

## Slide 8 · Kết quả (35 giây)

> CroCoAlign zero-shot đạt F1 strict 0,529, tinh chỉnh lên 0,565, thấp hơn nhiều so với 87,5 trong bài báo vì Hán cổ
> nằm ngoài dữ liệu huấn luyện. Cả bốn hệ dùng quy hoạch động đều trên 0,87. Đáng chú ý là Gale–Church, chỉ đếm độ dài
> câu, đã đạt 0,879. Baseline Hán-Việt cao nhất với 0,934, nhưng ba hệ đầu bảng chỉ chênh nhau vài nhóm, nằm trong sai
> số.

**Dẫn:** Con số quan trọng nhất là so sánh giữa CroCoAlign và cải tiến 1.

## Slide 9 · Kết quả chính (10 giây)

> Cùng một bộ mã hoá LaBSE, chỉ thay cách ra quyết định: từ 0,529 lên 0,918. Vấn đề nằm ở cách chọn liên kết, không nằm
> ở biểu diễn câu.

**Dẫn:** Hình sau cho thấy điều đó trực quan hơn.

## Slide 10 · Minh hoạ (25 giây)

> Đây là mục 7. Ô xám là đáp án, nằm dọc đường chéo vì bản dịch giữ thứ tự. Bên trái, CroCoAlign: nhiều đoạn trên đường
> chéo bị bỏ trống, và có những chấm đỏ nằm rất xa. Bên phải, quy hoạch động bám sát đường chéo.

**Dẫn:** Những chấm đỏ đó cụ thể là lỗi gì?

## Slide 11 · Phân tích lỗi (40 giây)

> Trên mục 7, CroCoAlign sai 36 trên 75 câu, theo ba kiểu. Nối xa: zh_9 bị nối sang vi_56, lệch 46 câu. Bỏ sót: zh_7 có
> nội dung rõ nhưng xác suất chưa tới 0,5. Bắt thiếu: zh_6 chỉ nối được một trong ba câu dịch. Cả ba kiểu đều xuất phát
> từ việc quyết định từng cặp độc lập. Ràng buộc đơn điệu loại bỏ cả ba: ứng viên ở xa bị chặn, còn cặp có điểm vừa phải
> vẫn được chọn nếu nằm trên đường đi tối ưu.

**Dẫn:** Từ đó, nhóm rút ra hai kết luận.

## Slide 12 · Kết luận (45 giây)

> Thứ nhất, với bản dịch giữ thứ tự như ĐVSKTT, ràng buộc đơn điệu quan trọng hơn chất lượng biểu diễn câu. Thứ hai, khi
> đã có ràng buộc thứ tự, tín hiệu tương đồng đơn giản là đủ: chỉ độ dài đã được 0,879, phiên âm được 0,934, không cần
> mạng nơ-ron.
>
> Về hạn chế, tập đáp án còn nhỏ, gán bằng Claude Code và chưa có người gán độc lập kiểm định. Hệ vẫn sai ở các nhóm một
> câu Hán ứng ba câu Việt trở lên, vì nằm ngoài các bước của bộ giải mã. Hướng tiếp theo là mở rộng và kiểm định đáp án,
> bổ sung các bước 1-3 và 2-2, và tinh chỉnh CroCoAlign trên dữ liệu Hán cổ – Việt.

## Slide 13 · Cảm ơn (5 giây)

> Em xin trân trọng cảm ơn quý thầy cô đã lắng nghe, và xin được nhận câu hỏi.

---

## Bốn lý do cách làm cần nói rõ

| Quyết định | Lý do |
|---|---|
| Tách câu Hán nhờ phiên âm | Không có mô hình tách câu văn ngôn; dùng dấu câu của người biên soạn để tránh lỗi lan |
| Tự xây đáp án độc lập | Không có đáp án sẵn; đáp án sinh từ LaBSE sẽ thiên vị hệ dùng LaBSE |
| Quy hoạch động | Bản dịch giữ thứ tự, còn CroCoAlign quyết định từng cặp nên không tận dụng được; DP là cách chuẩn để áp ràng buộc thứ tự |
| Giữ LaBSE của checkpoint (cải tiến 1) | Thí nghiệm có kiểm soát: cùng biểu diễn, chỉ đổi cách ra quyết định, để chứng minh nguyên nhân |

## Câu hỏi dễ gặp: lỗi có lan sang câu sau không?

Không. Quy hoạch động tối ưu toàn cục và được phép bỏ qua một câu với một khoản phạt nhỏ, nên một nhóm lấy thừa hoặc
thiếu chỉ ảnh hưởng tối đa nhóm ngay sau. Trên cả ba mục, chuỗi lỗi liên tiếp dài nhất của cải tiến 1 là 2 nhóm
(ví dụ mục 7: zh_5, zh_6 sai, từ zh_7 đúng trở lại).
