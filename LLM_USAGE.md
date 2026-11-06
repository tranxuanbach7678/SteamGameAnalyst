# LLM Usage Log (Nhật Ký Sử Dụng Mô Hình Ngôn Ngữ Lớn)

> **Môn học:** CSC17104 – Lập Trình Cho Khoa Học Dữ Liệu (Programming for Data Science)  
> **Trường:** Đại học Khoa học Tự nhiên – ĐHQG TP.HCM (HCMUS - FIT)  
> **Dự án cuối kỳ:** Steam Games Ecosystem & Commercial Performance Analysis  
> **Mục đích tài liệu:** Minh bạch hóa toàn bộ quá trình tham khảo và sử dụng các công cụ LLM (Google Gemini, Antigravity, ChatGPT, Claude, GitHub Copilot) theo đúng quy định tại **Mục 6 - Academic integrity and use of LLMs** trong đề cương dự án.

---

## 1. Nguyên Tắc Sử Dụng LLM Của Nhóm
Nhóm tuân thủ nghiêm ngặt quy định liêm chính học thuật của môn học:
- **Vai trò của LLM:** Được sử dụng như một **Gia sư (Tutor) & Trợ lý kỹ thuật (Technical Assistant)** nhằm giải thích khái niệm lý thuyết, tra cứu cú pháp thư viện, phát hiện lỗi parser (bug fixing), và gợi ý cấu trúc tài liệu.
- **Không ủy thác:** Toàn bộ việc xây dựng câu hỏi nghiên cứu, thực thi code phân tích, diễn giải ý nghĩa thực tế của kết quả kinh doanh và viết báo cáo phản ánh (reflections) đều do các thành viên nhóm tự thực hiện, kiểm tra và bảo đảm hiểu rõ từng dòng lệnh khi vấn đáp cá nhân.

---

## 2. Bảng Tổng Hợp Các Phiên Hỏi Đáp Chính

| STT | Thời gian | Công cụ | Thành viên | Mục đích / Câu hỏi đặt ra (Prompt) | Kết quả LLM cung cấp | Cách nhóm kiểm chứng & tùy biến lại |
|:---:|:---:|:---:|:---:|:---|:---|:---|
| **1** | 06/11/2026 | Google Gemini / Antigravity | Thành viên 1 | *"Khi đọc file `games.csv` kích thước ~414MB của Steam, phát hiện số lượng giá trị trong dòng dữ liệu là 40 nhưng header chỉ có 39. Làm sao để phát hiện và khắc phục lỗi lệch cột này?"* | Phân tích dòng header và chỉ ra cụm từ `'DiscountDLC count'` bị dính liền do thiếu dấu phẩy ngăn cách giữa hai cột `'Discount'` và `'DLC count'`. Gợi ý truyền danh sách `names` chuẩn 40 cột vào `pd.read_csv`. | Nhóm đã viết script Python thuần kiểm tra cấu trúc dòng bằng `csv.reader`, xác nhận cột 8 và 9 thực tế là Discount và DLC count, sau đó gói vào hàm `load_steam_raw_data` trong `src/data_loader.py`. |
| **2** | 06/11/2026 | ChatGPT (GPT-4o) | Thành viên 2 | *"Cách triển khai Bootstrap 95% Confidence Interval cho trung vị (median) của thời gian chơi giữa hai nhóm game có và không có DLC bằng NumPy / SciPy?"* | Cung cấp hàm mẫu sử dụng `scipy.stats.bootstrap` và vòng lặp `np.random.choice` tính sai số percentile. | Nhóm điều chỉnh số lần lấy mẫu (n_resamples=1000), xử lý giá trị ngoại lai (playtime = 0), và kết hợp thêm kiểm định phi tham số Mann-Whitney U test do phân phối thời gian chơi lệch phải nặng. |
| **3** | 06/11/2026 | Claude 3.5 Sonnet | Thành viên 3 | *"Làm sao để thiết lập Time-based Train/Test split và scikit-learn Pipeline có cả OneHotEncoder và StandardScaler mà không bị rò rỉ dữ liệu (data leakage)?"* | Hướng dẫn lọc theo năm phát hành `Release_Year` (Train $\le$ 2021, Test 2022–2024), tách riêng các cột rò rỉ (Reviews, Recommendations, Playtime), và dùng `ColumnTransformer`. | Nhóm đã tự kiểm tra danh sách loại trừ (excluded leaky columns), thiết lập metric đánh giá là PR-AUC và F1-macro do dữ liệu mất cân bằng (imbalanced target), kèm Permutation Importance. |
| **4** | 06/11/2026 | GitHub Copilot | Cả nhóm | Tra cứu cú pháp vẽ biểu đồ nâng cao với Seaborn (`sns.barplot`, `hue`, căn chỉnh format nhãn trên cột). | Tự động gợi ý snippet vòng lặp `ax.patches` để gán nhãn giá trị trực quan. | Nhóm tối ưu và đóng gói thành hàm `annotate_bars` trong module `src/visualization.py`. |

---

## 3. Bản Cam Kết Liêm Chính Học Thuật (Academic Honesty Declaration)
Chúng tôi cam kết:
1. Mọi code và văn bản trong đồ án này đều được kiểm chứng và hiểu rõ bởi các thành viên nhóm.
2. Không sao chép nguyên văn notebook từ Kaggle hay các nguồn công khai khác mà không trích dẫn nguồn.
3. Sẵn sàng giải thích chi tiết cơ chế hoạt động của mọi dòng code trong phần vấn đáp cá nhân (Q&A).

