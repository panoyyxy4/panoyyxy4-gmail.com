# Tối ưu hóa danh mục đầu tư bằng chiến lược MACD + RSI

Web app **Streamlit** tái hiện **đúng** quy trình trong tiểu luận *Quản lý danh mục đầu tư*
(và notebook gốc): tối ưu tham số → chọn danh mục → đầu tư & tái cân bằng →
**kiểm định kết quả mô hình** → **phân tích chiến lược**.

Thị trường HOSE · Giai đoạn học **2020** → Giai đoạn đầu tư **2021–2023**.

> Ở chế độ *Tái hiện nhanh*, app cho ra **đúng** kết quả gốc trên tệp dữ liệu kèm theo:
> lợi nhuận **189,90%**, Sharpe **1,392**, sụt giảm tối đa **−32,14%** (danh mục DRH, DBC, CTS, VIX).

---

## 1. App làm gì

| Bước | Nội dung |
|------|----------|
| **Học (2020)** | Tối ưu 5 tham số MACD + RSI cho từng mã bằng thuật toán **tối ưu bầy đàn (PSO, nevergrad)**, tối đa hóa tỉ số Sharpe. |
| **Chọn danh mục** | Chấm điểm theo phân vị lợi nhuận và Sharpe (50% – 50%), chọn **4 mã** điểm cao nhất. |
| **Đầu tư (2021–2023)** | Chia đều một tỉ đồng, **đóng băng** tham số, **tái cân bằng** theo tháng/quý/năm, giao dịch toàn bộ vào/ra, khớp tại **giá mở cửa phiên kế tiếp**, phí 0,15%. |
| **Kiểm định** | Kiểm định **t một mẫu** (lợi nhuận ngày > 0; vượt lãi suất phi rủi ro) và **Wilcoxon** (so với mua & giữ, so với VN-Index). Thanh trượt mức ý nghĩa α. |
| **Phân tích** | So sánh với mua & giữ và VN-Index; theo năm; theo từng mã; so sánh tần suất tái cân bằng. |

**Quy tắc tín hiệu**
- **MUA:** MACD cắt **lên** đường tín hiệu **và** RSI dưới ngưỡng trên.
- **BÁN:** MACD cắt **xuống** đường tín hiệu **hoặc** RSI vượt ngưỡng trên.

**Hai chế độ chạy (chọn ở thanh bên):**
- **Tái hiện nhanh** — dùng đúng 4 mã (DRH, DBC, CTS, VIX) và bộ tham số đã tối ưu trong bài → ra đúng kết quả gốc tức thì.
- **Chạy lại tối ưu hóa** — chạy PSO trên toàn bộ mã trong dữ liệu rồi tự chọn 4 mã (cần thư viện `nevergrad`; mất khoảng 1–3 phút, kết quả được lưu lại).

---

## 2. Cấu trúc thư mục

```
.
├── app.py             # Toàn bộ web app
├── requirements.txt   # Thư viện cần cài
├── README.md          # Tài liệu này
└── HOSE_2020_2023.csv # (khuyến nghị) dữ liệu ~7,5 MB, để app tự nạp khi deploy
```

### Định dạng dữ liệu (CSV)

Dạng **dài** — mỗi dòng là một mã trong một phiên. App đọc các cột chuẩn sau (tự nhận diện):

| Cột | Ý nghĩa |
|-----|---------|
| `date` | ngày giao dịch (định dạng `M/D/YYYY`, ví dụ `1/2/2020`) |
| `ticker` | mã cổ phiếu / chỉ số (app tự chuyển thành chữ HOA) |
| `adj_open` | giá mở cửa **đã điều chỉnh** (nếu thiếu, app dùng `open`) |
| `adj_close` | giá đóng cửa **đã điều chỉnh** (nếu thiếu, app dùng `close`) |

> Mã chỉ số VN-Index khai báo ở ô **"Tên mã chỉ số"** (mặc định `VNINDEX`). Đổi lại nếu dữ liệu ghi khác.

---

## 3. Chạy thử trên máy

```bash
pip install -r requirements.txt
streamlit run app.py
```

Mở `http://localhost:8501`, tải tệp CSV ở thanh bên.

---

## 4. Đưa lên GitHub và deploy trên Streamlit Cloud

### Bước 1 — Tạo repo GitHub
1. Tạo repo mới (ví dụ `macd-rsi-portfolio`), để **Public**.
2. Tải lên: `app.py`, `requirements.txt`, `README.md`, và `HOSE_2020_2023.csv` (7,5 MB — dưới giới hạn 100 MB của GitHub, nên commit để app tự nạp).

Hoặc dùng dòng lệnh:
```bash
git init
git add app.py requirements.txt README.md HOSE_2020_2023.csv
git commit -m "Web app MACD + RSI"
git branch -M main
git remote add origin https://github.com/<tên-tài-khoản>/macd-rsi-portfolio.git
git push -u origin main
```

### Bước 2 — Deploy
1. Vào **https://share.streamlit.io** → đăng nhập bằng GitHub.
2. **Create app** → **Deploy a public app from GitHub**.
3. Chọn: Repository = `<tên-tài-khoản>/macd-rsi-portfolio`, Branch = `main`, Main file path = `app.py`.
4. (Quan trọng cho mã QR trong bài) đặt **App URL** đúng là `macd-rsi-hose` để có địa chỉ
   `https://macd-rsi-hose.streamlit.app` — trùng với mã QR đã chèn trong tiểu luận.
5. **Deploy**. Chờ ít phút.

### Bước 3 — Mã QR cho bài
- Nếu đặt URL đúng `macd-rsi-hose`, mã QR trong phần Phụ lục của tiểu luận hoạt động ngay.
- Nếu dùng URL khác, tạo lại mã QR trỏ tới địa chỉ đó:
  ```bash
  pip install qrcode[pil]
  python -c "import qrcode; qrcode.make('https://<địa-chỉ-của-bạn>.streamlit.app').save('qr_app.png')"
  ```
  rồi thay ảnh trong tiểu luận.

---

## 5. Ghi chú phương pháp (để đối chiếu kỹ thuật)

- **Chỉ báo:** MACD = EMA(nhanh) − EMA(chậm), đường tín hiệu = EMA của MACD; RSI tính bằng **trung bình trượt đơn giản** theo chu kỳ.
- **Khớp lệnh:** lệnh phát sinh cuối phiên được khớp tại **giá mở cửa phiên kế tiếp** (loại bỏ thiên lệch nhìn trộm tương lai); chỉ báo giai đoạn đầu tư có **90 phiên khởi động** từ cuối 2020.
- **Tái cân bằng (tại giá đóng cửa):** tính NAV → mục tiêu mỗi ngăn = tỉ trọng × NAV → khối lượng dịch chuyển → trừ phí 0,15% → đặt lại các ngăn về tỉ trọng mục tiêu.
- **Cấu hình:** vốn 1 tỉ đồng, phí 0,15%/lệnh, lãi suất phi rủi ro 4%/năm, 252 phiên/năm, hạt giống PSO = 42 (tái lập kết quả).
- Kiểm định t giả định lợi nhuận theo ngày độc lập; thực tế có thể có tương quan chuỗi nhẹ, nên các giá trị p mang tính tham khảo định hướng.
- **Hiệu suất trong quá khứ không bảo đảm cho kết quả trong tương lai.**

---

*Tiểu luận cuối kỳ — Môn Quản lý danh mục đầu tư.*
