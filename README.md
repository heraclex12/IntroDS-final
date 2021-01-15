# Đồ án cuối kì môn Nhập môn Khoa học dữ liệu

### STT: 22
### Thành viên 1:
- Họ và tên: Trần Trung Hiếu
- MSSV: 1712442
### Thành viên 2:
- Họ và tên: Lê Long Hồ
- MSSV: 1712447
### Thành viên 3:
- Họ và tên: Nguyễn Đình Thiên Phúc
- MSSV: 18120144


## Thông tin đề tài:
Dữ liệu được thu thập từ trang [TMDB](https://www.themoviedb.org/)

**Câu hỏi**: Phân loại một bộ phim có phù hợp với lứa tuổi dưới 18 hay không?
- Input: ảnh poster và nội dung tóm tắt của bộ phim
- Output: 1-Trên 18 tuổi, 0-ngược lại

---
## Danh sách các tệp:
- [movieCrawling.py](https://github.com/heraclex12/IntroDS-final/blob/main/movieCrawling.py): mã nguồn thu thập dữ liệu.
- [1712442_1712447_18120144.ipynb](https://github.com/heraclex12/IntroDS-final/blob/main/1712442_1712447_18120144.ipynb): notebook chứa mã nguồn xử lí dữ liệu và mô hình hóa.
- [resources](https://github.com/heraclex12/IntroDS-final/tree/main/resources): thư mục chứa các tệp hỗ trợ quá trình xử lí dữ liệu
  - vi_stopwords.txt
  - en_stopwords.txt
  - age_restricted.txt
- [movie_final.csv](https://github.com/heraclex12/IntroDS-final/blob/main/movie_final.csv): tệp bảng dữ liệu về thông tin các bộ phim
- [poster_img_pixels.csv](https://github.com/heraclex12/IntroDS-final/blob/main/poster_img_pixels.csv): tệp histogram của ảnh poster phim
