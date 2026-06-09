import pandas as pd
import re
import os

# ==========================================
# 1. BỘ TỪ ĐIỂN HEURISTIC ĐA KHÍA CẠNH
# ==========================================
HEURISTIC_DICT = {
    'Size': {
        0: [r"chật", r"trật", r"rộng", r"kích chân", r"nhấc gót", r"tuột gót", r"không vừa", r"sai size", r"tăng size", r"lùi size", r"đau mũi", r"kích mũi"],
        1: [r"vừa vặn", r"chuẩn size", r"vừa in", r"đúng size", r"ôm chân", r"vừa y", r"chuẩn phom", r"fit chân"],
        2: [r"hơi rộng", r"hơi chật", r"vừa tạm", r"rộng xíu", r"chật xíu"]
    },
    'Price': {
        0: [r"đắt", r"mắc", r"chát", r"phí tiền", r"giá cao", r"không đáng", r"xót tiền"],
        1: [r"rẻ", r"hạt dẻ", r"hời", r"sale", r"giá sinh viên", r"giá mềm", r"ngon bổ rẻ", r"đáng tiền", r"giá tốt", r"giá ok", r"giá êm"],
        2: [r"tầm giá", r"tiền nào của nấy", r"hợp giá tiền", r"đúng giá", r"giá phù hợp", r"giá tạm"]
    },
    'Shipping': {
        0: [r"giao lâu", r"ship lâu", r"chậm", r"ngâm hàng", r"chờ mòn mỏi", r"thái độ shipper", r"shipper tệ"],
        1: [r"giao nhanh", r"ship nhanh", r"hỏa tốc", r"nhanh chóng", r"shipper nhiệt tình", r"shipper dễ thương", r"giao sớm", r"thần tốc"],
        2: [r"giao bình thường", r"thời gian giao ok", r"ship cũng được"]
    },
    'Outlook': {
        0: [r"xấu", r"phèn", r"sai màu", r"lệch màu", r"không giống hình", r"khác hình", r"ọp ẹp", r"form lỗi", r"phom méo", r"dấu keo", r"vết dơ", r"vết bẩn"],
        1: [r"đẹp", r"xinh", r"xỉu", r"lung linh", r"sang", r"đúng màu", r"y hình", r"mượt", r"xịn xò", r"chuẩn màu", r"dáng đẹp", r"phom đẹp", r"mẫu đẹp"],
        2: [r"mã tạm", r"nhìn tạm", r"mẫu mã bình thường", r"cũng được", r"ổn áp"]
    },
    'Quality': {
        0: [r"cứng", r"đau chân", r"cước chân", r"trầy gót", r"rách", r"bong keo", r"hôi", r"mùi nồng", r"mùi keo", r"trơn trượt", r"dỏm", r"lởm", r"mỏng", r"hỏng", r"nhựa", r"lót mỏng"],
        1: [r"êm", r"mềm", r"đi thích", r"đi sướng", r"nhẹ", r"bền", r"chắc chắn", r"không đau", r"thoải mái", r"chất lượng tốt", r"chất ok", r"xịn"],
        2: [r"chất tạm", r"đi tạm", r"chất liệu bình thường", r"vẫn đi được"]
    },
    'Shop_Service': {
        0: [r"không rep", r"seen", r"bơ khách", r"shop thái độ", r"đóng gói sơ sài", r"bọc sơ sài", r"hộp móp", r"nát hộp", r"nhàu nát", r"gói hàng chán", r"phục vụ kém"],
        1: [r"shop nhiệt tình", r"tư vấn có tâm", r"đóng gói cẩn thận", r"bọc kỹ", r"hộp đẹp", r"nguyên vẹn", r"rep nhanh", r"hỗ trợ tốt", r"uy tín", r"thư cảm ơn"],
        2: [r"shop bình thường", r"đóng gói tạm", r"có bọc"]
    },
    'General': {
        0: [r"thất vọng", r"tệ", r"chán", r"đừng mua", r"né gấp", r"chạy ngay đi", r"bai bai", r"khuyên không nên"],
        1: [r"tuyệt vời", r"hoàn hảo", r"10 điểm", r"nên mua", r"xuất sắc", r"5 sao", r"okela", r"duyệt", r"ưng ý", r"quá đã"],
        2: [r"bình thường", r"tạm ổn", r"chấp nhận được", r"không có gì đặc sắc", r"cũng ok"]
    },
    'Others': {
        0: [r"thiếu quà", r"không có hộp", r"không có bill"],
        1: [r"tặng kèm", r"có quà", r"tặng tất", r"tặng vớ", r"kèm dây giày", r"fullbox", r"đầy đủ phụ kiện"],
        2: []
    }
}

NEGATION_WORDS = [r"không", r"k", r"ko", r"chưa", r"chả", r"đếch"]

# ==========================================
# 2. HÀM KHÔI PHỤC TỰ ĐỘNG
# ==========================================
def recover_all_missing_labels(row):
    review = str(row['Review_Cleaned']).lower()
    
    def is_negated(keyword_match, text):
        start_idx = keyword_match.start()
        prefix = text[max(0, start_idx - 15):start_idx]
        for neg in NEGATION_WORDS:
            if re.search(r'\b' + neg + r'\b', prefix):
                return True
        return False

    for aspect in HEURISTIC_DICT.keys():
        current_label = row[aspect]
        if current_label == -1:
            recovered_label = -1
            for sentiment, keywords in HEURISTIC_DICT[aspect].items():
                for kw in keywords:
                    matches = list(re.finditer(kw, review))
                    if matches:
                        match = matches[0]
                        if is_negated(match, review):
                            if sentiment == 1: recovered_label = 0 
                            elif sentiment == 0: recovered_label = 1 
                            else: recovered_label = 2
                        else:
                            recovered_label = sentiment
                        break 
                if recovered_label != -1:
                    break
            row[aspect] = recovered_label
    return row

# ==========================================
# 3. ÁP DỤNG HÀNG LOẠT CHO 3 FILE DỮ LIỆU
# ==========================================
def process_datasets():
    # Danh sách các file cần xử lý
    files_to_process = [
        "Shoes_Train_Preprocessed.csv",
        "Shoes_Validate_Preprocessed.csv",
        "Shoes_Test_Preprocessed.csv"
    ]

    print("=== BẮT ĐẦU KHÔI PHỤC NHÃN BẰNG HEURISTIC ===\n")

    for file_name in files_to_process:
        if not os.path.exists(file_name):
            print(f"⚠️ Không tìm thấy file '{file_name}'. Vui lòng kiểm tra lại đường dẫn. Đang bỏ qua...")
            continue
            
        print(f"-> Đang xử lý file: {file_name} ...")
        df = pd.read_csv(file_name)
        
        # Đếm trước khi khôi phục
        missing_before = (df[list(HEURISTIC_DICT.keys())] == -1).sum().sum()
        
        # Áp dụng Heuristic
        df = df.apply(recover_all_missing_labels, axis=1)
        
        # Đếm sau khi khôi phục
        missing_after = (df[list(HEURISTIC_DICT.keys())] == -1).sum().sum()
        recovered_count = missing_before - missing_after
        
        # Tạo tên file mới
        new_file_name = file_name.replace(".csv", "_Recovered.csv")
        df.to_csv(new_file_name, index=False)
        
        print(f"   + Số nhãn -1 ban đầu : {missing_before}")
        print(f"   + Đã cứu thành công  : {recovered_count} nhãn")
        print(f"   + Đã lưu file mới tại: {new_file_name}\n")

    print("=== HOÀN TẤT QUÁ TRÌNH XỬ LÝ ===")

if __name__ == "__main__":
    process_datasets()