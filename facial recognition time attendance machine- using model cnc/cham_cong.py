import openpyxl
from datetime import datetime, date
import datetime
import requests
def chamcong(name,final):

    workbook = openpyxl.load_workbook(r'E:\code\demo\project\lvtn\pretrained_models\COLAP\excel\output.xlsx')
    sheet = workbook.active
    last_row = sheet.max_row


    new_employee = {
        'name': 'Đỗ Văn D',
        'date': '2023-07-02',
        'in': '08:30',
        'statr': 'đúng giờ',
        'check_in': '8:00',
        'check_out': '12:30',
    }
    fi =['xxx114xxxx','20511400XX','2051140058']
    D = ['tu dong hoa','co khi','co khi']
    current_time = datetime.datetime.now()
    date_part = current_time.strftime("%Y-%m-%d")
    time_part = current_time.strftime("%H:%M:%S")


    print("Ngày tháng năm:", date_part)
    print("Giờ phút giây:", time_part)

    out_time = datetime.datetime.strptime(new_employee['check_in'], "%H:%M").time()
    check_out_time = datetime.datetime.strptime(new_employee['check_out'], "%H:%M").time()
    
    time_obj = datetime.datetime.strptime(time_part, "%H:%M:%S").time()

    # Chuyển đổi out_time và time_obj thành datetime.datetime
    out_datetime = datetime.datetime.combine(current_time.date(), out_time)
    time_datetime = datetime.datetime.combine(current_time.date(), time_obj)
    check_out_time_datetime = datetime.datetime.combine(current_time.date(), check_out_time)
    # Tính toán thời gian làm việc
    ditre = 0
    print(check_out_time_datetime)
    if (time_datetime < out_datetime  ):
        new_employee['state'] = 'đúng giờ'
        print("đúng giờ ")
    if ((time_datetime > out_datetime ) & (time_datetime< check_out_time_datetime)  ):  
        ditre = time_datetime - out_datetime
        new_employee['state'] = f'đi trễ:{ditre}'
        print(ditre)
    if(time_datetime > check_out_time_datetime):
        new_employee['state'] = f'ra về'
        print(ditre)

    new_employee['in'] = time_part
    new_employee['date'] = date_part
    new_employee['name'] = name
    sheet[f'A{last_row + 1}'] = last_row
    sheet[f'B{last_row + 1}'] = new_employee['name']
    sheet[f'C{last_row + 1}'] = new_employee['date']
    sheet[f'D{last_row + 1}'] = new_employee['in']
    sheet[f'E{last_row + 1}'] = new_employee['state']

    url = f"https://script.google.com/macros/s/AKfycbw8KBUNLUDfVi9IE9zdrdj6IeThBaGVYq255k01QEc-kbGl2H05k7F6O6DPbVIC3uBw/exec?value1={new_employee['name']}&value2=Am20a&value3={fi[final]}&value4={D[final]}"

# Thực hiện yêu cầu HTTP GET
    response = requests.get(url)

    # Kiểm tra mã trạng thái HTTP
    if response.status_code == 200:
        # Lấy nội dung phản hồi
        data = response.text
        print(data)
    else:
        print(f"Lỗi: {response.status_code}")
    # Lưu file Excel
    workbook.save(r'E:\code\demo\project\lvtn\pretrained_models\COLAP\excel\output.xlsx')

# Mở file Excel

# name ="lê hữu bảo"
# chamcong("name",1)