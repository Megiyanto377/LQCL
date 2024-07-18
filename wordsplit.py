import os
import glob

os.chdir('IDLRWV2_3')

current_path = os.getcwd()

dir_list = os.listdir(current_path)
length = len(dir_list)
fold_name= ['test','train','val']
final_list = []

# Words provided
words = [
    "ada", "agak", "agama", "agar", "aja", "akan", "aku", "anak", "anda", "antara", "apa", "atas", "atau", 
    "aturan", "awal", "bagi", "bagian", "bahasa", "bahkan", "bahwa", "baik", "bakal", "balik", "banget", 
    "banyak", "baru", "beda", "begitu", "belajar", "benar", "berapa", "berarti", "berita", "bersama", "besar", 
    "besok", "biar", "biasa", "bicara", "bikin", "bilang", "bisa", "boleh", "buat", "bukan", "bulan", "cara", 
    "cukup", "cuma", "daerah", "dalam", "dan", "dapat", "dari", "data", "datang", "dengan", "depan", "dia", 
    "diri", "dua", "dulu", "dunia", "emang", "enggak", "gak", "gini", "gitu", "hadir", "hal", "hanya", "harga", 
    "hari", "harus", "hasil", "hidup", "hukum", "ibu", "ingin", "ini", "isu", "itu", "jadi", "jalan", "jam", 
    "jangan", "jauh", "jelas", "juga", "kabar", "kalau", "kali", "kalian", "kami", "kamu", "kasih", "kasus", 
    "kata", "keluar", "keluarga", "kembali", "kemudian", "kenapa", "kepada", "kepala", "kerja", "kita", "kok", 
    "kondisi", "kurang", "lagi", "lain", "lakukan", "lalu", "lama", "langsung", "lebih", "lihat", "luar", "lupa", 
    "maka", "makan", "malam", "mana", "manusia", "masa", "masalah", "masih", "masuk", "masyarakat", "mata", "mau", 
    "media", "melihat", "memang", "memberikan", "membuat", "menarik", "mengatakan", "menjadi", "menurut", "merasa", 
    "mereka", "muda", "mudah", "mulai", "muncul", "mungkin", "naik", "nama", "namanya", "nanti", "negara", "negeri", 
    "oleh", "orang", "pada", "pagi", "pak", "pakai", "paling", "para", "partai", "pas", "pasti", "pemerintah", 
    "pengen", "penting", "perempuan", "perlu", "pertama", "pertanyaan", "pihak", "politik", "program", "proses", 
    "publik", "pun", "rumah", "saja", "sakit", "salah", "sama", "sampai", "sana", "sangat", "sarapan", "satu", 
    "saya", "sebenarnya", "sebetulnya", "sedang", "segala", "sekali", "selama", "selamat", "semangat", "sempat", 
    "sendiri", "seperti", "sering", "siap", "siapa", "sini", "situ", "soal", "sosial", "suara", "sudah", "suka", 
    "supaya", "tadi", "tahu", "tahun", "telah", "tempat", "tengah", "tentang", "terima", "terus", "tetap", "tidak", 
    "tinggi", "tua", "uang", "udah", "untuk", "waktu", "warga", "yang"
]

list_kata_1 = [word for word in words if 3 <= len(word) <= 4]
list_kata_2 = [word for word in words if 5 <= len(word) <= 6]
list_kata_3 = [word for word in words if 7 <= len(word) <= 8]
list_kata_4 = [word for word in words if len(word) >= 9]


for i in range(length):

    path_ = str(os.path.join(current_path, dir_list[i]))

    if ('desktop.ini' or '.ipynb') in path_:

        continue
    os.chdir(path_)
    arr = [dir_list[i]]
    arr = str(arr)
    arr = arr.replace("'","")
    arr = arr.replace("[","")
    arr = arr.replace("]","")
    if arr in list_kata_1 :
        percentage = 1.0

    elif arr in list_kata_2 :
        percentage = 1.0
    elif arr in list_kata_3 :
        percentage = 1.0
    elif arr in list_kata_4 :
        percentage = 0.7

    for j in range(len(fold_name)):

        cur_path = os.path.join(path_, fold_name[j])

        dir_cur_list = os.listdir(cur_path)

        length_cur = len(dir_cur_list)

        length_cur_20 = int(length_cur*percentage)

        for k in range(length_cur):
            if k > length_cur_20:
                if ('desktop.ini' or '.ipynb_checkpoints') in (os.path.join(cur_path,dir_cur_list[k])):
                    k = k-1
                    continue
                try:
                    os.remove(os.path.join(cur_path, dir_cur_list[k]))
                except IsADirectoryError:
                    k = k-1
                    pass

