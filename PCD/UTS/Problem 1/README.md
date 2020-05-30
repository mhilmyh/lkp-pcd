## Identitas
Nama : Muhamad Hilmy Haidar  
NIM : G64170030  

## Daftar Isi
Pada folder ini berisi beberapa file yaitu :  
1. Folder `images/*` yang berisi file-file gambar yang digunakan dan dihasilkan.  
2. File `uts_pcd.py` yang merupakan code jawaban dari problem yang diberikan  

## Problem 1
Pada soal pertama, diberikan sebuah gambar `tomato_2.jpg`. 
Kita akan melakukan segmentasi untuk mendapatkan buah tomat 
yang matang. Berikut adalah _fungsi_ yang digunakan untuk menjawab
permasalahan tersebut.  

### Color Space
Apabila kita melihat gambar `tomato_2.jpg`, kita akan dapat melihat 
perbedaan warna yang signifikan dari buah tomat yang matang dan yang tidak. Terdapat
dua kode untuk conversi color space yaitu :    
```python
# Konversi dari RGB ke YUV
def rgb2yuv(gambar):
    row, col, _ = gambar.shape
    hasil = np.zeros_like(gambar)

    for i in range(row):
        for j in range(col):
            b, g, r = gambar[i, j]
            y = min(255, max(0, 0.29900 * r + 0.58700 * g + 0.11400 * b))
            u = min(255, max(0, -0.147108 * r - 0.288804 * g + 0.435912 * b + 127.5))
            v = min(255, max(0, 0.614777 * r - 0.514799 * g - 0.099978 * b + 127.5))
            hasil[i, j] = [y, u, v]

    return hasil, hasil[:, :, 0], hasil[:, :, 1], hasil[:, :, 2]
```    
```python
# Konversi dari RGB ke HSV
def rgb2hsv(gambar):
    row, col, _ = gambar.shape
    hasil = np.zeros_like(gambar)

    for i in range(row):
        for j in range(col):
            b, g, r = gambar[i, j]
            r, g, b = r / 255, g / 255, b / 255

            h = 0
            mx = max(r, g, b)
            mn = min(r, g, b)
            df = mx - mn

            if mx == r:
                h = (60 * ((g - b) / df) + 360) % 360
            elif mx == g:
                h = (60 * ((b - r) / df) + 120) % 360
            elif mx == b:
                h = (60 * ((r - g) / df) + 240) % 360
            if mx == 0:
                s = 0
            else:
                s = df / mx

            v = mx
            h = h * 255 / 360
            s *= 255
            v *= 255

            hasil[i, j] = h, s, v

    return hasil, hasil[:, :, 0], hasil[:, :, 1], hasil[:, :, 2]
```  

Untuk konversi dari RGB ke LAB bisa menggunakan package yaitu `cv2.cvtColor(src, cv2.COLOR_BGR2LAB)`.  

### Plot Histogram
Untuk menampilkan plot histogram gunakan code :  
```python
import matplotlib.pyplot as plt

plt.hist(tomat_v.ravel(), 256, [0, 256])
plt.show()
```  

### Fungsi Masking  
Fungsi masking yang digunakan di sini berasal dari package opencv bitwise-and yaitu `cv2.bitwise_and(src, dst, mask)`. 
Fungsi tersebut tidak akan mengambil nilai pixel di src jika pixel di masknya bernilai nol.