## Identitas
Nama : Muhamad Hilmy Haidar  
NIM : G64170030  

## Daftar Isi
Pada folder ini berisi beberapa file yaitu :  
1. Folder `images/*` yang berisi file-file gambar yang digunakan dan dihasilkan.  
2. File `uts_pcd.py` yang merupakan code jawaban dari problem yang diberikan  

## Problem 3
Pada soal ketiga, diberikan dua buah gambar `flowers.tif` dan `flower-template.tif`. 
Kita akan melakukan template matching dari gambar yang diberikan dengan fungsi-fungsi yang ada.

### Template Matching  
Untuk melakukan template matching dengan cross correlation, kita bisa menggunakan fungsi 
dari opencv yaitu `cv2.matchTemplate(src, templt, method)` dengan argumen pada method yang
diberikan yaitu `cv2.TM_CCORR_NORMED` untuk cross corelation