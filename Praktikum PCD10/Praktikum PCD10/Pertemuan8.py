import cv2 #memproses citra / memanggil gambar
import math
import imutils #unntuk melakukan perubahan ukuran
import sys #management system pengelolaan proses
import numpy as np # untuk menghitung histogram dari gambar / perhitungan matematika
from PyQt5 import QtCore, QtWidgets # untuk proses gui
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
from matplotlib import pyplot as plt #untuk proses histogram
from skimage import data, exposure
from skimage.feature import hog



class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self). __init__()
        loadUi('GUI.ui', self)
        self.Image = None
        #operasi titik
        self.button_LoadCitra.clicked.connect(self.fungsiopen)
        self.button_save.clicked.connect(self.grayscale)
        self.actionOperasi_Pencerahan.triggered.connect(self.brightness)
        self.actionSimple_Contrast.triggered.connect(self.contrast)
        self.actionContrast_Streching.triggered.connect(self.contrastStreching)
        self.actionNegative.triggered.connect(self.negative)
        self.actionBinner.triggered.connect(self.biner)

        #histogram
        self.actionHistogram_GreyScale.triggered.connect(self.greyHistogram)
        self.actionHistogram_RGB.triggered.connect(self.HistogramRGB)
        self.actionHistogram_Equalization.triggered.connect(self.HistogramEqual)

        #operasi geometri
        self.actionTranslasi.triggered.connect(self.trasnlasi)
        self.action90_Derajat.triggered.connect(self.rotasi90)
        self.action45_Derajat.triggered.connect(self.rotasi45)
        self.action180_Derajat.triggered.connect(self.rotasi180)
        self.action_45_Derajat.triggered.connect(self.rotasimin45)
        self.action_90_Derajat.triggered.connect(self.rotasimin90)
        self.actionCrop.triggered.connect(self.crop)

        #operasi aritmatika
        self.action2x.triggered.connect(self.zoomin2)
        self.action3x.triggered.connect(self.zoomin3)
        self.action4x.triggered.connect(self.zoomin4)
        self.action0_75.triggered.connect(self.zoomout075)
        self.action0_5.triggered.connect(self.zoomout05)
        self.action0_25.triggered.connect(self.zoomout025)
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionKali_dan_Bagi.triggered.connect(self.aritmatikaKB)

        #operasi boolean
        self.actionOperasi_AND.triggered.connect(self.operasiAND)
        self.actionOperasi_OR.triggered.connect(self.operasiOR)
        self.actionOperasi_XOR.triggered.connect(self.operasiXOR)

        #operasi spasial
        self.actionKonvolusi_A.triggered.connect(self.FilteringCliked)
        self.actionKonvolusi_B.triggered.connect(self.Filterring2)
        self.actionKernel_1_9.triggered.connect(self.Mean3x3)
        self.actionKernel_1_4.triggered.connect(self.Mean2x2)
        self.actionGaussian_Filter.triggered.connect(self.Gaussian)
        self.actionke_1.triggered.connect(self.Sharpening1)
        self.actionke_2.triggered.connect(self.Sharpening2)
        self.actionke_3.triggered.connect(self.Sharpening3)
        self.actionke_4.triggered.connect(self.Sharpening4)
        self.actionke_5.triggered.connect(self.Sharpening5)
        self.actionke_6.triggered.connect(self.Sharpening6)
        self.actionLaplace.triggered.connect(self.Laplace)
        self.actionMedianFilter.triggered.connect(self.Median)
        self.actionMaxFilter.triggered.connect(self.Max)
        self.actionMinFilter.triggered.connect(self.Min)

        # Transformasi Fourier
        self.actionDFT_Smoothing_Image.triggered.connect(self.SmoothImage)
        self.actionDFT_Edge_Detection.triggered.connect(self.EdgeDetec)

        # Deteksi Tepi
        self.actionOperasi_Sobel.triggered.connect(self.Opsobel)
        self.actionOperasi_Prewitt.triggered.connect(self.Opprewitt)
        self.actionOperasi_Robert.triggered.connect(self.Oprobert)
        self.actionOperasi_Canny.triggered.connect(self.OpCanny)

        # Morfologi
        self.actionDilasi.triggered.connect(self.MortlgiDilasi)
        self.actionErosi.triggered.connect(self.MorflgiErosi)
        self.actionOpening.triggered.connect(self.MorflgiOpening)
        self.actionClosing.triggered.connect(self.MorflgiClosing)
        self.actionSkeletonizing.triggered.connect(self.MorfSkeleton)
        # Segmentasi Citra
        self.actionBinary.triggered.connect(self.Binary)
        self.actionBinar_Invers.triggered.connect(self.BinaryInvers)
        self.actionTrunc.triggered.connect(self.Trunc)
        self.actionTo_Zero.triggered.connect(self.ToZero)
        self.actionTo_Zero_Invers.triggered.connect(self.ToZeroInvers)
        self.actionMean_Thresholding.triggered.connect(self.Meanthres)
        self.actionGaussian_Thresholding.triggered.connect(self.Gausthres)
        self.actionOtsu_Thresholding.triggered.connect(self.Otsuthres)
        self.actionContur.triggered.connect(self.Contour)

        #color prosseing
        self.actionTracking.triggered.connect(self.track)
        self.actionPicker.triggered.connect(self.pick)

        #cascade
        self.actionObject_Detection.triggered.connect(self.objectdetection)
        self.actionHistogram_of_Gradient.triggered.connect(self.HOG)
        self.actionHaar_Cascade_Eye_and_Face.triggered.connect(self.FaceandEye)
        self.actionHaar_Cascade_Pedestrian_Detection.triggered.connect(self.Pedestrian)
        self.actionCircle_Hough_Transform.triggered.connect(self.CircleHough)
        self.actionHistogram_of_Gradient_Jalan.triggered.connect(self.HOGJalan)

    def fungsiopen(self):
        self.image = cv2.imread('ht.jpg')
        self.displayImage(1)

    def grayscale(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)

    def brightness(self):
        try:
            self.image = cv2.cvtColor(self.image,cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.image.item(i,j)
                b = np.clip(a + brightness, 0, 255)

                self.image.itemset((i,j), b)

        self.displayImage(1)

    def contrast(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        contrast = 1.7
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = np.clip(a * contrast, 0, 255)

                self.image.itemset((i, j), b)

        self.displayImage(1)

    def contrastStreching(self):
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        minv = np.min(self.image)
        maxv = np.max(self.image)
        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = float(a - minv) / (maxv -  minv) * 255

                self.image.itemset((i, j), b)

        self.displayImage(1)

    def negative(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(255 - self.image[i, j, 0], 0, 255)
        self.image = gray
        self.displayImage(2)

    def biner(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        H, W = self.image.shape[:2]
        for i in np.arange(H):
            for j in np.arange(W):
                a = img.item(i, j)
                if a == 180:
                    a = 0
                elif a < 180:
                    a = 1
                else:
                    a = 255

        self.image = img
        print(self.image)
        self.displayImage(2)



    def displayImage(self, windows):
        qformat = QImage.Format_Indexed8

        if len(self.image.shape) == 3:
            if(self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)

        img = img.rgbSwapped()

        # self.label.setPixmap(QPixmap.fromImage(img))

        if windows == 1:
            self.label.setPixmap(QPixmap.fromImage(img))

        if windows == 2:
            self.label_2.setPixmap(QPixmap.fromImage(img))

    def greyHistogram(self):
        H, W = self.image.shape[:2]
        gray = np.zeros((H, W), np.uint8)
        for i in range(H):
            for j in range(W):
                gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                     0.587 * self.image[i, j, 1] +
                                     0.114 * self.image[i, j, 2], 0, 255)
        self.image = gray
        self.displayImage(2)
        plt.hist(self.image.ravel(), 255, [0, 255]) #untuk membuat histogram yg diambil dari pixel gambar
        plt.show() #menampilkan histogram

    def HistogramRGB(self):
        color = ('b', 'g', 'r') # array tp tidak bisa di ubah
        for i, col in enumerate(color): #melakukan perullangan berdasarkan warna diatas
            histo = cv2.calcHist([self.image], [i], None, [256], [0, 256]) #menghitung histogram dari sekumpulan array dari gambar
        plt.plot(histo, color=col) #untuk ploting ke histogram
        plt.xlim([0, 256])#mengatur batas sumbu x 256
        self.displayImage(2)
        plt.show()

    def HistogramEqual(self):
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256]) #mengubah image array menjadi 2dimensi
        cdf = hist.cumsum() #menghitung jumlah komulatif array pada sumbu tertentu
        cdf_normalized = cdf * hist.max() / cdf.max() # untuk melakukan normalisasi
        cdf_m = np.ma.masked_equal(cdf, 0) # masking / menutup array dgn nilai yg sama
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) #perhitungan
        cdf = np.ma.filled(cdf_m, 0).astype('uint8') #mnegisi nilai array dgn nilai skalar
        self.image = cdf[self.image] # mengganti nilai menjadi nilai komulatif
        self.displayImage(2)

        plt.plot(cdf_normalized, color='b') #ploting sesuai normalisasi
        plt.hist(self.image.flatten(), 256, [0, 256], color='r') #membuat histogram sesuai dgn nilai array
        plt.xlim([0, 256]) # mengatur sumbu x
        plt.legend(('cdf', 'histogram'), loc='upper left') #membuat text di histogram atas kiri
        plt.show()

    def trasnlasi(self):
        H, W = self.image.shape[:2]
        quarter_H, quarter_W = H / 6, W / 6 #
        T = np.float32([[1, 0, quarter_W], [0, 1, quarter_H]]) #matriks berupa numpy 32bit
        img = cv2.warpAffine(self.image, T, (W, H)) #menggeser image dengan matriks
        self.image = img
        self.displayImage(2)

    def rotasimin45(self):
        self.rotasi(-45)

    def rotasimin90(self):
        self.rotasi(-90)

    def rotasi180(self):
        self.rotasi(180)

    def rotasi45(self):
        self.rotasi(45)

    def rotasi90(self): #untuk derajat rotasi
        self.rotasi(90)

    def rotasi(self, degree):
        H, W = self.image.shape[:2]
        rotationMatrix = cv2.getRotationMatrix2D((W / 2, H / 2), degree, 0.7) #

        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        nW = int((H * sin) + (W * cos))
        nH = int((H * cos) + (W * sin))

        rotationMatrix[0, 2] += (nW / 2) - W / 2
        rotationMatrix[1, 2] += (nH / 2) - H / 2

        rot_image = cv2.warpAffine(self.image, rotationMatrix, (H, W)) #untuk merotasi image
        self.image = rot_image
        self.displayImage(2)

    def crop(self):
        x = 0 #variabel awal
        y = 0 #variabel awal
        a = 300 #variabel akhir
        b = 300 #variabel akhir
        citra = self.image[x:a, y:b] #menyimpan citra
        cv2.imshow('Original', self.image) #memunculkan img
        cv2.imshow('Crop', citra) #memunculkan img
        cv2.waitKey()

    def zoomin2(self):
        skala = 2
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomin3(self):
        skala = 3
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomin4(self):
        skala = 4
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomout075(self):
        skala = 0.75
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def zoomout05(self):
        skala = 0.5
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def zoomout025(self):
        skala = 0.25
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom Out', resize_image)
        cv2.waitKey()

    def aritmatika(self):
        image1 = cv2.imread('jt.jpg', 0)
        image2 = cv2.imread('ht.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Tambah', image_tambah)
        cv2.imshow('Image Kurang', image_kurang)
        cv2.waitKey()

    def aritmatikaKB(self):
        image1 = cv2.imread('jt.jpg', 0)
        image2 = cv2.imread('ht.jpg', 0)
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Kali', image_kali)
        cv2.imshow('Image Bagi', image_bagi)
        cv2.waitKey()

    def operasiAND(self): #menampilkan irisan area
        image1 = cv2.imread('jt.jpg', 0)
        image2 = cv2.imread('ht.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_and(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi AND', operasi)
        cv2.waitKey()

    def operasiOR(self): #menampilkan gabungan area
        image1 = cv2.imread('jt.jpg', 0)
        image2 = cv2.imread('ht.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_or(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi OR', operasi)
        cv2.waitKey()

    def operasiXOR(self): #menampilkan area diluar irisan
        image1 = cv2.imread('jt.jpg', 0)
        image2 = cv2.imread('ht.jpg', 0)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasi = cv2.bitwise_xor(image1, image2)
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image Operasi XOR', operasi)
        cv2.waitKey()

    def Konvolusi(self, X, F):  # Fungsi konvolusi 2D
        X_height = X.shape[0]  # membaca ukuran tinggi dan lebar citra
        X_width = X.shape[1]

        F_height = F.shape[0]  # membaca ukuran tinggi dan lebar kernel
        F_width = F.shape[1]

        H = (F_height) // 2 #mencari titik tengah kernel
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        for i in np.arange(H + 1, X_height - H):  # mengatur pergerakan karnel
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]
                        w = F[H + k, W + l]
                        sum += (w * a)  # menampung nilai total perkalian w kali a
                out[i, j] = sum  # menampung hasil
        return out

    def FilteringCliked(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY) #mengubah citra menjadi greyscale
        kernel = np.array(
            [
                [1, 1, 1],  # array kernelnya
                [1, 1, 1],
                [1, 1, 1]
            ])

        img_out = self.Konvolusi(img, kernel) #parameter dari fungsi konvolusi 2d
        print('---Nilai Pixel Filtering A--- \n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic') # bicubic untuk memperhalus tepi citra
        plt.xticks([]), plt.yticks([])
        plt.show()  # Menampilkan gambar

    def Filterring2(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            [
                [6, 0, -6],  # array kernelnya
                [6, 1, -6],
                [6, 0, -6]
            ]
        )

        img_out = self.Konvolusi(img, kernel)
        print('---Nilai Pixel Filtering B---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([]) #mengatur lokasi tick dan label sumbu x dan y
        plt.show()  # Menampilkan gambar

    def Mean2x2(self):  # Fungsi Mean  2x2       #D2
        mean = (1.0 / 4) * np.array(  # mengganti intensitas pixel dengan nilai rata-rata pixel
            [  # array kernel 3x3
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 2x2 ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([]) #mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    def Mean3x3(self):  # Fungsi Mean 3x3         #D2
        mean = (1.0 / 9) * np.array(  # Penapis rerata 1/9
            [  # array kernel 3x3
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('---Nilai Pixel Mean Filter 3x3---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([]) #mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    def Gaussian(self):  # proses penghalusan dan pengaburan citra
        gausian = (1.0 / 345) * np.array(
            [  # Kernel gaussian
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = self.Konvolusi(img, gausian)
        print('---Nilai Pixel Gaussian ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([]) #mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    def Sharpening1(self):  # D4
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 8, -1],
                [-1, -1, -1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel i ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening2(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [-1, -1, -1],
                [-1, 9, -1],
                [-1, -1, -1]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        print('---Nilai Pixel Kernel ii ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening3(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, -1, 0],
                [-1, 5, -1],
                [0, -1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iii ---\n', img_out) # memunculkan nilai pixel pada terminal
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening4(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 5, -2],
                [1, 2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel iv ---\n', img_out) # memunculkan nilai pixel pada terminal
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening5(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [1, -2, 1],
                [-2, 4, -2],
                [1, -2, 1]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel v ---\n', img_out) # memunculkan nilai pixel pada terminal
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Sharpening6(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = np.array(
            [
                [0, 1, 0],
                [1, -4, 1],
                [0, 1, 0]
            ]
        )
        img_out = self.Konvolusi(img, sharpe)
        print('---Nilai Pixel Kernel vi ---\n', img_out) # memunculkan nilai pixel pada terminal
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Laplace(self): #menandai bagian yang menjadi detail citra dan memperbaiki serta mengubah citra
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = (1.0 / 16) * np.array(
            [
                [0, 0, -1, 0, 0],
                [0, -1, -2, -1, 0],
                [-1, -2, 16, -2, -1],
                [0, -1, -2, -1, 0],
                [0, 0, -1, 0, 0]
            ])
        img_out = self.Konvolusi(img, sharpe)
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    def Median(self):  # mencari nilai tengah pixel setelah diurutkan
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # mengubah citra ke grayscale
        img_out = img.copy()
        H, W = img.shape[:2]  # tinggi dan lebar citra

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                neighbors = []  # menampung nilai pixel
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # menampung hasil
                        neighbors.append(a)  # menambahkan a ke neighbors
                neighbors.sort()  # untuk mengurutkan neighbors
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Median Filter---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Max(self):  #proses menggantikan nilai piksel dengan nilai piksel maksimum yang dipengaruhi piksel tetangga.
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # mengecek nilai setiap pixel
            for j in np.arange(3, W - 3):
                max = 0
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai hasil baca pixel
                        if a > max:
                            max = a
                        b = max
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Maximun Filter ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Min(self): #proses menggantikan nilai piksel dengan nilai piksel minimum yang dipengaruhi piksel tetangga.
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # mengecek nilai setiap pixel
            for j in np.arange(3, W - 3):
                min = 0
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai hasil baca pixel
                        if a < min:
                            min = a
                        b = min
                img_out.itemset((i, j), b)
        print('---Nilai Pixel Minimun Filter ---\n', img_out) # memunculkan nilai pixel pada terminal
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

        #E1
    def SmoothImage(self):
            x = np.arange(256)
            y = np.sin(2 * np.pi * x / 3)
            y += max(y)
            img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

            plt.imshow(img)
            img = cv2.imread('jantung.jpg', 0) #image noise yg di convert ke greyscale

            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT) #variabel dft untuk melakukan proses transformasi diskrit
            dft_shift = np.fft.fftshift(dft) #proses nilai koordinat fft supaya array berada di tengah (titik origin)

            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1]))) #menghitung spektrum

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.zeros((rows, cols, 2), np.uint8)
            r = 50
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0]) * 2 + (y - center[1]) * 2 <= r * r

            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift) #Proses mengembalikan titik Origin

            img_back = cv2.idft(f_ishift) #mengkonversi citra dalam frekuensi jadi salam spasial
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1]) # proses mengembalikan nilai real dan imaginer

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(magnitude_spectrum, cmap='gray')
            ax2.title.set_text('FFT of Image')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('Inverse Fourier')
            plt.show()

        # E2
    def EdgeDetec(self):
            x = np.arange(256)
            y = np.sin(2 * np.pi * x / 3)

            y += max(y)

            img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

            plt.imshow(img)
            img = cv2.imread("jantung.jpg", 0)

            dft = cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
            dft_shift = np.fft.fftshift(dft)

            magnitude_spectrum = 20 * np.log((cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))

            rows, cols = img.shape
            crow, ccol = int(rows / 2), int(cols / 2)
            mask = np.ones((rows, cols, 2), np.uint8)
            r = 80
            center = [crow, ccol]
            x, y = np.ogrid[:rows, :cols]
            mask_area = (x - center[0] * 2 + (y - center[1])) * 2 <= r * r

            mask[mask_area] = 1

            fshift = dft_shift * mask
            fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
            f_ishift = np.fft.ifftshift(fshift)

            img_back = cv2.idft(f_ishift)
            img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

            fig = plt.figure(figsize=(12, 12))
            ax1 = fig.add_subplot(2, 2, 1)
            ax1.imshow(img, cmap='gray')
            ax1.title.set_text('Input Image')
            ax2 = fig.add_subplot(2, 2, 2)
            ax2.imshow(magnitude_spectrum, cmap='gray')
            ax2.title.set_text('FFT of Image')
            ax3 = fig.add_subplot(2, 2, 3)
            ax3.imshow(fshift_mask_mag, cmap='gray')
            ax3.title.set_text('FFT + Mask')
            ax4 = fig.add_subplot(2, 2, 4)
            ax4.imshow(img_back, cmap='gray')
            ax4.title.set_text('Inverse fourier')
            plt.show()

        # F1
    def Opsobel(self):
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            X = np.array([[-1, 0, 1],
                          [-2, 0, 2],
                          [-1, 0, 1]])
            Y = np.array([[-1, -2, -1],
                          [0, 0, 0],
                          [1, 2, 1]])
            img_Gx = self.Konvolusi(img, X)
            img_Gy = self.Konvolusi(img, Y)
            img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
            img_out = (img_out / np.max(img_out)) * 255
            print('---Nilai Pixel Operasi Sobel--- \n', img_out)
            self.Image = img
            self.displayImage(2)
            plt.imshow(img_out, cmap='gray', interpolation='bicubic')
            plt.show()

    def Opprewitt(self):
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            prewit_X = np.array([[-1, 0, 1],
                                 [-1, 0, 1],
                                 [-1, 0, 1]])
            prewit_Y = np.array([[-1, -1, -1],
                                 [0, 0, 0],
                                 [1, 1, 1]])
            img_Gx = self.Konvolusi(img, prewit_X)
            img_Gy = self.Konvolusi(img, prewit_Y)
            img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
            img_out = (img_out / np.max(img_out)) * 255
            print('---Nilai Pixel Operasi Prewitt --- \n', img_out)
            self.Image = img
            self.displayImage(2)
            plt.imshow(img_out, cmap='gray', interpolation='bicubic')
            plt.show()

    def Oprobert(self):
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            RX = np.array([[1, 0, 0],
                           [0, -1, 0],
                           [0, 0, 0]])
            RY = np.array([[0, 1, 0],
                           [-1, 0, 0],
                           [0, 0, 0]])
            img_Gx = self.Konvolusi(img, RX)
            img_Gy = self.Konvolusi(img, RY)
            img_out = np.sqrt((img_Gx * img_Gx) + (img_Gy * img_Gy))
            img_out = (img_out / np.max(img_out)) * 255
            print('---Nilai Pixel Operasi Robert--- \n', img_out)
            self.Image = img
            self.displayImage(2)
            plt.imshow(img_out, cmap='gray', interpolation='bicubic')
            plt.show()

    def OpCanny(self):
            # Langkah ke 1 (Reduksi Noise0
            img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
            gaus = (1.0 / 57) * np.array(
                [[0, 1, 2, 1, 0],
                 [1, 3, 5, 3, 1],
                 [2, 5, 9, 5, 2],
                 [1, 3, 5, 3, 1],
                 [0, 1, 2, 1, 0]])
            img_out = self.Konvolusi(img, gaus)
            img_out = img_out.astype("uint8")
            cv2.imshow("Noise Reduction", img_out)

            # Langkah ke 2 (Finding Gradien)
            Gx = np.array([[-1, 0, 1],
                           [-2, 0, 2],
                           [-1, 0, 1]])
            Gy = np.array([[-1, -2, -1],
                           [0, 0, 0],
                           [1, 2, 1]])
            konvolusi_x = self.Konvolusi(img, Gx)
            konvolusi_y = self.Konvolusi(img, Gy)
            theta = np.arctan2(konvolusi_y, konvolusi_x)
            theta = theta.astype("uint8")
            cv2.imshow("Finding Gradien", theta)

            # Langkah Ke 3 (Non Maximum suppression)
            H, W = img.shape[:2]
            Z = np.zeros((H, W), dtype=np.int32)

            angle = theta * 180. / np.pi
            angle[angle < 0] += 180
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    try:
                        q = 255
                        r = 255

                        # Angle 0
                        if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                            q = img_out[i, j + 1]
                            r = img_out[i, j - 1]
                        # Angle 45
                        elif (22.5 <= angle[i, j] < 67.5):
                            q = img_out[i + 1, j - 1]
                            r = img_out[i - 1, j + 1]
                        # Angle 90
                        elif (67.5 <= angle[i, j] < 112.5):
                            q = img_out[i + 1, j]
                            r = img_out[i - 1, j]
                        # Angle 135
                        elif (112.5 <= angle[i, j] < 157.5):
                            q = img_out[i + 1, j - 1]
                            r = img_out[i - 1, j + 1]
                        if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                            Z[i, j] = img_out[i, j]
                        else:
                            Z[i, j] = 0

                    except IndexError as e:
                        pass

            img_N = Z.astype("uint8")
            cv2.imshow("Non Maximum Supression", img_N)

            # Langkah ke 4 (Hysterisis Tresholding)
            weak = 80
            strong = 110
            for i in np.arange(H):
                for j in np.arange(W):
                    a = img_N.item(i, j)
                    if (a > weak):
                        b = weak
                        if (a > strong):
                            b = 255
                    else:
                        b = 0
                    img_N.itemset((i, j), b)
            img_H1 = img_N.astype("uint8")
            cv2.imshow("Hysterisis part 1", img_H1)
            print('---Nilai Pixel Hysterisis Part 1--- \n', img_H1)

            # Hysteresis Thresholding eliminasi titk tepi lemah jika tidak terhubung dengan tetangga tepi kuat
            strong = 255
            for i in range(1, H - 1):
                for j in range(1, W - 1):
                    if (img_H1[i, j] == weak):
                        try:
                            if ((img_H1[i + 1, j - 1] == strong) or
                                    (img_H1[i + 1, j] == strong) or
                                    (img_H1[i + 1, j + 1] == strong) or
                                    (img_H1[i, j - 1] == strong) or
                                    (img_H1[i, j + 1] == strong) or
                                    (img_H1[i - 1, j - 1] == strong) or
                                    (img_H1[i - 1, j] == strong) or
                                    (img_H1[i - 1, j + 1] == strong)):
                                img_H1[i, j] = strong
                            else:
                                img_H1[i, j] = 0
                        except IndexError as e:
                            pass
            img_H2 = img_H1.astype("uint8")
            cv2.imshow("Hysteresis part 2", img_H2)
            print('---Nilai Pixel Hysterisis Part 1--- \n', img_H2)
# MORFOLOGI
    def MortlgiDilasi(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Membaca dan merubah citra menjadi grayscale
        ret, imgthres = cv2.threshold(img, 127, 255, 0) # Merubah Citra yang sudah di graysacle menjadi citra biner
        strel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)) # Inisialisasi Strel
        dilasi = cv2.dilate(imgthres, strel, iterations=1) # Membuat fungsi dilasi
        cv2.imshow("Hasil Dilasi", dilasi)
        print('---Nilai Pixel Dilasi--- \n', dilasi)

    def MorflgiErosi(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        erosi = cv2.erode(imgthres, strel, iterations=1)
        cv2.imshow("Hasil Erosi", erosi)
        print('---Nilai Pixel Erosi--- \n', erosi)

    def MorflgiOpening(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        opening = cv2.morphologyEx(imgthres, cv2.MORPH_OPEN, strel)
        cv2.imshow("Hasil Opening", opening)
        print('---Nilai Pixel Opening--- \n', opening)

    def MorflgiClosing(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        ret, imgthres = cv2.threshold(img, 127, 255, 0)
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))
        closing = cv2.morphologyEx(imgthres, cv2.MORPH_CLOSE, strel)
        cv2.imshow("Hasil Closing", closing)
        print('---Nilai Pixel Closing--- \n', closing)

    def MorfSkeleton(self):
        img = cv2.imread("ht.jpg")
        imgg = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        ret, imgg = cv2.threshold(imgg, 127, 255, 0)

        skel = np.zeros(imgg.shape, np.uint8) #inisialisasi skel
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        while True:
            open = cv2.morphologyEx(imgg, cv2.MORPH_OPEN, strel)
            # mengurangi gambar dari yang asli
            temp = cv2.subtract(imgg, open)
            # erosi citra asli dan perbaikan kerangka
            eroded = cv2.erode(imgg, strel)
            skel = cv2.bitwise_or(skel, temp)
            imgg = eroded.copy()
            # jika tidak ada pixel putih yang tersisa
            if cv2.countNonZero(imgg) == 0:
                break
            cv2.imshow("Skeleton", skel)
            print("---------Nilai pixel------\n", skel)
            cv2.imshow("Origin", img)
            print("---------Nilai pixel------\n", img)

    # SEGMENTASI
    # Thresholding merupakan proses mengubah citra grayscale menjadi citra biner berdasarkan nilai ambang T
    def Binary(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY) # Membaca citra dan merubahnya menjadi grayscale
        T = 127  # inisialisasi nilai ambang
        MAX = 255 # Inisialisasi nilai max derajat keabuan
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_BINARY)
        cv2.imshow('Binary', thres)
        print("---------Nilai pixel Binary------\n", thres)

    def BinaryInvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_BINARY_INV)
        cv2.imshow('Binaryinvers', thres)
        print("---------Nilai pixel Binary Invers------\n", thres)

    def Trunc(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TRUNC)
        cv2.imshow('Trunc', thres)
        print("---------Nilai pixel Trunc------\n", thres)

    def ToZero(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TOZERO)
        cv2.imshow('ToZero', thres)
        print("---------Nilai pixel ToZero------\n", thres)

    def ToZeroInvers(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 127
        MAX = 255
        ret, thres = cv2.threshold(img, T, MAX, cv2.THRESH_TOZERO_INV)
        cv2.imshow('ToZeroInvers', thres)
        print("---------Nilai pixel ToZero Invers------\n", thres)

        # H2
    def Meanthres(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow('MeanThresholding', thres)
        print("---------Nilai pixel MeanThres------\n", thres)

    def Gausthres(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        thres = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 3, 6)
        cv2.imshow('GaussianTresholding', thres)
        print("---------Nilai pixel GaussThress------\n", thres)

    def Otsuthres(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 130
        ret, thres = cv2.threshold(img, T, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        cv2.imshow('OtsuTresholding', thres)
        print('---Nilai Pixel Closing--- \n', thres)

        # H3
    def Contour(self):
        img = cv2.imread('contur.jpg') # Membaca satu citra
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) # Mengubah menjadi grayscale
        T = 127 # inisialisasi nilai ambang
        max = 255 # inisialisasi nilai max keabuan
        ret, hasil = cv2.threshold(gray, T, max, cv2.THRESH_BINARY)
        contours, hierarchy = cv2.findContours(image=hasil, mode=cv2.RETR_LIST, method=cv2.CHAIN_APPROX_SIMPLE) # Mengekstrak contur
        # mode=cv2.RETR_LIST --> mode retrieval data untuk mengambil semua kontur dan merekontruksi hierakrki secara penuh dari kontur bersarang
        # method=cv2.CHAIN_APPROX_SIMPLE --> Aprroximation method,, menyimpan hanya dua poin dari satu garis saja
        image_copy = img.copy()
        i = 0 # Menentukan titik tengah kontur
        for contour in contours:
            if i == 0:
                i = 1
                continue
            approx = cv2.approxPolyDP(contour, 0.01 * cv2.arcLength(contour, True), True) # Mengambil nilai aprroximation poligon
            cv2.drawContours(image=image_copy, contours=contours, contourIdx=-1, color=(0, 255, 0), thickness=3)
            M = cv2.moments(contour)
            #Memeriksa jika poligon dengan 4 sisi, apakah persegi atau persegi panjang
            if M['m00'] != 0.0:
                x = int(M['m10'] / M['m00'])
                y = int(M['m01'] / M['m00'])
            if len(approx) == 3:
                cv2.putText(image_copy, 'Triangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 4:
                cv2.putText(image_copy, 'Rectangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 10:
                cv2.putText(image_copy, 'Star', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            elif len(approx) == 4:
                cv2.putText(image_copy, 'rectangle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
            else:
                cv2.putText(image_copy, 'circle', (x, y),cv2.FONT_HERSHEY_SIMPLEX, 0.6, (17, 32, 242), 2)
        cv2.imshow('Original', img)
        cv2.imshow('Contour', image_copy)

    def track(self):
        cam = cv2.VideoCapture(0) #membuka webcam
        while True:
            _, frame = cam.read()
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # mengconvert dari warna rgb ke hsv
            lower_color = np.array([0,0,0]) #nilai batas atas warna hsv
            upper_color = np.array([255,255,140]) #nilai batas bawah warna hsv
            mask = cv2.inRange(hsv, lower_color, upper_color) #masking dari nilai batas atas dan bawah
            result = cv2.bitwise_and(frame, frame, mask=mask) #hasil color tracking dgn masking dan frame awal
            cv2.imshow("Frame", frame) #menampilkan frame webcam awal
            cv2.imshow("Mask", mask) #menampilkan hasil masking
            cv2.imshow("Result", result) # menampilkan hasil akhir
            key = cv2.waitKey(1) #waitkey untuk menahan layar selama sekian milidetik
            if key == 27: #dengan menekan keyboard esc maka program akan langsung keluar
                break
        cam.release()
        cv2.destroyAllWindows()

    def pick(self):
            def nothing(x):
                pass
            # untuk mengambil nilai hsv menggunakan trackbar (hue,saturation,value )
            cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # inisialisasi webcam
            cv2.namedWindow("Trackbar") #membuat trackbar

            cv2.createTrackbar("L-H", "Trackbar", 0, 179, nothing)
            cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
            cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)
            cv2.createTrackbar("U-H", "Trackbar", 179, 179, nothing)
            cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)
            cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)

            while True:
                _, frame = cam.read() #membka webcam
                hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert webcam dari rgb kr hsv

                #memberikan nilai posisi pada tiap tiap trackbar
                L_H = cv2.getTrackbarPos("L-H", "Trackbar") #lower
                L_S = cv2.getTrackbarPos("L-S", "Trackbar") #lower
                L_V = cv2.getTrackbarPos("L-V", "Trackbar") #lower
                U_H = cv2.getTrackbarPos("U-H", "Trackbar") #upper
                U_S = cv2.getTrackbarPos("U-S", "Trackbar") #upper
                U_V = cv2.getTrackbarPos("U-V", "Trackbar") #upper

                #memberikan nilai batas atas dan bawah
                lower_color = np.array([L_H, L_S, L_V])
                upper_color = np.array([U_H, U_S, U_V])
                mask = cv2.inRange(hsv, lower_color, upper_color) #membuat masking dari nilai batas atas dan bawah
                result = cv2.bitwise_and(frame, frame, mask=mask) #hasil dari frame awal dan masking
                cv2.imshow("Frame", frame)
                cv2.imshow("Mask", mask)
                cv2.imshow("Hasil", result)

                key = cv2.waitKey(1) #waitkey untuk menahan layar selama sekian milidetik
                if key == 27: #dengan menekan keyboard esc maka program akan langsung keluar
                    break
            cam.release()
            cv2.destroyAllWindows()

    def objectdetection(self):
        cam = cv2.VideoCapture('cctv1.mp4')  # Membaca file video dan menginisialisasi camera
        car_cascade = cv2.CascadeClassifier('car.xml')  # Memuat file klasifikasi yang berisi deskripsi fitur mobil
        while True:  # Looping video
            ret, frame = cam.read()  # Membaca setiap frame dari video
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah gambar ke dalam grayscale
            cars = car_cascade.detectMultiScale(gray, 1.1, 3)  # Mencari mobil pada frame dengan detektor mobil
            for (x, y, w, h) in cars:  # Loop setiap mobil yang terdeteksi dengan detektor mobil
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),1)  # Mencari mobil pada frame dengan detektor mobil
            cv2.imshow('Video', frame)  # Menampilkan frame mobil yang terdeteksi

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Menekan tombol q untuk menghentikan loop
                break
        cam.release()
        cv2.destroyAllWindows()

    def HOG(self):
        image=data.astronaut()  # Memuat citra astronaut dari library data scikit-image sebagai citra input
        fd, hog_image =hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,multichannel=True)
        # Ekstraksi fitur HOG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True,sharey=True)
        # Membuat sebuah figur matplotlib dan dua axis, yang pertama menampilkan input dan yang kedua menampilkan citra hog
        ax1.axis('off')  # Menghilangkan garis sumbu pada axis pertama
        ax1.imshow(image, cmap=plt.cm.gray)  # Menmapilkan citra input pada axis pertama dengan colormap gray
        ax1.set_title('Input image')  # Menambahkan judul pada axis pertama

        # Rescale histogram for better display
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # Normalisasi nilai intensitas citra HOG ke dalam range 0-10
        ax2.axis('off')  # Menghilangkan garis sumbu pada axis kedua
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)  # Menampilkan citra HOG pada axis kedua dengan colormap gray
        ax2.set_title('Histogram of Oriented Gradients')  # Menambahkan judul pada axis kedua
        plt.show()

    def HOGJalan(self):
        hog = cv2.HOGDescriptor()  # Membuat objek detektor HOG
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Menentukan Support Vector Machine (SVM) yang digunakan oleh detektor HOG dengan menggunakan detektor default
        Photo = cv2.imread("pict1.jpg")  # Membaca citra jalan

        Photo = imutils.resize(Photo, width=min(400, Photo.shape[0]))
        # Melakukan resizing citra untuk mempercepat proses deteksi dengan memastikan lebar tidak >400 pixel

        (regions, _) = hog.detectMultiScale(Photo, winStride=(4, 4), padding=(4, 4),scale=1.05)
        # Melakukan deteksi pejalan kaki pada citra menggunakan detektor HOG
        for (x, y, w, h) in regions:  # Looping pada setiap region pejalan kaki yang terdeteksi pada citra
            cv2.rectangle(Photo, (x, y), (x + w, y + h), (0, 0, 255),2)  # Menggambar kotak pada setiap region pejalan kaki yang terdeteksi pada citra
        cv2.imshow("image", Photo)  # Menampilkan citra dengan kotak-kotak deteksi pejalan kaki
        cv2.waitKey()

    def FaceandEye(self):
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Inisialisasi klasifier dengan menggunakan file xml yang disediakan
        Image = cv2.imread('pict2.jpg')  # Membaca gambar
        gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)  # Mengubah gambar menjadi grayscale
        faces = face_classifier.detectMultiScale(gray, 1.3, 5)  # Mendeteksi wajah pada gambar
        if faces == ():  # Jika tidak ada wajah yang terdeteksi
            print("No faces found")
        for (x, y, w, h) in faces:
            cv2.rectangle(Image, (x, y), (x + w, y + h), (127, 0, 255),2)  # Menggambar kotak pada wajah yang terdeteksi
            cv2.imshow('Face Detection', Image)  # Menampilkan gambar yang telah diberi kotak pada wajah
            cv2.waitKey(0)

    def Pedestrian(self):
        body_classifier = cv2.CascadeClassifier('haarcascade_fullbody.xml')  # Inisialisasi klasifier dengan menggunakan file xml
        cap = cv2.VideoCapture('cctv2.mp4')  # Membaca video
        while cap.isOpened():  # Looping setiap frame
            ret, frame = cap.read()  # Mengambil setiapframe video
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5,interpolation=cv2.INTER_LINEAR)
            # Mengubah ukuran frame agar menjadi lebih kecil dengan skala 0.5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah setiap fame ke dalam gambar grayscale
            # Pass frame to our body classifier
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)  # Menerapkan algoritma deteksi tubuh manusia
            # Extract bounding boxes for any bodies identified
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255),2)  # Menambahkan kotak di sekitar objek pejalan kaki
                cv2.imshow('Pedestrians', frame)  # Menampilkan hasil video dengan deteksi tubuh manusia
            if cv2.waitKey(1) == 13:  # 1 (enter) 3 is the Enter Key
                break
        cap.release()
        cv2.destroyAllWindows()

    def CircleHough(self):
        img = cv2.imread('pict3.jpg', 0)  # Membaca gambar dan mengubahnya ke grayscale
        img = cv2.medianBlur(img, 5)  # Median Filtering pada gambar untuk menghilangkan noise
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Mengubah gambar grayscale menjadi berwarna
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,maxRadius=0)  # Melakukan transformasi Hough lingkaran untuk mendeteksi lingkaran pada gambar
        circles = np.uint16(np.around(circles))  # Mengkonversi nilai koordinat lingkaran menjadi bilangan bulat (integer)
        for i in circles[0, :]:  # Looping
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Menggambar lingkaran pada gambar
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # Menggambar titik pada pusat lingkaran
        cv2.imshow('detected circles', cimg)  # Menampilkan gambar berwarna
        cv2.waitKey(0)
        cv2.destroyAllWindows()



app = QtWidgets.QApplication(sys.argv)
window = ShowImage()
window.setWindowTitle('Pertemuan 9')
window.show()
sys.exit(app.exec())


# image = cv2.imread('ht.jpg', cv2.IMREAD_COLOR)
# cv2.imshow('Gambar 1', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
