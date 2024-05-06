import sys
import cv2
import matplotlib.pyplot as plt
from PyQt5 import QtCore, QtWidgets
from PyQt5.QtCore import *
from PyQt5.QtGui import *
from PyQt5.QtWidgets import *
from PyQt5.uic import loadUi
import numpy as np
import imutils
from skimage import data, exposure
from skimage.feature import hog
import dlib



class ShowImage(QMainWindow):
    def __init__(self):
        super(ShowImage, self).__init__()
        loadUi('showgui.ui', self)
        self.image = None
        self.loadButton.clicked.connect(self.loadClicked)
        self.grayButton.clicked.connect(self.grayClicked)
        self.actionCitra_Kecerahan.triggered.connect(self.brightness)
        self.actionCitra_Kontras.triggered.connect(self.contrast)
        self.actioncontrast_stretching.triggered.connect(self.contrastStretching)
        self.actionCitra_Negative.triggered.connect(self.negative)
        self.actionCitra_Biner.triggered.connect(self.biner)
        self.actionHistogram_Grayscale.triggered.connect(self.grayHistogram)
        self.actionHistogram_RGB.triggered.connect(self.HistogramRGB)
        self.actionHistogram_Equalization.triggered.connect(self.EqualHistogram)
        self.actionTranslasi.triggered.connect(self.Translasi)

        # rotasi
        self.action_45_Derajat.triggered.connect(self.rotasi_minus45)
        self.action90_Derajat.triggered.connect(self.rotasi90derajat)
        self.action45_Derajat.triggered.connect(self.rotasi45derajat)
        self.actionminus_90_Derajat.triggered.connect(self.rotasi_minus90)
        self.action180_Derajat.triggered.connect(self.rotasi180derajat)
        #transpose
        self.actionTranspose.triggered.connect(self.transpose)

        #zoom in
        self.action2x.triggered.connect(self.zoomIn2x)
        self.action3x.triggered.connect(self.zoomIn3x)
        self.action4x.triggered.connect(self.zoomIn4x)
        #zoom out
        self.action1_2.triggered.connect(self.zoomOut1_2)
        self.action1_4.triggered.connect(self.zoomOut1_4)
        self.action3_4.triggered.connect(self.zoomOut3_4)
        #crop
        self.actionCrop.triggered.connect(self.crop)
        #aritmatika
        self.actionTambah_dan_Kurang.triggered.connect(self.aritmatika)
        self.actionBoolean.triggered.connect(self.boolean)

        # operasi spasial
        self.actionFiltering_1.triggered.connect(self.Filtering1)
        self.actionFiltering_2.triggered.connect(self.Filtering2)
        self.action2x2.triggered.connect(self.Mean2x2)
        self.action3x3.triggered.connect(self.Mean3x3)
        self.actionGaussian.triggered.connect(self.Gaussian)
        self.actionSharpening_1.triggered.connect(self.Sharpening1)
        self.actionSharpening_2.triggered.connect(self.Sharpening2)
        self.actionSharpening_3.triggered.connect(self.Sharpening3)
        self.actionSharpening_4.triggered.connect(self.Sharpening4)
        self.actionSharpening_5.triggered.connect(self.Sharpening5)
        self.actionSharpening_6.triggered.connect(self.Sharpening6)
        self.actionLaplace.triggered.connect(self.Laplace)
        self.actionMedian_Filtering.triggered.connect(self.Median)
        self.actionMax_Filtering.triggered.connect(self.Max)
        self.actionMin_Filtering.triggered.connect(self.Min)

        # DFT Smoothing Image
        self.actionFourier.triggered.connect(self.fourier)
        self.actionEdge.triggered.connect(self.edge)
        self.actionCanny_Edge_2.triggered.connect(self.canny_edge)
        self.actionSobel.triggered.connect(self.Sobel)
        self.actionPrewitt.triggered.connect(self.Prewitt)
        self.actionRobert.triggered.connect(self.Robert)

        # Morfologi
        self.actionmorfologi_2.triggered.connect(self.morfologi)

        # Thresholding
        self.actionBinary.triggered.connect(self.thresoldingbinary)
        self.actionBinary_Invers.triggered.connect(self.thresholdinginvbiner)
        self.actionTrunc.triggered.connect(self.tresholdingtrunc)
        self.actionTo_Zero.triggered.connect(self.thresholdingtozero)
        self.actionTo_Zero_Invers.triggered.connect(self.thresholdinginversetozero)

        # Lokal Thresholding
        self.actionMean_Thresholding.triggered.connect(self.meanthresholding)
        self.actionGaussian_Thresholding.triggered.connect(self.gaussianthresholding)
        self.actionOtsu_Thresholding.triggered.connect(self.otsuthresholding)

        self.actionContour.triggered.connect(self.contour)

        #color processing
        self.actionTracking.triggered.connect(self.track)
        self.actionPicker.triggered.connect(self.pick)

        #cascade
        self.actionJalan.triggered.connect(self.HOGJalan)
        self.actionObjectDetection.triggered.connect(self.objectdetection)
        self.actionHOG.triggered.connect(self.HOG)
        self.actionFace_and_Eye.triggered.connect(self.FaceandEye)
        self.actionPedestrian.triggered.connect(self.Pedestrian)
        self.actionCircle_Hough.triggered.connect(self.CircleHough)

        # face detection
        self.actionFacial_Landmark.triggered.connect(self.FacialLandmark)

    def loadClicked(self):
        self.image = cv2.imread('bangundatar.png')
        if self.image is not None:
            self.displayImage(1)

    def displayImage(self, windows=1):
        if self.image is None:
            return

        qformat = QImage.Format_Indexed8
        if len(self.image.shape) == 3:  # row[0],col[1],channel[2]
            if (self.image.shape[2]) == 4:
                qformat = QImage.Format_RGBA8888
            else:
                qformat = QImage.Format_RGB888

        img = QImage(self.image, self.image.shape[1], self.image.shape[0], self.image.strides[0], qformat)
        img = img.rgbSwapped()

        if windows == 1:
            self.imgLabel.setPixmap(QPixmap.fromImage(img))
            self.imgLabel.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel.setScaledContents(True)

        if windows == 2:
            gray_pixmap = QPixmap.fromImage(img).toImage().convertToFormat(QImage.Format_Grayscale8)
            self.imgLabel2.setPixmap(QPixmap.fromImage(gray_pixmap))
            self.imgLabel2.setAlignment(QtCore.Qt.AlignHCenter | QtCore.Qt.AlignVCenter)
            self.imgLabel2.setScaledContents(True)

    def grayClicked(self):
        # if self.image is not None:
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
        # error handling
        try :
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except :
            pass

        H, W = self.image.shape[:2]
        brightness = 80
        for i in range(H):
            for j in range(W):
                a = self.image.item(i,j)
                b = np.clip(a + brightness, 0, 255)

                self.image.itemset((i, j), b)
        self.displayImage(1)

    def contrast(self):
        # error handling
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

    def contrastStretching(self):
        # error handling
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]
        minV = np.min(self.image)
        maxV = np.max(self.image)

        for i in range(H):
            for j in range(W):
                a = self.image.item(i, j)
                b = float(a - minV) / (maxV - minV) * 255

                self.image.itemset((i, j), b)

        self.displayImage(1)

    def negative(self):
        # error handling
        try:
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        H, W = self.image.shape[:2]

        for i in range(H):
            for j in range(W):
                pixel_value = self.image.item(i, j)
                negative_value = 255 - pixel_value

                self.image.itemset((i, j), negative_value)

        self.displayImage(1)

    def biner(self):
        # Error handling
        try:
            # Convert the image to grayscale if it's not already
            self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        except:
            pass

        # Get the dimensions of the image
        H, W = self.image.shape[:2]

        # Iterate through each pixel of the image
        for i in range(H):
            for j in range(W):
                # Get the pixel value at (i, j)
                pixel_value = self.image.item(i, j)
                # If the pixel value is greater than or equal to the threshold, set it to 255 (white), otherwise set it to 0 (black)
                if pixel_value == 100:
                    self.image.itemset((i, j), 0)
                elif pixel_value < 180:
                    self.image.itemset((i, j), 1)
                elif pixel_value > 180:
                    self.image.itemset((i, j), 255)
                else:
                    self.image.itemset((i, j), 0)

        # Display the binary image
        self.displayImage(2)

    def grayHistogram(self):
        # if self.image is not None:
            H, W = self.image.shape[:2]
            gray = np.zeros((H, W), np.uint8)
            for i in range(H):
                for j in range(W):
                    gray[i, j] = np.clip(0.299 * self.image[i, j, 0] +
                                         0.587 * self.image[i, j, 1] +
                                         0.114 * self.image[i, j, 2], 0, 255)
            self.image = gray
            self.displayImage(2)
            plt.hist(self.image.ravel(), 255, [0, 255]) #membuat histogram nya dari pixel
            plt.show()

    def HistogramRGB(self):
        color = ('b', 'g', 'r') #warna blue green dan red
        for i, col in enumerate(color): #perulangan untuk setiap warna
            # fungsi untuk menghitung histogram dari setiap kumpulan array
            histo=cv2.calcHist([self.image],[i], None, [256], [0, 256])
            plt.plot(histo,color=col) #plotting histogram
            plt.xlim([0, 256]) #mengatur batas sumbu x
        plt.show()

    def EqualHistogram(self):
        # mengubah image array jadi 1 dimensu
        hist, bins = np.histogram(self.image.flatten(), 256, [0, 256])
        cdf = hist.cumsum() # menghitung jumlah kumualtif array
        cdf_normalized = cdf * hist.max() / cdf.max() # normalisasi
        cdf_m = np.ma.masked_equal(cdf, 0) # nge masking
        cdf_m = (cdf_m - cdf_m.min()) * 255 / (cdf_m.max() - cdf_m.min()) #perhitungannya
        cdf = np.ma.filled(cdf_m, 0).astype("uint8") #mengisi nilai array dengan skalar
        self.image = cdf[self.image] #mengganti nilai array menjadi nilai kumulatif
        self.displayImage(2)

        # membuat plotting
        plt.plot(cdf_normalized, color='b')
        plt.hist(self.image.flatten(), 256, [0, 256], color='r')
        plt.xlim([0, 256])
        plt.legend(('cdf', 'histogram'), loc='upper left')
        plt.show()

    def Translasi(self):
        h, w = self.image.shape[:2]
        quarter_h, quarter_w = h / 4, w / 4
        T = np.float32([[1, 0, quarter_w], [0, 1, quarter_h]]) #matriks
        img = cv2.warpAffine(self.image, T, (w, h))
        self.image = img
        self.displayImage(2)

    def transpose(self):
        self.image = cv2.transpose(self.image)
        self.displayImage(2)

    def rotasi(self, degree):
        h, w = self.image.shape[:2] #mengambil lebar dan tinggi dari gambar
        rotationMatrix = cv2.getRotationMatrix2D((w / 2, h / 2), degree, 1) #membuat rotasi matriks
        #mengambil nilai cosinus dari sudut rotasi dari matriks rotasi
        cos = np.abs(rotationMatrix[0, 0])
        sin = np.abs(rotationMatrix[0, 1])
        # menghitunf dimensi baru dari gambar yang telah dirotasi
        nW = int((h * sin) + (w * cos))
        nH = int((h * cos) + (w * sin))
        # memnyesuaikan matriks rotasi
        rotationMatrix[0, 2] += (nW / 2) - w / 2
        rotationMatrix[1, 2] += (nH / 2) - h / 2
        # memperbarui image dengan gambar yang sudha diputar
        rot_image = cv2.warpAffine(self.image, rotationMatrix, (nH, nW))
        self.image = rot_image
        self.displayImage(2)

    def rotasi_minus45(self):
        self.rotasi(-45)
    def rotasi45derajat(self):
        self.rotasi(45)
    def rotasi_minus90(self):
        self.rotasi(-90)
    def rotasi90derajat(self):
        self.rotasi(90)
    def rotasi180derajat(self):
        self.rotasi(180)


    def zoomIn(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()

    def zoomIn2x(self):
        self.zoomIn(2)
    def zoomIn3x(self):
        self.zoomIn(3)
    def zoomIn4x(self):
        self.zoomIn(4)

    def zoomOut(self, skala):
        resize_image = cv2.resize(self.image, None, fx=skala, fy=skala, interpolation=cv2.INTER_CUBIC)
        cv2.imshow('Original', self.image)
        cv2.imshow('Zoom In', resize_image)
        cv2.waitKey()
    def zoomOut1_2(self):
        self.zoomOut(0.5)
    def zoomOut1_4(self):
        self.zoomOut(0.25)
    def zoomOut3_4(self):
        self.zoomOut(.75)

    def crop(self):
        # Tentukan koordinat atau posisi x (row) dan y (coloum) awal yang diawali dari ujung kiri atas
        x = 100
        y = 100
        # Tentukan koordinat atau posisi x (row) dan y (coloum) akhir berakhir di ujung kanan bawah
        width = 300
        height = 100
        cropped_image = self.image[y:y + height, x:x + width]
        cv2.imshow('Original', self.image)
        cv2.imshow('Cropped', cropped_image)
        cv2.waitKey()

    def aritmatika(self):
        image1 = cv2.imread('ip15.jpg', 0)
        image2 = cv2.imread('ip7.jpg', 0)
        image_tambah = image1 + image2
        image_kurang = image1 - image2
        image_kali = image1 * image2
        image_bagi = image1 / image2
        cv2.imshow('Image 1 Original', image1)
        cv2.imshow('Image 2 Original', image2)
        cv2.imshow('Image 2 Tambah', image_tambah)
        cv2.imshow('Image 2 Kurang', image_kurang)
        cv2.imshow('Image 2 Kali', image_kali)
        cv2.imshow('Image 2 Bagi', image_bagi)
        cv2.waitKey()

    def boolean(self):
        image1 = cv2.imread('ip15.jpg', 1)
        image2 = cv2.imread('ip7.jpg', 1)
        image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)
        image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        operasiAnd = cv2.bitwise_and(image1, image2)
        operasiOr = cv2.bitwise_or(image1, image2)
        operasiXor = cv2.bitwise_xor(image1, image2)
        cv2.imshow("Image 1 Original", image1)
        cv2.imshow("Image 2 Original", image2)
        cv2.imshow("Operasi AND", operasiAnd)
        cv2.imshow("Operasi OR", operasiOr)
        cv2.imshow("Operasi XOR", operasiXor)
        cv2.waitKey()

    def Konvolusi(self, X, F):
        # membaca ukuran tinggi dan lebar citra
        X_height = X.shape[0]
        X_width = X.shape[1]
        # membaca ukuran tinggi dan lebar kernel
        F_height = F.shape[0]
        F_width = F.shape[1]
        # mencari titik tengah kernel
        H = (F_height) // 2
        W = (F_width) // 2

        out = np.zeros((X_height, X_width))

        # mengatur pergerakan kernel dengan for
        for i in np.arange(H + 1, X_height - H):
            for j in np.arange(W + 1, X_width - W):
                sum = 0
                for k in np.arange(-H, H + 1):
                    # mengambil piksel dari zona
                    for l in np.arange(-W, W + 1):
                        a = X[i + k, j + l]  # mengambil piksel yang ada pada dalam citra
                        w = F[H + k, W + l]  # mengambil bobot yang ada pada kernel
                        sum += (w * a)  # menampung nilai total perkalian w * a
                out[i, j] = sum  # menampung hasil
        return out

    def Filtering1(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)  # mengubah citra menjadi greyscale
        kernel = np.array(
            # array kernel
            [
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ])

        # parameter dari fungsi konvolusi 2d
        img_out = self.Konvolusi(img, kernel)
        print('Nilai Pixel Filtering A \n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')  # bicubic memperhalus tepi citra
        plt.xticks([]), plt.yticks([])
        plt.show()  # Menampilkan gambar

    def Filtering2(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        kernel = np.array(
            # array kernel
            [
                [6, 0, -6],
                [6, 1, -6],
                [6, 0, -6]
            ]
        )

        img_out = self.Konvolusi(img, kernel)
        # memunculkan nilai pixel pada terminal
        print('Nilai Pixel Filtering B\n', img_out)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        # mengatur lokasi tick dan label sumbu x dan y
        plt.xticks([]), plt.yticks([])
        plt.show()  # Menampilkan gambar

    # filtering 1 menghasilkan gmabar yang buram karena kernel rata2 mencampur nilai piksel di sekitar setiap piksel
    # filtering 2 menghasilkan gambar lebih jelas
    def Mean2x2(self):
        # mengganti intensitas pixel dengan rata-rata pixel
        mean = (1.0 / 4) * np.array(
            [  # array kernel 2x2
                [0, 0, 0],
                [0, 1, 1],
                [0, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('Nilai Pixel Mean Filter 2x2 \n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # mengatur lokasi tick dan label sumbu x dan y
        plt.show()

    def Mean3x3(self):
        # kernel 1/9
        mean = (1.0 / 9) * np.array(
            [  # array kernel 3x3
                [1, 1, 1],
                [1, 1, 1],
                [1, 1, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = self.Konvolusi(img, mean)
        print('Nilai Pixel Mean Filter 3x3\n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # mengatur lokasi tick, label sumbu x dan y
        plt.show() # menampilkan gambar

    # mean 2x2 efek blur lebih ringan tidak terlalu tajam
    # mean 3x3 efek blur lebih kuat dan detail si gambar lebih halus

    def Gaussian(self):
        gausian = (1.0 / 345) * np.array(
            # Kernel gaussian
            [
                [1, 5, 7, 5, 1],
                [5, 20, 33, 20, 5],
                [7, 33, 55, 33, 7],
                [5, 20, 33, 20, 5],
                [1, 5, 7, 5, 1]
            ]
        )
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        img_out = self.Konvolusi(img, gausian)
        print('Nilai Pixel Gaussian \n', img_out)  # memunculkan nilai pixel p
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])  # mengatur lokasi tick, label sumbu x dan y
        plt.show()
        # lebih ke penghilang noise, pengaburan citra dan penghalusan gambar

    def Sharpening1(self):
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
        print('Nilai Pixel Kernel i \n', img_out)  # memunculkan nilai pixel pada terminal
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
        print('Nilai Pixel Kernel ii \n', img_out)  # memunculkan nilai pixel pada terminal
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
        print('Nilai Pixel Kernel iii \n', img_out)  # memunculkan nilai pixel pada terminal
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
        print('Nilai Pixel Kernel iv \n', img_out)  # memunculkan nilai pixel pada terminal
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
        print('Nilai Pixel Kernel v \n', img_out)  # memunculkan nilai pixel pada terminal
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
        print('Nilai Pixel Kernel vi \n', img_out)  # memunculkan nilai pixel pada terminal
        cv2.imshow('Original', img)
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.show()
        cv2.waitKey()

    # sharpening 1-5 sama sama untuk menajamkakn citra nya tergantung dari nilai kernel nya
    # sharpening 6 menajamkan bagian tepi
    def Laplace(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        sharpe = (1.0 / 16) * np.array(  # nilai kernel
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

    # menajamkan semua tepi dengan signifikan dibandingkan dengan fungsi sharpening
    # menandai bagian yang menjadi detail citra dan memperbaiki serta mengubah citra

    def Median(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)  # mengubah ke grayscale
        img_out = img.copy()
        H, W = img.shape[:2]  # tinggi dan lebar citra

        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                neighbors = []  # menampung nilai pixel
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        # menampung hasil
                        a = img.item(i + k, j + l)
                        # menambahkan a ke neighbors
                        neighbors.append(a)
                neighbors.sort()
                median = neighbors[24]
                b = median
                img_out.itemset((i, j), b)
        print('Nilai Pixel Median Filter\n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # mencari nilai tengah pixel setelah diurutkan
    # menghilangkan noise acak dan menghaluskan detail detail penting

    def Max(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        # cek nilai setiap pixel
        for i in np.arange(3, H - 3):
            for j in np.arange(3, W - 3):
                max = 0
                # mencari nilai maximum
                for k in np.arange(-3, 4):
                    for l in np.arange(-3, 4):
                        # menampung nilai baca pixel
                        a = img.item(i + k, j + l)
                        if a > max:
                            max = a
                        b = max
                img_out.itemset((i, j), b)
        print('Nilai Pixel Maximun Filter \n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    def Min(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_RGB2GRAY)
        img_out = img.copy()
        H, W = img.shape[:2]

        for i in np.arange(3, H - 3):  # mengecek nilai setiap pixel
            for j in np.arange(3, W - 3):
                min = 0
                for k in np.arange(-3, 4):  # mencari nilai maximun
                    for l in np.arange(-3, 4):
                        a = img.item(i + k, j + l)  # untuk menampung nilai baca pixel
                        if a < min:
                            min = a
                        b = min
                img_out.itemset((i, j), b)
        print('Nilai Pixel Minimun Filter \n', img_out)  # memunculkan nilai pixel
        plt.imshow(img_out, cmap='gray', interpolation='bicubic')
        plt.xticks([]), plt.yticks([])
        plt.show()

    # max mengganti nilai pixel dengan maksimum yang dipengaruhi piksel tetangga
    # min mengganti nilai pixel dengan minimum efek yang dipengaruhi piksel tetangga sih di foto ini jadi item

    def fourier(self):
        x=np.arange(256)
        y=np.sin(2*np.pi*x/3)

        y+=max(y)

        img= np.array([[y[j]*127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        # membaca image citra bernoise
        plt.imshow(img)
        img= cv2.imread('nosie.jpg', 0) # 0 untuk menjadi grayscale

        # mengkonversikan ke float 32 dan menghasilkan array 2 dimensi
        dft= cv2.dft(np.float32(img), flags=cv2.DFT_COMPLEX_OUTPUT)
        # rearange nilai koordinat pusat nya diubah ke center
        dft_shift= np.fft.fftshift(dft)
        # perhitungan spectrum
        magnitude_spectrum = 20*np.log((cv2.magnitude(dft_shift[:,:,0], dft_shift[:,:,1]))) # 0 chanel rill dan 1 chanel imaginer

        # proses untuk memposisikan center
        rows, cols = img.shape
        crow, ccol = int(rows/2), int(cols/2)
        # masking
        mask = np.zeros((rows, cols, 2), np.uint8)
        # radius mask
        r = 50
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]

        # membuat si mask nya supaya lebih jelas bagian mask nya
        mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r*r)
        mask[mask_area] = 1 # membuat nilai mask jadi 1 sisa nya 0

        # mengalikan dft dengan si mask
        fshift = dft_shift * mask
        # melakukan proses mencari nilai spectrum
        fshift_mask_mag = 20*np.log(cv2.magnitude(fshift[:,:,0], fshift[:,:,1]))
        # proses mengembalikan proses origin dari tengah
        f_i_shift = np.fft.ifftshift(fshift)

        # mengembalikan citra ke bentuk spasial
        img_back = cv2.idft(f_i_shift)
        # proses magnitude, di kembalikan nilai real dan imaginer menjadi spasial
        img_back = cv2.magnitude(img_back[:,:,0], img_back[:,:,1])

        # plotting
        fig= plt.figure(figsize=(12,12))
        ax1 = fig.add_subplot(2,2,1)
        ax1.imshow(img, cmap='gray')
        ax1.title.set_text('Input Image')

        ax2 = fig.add_subplot(2, 2, 2)
        ax2.imshow(magnitude_spectrum, cmap='gray')
        ax2.title.set_text('FFT of Image')

        ax3 = fig.add_subplot(2, 2, 3)
        ax3.imshow(fshift_mask_mag, cmap='gray')
        ax3.title.set_text('FFT + Mask')

        ax1 = fig.add_subplot(2, 2, 4)
        ax1.imshow(img_back, cmap='gray')
        ax1.title.set_text('Inverse Fourier')

        plt.show()

    def edge(self):
        x = np.arange(256)
        y = np.sin(2 * np.pi * x / 3)

        y += max(y)

        img = np.array([[y[j] * 127 for j in range(256)] for i in range(256)], dtype=np.uint8)

        # membaca image citra bernoise
        plt.imshow(img)

        img= cv2.imread('nosie.jpg', 0)

        dft= cv2.dft(np.float32(img), flags= cv2.DFT_COMPLEX_OUTPUT)
        dft_shift = np.fft.fftshift(dft)

        # perhitungan spectrum
        magnitude_spectrum = 20 * np.log(
            (cv2.magnitude(dft_shift[:, :, 0], dft_shift[:, :, 1])))  # 0 chanel rill dan 1 chanel imaginer

        # proses untuk memposisikan center
        rows, cols = img.shape
        crow, ccol = int(rows / 2), int(cols / 2)
        # masking
        mask = np.ones((rows, cols, 2), np.uint8)
        # radius mask
        r = 80
        center = [crow, ccol]
        x, y = np.ogrid[:rows, :cols]

        # membuat si mask nya supaya lebih jelas bagian mask nya
        mask_area = ((x - center[0]) ** 2 + (y - center[1]) ** 2 <= r * r)
        mask[mask_area] = 0  # membuat nilai mask jadi 1 sisa nya 0

        # mengalikan dft dengan si mask
        fshift = dft_shift * mask
        # melakukan proses mencari nilai spectrum
        fshift_mask_mag = 20 * np.log(cv2.magnitude(fshift[:, :, 0], fshift[:, :, 1]))
        # proses mengembalikan proses origin dari tengah
        f_i_shift = np.fft.ifftshift(fshift)

        # mengembalikan citra ke bentuk spasial
        img_back = cv2.idft(f_i_shift)
        # proses magnitude, di kembalikan nilai real dan imaginer menjadi spasial
        img_back = cv2.magnitude(img_back[:, :, 0], img_back[:, :, 1])

        # plotting
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

        ax1 = fig.add_subplot(2, 2, 4)
        ax1.imshow(img_back, cmap='gray')
        ax1.title.set_text('Inverse Fourier')

        plt.show()

    def Sobel(self):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Inisialisasi kernel Sobel sumbu X
        kernel_sobel_x = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]])

        # Inisialisasi kernel Sobel sumbu Y
        kernel_sobel_y = np.array([[1, 2, 1], [0, 0, 0], [-1, -2, -1]])

        # Konvolusi citra terhadap kernel Sobel sumbu X dan Y
        Gx = cv2.filter2D(gray, cv2.CV_64F, kernel_sobel_x)
        Gy = cv2.filter2D(gray, cv2.CV_64F, kernel_sobel_y)

        # Hitung Gradien
        gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        # Normalisasi panjang gradien dalam range 0-255
        gradient_normalized = ((gradient_magnitude / gradient_magnitude.max()) * 255).astype(np.uint8)

        # Menampilkan citra keluaran dalam color map 'gray' dan dengan interpolasi 'bicubic'
        plt.imshow(gradient_normalized, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()

    def turunan_pertama(self, x1, x2, x3, x4, x5, x6, x7, x8, x9, y1, y2, y3, y4, y5, y6, y7, y8, y9):
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Inisialisasi kernel sumbu X
        kernel_x = np.array([[x1, x2, x3], [x4, x5, x6], [x7, x8, x9]])

        # Inisialisasi kernel sumbu Y
        kernel_y = np.array([[y1, y2, y3], [y4, y5, y6], [y7, y8, y9]])

        # Konvolusi citra terhadap kernel sumbu X dan Y
        Gx = cv2.filter2D(gray, cv2.CV_64F, kernel_x)
        Gy = cv2.filter2D(gray, cv2.CV_64F, kernel_y)

        # Hitung Gradien
        gradient_magnitude = np.sqrt(Gx ** 2 + Gy ** 2)

        # Normalisasi panjang gradien dalam range 0-255
        gradient_normalized = ((gradient_magnitude / gradient_magnitude.max()) * 255).astype(np.uint8)

        # Menampilkan citra keluaran dalam color map 'gray' dan dengan interpolasi 'bicubic'
        plt.imshow(gradient_normalized, cmap='gray', interpolation='bicubic')
        plt.axis('off')
        plt.show()

    def Prewitt(self):
        self.turunan_pertama(-1, 0, 1, -1, 0, 1, -1, 0, 1, -1, -1, -1, 0, 0, 0, 1, 1, 1)

    def Robert(self):
        self.turunan_pertama(0, 0, -1, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 1, 0, 0, 0, 0)

    def canny_edge(self):
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # array numpy 5x5
        conv = (1 / 345) * np.array(
            [[1, 5, 7, 5, 1],
             [5, 20, 33, 20, 5],
             [7, 33, 55, 33, 7],
             [5, 20, 33, 20, 5],
             [1, 5, 7, 5, 1]])

        # menginisialisasi fungsi konvolusi
        out_img = self.Konvolusi(img, conv)
        # mengubah tipe data
        out_img = out_img.astype("uint8")
        # menampilkan gambarnya
        cv2.imshow("Noise reduction", out_img)

        # finding gradient
        # dua array numpy berukuran 3x3
        Sx = np.array([[-1, 0, 1],
                       [-2, 0, 2],
                       [-1, 0, 1]])
        Sy = np.array([[-1, -2, -1],
                       [0, 0, 0],
                       [1, 2, 1]])
        # meneraptkan filter sobel vertikal dan horizontal
        img_x = self.Konvolusi(out_img, Sx)
        img_y = self.Konvolusi(out_img, Sy)
        # memnghitung besarnya gradien setiap pixel nya
        img_out = np.sqrt(img_x * img_x + img_y * img_y)
        # menormalisasi gambar gradien
        img_out = (img_out / np.max(img_out)) * 255
        # memunculkan gambar hasil finding gradien nya
        cv2.imshow("finding Gradien", img_out)
        # menghitung arah gradien
        theta = np.arctan2(img_y, img_x)

        # non-maximum suppression
        # menghitung arah gradien dalam derajat
        angle = theta * 180. / np.pi
        # untuk menangani nilai sudut gradien negatif
        angle[angle < 0] += 180
        # untuk mendapatkan tinggi dan lebar dari gambar
        H, W = img.shape[:2]
        # membuat array numpy Z
        Z = np.zeros((H, W), dtype=np.int32)
        # looping
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                try:
                    q = 255
                    r = 255

                    # angle 0
                    if (0 <= angle[i, j] < 22.5) or (157.5 <= angle[i, j] <= 180):
                        q = img_out[i, j + 1]
                        r = img_out[i, j - 1]
                    # angle 45
                    elif (22.5 <= angle[i, j] < 67.5):
                        q = img_out[i + 1, j - 1]
                        r = img_out[i - 1, j + 1]
                    # angle 90
                    elif (67.5 <= angle[i, j] < 112.5):
                        q = img_out[i + 1, j]
                        r = img_out[i - 1, j]
                    # angle 135
                    elif (112.5 <= angle[i, j] < 157.5):
                        q = img_out[i - 1, j - 1]
                        r = img_out[i + 1, j + 1]
                    if (img_out[i, j] >= q) and (img_out[i, j] >= r):
                        Z[i, j] = img_out[i, j]
                    else:
                        Z[i, j] = 0
                except IndexError as e:
                    pass
        img_N = Z.astype("uint8")
        cv2.imshow("Non Maximum Suppression", img_N)
        # hysteresis thresholding part 1
        weak = 15
        strong = 90
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
        cv2.imshow("hysteresis part 1", img_H1)
        # hysteresis thresholding part 2
        strong = 255
        for i in range(1, H - 1):
            for j in range(1, W - 1):
                if (img_H1[i, j] == weak):
                    try:
                        if ((img_H1[i + 1, j - 1] == strong) or (img_H1[i + 1, j] == strong) or (
                                img_H1[i + 1, j + 1] == strong) or (img_H1[i, j - 1] == strong) or (
                                img_H1[i, j + 1] == strong) or (img_H1[i - 1, j - 1] == strong) or (
                                img_H1[i - 1, j] == strong) or (img_H1[i - 1, j + 1] == strong)):
                            img_H1[i, j] = strong
                        else:
                            img_H1[i, j] = 0
                    except IndexError as e:
                        pass
        img_H2 = img_H1.astype("uint8")
        cv2.imshow("hysteresis part 2", img_H2)

    def morfologi(self):
        # membaca gambar
        img = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)

        # Konversi ke citra biner
        _, binary_img = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

        # Inisialisasi Strel
        strel = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))

        # Erosi
        erosion_img = cv2.erode(binary_img, strel)

        # Dilasi (memperluas dan memperbesar objek)
        dilation_img = cv2.dilate(binary_img, strel)

        # Opening
        opening_img = cv2.morphologyEx(binary_img, cv2.MORPH_OPEN, strel)

        # Closing
        closing_img = cv2.morphologyEx(binary_img, cv2.MORPH_CLOSE, strel)

        # Skeletonizing
        # numpy zeros untuk binary image, menyimpna hasil skeletonizing
        skel = np.zeros(binary_img.shape, np.uint8)
        # membuat kernel untuk operasi morfologi
        element = cv2.getStructuringElement(cv2.MORPH_CROSS, (5, 5))

        #looping
        while True:
            eroded = cv2.morphologyEx(binary_img, cv2.MORPH_ERODE, element) # erosi
            temp = cv2.morphologyEx(eroded, cv2.MORPH_DILATE, element) # dilasi
            temp = cv2.subtract(binary_img, temp) # mendapatkan tepi objek
            skel = cv2.bitwise_or(skel, temp) # menggabungkan hasil operasi seelumnya
            binary_img[:] = eroded[:] #

            if cv2.countNonZero(binary_img) == 0:
                break

        # Tampilkan hasil operasi morfologi
        cv2.imshow('Erosion', erosion_img)
        cv2.imshow('Dilation', dilation_img)
        cv2.imshow('Opening', opening_img)
        cv2.imshow('Closing', closing_img)
        cv2.imshow('Skeleton', skel)

        # Tunggu tombol keyboard dan tutup semua jendela saat tombol ditekan
        cv2.waitKey(0)
        cv2.destroyAllWindows()
        return erosion_img, dilation_img, opening_img, closing_img

    def thresoldingbinary(self):
        # Load gambar
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Thresholding Binary
        _, binary_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY)
        # menampilkan gambar
        cv2.imshow('Binary Threshold', binary_threshold)
        print(binary_threshold)

    def thresholdinginvbiner(self):
        # Load gambar
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Thresholding Inversi Biner
        _, inverse_binary_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        # menampilkan citra
        cv2.imshow('Inverse Binary Threshold', inverse_binary_threshold)
        print(inverse_binary_threshold)

    def tresholdingtrunc(self):
        # Load gambar
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Thresholding Trunc
        _, trunc_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_TRUNC)
        # menampilkan citra
        cv2.imshow('Trunc Threshold', trunc_threshold)
        print(trunc_threshold)

    def thresholdingtozero(self):
        # Load gambar
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Thresholding to Zero
        _, to_zero_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO)
        # menampilkan citra
        cv2.imshow('To Zero Threshold', to_zero_threshold)
        print(to_zero_threshold)

    def thresholdinginversetozero(self):
        # Load gambar
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Thresholding Inversi to Zero
        _, inverse_to_zero_threshold = cv2.threshold(image, 127, 255, cv2.THRESH_TOZERO_INV)
        # Menampilkan hasil
        cv2.imshow('Inverse To Zero Threshold', inverse_to_zero_threshold)
        print(inverse_to_zero_threshold)

    def meanthresholding(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        print("Pixel Awal: ", image)
        mean_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Mean Thresholding", mean_threshold)
        print("Pixel Mean Thresolding", mean_threshold)

    def gaussianthresholding(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        gauss_threshold = cv2.adaptiveThreshold(image, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 3, 2)
        cv2.imshow("Gaussian Thresholding", gauss_threshold)
        print("Pixel Gaussian Thresholding", gauss_threshold)

    def otsuthresholding(self):
        image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        T = 130
        ret, otsuthreshold = cv2.threshold(image, T, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        cv2.imshow("Otsu Thresholding", otsuthreshold)
        print("Pixel Otsu Thresholding", otsuthreshold)

    def contour(self):
        # Konversi citra RGB ke grayscale
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        # Threshold citra dengan nilai T=127
        _, thresh = cv2.threshold(gray, 127, 255, cv2.THRESH_BINARY)

        # Ekstrak kontur
        contours, _ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

        # Mendapatkan poligon pendekatan dan titik tengah kontur untuk setiap kontur
        approximate_polygons = []
        contour_centers = []
        for contour in contours:
            epsilon = 0.01 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            approximate_polygons.append(approx)
            M = cv2.moments(contour)
            cX = int(M["m10"] / M["m00"])
            cY = int(M["m01"] / M["m00"])
            contour_centers.append((cX, cY))

        # Gambar kontur pada citra
        for contour in contours:
            cv2.drawContours(self.image, [contour], -1, (0, 0, 0), 2)

        # Memeriksa jenis poligon
        for i, polygon in enumerate(approximate_polygons):
            num_sides = len(polygon)
            center = contour_centers[i]
            shape = "Undefined"
            if num_sides == 3:
                shape = "Triangle"
            elif num_sides == 4:
                # Memeriksa jika poligon dengan 4 sisi, apakah persegi atau persegi panjang
                x, y, w, h = cv2.boundingRect(polygon)
                aspect_ratio = float(w) / h
                shape = "Square" if 0.95 <= aspect_ratio <= 1.05 else "Rectangle"
            elif num_sides == 5:
                shape = "Pentagon"
            elif num_sides == 6:
                shape = "Hexagon"
            else:
                shape = "Circle"
            # Menampilkan hasil deteksi pada citra
            cv2.putText(self.image, shape, (center[0] - 20, center[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0),
                        2)
            cv2.circle(self.image, center, 3, (0, 0, 0), -1)
        
        # Menampilkan citra dengan hasil deteksi
        cv2.imshow("Detected Shapes", self.image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def track(self):
        cam = cv2.VideoCapture(0) #membuka webcam
        while True:
            _, frame = cam.read() #membaca cam
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) # convert dari warna rgb ke hsv
            lower_color = np.array([0,0,0]) # batas atas warna hsv
            upper_color = np.array([255,255,140]) # batas bawah warna hsv
            mask = cv2.inRange(hsv, lower_color, upper_color) #masking  nilai batas atas dan bawah
            result = cv2.bitwise_and(frame, frame, mask=mask) #hasil color tracking sama masking dan frame awal
            cv2.imshow("Frame", frame) #menampilkan frame webcam 
            cv2.imshow("Mask", mask) #menampilkan hasil masking
            cv2.imshow("Result", result) # menampilkan hasil 
            key = cv2.waitKey(1) 
            if key == 27: #esc for exit
                break
        cam.release()
        cv2.destroyAllWindows()

    def pick(self):
        def nothing(x):
            pass
        # mengambil nilai hsv menggunakan trackbar 
        cam = cv2.VideoCapture(0, cv2.CAP_DSHOW) # inisialisasi webcam
        cv2.namedWindow("Trackbar") #membuat trackbar
        # u = upper, l = lower
        cv2.createTrackbar("L-H", "Trackbar", 0, 179, nothing) 
        cv2.createTrackbar("L-S", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("L-V", "Trackbar", 0, 255, nothing)
        cv2.createTrackbar("U-H", "Trackbar", 179, 179, nothing)
        cv2.createTrackbar("U-S", "Trackbar", 255, 255, nothing)
        cv2.createTrackbar("U-V", "Trackbar", 255, 255, nothing)

        while True:
            _, frame = cam.read() # membuka webcam
            hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV) #convert webcam dari rgb ke hsv

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
        cam = cv2.VideoCapture('video1.mp4')  # Membaca file video 
        car_cascade = cv2.CascadeClassifier('car.xml')  # file klasifikasi yang berisi deskripsi mobil
        while True: 
            ret, frame = cam.read()  # Membaca setiap frame video
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah gambar ke dalam grayscale
            cars = car_cascade.detectMultiScale(gray, 1.2, 6)  # Mencari mobil pada frame 
            for (x, y, w, h) in cars:  # Loop setiap mobil yang terdeteksi dengan detektor mobil
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0),2)  # Mencari mobil pada frame dengan detektor mobil
            cv2.imshow('Video', frame)  # Menampilkan frame mobil yang terdeteksi

            if cv2.waitKey(10) & 0xFF == ord('q'):  # Menekan tombol q untuk menghentikan loop
                break
        cam.release()
        cv2.destroyAllWindows()

    def HOG(self):
        # mengambil image dari astronaut
        image = data.astronaut()

        # Ekstraksi fitur Histogram of Oriented Gradients (HOG) dari gambar
        fd, hog_image = hog(image, orientations=8, pixels_per_cell=(16, 16), cells_per_block=(1, 1), visualize=True,
                            channel_axis=-1)

        # Membuat plot untuk menampilkan gambar asli dan HOG
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 6), sharex=True, sharey=True)

        # Plot gambar asli
        ax1.axis('off')
        ax1.imshow(image, cmap=plt.cm.gray)
        ax1.set_title('Input image')

        # Rescale histogram untuk tampilan yang lebih baik
        hog_image_rescaled = exposure.rescale_intensity(hog_image, in_range=(0, 10))
        # Plot HOG
        ax2.axis('off')
        ax2.imshow(hog_image_rescaled, cmap=plt.cm.gray)
        ax2.set_title('Histogram of Oriented Gradients')

        plt.show()

    def HOGJalan(self):
        hog = cv2.HOGDescriptor()  # Membuat objek detektor HOG
        # Menentukan Support Vector Machine (SVM) yang digunakan oleh detektor HOG
        hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        # Membaca citra jalan.jpg
        Photo = cv2.imread("jalan.jpg")
        # Melakukan resizing citra untuk mempercepat proses deteksi
        Photo = imutils.resize(Photo, width=min(400, Photo.shape[0]))
        # Melakukan deteksi pejalan kaki pada citra menggunakan detektor HOG
        (regions, _) = hog.detectMultiScale(Photo, winStride=(4, 4), padding=(4, 4),scale=1.05)

        for (x, y, w, h) in regions:  # Looping pada setiap region pejalan kaki yang terdeteksi pada citra
            cv2.rectangle(Photo, (x, y), (x + w, y + h), (0, 0, 255),2)  # Menggambar kotak pada setiap pejalan kaki yang terdeteksi

        cv2.imshow("image", Photo)  # Menampilkan citra dengan kotak-kotak deteksi pejalan kaki
        cv2.waitKey()

    def FaceandEye(self):
        # face klasifikasi
        face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        # membaca image faceandeye.jpg
        Image = cv2.imread('faceandeye.jpg')
        # mengecek apakah berhasil di load
        if Image is None:
            print("Error: Unable to load image")
        else:
            # Convert the image to grayscale
            gray = cv2.cvtColor(Image, cv2.COLOR_BGR2GRAY)
            # mendeteksi wajah grayscale
            faces = face_classifier.detectMultiScale(gray, 1.3, 5)
            # mengecek jika tidak ada wajah yang terdeteksi
            if len(faces) == 0:
                print("No faces found")
            else:
                # menggambar kotak di bagian wajah yang terdeteksi
                for (x, y, w, h) in faces:
                    cv2.rectangle(Image, (x, y), (x + w, y + h), (127, 0, 255), 2)

                # menampilkan hasil face detection
                cv2.imshow('Face Detection', Image)
                cv2.waitKey(0)

        # menutup windows
        cv2.destroyAllWindows()

    def Pedestrian(self):
        body_classifier = cv2.CascadeClassifier(
            'haarcascade_fullbody.xml')  # Inisialisasi klasifier dengan menggunakan file xml
        cap = cv2.VideoCapture('video2.mp4')  # Membaca si video
        while cap.isOpened():  # Looping setiap frame
            ret, frame = cap.read()  # Mengambil setiap frame video
            frame = cv2.resize(frame, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_LINEAR)
            # Mengubah ukuran frame agar menjadi lebih kecil dengan skala 0.5
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)  # Mengubah setiap fame ke dalam gambar grayscale
            # Pass frame to our body classifier
            bodies = body_classifier.detectMultiScale(gray, 1.2, 3)  # algoritma deteksi tubuh manusia
            # Extract bounding boxes for any bodies identified
            for (x, y, w, h) in bodies:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255),
                              2)  # Menambahkan kotak di objek pejalan kaki
                cv2.imshow('Pedestrian', frame)  # hasil video dengan deteksi tubuh manusia
            if cv2.waitKey(10) & 0xFF == ord('q'):  # tombol q untuk menghentikan loop
                break
        cap.release()
        cv2.destroyAllWindows()

    def CircleHough(self):
        img = cv2.imread('circlehough.png', 0)  # Membaca gambar lalu mengubahnya ke grayscale
        img = cv2.medianBlur(img, 5)  # Median Filtering pada gambar
        cimg = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)  # Mengubah gambar menjadi berwarna
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0,maxRadius=0)  # Melakukan transformasi Hough lingkaran
        circles = np.uint16(np.around(circles))  # Mengkonversi nilai koordinat lingkaran menjadi integer
        for i in circles[0, :]:  # Looping
            # draw the outer circle
            cv2.circle(cimg, (i[0], i[1]), i[2], (0, 255, 0), 2)  # Menggambar lingkaran pada gambar
            # draw the center of the circle
            cv2.circle(cimg, (i[0], i[1]), 2, (0, 0, 255), 3)  # titik pada pusat lingkaran
        cv2.imshow('detected circles', cimg)  # gambar berwarna
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    def FacialLandmark(self):
        PREDICTOR_PATH = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(PREDICTOR_PATH)
        detector = dlib.get_frontal_face_detector()

        class TooManyFaces(Exception):
            pass

        class NoFaces(Exception):
            pass

    def get_landmarks(im):
        rects = detector(im, 1)

        if len(rects) > 1:
            raise TooManyFaces
        if len(rects) == 0:
            raise NoFaces
        return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

    def annotate_landmarks(im, landmarks):
        im = im.copy()
        for idx, point in enumerate(landmarks):
            pos = (point[0, 0], point[0, 1])
            cv2.putText(im, str(idx), pos,
                        fontFace=cv2.FONT_HERSHEY_SCRIPT_SIMPLEX,
                        fontScale=0.4, color=(0, 0, 255))
            cv2.circle(im, pos, 3, color=(0, 255, 255))
        return im

        image = cv2.imread('jeno.jpeg')
        landmarks = get_landmarks(image)
        image_with_landmarks = annotate_landmarks(image, landmarks)
        cv2.imshow('Result', image_with_landmarks)
        cv2.imwrite('image_with_landmarks.jpg', image_with_landmarks)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = ShowImage()
    window.setWindowTitle('showgui')
    window.show()
    sys.exit(app.exec_())
