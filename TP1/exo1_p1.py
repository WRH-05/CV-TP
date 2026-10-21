# =====================================================
# TP : Vision Artificielle 
# Acquisition, Numérisation et Transformations
# =====================================================

# =======================
# 1) Importation
# =======================
import cv2
import numpy as np
import matplotlib.pyplot as plt

# =======================
# 2) Lecture de l'image
# =======================
img = cv2.imread("TP1 VA/rgb.png")
if img is None:
    print("Erreur: image non trouvée.")
    exit()

r = img.copy()
g = img.copy()
b = img.copy()
r[:,:,1]=0;
r[:,:,2]=0;
g[:,:,0]=0;
g[:,:,2]=0;
b[:,:,0]=0;
b[:,:,1]=0;

plt.subplot(2,3,2)
plt.title("Image")
plt.axis("off")
plt.imshow(img);
plt.subplot(2,3,4)
plt.title("Red channel")
plt.axis("off")
plt.imshow(r);
plt.subplot(2,3,5)
plt.title("Green channel")
plt.axis("off")
plt.imshow(g);
plt.subplot(2,3,6)
plt.title("Blue channel")
plt.axis("off")
plt.imshow(b);

plt.show()


