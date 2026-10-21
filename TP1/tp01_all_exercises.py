from pathlib import Path

import cv2
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

BASE_DIR = Path(__file__).resolve().parent


def normalize_histogram(img_gray: np.ndarray):
    hist = cv2.calcHist([img_gray], [0], None, [256], [0, 256])
    return hist


def display_histogram(img: np.ndarray, title: str, ax):
    hist = normalize_histogram(img)
    ax.plot(hist, color="black")
    ax.set_title(title)
    ax.set_xlim(0, 256)
    ax.set_xlabel("Niveau de gris")
    ax.set_ylabel("Nombre de pixels")


# Exercice 1 : Lecture et analyse d'une image couleur

def exercice_1():
    img = cv2.imread(str(BASE_DIR / "rgb.png"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image rgb.png introuvable dans le dossier TP1 VA")

    print("Exercice 1")
    print(f"Taille : {img.shape[0]} x {img.shape[1]} x {img.shape[2]}")
    print(f"Type : {img.dtype}")

    b, g, r = cv2.split(img)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    fig, axes = plt.subplots(2, 3, figsize=(12, 8))
    axes[0, 0].imshow(rgb_img)
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")

    axes[0, 1].imshow(r, cmap="Reds")
    axes[0, 1].set_title("Canal R")
    axes[0, 1].axis("off")

    axes[0, 2].imshow(g, cmap="Greens")
    axes[0, 2].set_title("Canal G")
    axes[0, 2].axis("off")

    axes[1, 0].imshow(b, cmap="Blues")
    axes[1, 0].set_title("Canal B")
    axes[1, 0].axis("off")

    axes[1, 1].imshow(rgb_img[:, :, 0], cmap="gray")
    axes[1, 1].set_title("Composante Rouge (R)")
    axes[1, 1].axis("off")

    axes[1, 2].remove()
    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_1.png", dpi=200)
    plt.close(fig)


# Exercice 2 : Échantillonnage / réduction de résolution

def exercice_2():
    img = cv2.imread(str(BASE_DIR / "usthbc.jpg"), cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError("Image usthbc.jpg introuvable")

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    scales = [0.5, 0.25, 0.125]

    fig, axes = plt.subplots(1, 4, figsize=(16, 5))
    axes[0].imshow(gray, cmap="gray")
    axes[0].set_title("Originale")
    axes[0].axis("off")

    for ax, scale in zip(axes[1:], scales):
        resized = cv2.resize(gray, (0, 0), fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        ax.imshow(resized, cmap="gray")
        ax.set_title(f"Échantillonnage {scale}")
        ax.axis("off")

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_2.png", dpi=200)
    plt.close(fig)


# Exercice 3 : Quantification en niveaux de gris

def exercice_3():
    img = cv2.imread(str(BASE_DIR / "usthbs.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image usthbs.jpg introuvable")

    fig, axes = plt.subplots(2, 4, figsize=(18, 10))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("8 bits")
    axes[0, 0].axis("off")
    display_histogram(img, "Histogramme 8 bits", axes[1, 0])

    for idx, bit_count in enumerate([4, 2, 1], start=1):
        if bit_count == 1:
            q = np.where(img >= 128, 255, 0).astype(np.uint8)
        else:
            step = 256 / (2 ** bit_count)
            q = (img // step) * step
            q = np.clip(q, 0, 255).astype(np.uint8)

        axes[0, idx].imshow(q, cmap="gray")
        axes[0, idx].set_title(f"{bit_count} bits")
        axes[0, idx].axis("off")
        display_histogram(q, f"Histogramme {bit_count} bits", axes[1, idx])

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_3.png", dpi=200)
    plt.close(fig)


# Exercice 5 : Étirement d'histogramme (expansion dynamique)

def exercice_5():
    img = cv2.imread(str(BASE_DIR / "usthbc.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image usthbc.jpg introuvable")

    imin = np.min(img)
    imax = np.max(img)
    stretched = (img - imin) * 255 / (imax - imin) if imax != imin else img.copy()
    stretched = np.clip(stretched, 0, 255).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")
    display_histogram(img, "Histogramme original", axes[1, 0])

    axes[0, 1].imshow(stretched, cmap="gray")
    axes[0, 1].set_title("Image après étirement")
    axes[0, 1].axis("off")
    display_histogram(stretched, "Histogramme après étirement", axes[1, 1])

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_5.png", dpi=200)
    plt.close(fig)


# Exercice 7 : Égalisation d'histogramme

def exercice_7():
    img = cv2.imread(str(BASE_DIR / "usthbc.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image usthbc.jpg introuvable")

    equalized = cv2.equalizeHist(img)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes[0, 0].imshow(img, cmap="gray")
    axes[0, 0].set_title("Image originale")
    axes[0, 0].axis("off")
    display_histogram(img, "Histogramme original", axes[1, 0])

    axes[0, 1].imshow(equalized, cmap="gray")
    axes[0, 1].set_title("Image égalisée")
    axes[0, 1].axis("off")
    display_histogram(equalized, "Histogramme égalisé", axes[1, 1])

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_7.png", dpi=200)
    plt.close(fig)


# Exercice 9 : Seuillage

def exercice_9():
    img = cv2.imread(str(BASE_DIR / "usthbc.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image usthbc.jpg introuvable")

    _, thresholded = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(thresholded, cmap="gray")
    axes[1].set_title("Image seuillée (seuil = 127)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_9.png", dpi=200)
    plt.close(fig)


# Exercice 10 : Correction gamma

def exercice_10():
    img = cv2.imread(str(BASE_DIR / "usthbs.jpg"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image usthbs.jpg introuvable")

    gamma = 0.5
    gamma_corrected = np.array(255 * (img / 255.0) ** gamma, dtype="uint8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(gamma_corrected, cmap="gray")
    axes[1].set_title("Correction gamma (γ = 0.5)")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_10.png", dpi=200)
    plt.close(fig)


# Exercice 11 : Transformation logarithmique

def exercice_11():
    img = cv2.imread(str(BASE_DIR / "rgb.png"), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError("Image rgb.png introuvable")

    c = 255 / np.log(1 + np.max(img))
    log_img = c * np.log(1 + img)
    log_img = np.array(log_img, dtype="uint8")

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    axes[0].imshow(img, cmap="gray")
    axes[0].set_title("Image originale")
    axes[0].axis("off")

    axes[1].imshow(log_img, cmap="gray")
    axes[1].set_title("Transformation logarithmique")
    axes[1].axis("off")

    fig.tight_layout()
    fig.savefig(BASE_DIR / "exercice_11.png", dpi=200)
    plt.close(fig)


# Exercice 1 : Analyse de la matrice en niveaux de gris (à partir de deux images)

def exercice_gris():
    img_b = cv2.imread(str(BASE_DIR / "usthbc.jpg"), cv2.IMREAD_GRAYSCALE)
    img_s = cv2.imread(str(BASE_DIR / "usthbs.jpg"), cv2.IMREAD_GRAYSCALE)

    for name, img in [("usthbc.jpg", img_b), ("usthbs.jpg", img_s)]:
        print(f"{name}: forme = {img.shape}, type = {img.dtype}, min = {img.min()}, max = {img.max()}")


if __name__ == "__main__":
    exercice_1()
    exercice_2()
    exercice_3()
    exercice_5()
    exercice_7()
    exercice_9()
    exercice_10()
    exercice_11()
    exercice_gris()
    print("Tous les exercices ont été traités et les images sauvegardées dans le dossier TP1 VA.")
