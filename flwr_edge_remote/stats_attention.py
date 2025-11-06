import matplotlib.pyplot as plt

# Round / client IDs
clients = list(range(1, 21))

# ==========================
# CNNWithAttention — senza augmentation
# ==========================
acc_no_aug = [
    0.51991, 0.60841, 0.73673, 0.60841, 0.73673, 0.92920, 0.76991, 0.74336, 0.93805, 0.91372,
    0.90044, 0.93142, 0.90929, 0.94027, 0.92035, 0.93805, 0.93363, 0.94248, 0.94690, 0.94469
]

mask_benefit_no_aug = [
    0.0, 0.15044, 0.061947, -0.80088, -0.10177, 0.86726, 0.31858, -0.0044248, 0.25221, 0.42920,
    0.45133, 0.67257, 0.34513, 0.33186, 0.37168, 0.42035, 0.25664, 0.26549, 0.31416, 0.38496
]

# ==========================
# CNNWithAttention — con augmentation
# ==========================
acc_aug = [
    0.48009, 0.51991, 0.48009, 0.65929, 0.63938, 0.63717, 0.64381, 0.69690, 0.70796, 0.70133,
    0.80088, 0.75885, 0.82965, 0.92035, 0.90929, 0.82080, 0.94690, 0.93142, 0.79425, 0.93584
]

mask_benefit_aug = [
    0.0, 0.0, 0.0, 0.19912, 0.10177, 0.10619, 0.09292, 0.070796, 0.057522, 0.21681,
    0.38938, 0.26991, 0.49558, 0.70354, 0.60619, 0.12832, 0.49558, 0.72124, 0.33628, 0.47788
]

# ==========================
# Grafico Accuracy
# ==========================
plt.figure(figsize=(10,5))
plt.plot(clients, acc_no_aug, marker="o", linestyle="--", label="Without Data Augmentation")
plt.plot(clients, acc_aug, marker="o", linestyle="-", label="With Data Augmentation")
plt.title("Accuracy per Round - CNNWithAttention")
plt.xlabel("Round")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()

# ==========================
# Grafico Mask Benefit
# ==========================
plt.figure(figsize=(10,5))
plt.plot(clients, mask_benefit_no_aug, marker="s", linestyle="--", label="Without Data Augmentation")
plt.plot(clients, mask_benefit_aug, marker="s", linestyle="-", label="With Data Augmentation")
plt.title("Mask Benefit per Round - CNNWithAttention")
plt.xlabel("Round")
plt.ylabel("Mask Benefit")
plt.legend()
plt.grid(True)
plt.show()
