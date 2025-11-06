import matplotlib.pyplot as plt

# Round (client)
rounds = list(range(1, 21))

# Accuracy senza augmentation (3ª colonna corretta)
acc_no_aug = [
    0.4801, 0.4801, 0.5841, 0.7367, 0.6150, 0.6372, 0.5907, 0.6327, 0.6792, 0.6460,
    0.7124, 0.7965, 0.8274, 0.8230, 0.7876, 0.8938, 0.9469, 0.9314, 0.8628, 0.8606
]

# Accuracy con augmentation (dai log)
acc_aug = [
    0.48009, 0.51991, 0.48009, 0.53319, 0.48009, 0.66814, 0.73673, 0.69027, 0.80973, 0.86947,
    0.80088, 0.89159, 0.89159, 0.94027, 0.94690, 0.94690, 0.93363, 0.95354, 0.91593, 0.95354
]

plt.figure(figsize=(10,5))
plt.plot(rounds, acc_no_aug, label="Without Data Augmentation", marker="o", linestyle="--")
plt.plot(rounds, acc_aug, label="With Data Augmentation", marker="o", linestyle="-")

plt.xlabel("Rounds")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)
plt.show()
