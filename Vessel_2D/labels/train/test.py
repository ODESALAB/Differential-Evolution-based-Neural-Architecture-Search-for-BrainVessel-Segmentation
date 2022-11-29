import numpy as np
from torchvision.io import read_image

for i in range(0, 34):
    print(i)
    img = read_image(f"{i}.PNG")
    c1, c2, c3 = img[0, :, :], img[1, :, :], img[2, :, :]
    channels = [c1, c2, c3]
    for id, c in enumerate(channels):
        for idx in range(id, len(channels)):
            result = np.where((c == channels[idx]) == False)
            if len(result[0]) > 0 or len(result[1]) > 0:
                print("HATA")

print()