import scipy.io as sio
import numpy as np

path = "eye_feature_smooth/1/1_20160518.mat"
data = sio.loadmat(path)

print("=== keys ===")

for k in data.keys():
    print(k)

print("\n=== shapes ===")
for k in data.keys():
    if not k.startswith("__"):
        v = data[k]
        try:
            print(f"{k}: shape = {v.shape}")
        except:
            print(f"{k}: type = {type(v)}")

# for i in data['de_LDS1'][0]:
#     print(*i)