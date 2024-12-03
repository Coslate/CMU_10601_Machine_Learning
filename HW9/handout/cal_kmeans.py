 
from typing import Optional
import json
import time
import numpy as np

if __name__ == "__main__":
    # Generate plots
    D = np.array([(5.5, 3.1), (5.1, 4.8), (6.6, 3.0), (5.5, 4.6), (6.8, 3.8)])

    u0 = np.array([(5.3, 3.5)])
    u1 = np.array([(5.1, 4.2)])
    T = 1

    print(f"u0 = {u0}")
    print(f"u1 = {u1}")
    for t in range(T):
        print(f"t = {t}")
        dist1 = np.linalg.norm((D-u0), axis=1)
        dist2 = np.linalg.norm((D-u1), axis=1)
        print(f"dist1 = {dist1}")
        print(f"dist2 = {dist2}")
        z = ["u0"]*D.shape[0]

        for i, pt in enumerate(z):
            if dist1[i] < dist2[i]:
                z[i] = "u0"
            else:
                z[i] = "u1"

        mask0 = [item == 'u0' for item in z]
        mask1 = [item == 'u1' for item in z]

        u0_sel_pts = D[mask0]
        u1_sel_pts = D[mask1]

        u0 = np.mean(u0_sel_pts, axis=0)
        u1 = np.mean(u1_sel_pts, axis=0)
        print(f"u0 = {u0}")
        print(f"u1 = {u1}")
        print(f"u0_sel_pts = {u0_sel_pts}")
        print(f"u1_sel_pts = {u1_sel_pts}")


        

        






