# SVD Image Compression

This project demonstrates how **Singular Value Decomposition (SVD)** can be used to compress color images by progressively keeping more singular values. We look at how image quality improves as the number of k's increase, allowing us to see the trade-off between compression and reconstruction quality.

---

## Workflow
- Load a local RGB image (`model.png`).  
- Apply SVD separately to each color channel (R, G, B).  
- Reconstruct approximations of the image with increasing numbers of singular values (k=1 to k=24).  
- Display a grid of 25 images (original + 24 compressed) showing how the image sharpens as k increases.  
- Compare the original and compressed images side by side.

---

## Why SVD for image compression?

- **SVD** breaks down a matrix into components ordered by importance.
- By keeping only the top k singular values, we store the most significant features of the image and discard less important details.
- This shows how complex images can be approximated with a small number of components.

---

Image used for Compression:
![image](model.png)
