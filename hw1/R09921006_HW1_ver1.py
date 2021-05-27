import numpy as np
import cv2

# Read image

def readimg(path):
    img = cv2.imread(path)
    return img

img_path = './lena.bmp'
print('Reading image')
img_origin = readimg(img_path)
print('Shape of lena.bmp = {}, dtype={}'.format(img_origin.shape, img_origin.dtype))
cv2.imshow('Original lena.bmp', img_origin)
cv2.waitKey(1000)

# Part 1

img_upside_down = np.zeros(img_origin.shape, dtype=np.uint8) # must specific dtype=np.uint8 for cv2
img_right_side_left = np.zeros(img_origin.shape, dtype=np.uint8)
img_diagonally_flip = np.zeros(img_origin.shape, dtype=np.uint8)

for row in range(img_origin.shape[0]):
    img_upside_down[row, :, :] = img_origin[-1-row, :, :]
    for col in range(img_origin.shape[1]):
        img_right_side_left[:, col, :] = img_origin[:, -1-col, :]
        img_diagonally_flip[row, col, :] = img_origin[col, row, :]

print('Shape of upside-down lena.bmp = {}, dtype={}'.format(img_upside_down.shape, img_upside_down.dtype))
cv2.imwrite('upside_down_lena.bmp', img_upside_down)
cv2.imshow('Upside-down lena.bmp', img_upside_down)
cv2.waitKey(1000)

print('Shape of right-side-left lena.bmp = {}, dtype={}'.format(img_right_side_left.shape, img_right_side_left.dtype))
cv2.imwrite('right_side_left_lena.bmp', img_right_side_left)
cv2.imshow('Right_side_left lena.bmp', img_right_side_left)
cv2.waitKey(1000)

print('Shape of diagonally_flip lena.bmp = {}, dtype={}'.format(img_diagonally_flip.shape, img_diagonally_flip.dtype))
cv2.imwrite('diagonally_flip_lena.bmp', img_diagonally_flip)
cv2.imshow('Diagonally_flip lena.bmp', img_diagonally_flip)
cv2.waitKey(1000)

# Part 2

rotate_matrix = cv2.getRotationMatrix2D((img_origin.shape[1]/2, img_origin.shape[0]/2), -45, 1.0)
img_rotate_45_deg_clockwise = cv2.warpAffine(img_origin, rotate_matrix, (img_origin.shape[1], img_origin.shape[0]))
print('Shape of rotate_45_deg_clockwise lena.bmp = {}, dtype={}'.format(img_rotate_45_deg_clockwise.shape, img_rotate_45_deg_clockwise.dtype))
cv2.imwrite('rotate_45_deg_clockwise_lena.bmp', img_rotate_45_deg_clockwise)
cv2.imshow('Rotate_45_deg_clockwise lena.bmp', img_rotate_45_deg_clockwise)
cv2.waitKey(1000)

img_shrink_in_half = cv2.resize(img_origin, (img_origin.shape[1]//2, img_origin.shape[0]//2))
print('Shape of shrink_in_half lena.bmp = {}, dtype={}'.format(img_shrink_in_half.shape, img_shrink_in_half.dtype))
cv2.imwrite('shrink_in_half_lena.bmp', img_shrink_in_half)
cv2.imshow('Shrink_in_half lena.bmp', img_shrink_in_half)
cv2.waitKey(1000)

_, img_binarize_at_128 = cv2.threshold(img_origin, 128, 255, cv2.THRESH_BINARY)
print('Shape of binarize_at_128 lena.bmp = {}, dtype={}'.format(img_binarize_at_128.shape, img_binarize_at_128.dtype))
cv2.imwrite('binarize_at_128_lena.bmp', img_binarize_at_128)
cv2.imshow('Binarize_at_128 lena.bmp', img_binarize_at_128)
cv2.waitKey(1000)