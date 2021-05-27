import cv2
import math, sys
import matplotlib.pyplot as plt
import numpy as np

####### binarize ######
def img_binarize(img_in):
    return (img_in > 0x7f) * 0xff

####### yokoi core ####
def do_yokoi(img_down):
    # yokoi h op
    def h(b, c, d, e):
        if b == c and (d != b or e != b):
            return 'q'
        if b == c and (d == b and e == b):
            return 'r'

        return 's'

    # main part of yokoi connectivity
    res = np.zeros(img_down.shape, np.int)
    row, col = img_down.shape
    for i in range(row):
        for j in range(col):
            if img_down[i, j] > 0:
                if i == 0:
                    # top-left
                    if j == 0:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = 0, img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = 0, img_down[i + 1, j], img_down[i + 1, j + 1]
                    # top-right
                    elif j == col - 1:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], 0
                        x8, x4, x5 = img_down[i + 1, j - 1], img_down[i + 1, j], 0
                    # top-row
                    else:
                        x7, x2, x6 = 0, 0, 0
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = img_down[i + 1, j - 1], img_down[i + 1, j], img_down[i + 1, j + 1]
                elif i == row - 1:
                    # bottom-left
                    if j == 0:
                        x7, x2, x6 = 0, img_down[i - 1, j], img_down[i - 1, j + 1]
                        x3, x0, x1 = 0, img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = 0, 0, 0
                    # bottom-right
                    elif j == col - 1:
                        x7, x2, x6 = img_down[i - 1, j - 1], img_down[i - 1, j], 0
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], 0
                        x8, x4, x5 = 0, 0, 0
                    # bottom-row
                    else:
                        x7, x2, x6 = img_down[i - 1, j - 1], img_down[i - 1, j], img_down[i - 1, j + 1]
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = 0, 0, 0
                else:
                    # leftmost-row
                    if j == 0:
                        x7, x2, x6 = 0, img_down[i - 1, j], img_down[i - 1, j + 1]
                        x3, x0, x1 = 0, img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = 0, img_down[i + 1, j], img_down[i + 1, j + 1]
                    # rightmost-column
                    elif j == col - 1:
                        x7, x2, x6 = img_down[i - 1, j - 1], img_down[i - 1, j], 0
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], 0
                        x8, x4, x5 = img_down[i + 1, j - 1], img_down[i + 1, j], 0
                    #the rest, inner
                    else:
                        x7, x2, x6 = img_down[i - 1, j - 1], img_down[i - 1, j], img_down[i - 1, j + 1]
                        x3, x0, x1 = img_down[i, j - 1], img_down[i, j], img_down[i, j + 1]
                        x8, x4, x5 = img_down[i + 1, j - 1], img_down[i + 1, j], img_down[i + 1, j + 1]

                a1 = h(x0, x1, x6, x2)
                a2 = h(x0, x2, x7, x3)
                a3 = h(x0, x3, x8, x4)
                a4 = h(x0, x4, x5, x1)

                cnt = 0
                if a1 == 'r' and a2 == 'r' and a3 == 'r' and a4 == 'r':
                    cnt = 5
                else:
                    cnt = 0
                    for a_i in [a1, a2, a3, a4]:
                        if a_i == 'q':
                            cnt += 1
                
                res[i, j] = cnt
                
    return res

####### ib core ######
def do_ib(img_in):
    def h(c, d):
        # interior content
        if c == d:
            return c         
        # border content
        else:
            return 'b'

    # BG: 0, interior: 1, border: 2
    res = np.zeros(img_in.shape, np.int)
    row, col = img_in.shape
    for i in range(row):
        for j in range(col):
            if img_in[i, j] > 0:
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    # top-left
                    if j == 0:
                        x1, x4 = img_in[i, j + 1], img_in[i + 1, j]
                    # top-right
                    elif j == col - 1:
                        x3, x4 = img_in[i, j - 1], img_in[i + 1, j]
                    # top-row
                    else:
                        x1, x3, x4 = img_in[i, j + 1], img_in[i, j - 1], img_in[i + 1, j]
                elif i == row - 1:
                    # bottom-left
                    if j == 0:
                        x1, x2 = img_in[i, j + 1], img_in[i - 1, j]
                    # bottom-right
                    elif j == col - 1:
                        x2, x3 = img_in[i - 1, j], img_in[i, j - 1]
                    # bottom-row
                    else:
                        x1, x2, x3 = img_in[i, j + 1], img_in[i - 1, j], img_in[i, j - 1]
                else:
                    # leftmost-row
                    if j == 0:
                        x1, x2, x4 = img_in[i, j + 1], img_in[i - 1, j], img_in[i + 1, j]
                    # rightmost-colmn
                    elif j == col - 1:
                        x2, x3, x4 = img_in[i - 1, j], img_in[i, j - 1], img_in[i + 1, j]
                    # the rest, inner
                    else:
                        x1, x2, x3, x4 = img_in[i, j + 1], img_in[i - 1, j], img_in[i, j - 1], img_in[i + 1, j]

                x1 /= 255
                x2 /= 255
                x3 /= 255
                x4 /= 255
                a1 = h(1, x1)
                a2 = h(a1, x2)
                a3 = h(a2, x3)
                a4 = h(a3, x4)
                if a4 == 'b':
                    res[i, j] = 2
                else:
                    res[i, j] = 1
    
    return res

###### mp core ######
def do_mp(img_in):
    def h(a, m):
        # interior content
        if a == m:
            return 1         
        # border content
        else:
            return 0

    # BG: 0, p: 1, q: 2
    
    res = np.zeros(img_in.shape, np.int)
    row, col = img_in.shape
    for i in range(row):
        for j in range(col):
            if img_in[i, j] > 0:
                x1, x2, x3, x4 = 0, 0, 0, 0
                if i == 0:
                    # top-left
                    if j == 0:
                        x1, x4 = img_in[i, j + 1], img_in[i + 1, j]
                    # top-right
                    elif j == col - 1:
                        x3, x4 = img_in[i, j - 1], img_in[i + 1, j]
                    # top-row
                    else:
                        x1, x3, x4 = img_in[i, j + 1], img_in[i, j - 1], img_in[i + 1, j]
                elif i == row - 1:
                    # bottom-left
                    if j == 0:
                        x1, x2 = img_in[i, j + 1], img_in[i - 1, j]
                    # bottom-right
                    elif j == col - 1:
                        x2, x3 = img_in[i - 1, j], img_in[i, j - 1]
                    # bottom-row
                    else:
                        x1, x2, x3 = img_in[i, j + 1], img_in[i - 1, j], img_in[i, j - 1]
                else:
                    # leftmost-row
                    if j == 0:
                        x1, x2, x4 = img_in[i, j + 1], img_in[i - 1, j], img_in[i + 1, j]
                    # rightmost-colmn
                    elif j == col - 1:
                        x2, x3, x4 = img_in[i - 1, j], img_in[i, j - 1], img_in[i + 1, j]
                    # the rest, inner
                    else:
                        x1, x2, x3, x4 = img_in[i, j + 1], img_in[i - 1, j], img_in[i, j - 1], img_in[i + 1, j]
                
                # problem check
                if h(x1, 1) + h(x2, 1) + h(x3, 1) + h(x4, 1) >= 1 and img_in[i, j] == 2:
                    res[i, j] = 1
                else:
                    res[i, j] = 2
         
    return res


    ####### IO ############
img = cv2.imread('lena.bmp', cv2.IMREAD_GRAYSCALE)

#### thinning core ####
def do_thinnig():
    img_bin = img_binarize(img)
    
    row, col = (img_bin.shape)
    row, col = row // 8, col // 8
    res_final = np.zeros((row, col), np.int)
    
    for i in range(row):
        for j in range(col):
            res_final[i, j] = img_bin[8 * i, 8* j]
    
    step_cnt = 0
    while True:
        # use numpy copy to prevent from changing to same memory block
        
        res_old = res_final
        if id(res_old) == id(res_final):
            print('Same ID!')
            res_old = np.copy(res_final)
        
        '''
        Step1
        input : original symbolic image 
        marked-interior/border-pixel operator
        
        output : interior/border image
        '''
        res_ib = do_ib(res_final)
        
        '''
        Step2
        input : interior/border image
        pair relationship operator
        output : marked image
        '''
        res_mp = do_mp(res_ib)
        
        '''
        Step3
        input : original symbolic image +marked image
        marked-pixel connected shrink operator
        removable(by connected shrink operator on original symbolic image)
        marked(by marked image)
        delete those pixels satisfied the two conditions mentioned above

        output : thinned output image
        '''
        res_yokoi = do_yokoi(res_final)
        res_to_delete = (res_yokoi == 1) * 1
        
        for i in range(row):
            for j in range(col):
                if res_to_delete[i, j] == 1 and res_mp[i, j] == 1:
                    res_final[i, j] = 0
        
        '''
        compare the currently thinned image with the old one
        break if has not been changed since last iteration
        '''
        
        save_name = 'lena_thinned_step' + str(step_cnt) + '.png'
        cv2.imwrite(save_name, res_final)
        plt.imshow(res_final, cmap = 'gray')
        plt.show()
        step_cnt += 1
        
        if np.sum(res_old == res_final) == row * col:
            break
    

do_thinnig()        