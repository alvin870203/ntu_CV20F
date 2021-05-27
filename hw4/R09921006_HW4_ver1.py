from PIL import Image
import numpy as np


def dilation(originalImage, kernel):
    """
    :type originalImage: Image (from PIL)
    :type kernel: numpy array
    :return type: Image (from PIL)
    """
    # Get center position of kernel.
    centerKernel = tuple([x // 2 for x in kernel.shape])
    # New image with the same size and 'binary' format.
    dilationImage = Image.new('1', originalImage.size)
    # Scan each column in original image.
    for r in range(originalImage.size[0]):
        # Scan each row in original image.
        for c in range(originalImage.size[1]):
            # Get pixel value in original image at (r, c).
            originalPixel = originalImage.getpixel((r, c))
            # If this pixel is object (1, white).
            if (originalPixel != 0):
                # Paste kernel on original image at (r, c).
                # Scan each column in kernel.
                for x in range(kernel.shape[0]):
                    # Scan each row in kernel.
                    for y in range(kernel.shape[1]):
                        # Only paste '1' value from kernel.
                        if (kernel[x, y] == 1):
                            # Calculate destination x, y position.
                            destX = r + (x - centerKernel[0])
                            destY = c + (y - centerKernel[1])
                            # Avoid out of image range.
                            if ((0 <= destX < originalImage.size[0]) and \
                                (0 <= destY < originalImage.size[1])):
                                # Paste '1' value on original image.
                                dilationImage.putpixel((destX, destY), 1)
    # Return dilation image.
    return dilationImage

def erosion(originalImage, kernel):
    """
    :type originalImage: Image (from PIL)
    :type kernel: numpy array
    :return type: Image (from PIL)
    """
    # Get center position of kernel.
    centerKernel = tuple([x // 2 for x in kernel.shape])
    # New image with the same size and 'binary' format.
    erosionImage = Image.new('1', originalImage.size)
    # Scan each column in original image.
    for r in range(originalImage.size[0]):
        # Scan each row in original image.
        for c in range(originalImage.size[1]):
            # Flag of match.
            matchFlag = True
            # Scan each column in kernel.
            for x in range(kernel.shape[0]):
                # Scan each row in kernel.
                for y in range(kernel.shape[1]):
                    # Only check '1' value from kernel.
                    if (kernel[x, y] == 1):
                        # Calculate destination x, y position.
                        destX = r + (x - centerKernel[0])
                        destY = c + (y - centerKernel[1])
                        # Avoid out of image range.
                        if ((0 <= destX < originalImage.size[0]) and \
                            (0 <= destY < originalImage.size[1])):
                            # If this point doesn't match with kernel.
                            if (originalImage.getpixel((destX, destY)) == 0):
                                # Clear flag of match.
                                matchFlag = False
                                break
                        # It is edge point, it will never match.
                        else:
                            # Clear flag of match.
                            matchFlag = False
                            break
            # Full kernel is match in original image at (r, c).
            if (matchFlag):
                # Paste '1' value on original image.
                erosionImage.putpixel((r, c), 1)
    # Return erosion image.
    return erosionImage

def opening(originalImage, kernel):
    """
    :type originalImage: Image (from PIL)
    :type kernel: numpy array
    :return type: Image (from PIL)
    """
    return dilation(erosion(originalImage, kernel), kernel)

def closing(originalImage, kernel):
    """
    :type originalImage: Image (from PIL)
    :type kernel: numpy array
    :return type: Image (from PIL)
    """
    return erosion(dilation(originalImage, kernel), kernel)

def complement(originalImage):
    """
    :type originalImage: Image (from PIL)
    :return type: Image (from PIL)
    """
    # New image with the same size and 'binary' format.
    complementImage = Image.new('1', originalImage.size)
    # Scan each column in original image.
    for r in range(originalImage.size[0]):
        # Scan each row in original image.
        for c in range(originalImage.size[1]):
            # If this pixel is object (1, white).
            if (originalImage.getpixel((r, c)) == 0):
                # Paste '1' value on intersection image.
                complementImage.putpixel((r, c), 1)
            else:
                # Paste '0' value on intersection image.
                complementImage.putpixel((r, c), 0)
    return complementImage

def intersection(image1, image2):
    """
    :type image1: Image (from PIL)
    :type image2: Image (from PIL)
    :return type: Image (from PIL)
    """
    from PIL import Image
    # New image with the same size and 'binary' format.
    intersectionImage = Image.new('1', image1.size)
    # Scan each column in image 1.
    for r in range(image1.size[0]):
        # Scan each row in image 1.
        for c in range(image1.size[1]):
            # Get pixel value in image 1 at (r, c).
            image1Pixel = image1.getpixel((r, c))
            # Get pixel value in image 2 at (r, c).
            image2Pixel = image2.getpixel((r, c))
            # If those pixels are object (1, white).
            if (image1Pixel != 0 and image2Pixel != 0):
                # Paste '1' value on intersection image.
                intersectionImage.putpixel((r, c), 1)
            else:
                # Paste '0' value on intersection image.
                intersectionImage.putpixel((r, c), 0)
    return intersectionImage

def erosionWithCenter(originalImage, kernel, centerKernel):
    """
    :type originalImage: Image (from PIL)
    :type kernel: numpy array
    :type centerKernel: tuple
    :return type: Image (from PIL)
    """
    from PIL import Image
    # New image with the same size and 'binary' format.
    erosionImage = Image.new('1', originalImage.size)
    # Scan each column in original image.
    for r in range(originalImage.size[0]):
        # Scan each row in original image.
        for c in range(originalImage.size[1]):
            # Flag of match.
            matchFlag = True
            # Scan each column in kernel.
            for x in range(kernel.shape[0]):
                # Scan each row in kernel.
                for y in range(kernel.shape[1]):
                    # Only check '1' value from kernel.
                    if (kernel[x, y] == 1):
                        # Calculate destination x, y position.
                        destX = r + (x - centerKernel[0])
                        destY = c + (y - centerKernel[1])
                        # Avoid out of image range.
                        if ((0 <= destX < originalImage.size[0]) and \
                            (0 <= destY < originalImage.size[1])):
                            # If this point doesn't match with kernel.
                            if (originalImage.getpixel((destX, destY)) == 0):
                                # Clear flag of match.
                                matchFlag = False
                                break
                        # It is edge point, it will never match.
                        else:
                            # Clear flag of match.
                            matchFlag = False
                            break
            # Full kernel is match in original image at (r, c).
            if (matchFlag):
                # Paste '1' value on original image.
                erosionImage.putpixel((r, c), 1)
    # Return erosion image.
    return erosionImage

def hitmiss(originalImage, kernel_J, centerKernel_J, kernel_K, centerKernel_K):
    """
    :type originalImage: Image (from PIL)
    :type kernel_J: numpy array
    :type centerKernel_J: tuple
    :type kernel_K: numpy array
    :type centerKernel_K: tuple
    :return type: Image (from PIL)
    """
    return intersection(erosionWithCenter(originalImage, kernel_J, centerKernel_J), 
                erosionWithCenter(complement(originalImage), kernel_K, centerKernel_K))



######################################################

if __name__ == '__main__':

    # Define threshold of binary image.
    threshold = 128

    # Load image from file.
    originalImage = Image.open('lena.bmp')

    # Get width and height of image.
    width, height = originalImage.size
    # print ('width = %d, height = %d' %(width, height))

    # New image with the same size and 'binary' format.
    binaryImage = Image.new('1', originalImage.size)

    # Process image pixel by pixel.
    for c in range(width):
        for r in range(height):
            # Get pixel from original image.
            value = originalImage.getpixel((c, r))
            if (value >= threshold):
                value = 1
            else:
                value = 0
            # Put pixel to binary image.
            binaryImage.putpixel((c, r), value)

    ###################################################

    # Define kernel for dilation.
    kernel = np.array([\
        [0, 1, 1, 1, 0], \
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1], \
        [1, 1, 1, 1, 1], \
        [0, 1, 1, 1, 0]])
    # Get dilation image.
    dilationImage = dilation(binaryImage, kernel)
    # Save image fo file.
    dilationImage.save('dilation.bmp')

    ##################################################

    # Get erosion image.
    erosionImage = erosion(binaryImage, kernel)
    # Save image fo file.
    erosionImage.save('erosion.bmp')

    ###################################################
    
    # Get opening image.
    openingImage = opening(binaryImage, kernel)
    # Save image fo file.
    openingImage.save('opening.bmp')

    ###################################################

    # Get closing image.
    closingImage = closing(binaryImage, kernel)
    # Save image fo file.
    closingImage.save('closing.bmp')

    ###################################################

    # Define kernels for hit-and-miss.
    kernel_J = np.array([
        [1, 1], 
        [0, 1]])
    centerKernel_J = (1, 0)
    kernel_K = np.array([
        [1, 1], 
        [0, 1]])
    centerKernel_K = (0, 1)
    # Get hit-and-miss image.
    hitAndMissImage = hitmiss(binaryImage, 
        kernel_J, centerKernel_J, 
        kernel_K, centerKernel_K)
    # Save image fo file.
    hitAndMissImage.save('hit-and-miss.bmp')