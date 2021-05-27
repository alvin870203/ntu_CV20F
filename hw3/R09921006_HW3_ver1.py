from PIL import Image
import matplotlib.pyplot as plt
import numpy as np
import csv


def histogram(img, name):
    # Create histogram array with zeros.
    Histogram = np.zeros(256)

    # Process image pixel by pixel.
    for c in range(width):
        for r in range(height):
            # Get pixel from image.
            pixelValue = img.getpixel((c, r))
            # Record count in histogram array.
            Histogram[pixelValue] += 1

    # Clear plot.
    plt.gcf().clear()
    # Plot histogram.
    plt.bar(range(len(Histogram)), Histogram)
    
    plt.savefig(name)

    return Histogram


if __name__ == '__main__':
    # Load image from file.
    originalImage = Image.open('lena.bmp')

    # Get width and height of image.
    width, height = originalImage.size

    # Save original image
    originalImage.save('lena_origin.bmp')
    _ = histogram(originalImage, 'histogram_origin')

    # New image with the same size and 'grayscale' format.
    darkImage = Image.new('L', originalImage.size)

    # Process image pixel by pixel.
    for c in range(width):
        for r in range(height):
            # Get pixel from original image.
            pixelValue = originalImage.getpixel((c, r))
            # Assign 1/3 pixel value to dark image.
            darkImage.putpixel((c, r), pixelValue // 3)

    # Save image to file.
    # Save image with intensity divided by 3
    darkImage.save('lena_dark.bmp')
    darkHistogram = histogram(darkImage, 'histogram_dark.png')

    # Histogram Equalization
    # Look up table for transformation.
    transformationTable = np.zeros(256)

    # Deal with each value (0 ~ 255).
    for i in range(len(transformationTable)):
        transformationTable[i] = 255 * np.sum(darkHistogram[0:i + 1]) / width / height

    # New image with the same size and 'grayscale' format.
    histEquImage = Image.new('L', originalImage.size)

    # Process image pixel by pixel.
    for c in range(width):
        for r in range(height):
            # Get pixel from dark image.
            pixelValue = darkImage.getpixel((c, r))
            # Put pixel to histogram equalization image.
            histEquImage.putpixel((c, r), int(transformationTable[pixelValue]))
        
    # Save image to file.
    histEquImage.save('lena_equalization.bmp')
    _ = histogram(histEquImage, 'histogram_equalization.png')