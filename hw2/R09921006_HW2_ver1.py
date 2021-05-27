from PIL import Image

class Pixels2D():
    
    def __init__(self, pixels_or_image, width=None, size=None):
        if isinstance(pixels_or_image, Image.Image):
            self.data = list(pixels_or_image.getdata())
            self.width = pixels_or_image.width
        elif hasattr(pixels_or_image, '__getitem__'):
            if width == size == None:
                raise ValueError("Specify 'width' or 'size' in arguments.")

            self.data = list(pixels_or_image)
            self.width = width if width != None else size[0]
        else:
            raise ValueError("Pass 1D pixels list or image in argument 'pixels_or_image'.")

    
    def _get_index(self, xy):
        if isinstance(xy, int):
            return xy
        if isinstance(xy, tuple) and len(xy) == 2:
            x, y = xy
            return y * self.width + x
        raise IndexError()

    __getitem__ = lambda self, xy: self.data.__getitem__(self._get_index(xy))
    __setitem__ = lambda self, xy, value: self.data.__setitem__(self._get_index(xy), value)


def thresholding(img, at):
    result = Image.new('1', img.size)
    result.putdata(list(map(lambda x: int(x >= at), img.getdata())))
    return result

def histogram(img):
    result = [0] * 256
    for p in img.getdata():
        result[p] += 1
    return result

def draw_histogram(result):
    height = max(result) * 4 // 3
    result_img = Image.new('1', (256, height))

    result_data = Pixels2D([1] * 256 * height, width=256)
    for x, h in enumerate(result):
        for y in range(h):
            result_data[x, height - 1 - y] = 0

    result_img.putdata(result_data.data)
    return result_img.resize((height, height))

def connected_components(img_bin): # A Space-Efficient Two-Pass Algorithm That Uses a Local Equivalence Table
    pixels = Pixels2D(img_bin)
    labels = []
    pixels_label = [[-1] * img.width for h in range(img.height)]

    for y in range(img.height):
        for x in range(img.width): # first pass
            if pixels[x, y] != 1:
                continue

            result_label = -1
            if x > 0 and pixels_label[y][x-1] != -1:
                result_label = pixels_label[y][x-1]
            if y > 0 and pixels_label[y-1][x] != -1:
                _result = pixels_label[y-1][x]

                if result_label != -1 and result_label != _result: # Local Equivalence Table
                    for _x, _y in labels[result_label]: # second pass
                        pixels_label[_y][_x] = _result
                    labels[_result] += labels[result_label]
                    labels[result_label] = None
                
                result_label = _result

            if result_label == -1:
                result_label = len(labels)
                labels.append([(x, y)])
            else:
                labels[result_label].append((x, y))

            pixels_label[y][x] = result_label

    return filter(lambda x: type(x)==list and len(x)>=500, labels)

def draw_rectangle(img, left, right, top, bottom, color):
    # Draw top & bottom
    for x in range(left, right + 1):
        for i in range(3):
            img.putpixel((x, top+i), color)
            img.putpixel((x, bottom-i), color)

    # Draw left & right
    for y in range(top, bottom + 1):
        for i in range(3):
            img.putpixel((left+i, y), color)
            img.putpixel((right-i, y), color)

def draw_centroid(img, pos, color):
    x, y = pos
    width, height = 11, 11
    for r in range(-width // 2, width // 2 + 1):
        for i in range(-1, 2):
            img.putpixel((x + r, y+i), color)
    for r in range(-height // 2, height // 2 + 1):
        for i in range(-1, 2):
            img.putpixel((x+i, y + r), color)

if __name__ == '__main__':
    img = Image.open('lena.bmp')

    # 1
    img_bin = thresholding(img, 128)
    img_bin.save('binary_image_threshold_at_128.bmp')

    # 2
    draw_histogram(histogram(img)).save('histogram.bmp')

    # 3
    #RAINBOW = ((255, 0, 0), (255, 127, 0), (255, 255, 0), (0, 255, 0), (100, 100, 255), (127, 0, 255), (255, 0, 255))

    img_rec = Image.new('RGB', img.size)
    img_rec.putdata(list(map(lambda p: (p*255, p*255, p*255), img_bin.getdata())))
    #colors = iter(RAINBOW)
    for component in connected_components(img_bin):
        #color = next(colors)
        #fill_color = tuple(map(lambda c: int(c * 0.4), color))

        (left, top), (right, bottom) = component[0], component[0]
        centroid_x, centroid_y = 0, 0
        for x, y in component:
            if x < left:
                left = x
            if x > right:
                right = x
            if y < top:
                top = y
            if y > bottom:
                bottom = y
            #img_rec.putpixel((x, y), fill_color)
            centroid_x += x
            centroid_y += y

        draw_rectangle(img_rec, left, right, top, bottom, (0, 0, 255))
        draw_centroid(img_rec, (centroid_x // len(component), centroid_y // len(component)), (255, 0, 0))
    img_rec.save('connected_components.bmp')
