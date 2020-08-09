from PIL import Image
import os
import math
import glob

def pad_img(img, width, height):
    old_size = img.size
    new_size = (height, width)

    new_img = Image.new("RGB", new_size)
    new_img.paste(img, (math.floor((new_size[0] - old_size[0]) / 2),
                        math.ceil((new_size[1] - old_size[1]) / 2)))
    return new_img
    
def crop(img,height,width):
    imgwidth, imgheight = img.size
    # iterate over width and height creating the necessary patches
    for i in range(imgheight // height):
        for j in range(imgwidth // width):
            box = (j * width, i * height, (j+1) * width, (i+1) * height)
            yield img.crop(box)

def join_patches(patches, num_patches_X, num_patches_Y):
    final_width = 123
    final_height = 123
    
    # for patch_X in range(num_patches_X):
    #     final_width += patches[patch_X * patch_Y].size[0]
    # for patch_Y in range(num_patches_Y):
    #     final_height += patches[patch_X * patch_Y].size[1]
    
    out_img = Image.new('RGB', (final_height, final_width), 255)
    
    for patch_Y in range(num_patches_Y):
        for patch_X in range(num_patches_X):
            out_img.paste(patches[patch_X * patch_Y], (patch_X, patch_Y))

    return out_img
    

if __name__=='__main__':
    # open input image and set add padding
    img = Image.open('image.jpg')
    padded_img = pad_img(img, 123, 123) #123 is 3*41
    patch_height = 41
    patch_width = 41
    start_patch = 0

    # divide into patches
    for k, patch in enumerate(crop(padded_img, patch_height, patch_width), start_patch):
        out_img = Image.new('RGB', (patch_height, patch_width), 255)
        out_img.paste(patch)
        path=os.path.join('.\\patches\\',"IMG-%s.png" % k)
        out_img.save(path)
    
    # recombine every patch
    image_list = []
    for filename in glob.glob('patches/*.png'): #assuming gif
        patch = Image.open(filename)
        image_list.append(patch)
    
    merged_patches = join_patches(image_list, 3, 3)
    merged_patches.save("merged.png")