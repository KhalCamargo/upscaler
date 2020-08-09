from PIL import Image
import os
import math
import glob

def pad_img(img, width, height):
    ''' This function will pad given image, using the final width and height as 
    parameters to create the new image, and pasting the original image inside it'''
    old_size = img.size
    new_size = (height, width)

    new_img = Image.new("RGB", new_size)
    new_img.paste(img, (math.floor((new_size[0] - old_size[0]) / 2),
                        math.ceil((new_size[1] - old_size[1]) / 2)))
    return new_img
    
def crop(img,height,width):
    ''' This function will create a iterable mask (yield) containing the output patches of given image,
     sing the given width and height as the width and height of each patch'''
    img_width, img_height = img.size
    
    for i in range(img_height // height):
        for j in range(img_width // width):
            mask = (j * width, i * height, (j+1) * width, (i+1) * height)
            yield img.crop(mask)

def join_patches(patches, num_patches_X, num_patches_Y):
    ''' This function will join all the patches from the patch list together in a squared image'''
    final_width = num_patches_X * patches[0].size[0]
    final_height = num_patches_Y * patches[0].size[1]
    
    out_img = Image.new('RGB', (final_height, final_width), 0)
    
    k = 0
    for patch_Y in range(num_patches_Y):
        for patch_X in range(num_patches_X):
            patch = patches[k]
            w = patch.size[0]
            h = patch.size[1]
            out_img.paste(patch, (patch_X * w, patch_Y * h))
            k += 1

    return out_img
    

if __name__=='__main__':
    # open input image and set add padding
    img = Image.open('image2.jpg')
    padded_img = pad_img(img, 123, 123) #123 is 3 * 41
    patch_height = 41
    patch_width = 41
    start_patch = 0 # enumerate() needs this shit

    # divide into patches
    for k, patch in enumerate(crop(padded_img, patch_height, patch_width), start_patch):
        out_img = Image.new('RGB', (patch_height, patch_width), 0)
        out_img.paste(patch)
        os.makedirs(os.path.dirname('.\\patches\\'), exist_ok=True)
        path=os.path.join('.\\patches\\',"IMG-%s.jpg" % k)
        out_img.save(path)
    
    # recombine every patch
    image_list = []
    for filename in glob.glob('patches/*.jpg'): #assuming gif
        patch = Image.open(filename)
        image_list.append(patch)
    
    merged_patches = join_patches(image_list, 3, 3)
    merged_patches.save("merged.jpg")