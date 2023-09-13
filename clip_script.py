import cv2
import numpy as np

def split_image(img_path, split_num):
    img = cv2.imread(img_path)
    height, width, channels = img.shape

    step_w = width // split_num
    step_h = height // split_num

    tiles = []

    for i in range(split_num):
        for j in range(split_num):
            left = i * step_w
            upper = j * step_h
            right = left + step_w
            lower = upper + step_h

            tile = img[upper:lower, left:right]
            tiles.append(tile)

    return tiles

def upscale_tile(tile, scale_factor):
    return cv2.resize(tile, None, fx=scale_factor, fy=scale_factor, interpolation=cv2.INTER_CUBIC)

def enhance_contrast_brightness(image, alpha, beta):
    adjusted = cv2.convertScaleAbs(image, alpha=alpha, beta=beta)
    return adjusted

def main(img_path, split_num=8, scale_factor=8,alpha=1, beta=0):
    tiles = split_image(img_path, split_num)
    
    for idx, tile in enumerate(tiles):
        upscaled_tile = upscale_tile(tile, scale_factor)
        enhanced_tile = enhance_contrast_brightness(upscaled_tile, alpha, beta)
        
        cv2.imwrite(f"upscaled_enhanced/upscaled_enhanced_{idx}.png", enhanced_tile)

if __name__ == "__main__":
    main("eye.jpg")
