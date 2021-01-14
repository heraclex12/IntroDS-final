import matplotlib.image as mpimg
import numpy as np
import pandas as pd
import csv
import os
from joblib import Parallel, delayed


if __name__ == '__main__':
    img_path = 'poster_final/'
    def parse_img(filename):
        img = mpimg.imread(img_path + filename).reshape(-1)
        hist_img, _ = np.histogram(img, bins=256)
        return (filename, hist_img/len(img))

    imgs = Parallel(n_jobs=-1)(delayed(parse_img)(filename) for filename in os.listdir(img_path))
    with open('poster_img_pixels.csv', 'w') as img_file:
        writer = csv.writer(img_file)
        writer.writerow(['posterImagePath', 'imagePixels'])
        for img in imgs:
            writer.writerow(['poster_final/' + img[0], ' '.join(map(str, img[1]))])
