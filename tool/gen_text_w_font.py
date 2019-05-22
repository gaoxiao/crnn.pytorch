import os
import random
import uuid
from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

home = str(Path.home())

data_dir = os.path.join(home, 'data/ocr_data/font_gen/img')
gt_dir = os.path.join(home, 'data/ocr_data/font_gen/gt')
font_dir = os.path.join(home, 'data/fonts/')
if not os.path.isdir(data_dir):
    os.makedirs(data_dir)
if not os.path.isdir(gt_dir):
    os.makedirs(gt_dir)

# record_file = '{}/{}.txt'.format(gt_dir, uuid.uuid4())
record_file = '{}/{}.txt'.format(gt_dir, 'ALL')

zoom_factor = 1


def random_bg_color():
    f = random.randint
    return f(200, 255), f(200, 255), f(200, 255)


def random_fg_color():
    f = random.randint
    return f(0, 100), f(0, 100), f(0, 100)


def get_fonts():
    fonts = []

    path = os.path.join(font_dir, 'Schoolbell.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)

    # for f in os.listdir(font_dir):
    #     if f.endswith('ttf'):
    #         path = os.path.join(font_dir, f)
    #         font = ImageFont.truetype(path, size=18 * zoom_factor)
    #         fonts.append(font)
    return fonts


def write_record(record_list):
    with open(record_file, 'a') as res:
        for idx, word in record_list:
            res.write('{},{}\n'.format(idx, word))
    del record_list[:]


def main():
    words = []
    with open('google-10000-english-usa.txt') as f:
        for l in f:
            l = l.strip()
            if len(l) > 8:
                continue
            words.append(l)

    fonts = get_fonts()

    size = 100000
    record = []

    for idx in tqdm(range(size)):
        id_ = uuid.uuid4()
        image = Image.new("RGB", (100 * zoom_factor, 32 * zoom_factor), random_bg_color())
        draw = ImageDraw.Draw(image)
        font = random.choice(fonts)

        # w1, w2 = random.choice(words), random.choice(words)
        # text = '{} {}'.format(w1, w2)

        text = random.choice(words)

        draw.text((5, 5), text, fill=random_fg_color(), font=font)
        # draw.text((11, 11), text, fill=random_fg_color(), font=font)
        # draw.text((12, 12), text, fill=random_fg_color(), font=font)
        image.save('{}/{}.png'.format(data_dir, id_))
        record.append((id_, text))

        if idx % 100 == 0:
            write_record(record)

    write_record(record)


if __name__ == '__main__':
    main()
