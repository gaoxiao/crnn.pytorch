import os
import random
import shutil
import uuid
from pathlib import Path

from PIL import Image, ImageFont, ImageDraw
from tqdm import tqdm

home = str(Path.home())

data_dir = os.path.join(home, 'data/ocr_data/font_gen1/img')
gt_dir = os.path.join(home, 'data/ocr_data/font_gen1/gt')
font_dir = 'fonts'

shutil.rmtree(data_dir, ignore_errors=True)
shutil.rmtree(gt_dir, ignore_errors=True)

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

    path = os.path.join(font_dir, 'Almost Cartoon.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'AckiPreschool.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'BPchildfatty.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Children.ttf')
    font = ImageFont.truetype(path, size=16 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Children Sans.ttf')
    font = ImageFont.truetype(path, size=16 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Childrens Party Personal Use.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'ComingSoon.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'doves.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'elizajane.ttf')
    font = ImageFont.truetype(path, size=12 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Gruenewald VA normal.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'KatetheGreat.ttf')
    font = ImageFont.truetype(path, size=14 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Kindergarden.ttf')
    font = ImageFont.truetype(path, size=14 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'PizzaismyFAVORITE.ttf')
    font = ImageFont.truetype(path, size=14 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Schoolbell.ttf')
    font = ImageFont.truetype(path, size=18 * zoom_factor)
    fonts.append(font)
    path = os.path.join(font_dir, 'Tafelschrift.ttf')
    font = ImageFont.truetype(path, size=14 * zoom_factor)
    fonts.append(font)
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

    size = 10
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
