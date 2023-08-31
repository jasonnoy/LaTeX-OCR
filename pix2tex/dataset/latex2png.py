# mostly taken from http://code.google.com/p/latexmath2png/
# install preview.sty
import argparse
import os
import re
import sys
import io
import glob
import tempfile
import shlex
import subprocess
import traceback
import random
from io import BytesIO
from PIL import Image
from textimg_gen import augment_image
import matplotlib.pyplot as plt
import webdataset as wds


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\usepackage{amsmath}
\setmathfont{%s}
\begin{document}
\thispagestyle{empty}
%s
\end{document}
'''

    def __init__(self, math, dpi=250, font='latinmodern-math.otf'):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        self.math = math
        self.dpi = dpi
        self.font = font
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        workdir = tempfile.gettempdir()
        fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
        # print(self.BASE % (self.font, self.math))
        with os.fdopen(fd, 'w+') as f:
            document = self.BASE % (self.font, '\n'.join(self.math))
            # print(document)
            f.write(document)

        png, error_index = self.convert_file(
            texfile, workdir, return_bytes=return_bytes)
        # if os.path.exists(texfile):
        #     try:
        #         os.remove(texfile)
        #     except PermissionError:
        #         pass
        return png, error_index

    def convert_file(self, infile, workdir, return_bytes=False):
        infile = infile.replace('\\', '/')
        print("infile", infile)
        # Generate the PDF file
        #  not stop on error line, but return error line index,index start from 1
        cmd = 'xelatex -interaction=nonstopmode -output-directory %s %s' % (
            workdir.replace('\\', '/'), infile)
        p = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
        sout, serr = p.communicate()
        print("sout:", sout)
        # extract error line from sout
        error_index, _ = extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
        # extract success rendered equation
        if error_index != []:
            # offset index start from 0, same as self.math
            error_index = [int(_) - self.prefix_line - 1 for _ in error_index]
        # Convert the PDF file to PNG's
        pdffile = infile.replace('.tex', '.pdf')
        result, _ = extract(
            text=sout, expression="Output written on %s \((\d+)? page" % pdffile)
        if int(result[0]) != len(self.math):
            raise Exception(
                'xelatex rendering error, generated %d formula\'s page, but the total number of formulas is %d.' % (
                    int(result[0]), len(self.math)))
        pngfile = os.path.join(workdir, infile.replace('.tex', '.png'))

        cmd = 'convert -density %i -colorspace gray %s -quality 90 %s' % (
            self.dpi,
            pdffile,
            pngfile,
        )  # -bg Transparent -z 9
        if sys.platform == 'win32':
            cmd = 'magick ' + cmd
        p = subprocess.Popen(
            shlex.split(cmd),
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        sout, serr = p.communicate()
        if p.returncode != 0:
            raise Exception('PDFpng error', serr, cmd, os.path.exists(
                pdffile), os.path.exists(infile))
        if return_bytes:
            if len(self.math) > 1:
                png = [open(pngfile.replace('.png', '') + '-%i.png' %
                            i, 'rb').read() for i in range(len(self.math))]
            else:
                png = [open(pngfile.replace(
                    '.png', '') + '.png', 'rb').read()]
        else:
            # return path
            if len(self.math) > 1:
                png = [(pngfile.replace('.png', '') + '-%i.png' % i)
                       for i in range(len(self.math))]
            else:
                png = [(pngfile.replace('.png', '') + '.png')]
        basefile = infile.replace('.tex', '')
        tempext = ['.aux', '.pdf', '.log']
        if return_bytes:
            ims = glob.glob(basefile + '*.png')
            # for im in ims:
            #     os.remove(im)
        for te in tempext:
            tempfile = basefile + te
            # if os.path.exists(tempfile):
            #     os.remove(tempfile)
        return png, error_index


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def tex2pil(tex, return_error_index=False, **kwargs):
    if type(tex) != list:
        tex = [tex]
    pngs, error_index = Latex(tex, **kwargs).write(return_bytes=True)
    images = [Image.open(io.BytesIO(d)).convert('RGB') for d in pngs]
    return (images, error_index) if return_error_index else images


def extract(text, expression=None):
    """extract text from text by regular expression

    Args:
        text (str): input text
        expression (str, optional): regular expression. Defaults to None.

    Returns:
        str: extracted text
    """
    try:
        pattern = re.compile(expression)
        results = re.findall(pattern, text)
        return results, True if len(results) != 0 else False
    except Exception:
        traceback.print_exc()


def formula2img(str_latex, out_file, img_size=(5, 3), font_size=16):
    fig = plt.figure(figsize=img_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, str_latex, fontsize=font_size, verticalalignment='center', horizontalalignment='center')
    plt.savefig(out_file)


def formula2pil(str_latex, img_size=(10, 1), font_size=16):
    from PIL import Image

    def fig2img(fig):
        '''
        matplotlib.figure.Figure转为PIL image
        '''
        fig.canvas.draw()
        w, h = fig.canvas.get_width_height()
        # 将Image.frombytes替换为Image.frombuffer,图像会倒置
        img = Image.frombytes('RGB', (w, h), fig.canvas.tostring_rgb())
        return img

    fig = plt.figure(figsize=img_size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)
    ax.set_xticks([])
    ax.set_yticks([])
    plt.text(0.5, 0.5, str_latex, fontsize=font_size, verticalalignment='center', horizontalalignment='center')
    # Image.open(io.BytesIO(d))
    # plt.savefig(out_file)
    plt.savefig("test.png")
    byte_io = BytesIO()
    plt.savefig(byte_io)
    return Image.open(byte_io)


def tex2pil_with_augment(tex, use_xelatex=True, **kwargs):
    if use_xelatex:
        pil_imgs = tex2pil(tex, **kwargs)
    else:
        pil_imgs = [formula2pil(tex)]
    avail_background_colors = [
        'white', 'black', 'red', 'green', 'blue', 'yellow', 'cyan', 'magenta', 'gray', 'grey', 'maroon', 'navy',
        'olive', 'purple', 'silver', 'teal'
    ]
    rotation_angle = (-20, 20)
    tilt_angle = (-15, 15)
    alphas = (0.95, 1.05)
    betas = (0, 0)
    background_color = 'white'
    aug_imgs = []
    for image in pil_imgs:
        if random.random() < 0.25:
            background_color = random.choice(avail_background_colors)
        if random.random() < 0.5:
            rotation_angle, tilt_angle = random.uniform(*rotation_angle), random.uniform(*tilt_angle)
        else:
            rotation_angle, tilt_angle = 0, 0
        if random.random() < 0.5:
            alpha, beta = random.uniform(*alphas), random.randint(*betas)
        else:
            alpha, beta = 1, 0
        aug_image = augment_image(image, rotation_angle, tilt_angle, alpha, beta, background_color=background_color)
        aug_imgs.append(aug_image)
    return aug_imgs


def get_img_bytes(img, add_aug=True):
    # 将PIL图像转换为字节
    byte_io = BytesIO()
    if add_aug:
        save_quality = int(45 + random.random() * 40)
    else:
        save_quality = 75
    img.save(byte_io, format='JPEG', quality=save_quality)
    img_bytes = byte_io.getvalue()
    return img_bytes


def process_texes(texes, base_format="random", use_xelatex=True):
    format_base = {"equation": r"\begin{equation} \begin{aligned} %s \end{aligned} %s \end{equation}",
                   "displaymath": r"\begin{displaymath} \begin{aligned} %s \end{aligned} \end{displaymath}",
                   "normal": r"$%s$"}
    if base_format == "random":
        base_format = random.choice(list(format_base.keys()))
    tex_str = r" \\ ".join(texes)
    res_tex_str = tex_str

    if "{align" in tex_str:
        tex_str = tex_str.replace(r"\begin{align}", "").replace(r"\end{align}", "").replace(r"\begin{aligned}", "").replace(r"\end{aligned}", "").replace(r"\begin{align*}", "").replace(r"\end{align*}", "")
    base_str = format_base[base_format]
    if base_format == "equation":
        if random.random() < 0.5:
            nonumber = r"\nonumber"
        else:
            nonumber = ""
        tex_str = base_str % (tex_str, nonumber)
    else:
        tex_str = base_str % tex_str

    aug_imgs = tex2pil_with_augment(tex_str, use_xelatex=use_xelatex)
    return aug_imgs, res_tex_str


def check_validity(tex_str):
    filter_strs = [r"\hpic"]
    for f_str in filter_strs:
        if f_str in tex_str:
            return False
    return True


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--add_aug', type=bool, default=True)
    parser.add_argument('--debug', type=bool, default=False)

    args = parser.parse_args()

    tar_index = 0
    save_dir = os.path.join("/nxchinamobile2/shared/img_datasets/math_ocr/aminer_math", 'part-%03d' % args.rank)
    os.makedirs(save_dir, exist_ok=True)
    tar_path = os.path.join(save_dir, "%06d.tar" % tar_index)
    input_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_parts"
    input_file_path = os.path.join(input_path, f"part_{args.rank}.txt")
    sink = wds.TarWriter(tar_path)

    texes = []
    count = 0
    error_count = 0
    invalid_count = 0

    with open(input_file_path, "r") as f:
        for line in f:
            line = line.strip()
            if not check_validity(line):
                invalid_count += 1
                continue
            if r"\\" in line:
                try:
                    aug_imgs, tex_str = process_texes([line], use_xelatex=True)
                except Exception as e:
                    error_count += 1
                    continue
                if args.debug:
                    aug_imgs[0].save(f"./test/{count}.jpg")
                count += 1
                continue
            texes.append(line)
            if len(texes) >= 10 or random.random() < 0.15:
                try:
                    aug_imgs, tex_str = process_texes(texes, use_xelatex=True)
                except Exception as e:
                    error_count += 1
                    continue

                assert len(aug_imgs) == 1
                aug_img = aug_imgs[0]
                if args.debug:
                    aug_img.save(f"./test/{count}.jpg")
                texes = []

                # 将PIL图像转换为字节
                byte_io = BytesIO()
                if args.add_aug:
                    save_quality = int(45 + random.random() * 40)
                else:
                    save_quality = 75
                aug_img.save(byte_io, format='JPEG', quality=save_quality)
                img_bytes = byte_io.getvalue()

                # 将图像和文本保存到webdataset中
                key = "%03d%07d" % (args.rank, count)
                sink.write({
                    "__key__": key,
                    "jpg": img_bytes,
                    "txt": tex_str,
                })

                count += 1
                if error_count/count > 0.25:
                    print(f"Warning: more than 25% of images failed to process. ({error_count}/{count})")
                # 如果达到10,000个样本，分割.tar文件
                if count % 10000 == 0:
                    sink.close()
                    tar_index += 1
                    tar_path = os.path.join(args.save_dir, "%06d.tar" % tar_index)
                    sink = wds.TarWriter(tar_path)
                    print(f'rank {args.rank} finish {tar_path} error count: {error_count}.', flush=True)
    if count % 10000 != 0:
        sink.close()
        tar_index += 1
        tar_path = os.path.join(args.save_dir, "%06d.tar" % tar_index)
        print(f'rank {args.rank} finish {tar_path} error count: {error_count}.', flush=True)
