from retry import retry
import argparse
import glob
import os
import re
import sys
import io
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
from tqdm import tqdm
import matplotlib.font_manager as mfm
from matplotlib import mathtext
import numpy as np


class Latex:
    BASE = r'''
\documentclass[varwidth]{standalone}
\usepackage{fontspec,unicode-math}
\usepackage[active,tightpage,displaymath,textmath]{preview}
\usepackage{amsmath}
%s
\begin{document}
\thispagestyle{empty}
\begin{tiny}
%s
\end{tiny}
\end{document}
'''

    def __init__(self, math, dpi=550, fonts=None, debug=False, **kwargs):
        '''takes list of math code. `returns each element as PNG with DPI=`dpi`'''
        if fonts is None:
            fonts = ['lmodern']
        self.math = math
        self.dpi = dpi
        self.fonts = fonts
        self.debug = debug
        self.prefix_line = self.BASE.split("\n").index(
            "%s")  # used for calculate error formula index

    def write(self, return_bytes=False):
        # inline = bool(re.match('^\$[^$]*\$$', self.math)) and False
        try:
            # workdir = tempfile.gettempdir()
            workdir = "/nxchinamobile2/shared/jjh/projects/LaTeX-OCR/temp"
            fd, texfile = tempfile.mkstemp('.tex', 'eq', workdir, True)
            font = random.choice(self.fonts)
            with os.fdopen(fd, 'w+') as f:
                document = self.BASE % (font, '\n'.join(self.math))
                f.write(document)

            pngs, error_index = self.convert_file(
                texfile, workdir, return_bytes=return_bytes, debug=self.debug)
            return pngs, error_index
        finally:
            if not self.debug:
                if os.path.exists(texfile):
                    try:
                        os.remove(texfile)
                    except PermissionError:
                        pass

    def convert_file(self, infile, workdir, return_bytes=False, debug=False):
        infile = infile.replace('\\', '/')
        try:
            # Generate the PDF file
            #  not stop on error line, but return error line index,index start from 1
            cmd = 'xelatex -interaction=batchmode -output-directory %s %s' % (
                workdir.replace('\\', '/'), infile)
            p = subprocess.Popen(
                shlex.split(cmd),
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            sout, serr = p.communicate()
            # extract error line from sout
            error_index, _ = extract(text=sout, expression=r"%s:(\d+)" % os.path.basename(infile))
            # extract success rendered equation
            if error_index != []:
                # offset index start from 0, same as self.math
                error_index = [int(_) - self.prefix_line - 1 for _ in error_index]
            # Convert the PDF file to PNG's
            pdffile = infile.replace('.tex', '.pdf')
            # print(sout)
            # result, _ = extract(
            #     text=sout, expression="Output written on %s \((\d+)? page" % pdffile)
            # print(result)
            # if not result:
            #     raise Exception(
            #         'xelatex rendering error, file: %s' % pdffile)
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
            return png, error_index
        finally:
            # Cleanup temporaries
            if not debug:
                basefile = infile.replace('.tex', '')
                tempext = ['.aux', '.pdf', '.log']
                if return_bytes:
                    ims = glob.glob(basefile + '*.png')
                    for im in ims:
                        os.remove(im)
                for te in tempext:
                    tempfile = basefile + te
                    if os.path.exists(tempfile):
                        os.remove(tempfile)


__cache = {}


def tex2png(eq, **kwargs):
    if not eq in __cache:
        __cache[eq] = Latex(eq, **kwargs).write(return_bytes=True)
    return __cache[eq]


def matplot_lex2pil(text, size=64, color=(0.1, 0.1, 0.1), out=None, **kwds):
    """LaTex数学公式转图片

        text        - 文本字符串，其中数学公式须包含在两个$符号之间
        size        - 字号，整型，默认64
        color       - 颜色，浮点型三元组，值域范围[0,1]，默认深黑色
        out         - 文件名，仅支持后缀名为.png的文件名。若维None，则返回PIL图像对象
        kwds        - 关键字参数
                        dpi         - 输出分辨率（每英寸像素数），默认72
                        family      - 系统支持的字体，None表示当前默认的字体
                        weight      - 笔画轻重，可选项包括：normal（默认）、light和bold
        """

    assert out is None or os.path.splitext(out)[1].lower() == '.png', '仅支持后缀名为.png的文件名'

    for key in kwds:
        if key not in ['dpi', 'family', 'weight']:
            raise KeyError('不支持的关键字参数：%s' % key)

    dpi = kwds.get('dpi', 72)
    family = kwds.get('family', None)
    weight = kwds.get('weight', 'normal')

    bfo = BytesIO()  # 创建二进制的类文件对象
    prop = mfm.FontProperties(family=family, size=size, weight=weight)
    mathtext.math_to_image(text, bfo, prop=prop, dpi=dpi)
    im = Image.open(bfo)

    r, g, b, a = im.split()
    r, g, b = 255 - np.array(r), 255 - np.array(g), 255 - np.array(b)
    a = r / 3 + g / 3 + b / 3
    r, g, b = r * color[0], g * color[1], b * color[2]

    im = np.dstack((r, g, b, a)).astype(np.uint8)
    im = Image.fromarray(im).convert('RGB')

    if out:
        im.save(out)
    return im


def tex2pil(tex_texts, return_error_index=False, use_xelatex=False, **kwargs):
    if use_xelatex:
        pngs, error_index = Latex(tex_texts, **kwargs).write(return_bytes=True)
        images = [Image.open(io.BytesIO(d)).convert('RGB') for d in pngs]

    else:
        tex = tex_texts[0]
        png = matplot_lex2pil(tex)
        images = [png]
        error_index = None

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


def is_str_close(a):
    """
    判断括号是否闭合
    """
    b = []
    flag = True
    for i in a:
        if i == "{" or i == "[" or i == "(":
            # 左边的括号加进去
            b.append(i)
        elif i == "}":
            # 遇到右边括号}弹出最后面的一个{
            if len(b) == 0 or b.pop() != "{":
                return False
        elif i == "]":
            # 遇到右边括号]弹出最后面的一个[
            if len(b) == 0 or b.pop() != "[":
                return False
        elif i == ")":
            # 遇到右边括号)弹出最后面的一个(
            if len(b) == 0 or b.pop() != "(":
                return False
    # 判断最后列表b里面的左边括号是否全部被弹出
    if len(b) != 0:
        flag = False
    return flag


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


def process_single_tex(tex):
    tex = tex.replace("&", "")
    if "{align" in tex:
        tex = tex.replace(r"\begin{align}", "").replace(r"\end{align}", "").replace(r"\begin{aligned}",
                                                                                            "").replace(
            r"\end{aligned}", "").replace(r"\begin{align*}", "").replace(r"\end{align*}", "")
    if "{gather" in tex:
        tex = tex.replace(r"\begin{gather}", "").replace(r"\end{gather}", "")
    return tex


def process_batch_tex(batch_texes, format_base, base_format):
    res_tex_strs = []
    tex_strs = []
    for texes in batch_texes:
        texes = [process_single_tex(tex) for tex in texes]
        res_tex_strs.extend(texes)
        tex_str = r" \\ ".join(texes)

        base_str = format_base[base_format]
        if base_format == "equation":
            if random.random() < 0.5:
                nonumber = r"\nonumber"
            else:
                nonumber = ""
            tex_str = base_str % (tex_str, nonumber)
        else:
            tex_str = base_str % tex_str
        tex_strs.append(tex_str)
    return tex_strs, res_tex_strs


def process_tex(tex, format_base, base_format):
    tex = process_single_tex(tex)
    base_str = format_base[base_format]
    if base_format == "equation":
        if random.random() < 0.5:
            nonumber = r"\nonumber"
        else:
            nonumber = ""
        tex_str = base_str % (tex, nonumber)
    else:
        tex_str = base_str % tex
    return tex_str, tex


def process_texes(tex, base_format="random", use_xelatex=False, **kwargs):
    tex = tex[0][0]
    format_base = {"equation": r"\begin{equation} \begin{aligned} %s \end{aligned} %s \end{equation}",
                   "displaymath": r"\begin{displaymath} \begin{aligned} %s \end{aligned} \end{displaymath}",
                   "normal": r"$%s$"}
    if base_format == "random":
        base_format = random.choice(list(format_base.keys()))

    # tex_lines, res_tex_strs = process_batch_tex(batch_texes, format_base, base_format)
    tex_line, res_tex_str = process_tex(tex, format_base, base_format)

    math_fonts = [r'\usepackage{concmath}', r'\usepackage{cmbright}',
                  r'\usepackage{unicode-math} \setmathfont{Erewhon Math} \setmainfont{Erewhon Regular}',
                  r'\usepackage[sfdefault,lining]{FiraSans} \usepackage[fakebold]{firamath-otf} \renewcommand*\oldstylenums[1]{{\firaoldstyle #1}}',
                  r'\usepackage[sfdefault,scaled=.85]{FiraSans} \usepackage{newtxsf}',
                  r'\usepackage[sfmath]{kpfonts} \renewcommand*\familydefault{\sfdefault}',
                  r'\usepackage{kpfonts}',
                  r'\usepackage{lmodern}',
                  r'\usepackage{libertinus}',
                  r'\usepackage{libertine} \usepackage{libertinust1math}',
                  r'\usepackage{lxfonts}',
                  r'\usepackage{newpxtext,newpxmath}',
                  r'\usepackage{newtxtext,newtxmath}',
                  r'\usepackage{mathptmx}',
                  r'\usepackage{notomath}',
                  r'\usepackage[sc]{mathpazo} \linespread{1.05}',
                  r'\usepackage{pxfonts}',
                  r'\usepackage[notext]{stix} \usepackage{step}',
                  r'\usepackage{txfonts}',
                  ]

    # aug_imgs = tex2pil_with_augment(tex_str, use_xelatex=use_xelatex, fonts=math_fonts)
    pil_imgs = tex2pil([tex_line], fonts=math_fonts, use_xelatex=use_xelatex, **kwargs)
    return pil_imgs, [res_tex_str]


def check_validity(tex_str):
    if not is_str_close(tex_str):
        return False
    if re.match(r"[A-Za-z\s]+$", tex_str):
        return False
    return True


def preprocess_line(string):
    string = re.sub(" +", " ", string)
    return string


# @retry(delay=0.1)
def convert_pil_to_bytes(aug_img):
    byte_io = BytesIO()
    aug_img.save(byte_io, format='RGB')
    return byte_io.getvalue()


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--local_rank', type=int, default=0)
    parser.add_argument('--world_size', type=int, default=1)
    parser.add_argument('--rank', type=int, default=0)
    parser.add_argument('--batch_size', type=int, default=1, help="num of pages per xelatex compilation")
    parser.add_argument('--add_aug', type=bool, default=True)
    parser.add_argument('--debug', action='store_true')

    args = parser.parse_args()

    if args.debug:
        print("=================debug mode=================")

    tar_index = 0
    save_dir = os.path.join("/nxchinamobile2/shared/img_datasets/math_ocr/aminer_math", 'part-%03d' % args.rank)
    os.makedirs(save_dir, exist_ok=True)
    tar_path = os.path.join(save_dir, "%06d.tar" % tar_index)
    input_path = "/nxchinamobile2/shared/img_datasets/math_ocr/AMiner/all_parts_new"
    # input_file_path = os.path.join(input_path, f"part_{args.rank}.txt")
    input_file_path = os.path.join(input_path, "part_1717.txt")
    sink = wds.TarWriter(tar_path)

    texes = []
    batch = []
    count = 0
    error_count = 0
    invalid_count = 0

    with open(input_file_path, "r") as f:
        lines = tqdm(f.readlines())
        for line in lines:
            line = line.strip()
            if not check_validity(line):
                invalid_count += 1
                lines.set_description(f'rank {args.rank}')
                lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                continue
            line = preprocess_line(line)

            if r"\\" in line:
                batch.append([line])
            else:
                texes.append(line)
                if len(texes) >= 1 or random.random() < 0.15:
                    batch.append(texes)
                    texes = []

            if len(batch) >= args.batch_size:
                if args.debug:
                    pil_imgs, tex_strs = process_texes(batch, use_xelatex=False, debug=args.debug,
                                                       batch_size=args.batch_size)
                else:
                    try:
                        pil_imgs, tex_strs = process_texes(batch, use_xelatex=False, debug=args.debug,
                                                           batch_size=args.batch_size)
                    except Exception as e:
                        error_count += len(batch)
                        lines.set_description(f'rank {args.rank}')
                        lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                        continue

                if len(pil_imgs) != len(batch):
                    print(f"rank: {args.rank}, batch error")
                    error_count += len(batch)
                    lines.set_description(f'rank {args.rank}')
                    lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
                    continue

                batch = []

                for pil_img, tex_str in zip(pil_imgs, tex_strs):
                    if args.debug:
                        pil_img.save(f"./test/{count}.jpg")
                    # 将PIL图像转换为字节
                    img_bytes = convert_pil_to_bytes(pil_img)
                    # 将图像和文本保存到webdataset中
                    key = "%03d%07d" % (args.rank, count)
                    sink.write({
                        "__key__": key,
                        "jpg": img_bytes,
                        "txt": tex_str,
                    })
                    count += 1
                    lines.set_description(f'rank {args.rank}')
                    lines.set_postfix(count=count, invalid_count=invalid_count, error_count=error_count)
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
