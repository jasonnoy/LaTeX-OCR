# modified from https://tex.stackexchange.com/a/521639

import argparse
import re
import logging
from collections import Counter
import time
from functools import partial


class DemacroError(Exception):
    pass


def main():
    args = parse_command_line()
    data = read(args.input)
    data = pydemacro(data)
    if args.output is not None:
        write(args.output, data)
    else:
        print(data)


def parse_command_line():
    parser = argparse.ArgumentParser(description='Replace \\def with \\newcommand where possible.')
    parser.add_argument('input', help='TeX input file with \\def')
    parser.add_argument('--output', '-o', default=None, help='TeX output file with \\newcommand')
    return parser.parse_args()


def read(path):
    with open(path, mode='r') as handle:
        return handle.read()


def bracket_replace(string: str) -> str:
    '''
    replaces all layered brackets with special symbols
    '''
    layer = 0
    out = list(string)
    for i, c in enumerate(out):
        if c == '{':
            if layer > 0:
                out[i] = 'Ḋ'
            layer += 1
        elif c == '}':
            layer -= 1
            if layer > 0:
                out[i] = 'Ḍ'
    return ''.join(out)


def undo_bracket_replace(string):
    return string.replace('Ḋ', '{').replace('Ḍ', '}')


def sweep(t, cmds):
    num_matches = 0
    for c in cmds:
        nargs = int(c[1][1]) if c[1] != r'' else 0
        optional = c[2] != r''
        if nargs == 0:
            num_matches += len(re.findall(r'\\%s([\W_^\dĊ])' % c[0], t))
            if num_matches > 0:
                t = re.sub(r'\\%s([\W_^\dĊ])' % c[0], r'%s\1' % c[-1].replace('\\', r'\\'), t)
        else:
            matches = re.findall(r'(\\%s(?:\[(.+?)\])?' % c[0]+r'{(.+?)}'*(nargs-(1 if optional else 0))+r')', t)
            num_matches += len(matches)
            for i, m in enumerate(matches):
                r = c[-1]
                if m[1] == r'':
                    matches[i] = (m[0], c[2][1:-1], *m[2:])
                for j in range(1, nargs+1):
                    r = r.replace(r'#%i' % j, matches[i][j+int(not optional)])
                t = t.replace(matches[i][0], r)
    return t, num_matches


# def unfold(t):
#     #t = queue.get()
#     t = t.replace('\n', 'Ċ')
#     t = bracket_replace(t)
#     commands_pattern = r'\\(?:re)?newcommand\*?{\\(.+?)}[\sĊ]*(\[\d\])?[\sĊ]*(\[.+?\])?[\sĊ]*{(.*?)}'
#     cmds = re.findall(commands_pattern, t)
#     t = re.sub(r'(?<!\\)'+commands_pattern, 'Ċ', t)
#     cmds = sorted(cmds, key=lambda x: len(x[0]))
#     cmd_names = Counter([c[0] for c in cmds])
#     for i in reversed(range(len(cmds))):
#         if cmd_names[cmds[i][0]] > 1:
#             # something went wrong here. No multiple definitions allowed
#             del cmds[i]
#         elif '\\newcommand' in cmds[i][-1]:
#             logging.debug("Command recognition pattern didn't work properly. %s" % (undo_bracket_replace(cmds[i][-1])))
#             del cmds[i]
#     start = time.time()
#     try:
#         for i in range(10):
#             # check for up to 10 nested commands
#             if i > 0:
#                 t = bracket_replace(t)
#             t, N = sweep(t, cmds)
#             if time.time()-start > 5: # not optimal. more sophisticated methods didnt work or are slow
#                 raise TimeoutError
#             t = undo_bracket_replace(t)
#             if N == 0 or i == 9:
#                 #print("Needed %i iterations to demacro" % (i+1))
#                 break
#             elif N > 4000:
#                 raise ValueError("Too many matches. Processing would take too long.")
#     except ValueError:
#         pass
#     except TimeoutError:
#         pass
#     except re.error as e:
#         raise DemacroError(e)
#     t = remove_labels(t.replace('Ċ', '\n'))
#     # queue.put(t)
#     return t


def pydemacro(t: str) -> str:
    r"""Replaces all occurences of newly defined Latex commands in a document.
    Can replace `\newcommand`, `\def` and `\let` definitions in the code.

    Args:
        t (str): Latex document

    Returns:
        str: Document without custom commands
    """
    t = re.sub(r'%.*', '', t)
    return re.sub('\n+', '\n', re.sub(r'(?<!\\)%.*\n', '\n', t))


def replace(match):
    result = '\n' + match.group(0)
    return result


def dict_replace(match, dic):
    return dic[match.group(0)]


def sub_mods(tex, data):
    command_pattern = r'\\(?:re)?newcommand{(.+?)}{(.+)}'
    let_pattern = r'\\let(\\[a-zA-Z]+)\s*=(\\?\w+)*'
    def_pattern = r'\\def(\\[a-zA-Z]+){(.+)}'
    tex = re.sub(
        r'(\\let|\\def|\\(?:re)?newcommand)',
        replace,
        tex,
    )
    command_pairs = re.findall(command_pattern, tex)
    let_pairs = re.findall(let_pattern, tex)
    def_pairs = re.findall(def_pattern, tex)
    all_pairs = command_pairs + let_pairs + def_pairs
    pair_dict = dict(all_pairs)
    pass_patterns = [r"\b", r"\be", r"\beg", r"\begi", r"\begin", r"\end", r"\en", r"\e"]
    for p in pass_patterns:
        if p in pair_dict:
            pair_dict.pop(p)
    pattern = list(map(lambda s: '{!r}'.format(s), pair_dict.keys()))
    pattern = [p[1:-1] for p in pattern]
    data = re.sub(
        r'|'.join(pattern),
        partial(dict_replace, dic=pair_dict),
        data
    )
    return data


def write(path, data):
    with open(path, mode='w') as handle:
        handle.write(data)

    print('=> File written: {0}'.format(path))


if __name__ == '__main__':
    # main()
    tex = r"""\def\half{{\textstyle{1\over2}}}
\let\bl=\bigl
{\bl H}^{-1/4}"""

# r"""\let\la=\label \let\ci=\cite \let\re=\ref
# \let\se=\section \let\sse=\subsection \let\ssse=\subsubsection
# \def\nn{\nonumber} \def\bd{\begin{document}} \def\ed{\end{document}}
# \def\ds{\documentstyle} \let\fr=\frac \let\bl=\bigl \let\br=\bigr
# \let\Br=\Bigr \let\Bl=\Bigl
# \let\bm=\bibitem
# \let\na=\nabla
# \let\pa=\partial \let\ov=\overline
#
# %Kelly's shorthands
# \newcommand{\be}{\begin{equation}}
# \newcommand{\ee}{\end{equation}}
# \newcommand{\bea}{\begin{eqnarray}}
# \newcommand{\eea}{\end{eqnarray}}
# \newcommand{\ba}{\begin{array}}
# \newcommand{\ea}{\end{array}}"""
    print("res:\n", pydemacro(tex))
