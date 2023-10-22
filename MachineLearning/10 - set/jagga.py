import re
import shutil
from os import walk
import os


def main():
    imgs = next(walk("./set"), (None, None, []))[2]
    color = {"g": [], "p": [], "r": []}
    num = {"1": [], "2": [], "3": []}
    shape = {"d": [], "o": [], "s": []}
    shade = {"f": [], "o": [], "s": []}
    names = ["color", "shade", "num", "shape"]
    dirs = [color, shade, num, shape]
    for i in imgs:
        for idx, d in enumerate(dirs):
            for k in d:
                reg = "....\.png"
                reg2 = reg[:idx] + k + reg[idx + 1:]
                print(reg2)
                if re.search(reg2, i):
                    dirs[idx][k] = dirs[idx][k] + [i]
                    break
    for idx, ctg in enumerate(names):
        if not os.path.exists("./" + ctg):
            os.makedirs("./" + ctg)
            for k in dirs[idx]:
                if not os.path.exists("./" + ctg + "/" + k):
                    os.makedirs("./" + ctg + "/" + k)
    for d, n in zip(dirs, names):
        for k, val in d.items():
            for v in val:
                shutil.copy("./set/" + v, "./" + n + "/" + k)


if __name__ == "__main__":
    main()
