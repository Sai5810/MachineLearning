import os

from PIL import Image


def main():
    alphas = []
    if not os.path.exists("./alphas"):
        os.makedirs("./alphas")
    for filename in os.listdir("./realset"):
        f = os.path.join("./realset", filename)
        if os.path.isfile(f):
            alphas.append(Image.open(f).split()[-1])
            alphas[-1].save("./alphas/" + filename)
            print("./alphas/" + filename)
    print(1)



if __name__ == "__main__":
    main()
