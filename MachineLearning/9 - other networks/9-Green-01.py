from torchvision import datasets


def main():
    trainset = datasets.CIFAR10(root='./data')
    for idx, (img, cidx) in enumerate(trainset):
        if cidx == 3:
            print(f"Index: {idx}")
            img.resize((320, 320)).show()
            break


if __name__ == "__main__":
    main()
