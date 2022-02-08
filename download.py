import gdown
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Supply file type")
        exit()

    arg = sys.argv[1]
    if arg == "train":
        url = "https://drive.google.com/uc?id=1Yy03NNeiDzrWz28dHHu_xHMAlpZJpXIf"
    elif arg == "test":
        url = "https://drive.google.com/uc?id=1vPBAXoqgOtfUPFK0bu7-1rF-lWLVpsYo"
    else:
        pass

    output = "r3d.zip"
    gdown.download(url, output, quiet=False)