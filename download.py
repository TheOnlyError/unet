import gdown
import sys

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Supply file type")
        exit()

    arg = sys.argv[1]
    if arg == "rooms":
        url = "https://drive.google.com/uc?id=1YHYxe0EpstlZEN20VN-TbKcVjuMd9T4r"
    elif arg == "buildings":
        url = "https://drive.google.com/uc?id=10nahGzZtTAgAPic2lYBB4pRmAb-rxX9-"
    elif arg == "rooms_buildings":
        url = "https://drive.google.com/uc?id=1xYQLfODzw4Ggm8RqXNPx6sHv_ec_3vwp"
    elif arg == "combine":
        url = "https://drive.google.com/uc?id=1-Guz5fzvuors8f0-Wnd4xNaXOdZglQCs"
    else:
        print("invalid name")
        exit()

    print(url)
    output = "r3d.zip"
    gdown.download(url, output, quiet=False)