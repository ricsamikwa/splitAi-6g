import os
import tarfile
import urllib.request


# ============================================================
# Config
# ============================================================

URL = "https://s3.amazonaws.com/fast-ai-imageclas/imagenette2-160.tgz"

DATA_DIR = "data"
ARCHIVE = os.path.join(DATA_DIR, "imagenette2-160.tgz")
DATASET_DIR = os.path.join(DATA_DIR, "imagenette2-160")


# ============================================================
# Download
# ============================================================

def download_progress(block_num, block_size, total_size):
    downloaded = block_num * block_size

    if total_size > 0:
        percent = min(
            100.0,
            downloaded * 100.0 / total_size
        )

        print(
            f"\rDownloading: {percent:6.2f}%",
            end="",
            flush=True
        )


def main():

    os.makedirs(DATA_DIR, exist_ok=True)

    print("=" * 80)
    print("IMAGENETTE DATASET SETUP")
    print("=" * 80)

    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    if not os.path.exists(ARCHIVE):

        print("Downloading Imagenette...")
        print(f"Source: {URL}")
        print()

        urllib.request.urlretrieve(
            URL,
            ARCHIVE,
            reporthook=download_progress
        )

        print()
        print("Download complete.")

    else:

        print(
            f"Archive already exists: {ARCHIVE}"
        )

    # --------------------------------------------------------
    # Extract
    # --------------------------------------------------------

    if not os.path.exists(DATASET_DIR):

        print()
        print("Extracting dataset...")

        with tarfile.open(
            ARCHIVE,
            "r:gz"
        ) as tar:

            tar.extractall(DATA_DIR)

        print("Extraction complete.")

    else:

        print(
            f"Dataset already extracted: {DATASET_DIR}"
        )

    # --------------------------------------------------------
    # Check validation directory
    # --------------------------------------------------------

    val_dir = os.path.join(
        DATASET_DIR,
        "val"
    )

    print()

    if not os.path.exists(val_dir):

        print(
            "ERROR: validation directory was not found."
        )

        return

    classes = sorted([
        d
        for d in os.listdir(val_dir)
        if os.path.isdir(
            os.path.join(val_dir, d)
        )
    ])

    print(f"Validation directory: {val_dir}")
    print(f"Number of classes:    {len(classes)}")

    print()
    print("Class folders:")

    for cls in classes:
        class_dir = os.path.join(
            val_dir,
            cls
        )

        num_images = len([
            f
            for f in os.listdir(class_dir)
            if f.lower().endswith(
                (".jpeg", ".jpg", ".png")
            )
        ])

        print(
            f"  {cls}: {num_images} images"
        )

    print()
    print("=" * 80)
    print("Imagenette is ready.")
    print("=" * 80)


if __name__ == "__main__":
    main()