import pathlib
import urllib.request


PACKAGE_URL = "http://hl7.org.au/fhir/core/package.tgz"


def download_hl7_package(destination_dir: str = "playbook") -> None:
    """
    Download the HL7 AU FHIR core package.tgz file into the given directory.

    The original filename from the URL is preserved.
    """
    base_path = pathlib.Path(__file__).resolve().parent
    target_dir = base_path / destination_dir
    target_dir.mkdir(parents=True, exist_ok=True)

    filename = PACKAGE_URL.rsplit("/", 1)[-1]
    target_path = target_dir / filename

    print(f"Downloading {PACKAGE_URL} to {target_path} ...")
    urllib.request.urlretrieve(PACKAGE_URL, target_path)
    print("Download complete.")


if __name__ == "__main__":
    download_hl7_package()
