"""Build the optional local C ABI kernel. Never runs during package import."""

from pathlib import Path
import os
import subprocess


def main():
    root = Path(__file__).resolve().parents[1]
    source = root / "src/cpp/rolling.cpp"
    target = root / "src/gig/native/_rolling.so"
    subprocess.run([os.environ.get("CXX", "c++"), "-std=c++20", "-O3", "-fPIC", "-shared",
                    "-Wall", "-Wextra", "-Werror", str(source), "-o", str(target)], check=True)
    print(target)


if __name__ == "__main__":
    main()
