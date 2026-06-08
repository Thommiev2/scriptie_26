import subprocess
import argparse
import sys


def run_pip_command(command):
    try:
        subprocess.run([sys.executable, "-m", "pip"] + command, check=True)
    except subprocess.CalledProcessError as e:
        sys.exit(1)


def update_transformers(version):
    run_pip_command(["uninstall", "transformers", "-y"])
    run_pip_command(["install", f"transformers=={version}"])


if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="File to quickly switch between transformer versions for qwen3asr and the others")
    parser.add_argument("--upgrade", action="store_true", help="Include this flag to upgrade")
    args = parser.parse_args()

    version = "5.9.0" if args.upgrade else "4.57.6"

    update_transformers(version=version)
