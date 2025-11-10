import os
from pathlib import Path


def main() -> None:
    folders = [
        "environments",
        "models",
        "algorithms",
        "attacks",
        "defenses",
        "utils",
        "data/checkpoints",
        "data/logs",
        "data/plots",
        "data/videos",
        "configs",
    ]

    for folder in folders:
        Path(folder).mkdir(parents=True, exist_ok=True)

        package_root = Path(folder.split("/")[0])
        init_file = package_root / "__init__.py"
        if not init_file.exists():
            init_file.touch()

    print("Project structure created.")
    print("\nFolder structure:")
    for folder in folders:
        print(f"  {folder}/")


if __name__ == "__main__":
    main()

