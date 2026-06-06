import json, sys

def extract(path):
    with open(path) as f:
        nb = json.load(f)
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] == "code":
                print(f"\n# --- Cell {i} --- \n")
                print("".join(cell["source"]))

if __name__ == "__main__":
    extract(sys.argv[1])
