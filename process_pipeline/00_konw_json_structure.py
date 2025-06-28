import json
import argparse

def merge_keys_tree(obj, tree):
    if isinstance(obj, dict):
        for key, value in obj.items():
            if key not in tree:
                tree[key] = {}
            merge_keys_tree(value, tree[key])
    elif isinstance(obj, list):
        for item in obj:
            merge_keys_tree(item, tree)

def format_tree(tree, indent=0):
    lines = []
    for key, subtree in tree.items():
        lines.append('  ' * indent + f"- {key}")
        lines.extend(format_tree(subtree, indent + 1))
    return lines

def main():
    parser = argparse.ArgumentParser(description="Extract JSON key structure")
    parser.add_argument('-i', '--input', required=True, help='Path to input JSON file')
    parser.add_argument('-o', '--output', required=True, help='Path to output structure text file')
    args = parser.parse_args()

    with open(args.input, 'r', encoding='utf-8') as f:
        data = json.load(f)

    data = data['graphs']

    structure_tree = {}
    for entry in data:
        merge_keys_tree(entry, structure_tree)

    # Format tree as lines
    lines = format_tree(structure_tree)

    # Write the formatted structure to the output file
    with open(args.output, 'w', encoding='utf-8') as f:
        f.write("All field structure:\n")
        f.write('\n'.join(lines))

    print("🤓Structure written to:", args.output)

if __name__ == '__main__':
    main()
