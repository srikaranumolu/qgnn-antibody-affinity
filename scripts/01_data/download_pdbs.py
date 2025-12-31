"""Download PDB files from SAbDab (placeholder script)."""

import argparse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--out', default='data/raw')
    args = parser.parse_args()
    print('Would download PDBs into', args.out)


if __name__ == '__main__':
    main()

