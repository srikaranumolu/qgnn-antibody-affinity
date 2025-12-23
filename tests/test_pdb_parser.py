from src.data.pdb_parser import parse_pdb

def test_parse_pdb(tmp_path):
    p = tmp_path / 'sample.pdb'
    p.write_text('ATOM      1  N   ALA A   1      11.104  13.207   8.678  1.00 20.00           N\n')
    lines = parse_pdb(str(p))
    assert len(lines) == 1

