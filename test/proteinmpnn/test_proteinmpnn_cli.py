from pathlib import Path

from click.testing import CliRunner

from rfantibody.cli.inference import proteinmpnn


def test_proteinmpnn_forwards_original_bias_jsonl_options(monkeypatch, tmp_path):
    input_dir = tmp_path / "inputs"
    output_dir = tmp_path / "outputs"
    bias_aa_jsonl = tmp_path / "bias_aa.jsonl"
    bias_by_res_jsonl = tmp_path / "bias_by_res.jsonl"

    input_dir.mkdir()
    bias_aa_jsonl.write_text('{"A": -1.1, "F": 0.7}\n')
    bias_by_res_jsonl.write_text('{"H": [[0.0]]}\n')

    captured = {}

    def fake_run(cmd, cwd):
        captured["cmd"] = cmd
        captured["cwd"] = cwd

        class Result:
            returncode = 0

        return Result()

    monkeypatch.setattr("rfantibody.cli.inference.subprocess.run", fake_run)

    runner = CliRunner()
    result = runner.invoke(
        proteinmpnn,
        [
            "--input-dir", str(input_dir),
            "--output-dir", str(output_dir),
            "--bias_AA_jsonl", str(bias_aa_jsonl),
            "--bias_by_res_jsonl", str(bias_by_res_jsonl),
        ],
    )

    assert result.exit_code == 0, result.output
    assert captured["cmd"][0] == "python"
    assert "-bias_AA_jsonl" in captured["cmd"]
    assert str(bias_aa_jsonl.resolve()) in captured["cmd"]
    assert "-bias_by_res_jsonl" in captured["cmd"]
    assert str(bias_by_res_jsonl.resolve()) in captured["cmd"]
    assert Path(captured["cmd"][1]).name == "proteinmpnn_interface_design.py"
