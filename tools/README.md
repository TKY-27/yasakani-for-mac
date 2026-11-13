# Tools for checking the published files

日本語：このフォルダーには、公開したモデルファイルと品質の数字を、エンジンを信用せずに自分で確かめるための道具があります。ファイル形式の定義は
[docs/FORMAT.ja.md](../docs/FORMAT.ja.md) です。

Everything here is for checking what this project publishes without having to trust the engine.

| | |
|---|---|
| [`yskn_decode.py`](yskn_decode.py) | reads a `.yskn` model file without the engine: lists what is inside, turns a tensor back into ordinary weights, compares every tensor with the original checkpoint. Python and numpy, no GPU |
| [`yskn_fixture.py`](yskn_fixture.py), [`testdata/`](testdata/) | a small test file in the same format, and the values a reader must get from it |
| [`records/`](records/) | what `yskn_decode.py compare` gives for the two released models, tensor by tensor |
| [`quality/`](quality/) | the quality measurement: the 20 evaluation windows, a program that makes the reference from the original model, a program that measures a released model with the engine |

The file format is defined in [docs/FORMAT.md](../docs/FORMAT.md).

    python tools/yskn_decode.py selftest
    python tools/yskn_decode.py info    qwen38_27b_quality.yskn
    python tools/yskn_decode.py compare qwen38_27b_quality.yskn --original <folder with the *.safetensors of Qwen/Qwen3.8-27B>

What is not here: the conversion tool that makes a `.yskn` file from the original weights. It is not public
([docs/FORMAT.md](../docs/FORMAT.md), section 9).
