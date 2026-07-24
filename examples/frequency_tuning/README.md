# 周波数自動調整の例（Export Python Script の活用）

Multi-Region Editor の **[Export Python Script]** で書き出したスクリプトは、
次の形になっています。

```python
_SCALE = 0.001
_MESH_ORDER = 2
_OUT_MSH = 'cavity.msh'


def build_model(out_msh=_OUT_MSH, mesh_order=_MESH_ORDER, **user_vars):
    """geom を構築し .msh を書き出す。user_vars で変数表の値を上書き可能。"""
    # === user variables (変数表) — user_vars で上書き可、依存変数は再計算 ===
    a = user_vars.get('a', 40)
    L = user_vars.get('L', 50)
    gap = user_vars.get('gap', L / 10)   # 依存変数は上書き後の L から再計算される
    ...
    return out_msh


if __name__ == '__main__':
    build_model()
```

そのまま `python cavity_mesh.py` と実行すれば従来どおり `.msh` を生成し、
**外部スクリプトから `build_model(a=42, out_msh='trial.msh')` のように呼べば
パラメータを変えたメッシュを何度でも作り直せます**。変数表で `c = a + b` の
ように定義した依存変数は、`a` を上書きすると自動的に再計算されます。

## tune_frequency.py

半径 `a` を二分法で動かし、TM010 共振周波数を目標値（既定 2.856 GHz）に
合わせるサンプルです。

```bash
# デモ形状（ピルボックス）を自動生成して調整
python tune_frequency.py

# GUI から書き出したスクリプトを使う場合
python tune_frequency.py cavity_mesh.py --var a --target 2.856 --lo 30 --hi 60
```

各反復で `build_model(out_msh=..., a=<試行値>)` を呼んでメッシュを作り直し、
`solve_tm0_standing` で最低次モードの周波数を求めています。

実行例（ピルボックス、2 次要素）:

```
  [13] a=40.177002 -> 2.855926 GHz (誤差 -7.43e-05)

調整結果: a = 40.177002 (単位は形状の Unit)
```

解析解 `a = j01·c / (2π f) = 40.1766 mm`（`j01 = 2.40483`）とよく一致します。

## 応用のヒント

- `tune()` は単調性（半径↑ → 周波数↓）を仮定した二分法です。複数変数の同時
  最適化には `scipy.optimize` の `brentq` / `least_squares` などに置き換えられます。
- 誘電体入り空洞では `.msh` と同時に `.materials.json` が要るため、
  `build_model` に加えて材料 JSON も書き出す処理を足してください
  （GUI の Export MSH は両方を出力します）。
- メッシュサイズ `lc` も変数で定義しておけば `build_model(lc=...)` で
  収束性（h 収束）の検証に使えます。
