"""
日本周辺域に緯度経度グリッド（黒）と、指定した解像度の赤い格子線を重ねた地図画像を作成します。
- 出力は以下の3枚を作成:
  - ./result_0_25.png  （赤：0.25度格子）
  - ./result_0_5.png   （赤：0.5度格子）
  - ./result_1_0.png   （赤：1.0度格子）

- 地図範囲（経度・緯度の順）は [100, 180, 0, 70]（＝東経100–180度、北緯0–70度）
- 黒いグリッドは5度間隔（変更可）
- 利用ライブラリは環境指定（cartopy, matplotlib, numpy, japanize-matplotlib）内に限定

使い方:
    python src/WeatherLLM/plot_map/map.py

必要パッケージ（environmentの一部）:
    - cartopy
    - matplotlib
    - numpy
    - japanize-matplotlib
"""

from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt
import json

# cartopy関連
import cartopy.crs as ccrs
from cartopy.mpl.gridliner import LONGITUDE_FORMATTER, LATITUDE_FORMATTER
import matplotlib.ticker as mticker

# 日本語フォント
import japanize_matplotlib

# NaturalEarth のフィーチャは環境によっては初回ダウンロードが必要になるため、
# 失敗しても致命的にしない（海陸塗りつぶしはあれば使う）方針
try:
    import cartopy.feature as cfeature
    _HAS_FEATURE = True
except Exception:
    _HAS_FEATURE = False


def draw_map_with_red_grid(
    step_deg: float,
    out_path: str,
    extent=(100.0, 180.0, 0.0, 70.0),
    base_grid_interval: float = 5.0,
    dpi: int = 400,
    figsize=(12, 12),
    stations_json=None,
    save_svg: bool = True,
) -> None:
    """
    指定の範囲に、黒の基準グリッド（base_grid_interval度）と
    赤の高解像度グリッド（step_deg度）を重ねて描画し、PNG保存する。

    Parameters
    ----------
    step_deg : float
        赤い格子線の間隔（度）。例: 0.25, 0.5, 1.0
    out_path : str
        出力PNGファイルパス
    extent : tuple(float, float, float, float)
        (lon_min, lon_max, lat_min, lat_max)
    base_grid_interval : float
        黒いグリッド線の間隔（度）
    dpi : int
        出力画像のDPI
    """
    lon_min, lon_max, lat_min, lat_max = extent

    # 投影は PlateCarree（地理座標）を使用
    proj = ccrs.PlateCarree()
    fig = plt.figure(figsize=figsize, dpi=dpi)
    ax = plt.axes(projection=proj)
    ax.set_extent([lon_min, lon_max, lat_min, lat_max], crs=proj)
    ax.set_facecolor("white")

    # 海陸や海岸線（利用可能なら）
    if _HAS_FEATURE:
        try:
            ax.add_feature(cfeature.OCEAN, facecolor="#E8F4FF", zorder=0)      # 薄い水色
            ax.add_feature(cfeature.LAND, facecolor="#F3F3F3", zorder=0)       # 薄いグレー
            ax.add_feature(cfeature.COASTLINE, edgecolor="dimgray", linewidth=0.6, zorder=2)
            ax.add_feature(cfeature.BORDERS, edgecolor="gray", linewidth=0.4, zorder=2)
        except Exception:
            # 失敗しても続行（グリッド線が主目的）
            pass
    else:
        # フィーチャ無しでも最低限の海岸線は描く（ダウンロードが必要な場合あり）
        try:
            ax.coastlines(resolution="50m", color="dimgray", linewidth=0.6, zorder=2)
        except Exception:
            pass

    # 黒い基準グリッド（ラベル付き）
    gl = ax.gridlines(
        draw_labels=True,
        linewidth=0.8,
        color="black",
        alpha=0.6,
        linestyle="-",
        zorder=1,
    )
    gl.xlocator = mticker.MultipleLocator(base_grid_interval)
    gl.ylocator = mticker.MultipleLocator(base_grid_interval)
    gl.xformatter = LONGITUDE_FORMATTER
    gl.yformatter = LATITUDE_FORMATTER
    gl.xlabel_style = {"size": 9}
    gl.ylabel_style = {"size": 9}
    gl.top_labels = False
    gl.right_labels = False

    # 赤い高解像度グリッド（step_deg間隔）
    # 経線（縦線）
    lons = np.arange(lon_min, lon_max + 1e-10, step_deg)
    for lon in lons:
        ax.plot(
            [lon, lon],
            [lat_min, lat_max],
            color="crimson",
            linewidth=0.4,
            alpha=0.85,
            transform=proj,
            zorder=3,
        )
    # 緯線（横線）
    lats = np.arange(lat_min, lat_max + 1e-10, step_deg)
    for lat in lats:
        ax.plot(
            [lon_min, lon_max],
            [lat, lat],
            color="crimson",
            linewidth=0.4,
            alpha=0.85,
            transform=proj,
            zorder=3,
        )

    # 観測所（JSON）を緑の点でプロット
    try:
        if stations_json is None:
            stations_json = Path(__file__).resolve().parent / "kanku_chihou_56.json"
        stations_json = Path(stations_json)
        if stations_json.exists():
            with stations_json.open("r", encoding="utf-8") as f:
                _data = json.load(f)
            st_lons = []
            st_lats = []
            for item in _data:
                try:
                    _lon = float(item.get("lon"))
                    _lat = float(item.get("lat"))
                except Exception:
                    continue
                if (lon_min <= _lon <= lon_max) and (lat_min <= _lat <= lat_max):
                    st_lons.append(_lon)
                    st_lats.append(_lat)
            if st_lons and st_lats:
                ax.scatter(
                    st_lons,
                    st_lats,
                    s=50,
                    c="limegreen",
                    edgecolors="white",
                    linewidths=0.6,
                    alpha=0.95,
                    transform=proj,
                    zorder=4,
                    label="気象台"
                )
    except Exception:
        # JSON読み込みに失敗しても地図描画は継続
        pass

    # タイトル（日本語）
    center_lat = 0.5 * (lat_min + lat_max)
    km_per_deg_lat = 111.32  # 1度あたり南北距離[km]
    km_per_deg_lon = 111.32 * np.cos(np.deg2rad(center_lat))  # 中心緯度での1度あたり東西距離[km]
    lat_km = step_deg * km_per_deg_lat
    lon_km = step_deg * km_per_deg_lon
    ax.set_title(
        f"日本周辺域（東経{lon_min:.0f}–{lon_max:.0f}°, 北緯{lat_min:.0f}–{lat_max:.0f}°）\n"
        f"赤：{step_deg:g}°格子（南北約{lat_km:.1f} km, 東西約{lon_km:.1f} km@{center_lat:.1f}°N）, 黒：{base_grid_interval:g}°グリッド",
        fontsize=12,
        pad=10,
    )

    # 保存
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_path, bbox_inches="tight", facecolor="white")
    # 可能ならベクタ（SVG）も保存（拡大しても劣化なし）
    if save_svg:
        try:
            svg_path = out_path.with_suffix(".svg")
            plt.savefig(svg_path, bbox_inches="tight", facecolor="white")
        except Exception:
            pass
    plt.close(fig)


def main():
    # 指定範囲は "70 100 0 180" の読み替えとして
    #  (lat_max, lon_min, lat_min, lon_max) を意味すると解釈し、
    #  extent=(lon_min, lon_max, lat_min, lat_max) = (100, 180, 0, 70) を採用
    extent = (100.0, 180.0, 0.0, 70.0)

    # それぞれ1枚ずつ作成（ファイル名を明示）
    tasks = [
        (0.25, "result_0_25.png"),
        (0.5, "result_0_5.png"),
        (1.0, "result_1_0.png"),
    ]

    for step, fname in tasks:
        draw_map_with_red_grid(step_deg=step, out_path=fname, extent=extent, base_grid_interval=5.0)


if __name__ == "__main__":
    main()
