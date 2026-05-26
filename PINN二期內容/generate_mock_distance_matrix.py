import sys
from pathlib import Path

import numpy as np
import pandas as pd


if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")


# 二期抽水井與觀測井。若新資料的井號不同，先改這兩個清單。
pumping_wells = ["PW01", "PW05", "PW06", "PW07", "PW08", "PW09", "PW010", "PW011"]
obs_wells = ["PW02", "PW03", "PW04"]


# 井位座標，單位建議使用公尺。
# 這是一份示意座標，用於產生 mock 距離矩陣；若有實際測量座標，務必替換成真實座標。
coords = {
    # Pumping wells
    "PW01": (96, 30),
    "PW05": (0, 30),
    "PW06": (0, 0),
    "PW07": (24, 0),
    "PW08": (48, 0),
    "PW09": (72, 0),
    "PW010": (96, 0),
    "PW011": (120, 0),
    # Observation wells
    "PW04": (24, 15),
    "PW03": (48, 15),
    "PW02": (72, 15),
}


output_file = Path("Distance_Matrix_Phase2.csv")


def validate_inputs() -> None:
    all_wells = obs_wells + pumping_wells
    missing_coords = [well for well in all_wells if well not in coords]
    if missing_coords:
        raise ValueError(f"以下井位缺少座標，請補到 coords: {missing_coords}")

    duplicated = sorted({well for well in all_wells if all_wells.count(well) > 1})
    if duplicated:
        raise ValueError(f"obs_wells 與 pumping_wells 不可重複，重複井位: {duplicated}")


def build_distance_matrix() -> pd.DataFrame:
    all_wells = obs_wells + pumping_wells
    dist_matrix = np.zeros((len(all_wells), len(pumping_wells)), dtype=float)

    for i, source_well in enumerate(all_wells):
        for j, pumping_well in enumerate(pumping_wells):
            if source_well == pumping_well:
                dist_matrix[i, j] = 0.0
                continue

            x1, y1 = coords[source_well]
            x2, y2 = coords[pumping_well]
            dist_matrix[i, j] = float(np.hypot(x1 - x2, y1 - y2))

    return pd.DataFrame(dist_matrix, index=all_wells, columns=pumping_wells)


def main() -> None:
    validate_inputs()
    df_dist = build_distance_matrix()
    df_dist.to_csv(output_file, encoding="utf-8-sig")

    print(f"抽水井 ({len(pumping_wells)}): {pumping_wells}")
    print(f"觀測井 ({len(obs_wells)}): {obs_wells}")
    print(f"[Done] 已輸出 {output_file.resolve()}")
    print(df_dist.head())


if __name__ == "__main__":
    main()
