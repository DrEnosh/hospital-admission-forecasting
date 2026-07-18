"""
Stage 1 — Data Preparation
Dataset Type: Synthetic
Loads raw Excel files, creates daily/monthly/department aggregations.
"""
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import numpy as np
from src.config import (SOURCE_FILES, LOCATION_ORDER, DATA_CLEAN_DIR, WATERMARK)

def load_and_tag():
    """Load all locations, return a single combined DataFrame."""
    frames = []
    for loc in LOCATION_ORDER:
        path = SOURCE_FILES[loc]
        print(f"  Loading {loc} from {os.path.basename(path)} ...")
        df = pd.read_excel(path, sheet_name="Synthetic Admissions")
        df["Location"] = loc
        frames.append(df)
    combined = pd.concat(frames, ignore_index=True)
    combined["Admission Date"] = pd.to_datetime(combined["Admission Date"])
    combined.sort_values(["Location", "Admission Date"], inplace=True)
    return combined

def make_daily(combined):
    """Daily admission counts per location."""
    daily = (combined.groupby(["Location", "Admission Date"])
             .size()
             .reset_index(name="Admissions"))
    
    # Fill missing calendar days with 0
    all_frames = []
    for loc in LOCATION_ORDER:
        loc_df = daily[daily["Location"] == loc].copy()
        date_min, date_max = loc_df["Admission Date"].min(), loc_df["Admission Date"].max()
        full_range = pd.date_range(date_min, date_max, freq="D")
        loc_full = pd.DataFrame({"Admission Date": full_range})
        loc_full = loc_full.merge(loc_df[["Admission Date", "Admissions"]], on="Admission Date", how="left")
        loc_full["Admissions"] = loc_full["Admissions"].fillna(0).astype(int)
        loc_full["Location"] = loc
        all_frames.append(loc_full)
    
    return pd.concat(all_frames, ignore_index=True)

def make_monthly(daily):
    """Monthly admission counts per location."""
    daily_c = daily.copy()
    daily_c["Month"] = daily_c["Admission Date"].dt.to_period("M")
    monthly = (daily_c.groupby(["Location", "Month"])["Admissions"]
               .sum().reset_index())
    monthly["Month"] = monthly["Month"].dt.to_timestamp()
    return monthly

def make_department_monthly(combined):
    """Monthly counts per location × department."""
    combined_c = combined.copy()
    combined_c["Month"] = combined_c["Admission Date"].dt.to_period("M")
    dept = (combined_c.groupby(["Location", "Month", "Admitting Department"])
            .size().reset_index(name="Admissions"))
    dept["Month"] = dept["Month"].dt.to_timestamp()
    return dept

def make_hourly(combined):
    """Hourly admission profile per location."""
    combined_c = combined.copy()
    combined_c["Hour"] = pd.to_datetime(combined_c["Admission Time"], format="%H:%M:%S", errors="coerce").dt.hour
    hourly = (combined_c.groupby(["Location", "Hour"])
              .size().reset_index(name="Admissions"))
    return hourly

def make_dow(combined):
    """Day-of-week admission profile per location."""
    combined_c = combined.copy()
    combined_c["DayOfWeek"] = combined_c["Admission Date"].dt.day_name()
    combined_c["DayNum"] = combined_c["Admission Date"].dt.dayofweek
    dow = (combined_c.groupby(["Location", "DayOfWeek", "DayNum"])
           .size().reset_index(name="Admissions"))
    dow.sort_values(["Location", "DayNum"], inplace=True)
    return dow

def main():
    print("=" * 60)
    print(f"STAGE 1 — DATA PREPARATION  |  {WATERMARK}")
    print("=" * 60)
    
    # Load
    print("\n[1/6] Loading raw data ...")
    combined = load_and_tag()
    print(f"  Combined shape: {combined.shape}")
    combined.to_csv(os.path.join(DATA_CLEAN_DIR, "combined_raw.csv"), index=False)
    
    # Daily
    print("[2/6] Creating daily counts ...")
    daily = make_daily(combined)
    daily.to_csv(os.path.join(DATA_CLEAN_DIR, "daily_admissions.csv"), index=False)
    print(f"  Daily rows: {len(daily)}")
    
    # Monthly
    print("[3/6] Creating monthly counts ...")
    monthly = make_monthly(daily)
    monthly.to_csv(os.path.join(DATA_CLEAN_DIR, "monthly_admissions.csv"), index=False)
    print(f"  Monthly rows: {len(monthly)}")
    
    # Department
    print("[4/6] Creating department-level monthly counts ...")
    dept = make_department_monthly(combined)
    dept.to_csv(os.path.join(DATA_CLEAN_DIR, "department_monthly.csv"), index=False)
    print(f"  Department-monthly rows: {len(dept)}")
    
    # Hourly profile
    print("[5/6] Creating hourly profiles ...")
    hourly = make_hourly(combined)
    hourly.to_csv(os.path.join(DATA_CLEAN_DIR, "hourly_profile.csv"), index=False)
    
    # Day-of-week profile
    print("[6/6] Creating day-of-week profiles ...")
    dow = make_dow(combined)
    dow.to_csv(os.path.join(DATA_CLEAN_DIR, "dow_profile.csv"), index=False)
    
    print(f"\n[OK] Stage 1 complete — all CSVs saved to {DATA_CLEAN_DIR}")
    
    # Summary
    for loc in LOCATION_ORDER:
        loc_daily = daily[daily["Location"] == loc]
        loc_monthly = monthly[monthly["Location"] == loc]
        print(f"  {loc:15s}  days={len(loc_daily):5d}  months={len(loc_monthly):3d}  "
              f"total_admissions={loc_daily['Admissions'].sum():,}")

if __name__ == "__main__":
    main()
