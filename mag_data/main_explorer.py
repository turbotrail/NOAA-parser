import os
import argparse
from datetime import datetime, timezone
import logging

from satellite_mag_explorer.orbit_engine import load_satellite_by_name, generate_ephemeris
from satellite_mag_explorer.magnetic_model import compute_magnetic_field
from satellite_mag_explorer.orbit_analysis import detect_saa_encounters, find_magnetic_equator_crossings
from satellite_mag_explorer.visualization import plot_ground_track_with_bfield, plot_time_series, animate_orbit

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

def main():
    parser = argparse.ArgumentParser(description="Satellite Orbit Magnetic Field Explorer")
    parser.add_argument("--satellite", type=str, default="ISS (ZARYA)", help="Satellite name in Celestrak")
    parser.add_argument("--duration", type=float, default=24.0, help="Simulation duration in hours")
    parser.add_argument("--step", type=int, default=60, help="Time step in seconds")
    parser.add_argument("--output_dir", type=str, default="satellite_outputs", help="Output directory")
    parser.add_argument("--animate", action="store_true", help="Generate MP4 animation")
    
    args = parser.parse_args()
    
    os.makedirs(args.output_dir, exist_ok=True)
    
    logging.info(f"Loading satellite '{args.satellite}'...")
    try:
        sat = load_satellite_by_name(args.satellite)
    except Exception as e:
        logging.error(f"Error loading satellite: {e}")
        return

    start_time = datetime.now(timezone.utc)
    logging.info(f"Generating ephemeris for {args.duration} hours starting at {start_time}...")
    ephemeris_df = generate_ephemeris(sat, start_time, duration_hours=args.duration, step_seconds=args.step)
    
    logging.info("Computing magnetic field (IGRF-14)...")
    mag_df = compute_magnetic_field(ephemeris_df)
    
    csv_path = os.path.join(args.output_dir, "orbit_magnetic_field.csv")
    logging.info(f"Saving full dataset to {csv_path}...")
    mag_df.to_csv(csv_path, index=False)
    
    logging.info("Analyzing orbit events...")
    saa_df = detect_saa_encounters(mag_df, f_threshold=25000.0)
    if not saa_df.empty:
        logging.info(f"Found {len(saa_df)} SAA encounters.")
        saa_df.to_csv(os.path.join(args.output_dir, "saa_encounters.csv"), index=False)
    else:
        logging.info("No SAA encounters found (using F < 25,000 nT threshold).")
        
    equator_df = find_magnetic_equator_crossings(mag_df)
    logging.info(f"Found {len(equator_df)} magnetic equator crossings.")
    equator_df.to_csv(os.path.join(args.output_dir, "equator_crossings.csv"), index=False)
    
    logging.info("Generating plots...")
    fig_track = plot_ground_track_with_bfield(mag_df, title=f"{args.satellite} Ground Track with IGRF-14 F-field")
    track_path = os.path.join(args.output_dir, "ground_track.png")
    fig_track.savefig(track_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved ground track map to {track_path}")
    
    fig_ts = plot_time_series(mag_df)
    ts_path = os.path.join(args.output_dir, "time_series.png")
    fig_ts.savefig(ts_path, dpi=300, bbox_inches='tight')
    logging.info(f"Saved time series plot to {ts_path}")
    
    if args.animate:
        logging.info("Generating animation (this may take a while)...")
        # Downsample for animation to speed it up (e.g., one frame every 3 minutes if step is 60s)
        ani_df = mag_df.iloc[::3].reset_index(drop=True)
        ani_path = os.path.join(args.output_dir, "orbit_animation.mp4")
        animate_orbit(ani_df, ani_path, fps=15)
        logging.info(f"Saved animation to {ani_path}")
        
    logging.info("Analysis complete!")

if __name__ == "__main__":
    main()
