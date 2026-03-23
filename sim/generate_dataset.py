#!/usr/bin/env python3
"""
Data generation script for Koopman / DMDc identification of the glucose–insulin system.
This script uses the SimGlucose simulator to create multi‑day trajectories of
true blood glucose, CGM (optional), insulin delivery, and meal disturbances.
The resulting datasets are saved as CSV files, ready for delay embedding and
system identification.

Data generation script for Koopman / DMDc identification of the glucose–insulin system.
This script uses the SimGlucose simulator to create multi‑day trajectories of
true blood glucose, CGM (optional), insulin delivery, and meal disturbances.
The resulting datasets are saved as CSV files, ready for delay embedding and
system identification.

Improvements for system identification:
- Adds discrete time index (t_idx, t_minutes) for DMDc compatibility.
- Includes patient identifier column.
- Uses per‑patient seeds to ensure varied meal patterns.
- Saves long continuous trajectories (SPLIT_BY_DAY = False by default).
- Outputs a metadata JSON file with simulation parameters.
"""

import os
import json
import logging
from datetime import datetime, timedelta

import pandas as pd

from simglucose.controller.basal_bolus_ctrller import BBController
from simglucose.sensor.cgm import CGMSensor
from simglucose.actuator.pump import InsulinPump
from simglucose.patient.t1dpatient import T1DPatient
from simglucose.simulation.env import T1DSimEnv
from simglucose.simulation.scenario_gen import RandomScenario
from simglucose.simulation.sim_engine import SimObj, sim

# -----------------------------------------------------------------------------
# User configuration – adjust these parameters as needed
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# -----------------------------------------------------------------------------
# User configuration – Kaggle version
# -----------------------------------------------------------------------------
NUM_DAYS = 7
PATIENTS = [
    'adolescent#001',
    'adult#001',
    'child#001'
]

RANDOM_SEED = 42

# Kaggle writable output directory
OUTPUT_PATH = '/kaggle/working/data'

# Ensure folder exists
import os
os.makedirs(OUTPUT_PATH, exist_ok=True)

SAVE_CGM = True
SPLIT_BY_DAY = False

# Sampling interval (minutes) – SimGlucose default is 3 minutes
SAMPLING_MINUTES = 3

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# -----------------------------------------------------------------------------
# Utility functions
# -----------------------------------------------------------------------------
def create_environment(patient_name: str, seed: int, start_time: datetime) -> T1DSimEnv:
    """
    Create a SimGlucose environment with a random meal scenario.

    Parameters
    ----------
    patient_name : str
        Name of the patient (e.g., 'adolescent#001').
    seed : int
        Random seed for the CGM sensor and meal scenario.
    start_time : datetime
        Simulation start time (used by the scenario generator).

    Returns
    -------
    T1DSimEnv
        Configured simulation environment.
    """
    patient = T1DPatient.withName(patient_name)
    sensor = CGMSensor.withName('Dexcom', seed=seed)
    pump = InsulinPump.withName('Insulet')
    scenario = RandomScenario(start_time=start_time, seed=seed)
    env = T1DSimEnv(patient, sensor, pump, scenario)
    return env


def run_simulation(env: T1DSimEnv, controller: BBController, days: int) -> pd.DataFrame:
    sim_length = timedelta(days=days)
    sim_obj = SimObj(env, controller, sim_length, animate=False, path='/kaggle/working/')
    
    sim(sim_obj)              # run simulation
    results = sim_obj.results()  # <-- THIS is the actual dataframe
    
    return results


 
 

def extract_signals(results: pd.DataFrame, save_cgm: bool = True) -> pd.DataFrame:
    print("COLUMNS:", results.columns)

    df = pd.DataFrame()

    # Copy time from index
    df['time'] = results.index.astype(str)

    # CRITICAL: use .values to avoid index alignment
    df['BG'] = results['BG'].values
    df['insulin'] = results['insulin'].values
    df['meal'] = results['CHO'].values

    if save_cgm and 'CGM' in results.columns:
        df['CGM'] = results['CGM'].values

    # Time indices
    df['t_idx'] = range(len(df))
    df['t_minutes'] = df['t_idx'] * SAMPLING_MINUTES

    return df







def save_dataset(df: pd.DataFrame, patient_name: str, output_path: str, split_by_day: bool = False) -> None:
    """
    Save the extracted signals to CSV file(s).

    Parameters
    ----------
    df : pd.DataFrame
        Data to save (must contain a 'time' column).
    patient_name : str
        Name of the patient (used in filename).
    output_path : str
        Directory where files will be saved.
    split_by_day : bool, optional
        If True, split the data by calendar day and save one file per day.
        Otherwise, save a single file for the whole trajectory.
    """
    os.makedirs(output_path, exist_ok=True)
    # Sanitize patient name for filesystem
    safe_name = patient_name.replace('#', '_').replace(' ', '_')

    if not split_by_day:
        filename = f"{safe_name}.csv"
        filepath = os.path.join(output_path, filename)
        df.to_csv(filepath, index=False)
        logger.info(f"Saved {filepath}")
    else:
        # Add a date column for grouping
        df = df.copy()
        df['date'] = df['time'].dt.date
        for date, day_df in df.groupby('date'):
            day_filename = f"{safe_name}_{date.isoformat()}.csv"
            day_filepath = os.path.join(output_path, day_filename)
            day_df.drop(columns='date').to_csv(day_filepath, index=False)
            logger.info(f"Saved {day_filepath}")


def save_metadata(metadata: dict, output_path: str) -> None:
    """Save simulation metadata as a JSON file."""
    os.makedirs(output_path, exist_ok=True)
    filepath = os.path.join(output_path, 'metadata.json')
    with open(filepath, 'w') as f:
        json.dump(metadata, f, indent=2)
    logger.info(f"Saved metadata to {filepath}")


# -----------------------------------------------------------------------------
# Main script
# -----------------------------------------------------------------------------
def main():
    """Run simulations for all patients and save the datasets."""
    # Start time: beginning of today (as in the example)
    now = datetime.now()
    start_time = datetime.combine(now.date(), datetime.min.time())

    logger.info(f"Starting data generation. Patients: {PATIENTS}, days: {NUM_DAYS}, base seed: {RANDOM_SEED}")

    # Collect metadata for final JSON
    metadata = {
        "num_days": NUM_DAYS,
        "patients": PATIENTS,
        "base_seed": RANDOM_SEED,
        "sampling_minutes": SAMPLING_MINUTES,
        "save_cgm": SAVE_CGM,
        "split_by_day": SPLIT_BY_DAY
    }

    for patient in PATIENTS:
        logger.info(f"Simulating patient: {patient}")
        try:
            # Vary seed per patient to ensure different meal patterns
            # Use hash of patient name to modify the base seed
            patient_seed = RANDOM_SEED + (hash(patient) % 1000)

            env = create_environment(patient, patient_seed, start_time)
            controller = BBController()  # default basal‑bolus controller
            results = run_simulation(env, controller, NUM_DAYS)

            logger.debug(f"Columns in raw results: {list(results.columns)}")
            df = extract_signals(results, save_cgm=SAVE_CGM)

            # Add patient identifier column
            df['patient'] = patient

            save_dataset(df, patient, OUTPUT_PATH, split_by_day=SPLIT_BY_DAY)
        except Exception as e:
            logger.error(f"Simulation failed for {patient}: {e}")
            continue

    # Save overall metadata
    save_metadata(metadata, OUTPUT_PATH)

    logger.info("Data generation completed.")
    print("COLUMNS:", results.columns)
    print(results.head())



if __name__ == '__main__':
    main()