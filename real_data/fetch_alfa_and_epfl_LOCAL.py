"""
fetch_alfa_and_epfl_LOCAL.py
------------------------------
IMPORTANT: This script is NOT run as part of the project pipeline, and it
was NOT executed by Claude. kilthub.cmu.edu and zenodo.org are blocked in
the sandboxed environment this project was built in (confirmed via a
direct connectivity test -- both returned "host_not_allowed"). Rather than
fake having integrated these datasets, this file is a documented starting
point for YOU to run on your own machine, where you have normal internet
access, to add them for real.

What each dataset would add if you plug it in:

1) CMU ALFA (UAV Fault & Anomaly Detection)
   https://kilthub.cmu.edu/articles/dataset/ALFA_A_Dataset_for_UAV_Fault_and_Anomaly_Detection/12707963
   - 47 real autonomous fixed-wing flights, 23 with sudden full engine
     failure, real ground-truth fault timing.
   - NOTE: the aircraft used a single ELECTRIC motor, not a piston engine.
     Don't present this as piston-specific -- present it as validating the
     ANOMALY-DETECTION half of the system (sudden-failure signature
     recognition) using real flight telemetry, separate from the
     synthetic piston degradation-to-failure data used for RUL.
   - Reader/loader tools: https://github.com/castacks/alfa-dataset
     (this repo IS reachable from a normal machine; it has Python/MATLAB
     helpers for reading the ROS-bag-derived CSVs).

   Steps once downloaded:
     a. Download the processed CSVs from the kilthub page above.
     b. Use the loader tools in castacks/alfa-dataset to get per-flight
        time series (airspeed, throttle, IMU, etc.) with fault onset labels.
     c. Extract a handful of features per time window (mean/std of each
        channel, similar to this project's rolling-feature approach).
     d. Train a binary classifier: "pre-fault" vs "post-fault-onset" window.
     e. Report accuracy/F1 the same way real_validation_cwru.py does, and
        add it to the app as a 3rd real-data validation panel.

2) EPFL Fixed-Wing UAV Flight Log Dataset
   https://zenodo.org/records/10372783
   - ROSbag telemetry: airspeed, pressure, barometric altitude, IMU, motor
     control output, from real fixed-wing flights.
   - Good for validating the "operating context" features (altitude,
     environmental conditions) used in the piston-twin's feature set
     against real flight profiles, and/or building a more realistic
     flight-phase simulator for the synthetic generator.

   Steps once downloaded:
     a. Download from the Zenodo record above (Zenodo lets you download
        directly, no account needed for public records).
     b. Convert ROS bags to CSV (rosbag + pandas, or the `bagpy` package).
     c. Extract altitude/airspeed/motor-output trends to sanity-check the
        operating-condition ranges used in src/generate_data.py
        (altitude_ft, ambient_temp_C) against real flight envelopes.

General integration pattern for both: once you have a CSV with a
consistent schema, you can largely reuse this project's existing code --
preprocess.py's rolling-feature logic and train_models.py's model-training
loop are not piston-specific, they just expect a dataframe of sensor
columns. The main work is the ETL step (raw dataset -> clean CSV), which
is why it's left to you to run locally where the hosts aren't blocked.
"""

print(__doc__)
