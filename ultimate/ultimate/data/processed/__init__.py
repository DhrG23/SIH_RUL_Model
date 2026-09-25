"""
ultimate.data.processed -- 47 raw ALFA/carbonZ UAV flight-log folders,
one per real flight, each named for its failure condition (e.g.
"..._engine_failure", "..._no_failure", "..._no_ground_truth"). Each
folder holds that flight's per-ROS-topic CSV exports (battery, IMU,
GPS/global-position, airspeed, failure-status, ...) at each topic's
native sample rate. build_uav_dataset.py (in the package root) merges
each flight's CSVs on timestamp into one row-per-reading table and
concatenates all 47 into ../uav_processed_dataset.csv.

No further subpackages here -- the per-flight folders are data, not code
(hence no __init__.py inside each of them).
"""
