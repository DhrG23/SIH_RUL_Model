"""
ultimate.data -- datasets shipped with this package. Not imported for
code (there's none here), just marked as a package so the whole
`ultimate` tree is consistently importable/resource-loadable.

  uav_processed_dataset.csv   flattened, model-ready table built by
                               ../build_uav_dataset.py from processed/ --
                               this is what configs/uav_dataset_config.yaml
                               points prognostics.run_on_custom_data at,
                               and what replaced NASA C-MAPSS.
  processed/<flight_session>/ raw ALFA/carbonZ UAV flight-log CSVs, one
                               folder per real flight (47 flights). Each
                               folder's CSVs are per-topic exports
                               (battery, IMU, GPS, airspeed, failure
                               status, ...) at that topic's native rate;
                               build_uav_dataset.py merges them per flight.
                               Original .bag/.mat files were left out to
                               keep this package a reasonable size -- the
                               CSVs carry the same data in a form pandas
                               reads directly; see README.md.

NASA C-MAPSS (train_FD001.txt / test_FD001.txt / RUL_FD001.txt) is NOT
shipped here anymore -- see README.md, "Migrating off C-MAPSS".
"""
