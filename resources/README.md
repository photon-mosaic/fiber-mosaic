# Testing Data

Real fiber photometry recordings ported from GuPPy test suite. Each session has been truncated from the corresponding full recording in `testing_data/` to reduce file size while retaining enough data to exercise the relevant code paths — for sessions with TTL events, the duration was chosen to capture at least 5 complete pulses. To regenerate from the full recordings, run `src/guppy/testing/scripts/create_stubbed_testing_data.py`.

---

# TDT

Tucker-Davis Technologies (TDT) Synapse tank format. Each session is a folder of binary files (`.tev`, `.tsq`, `.Tbk`, `.Tdx`, `.tin`, `.tnt`).

## `tdt/Photo_63_207-181030-103332`

Standard clean recording used as the baseline TDT case. Also used for cross-correlation testing with a dual-region storename map (DMS + DLS). Duration: 157.5 s (5th port-entry event at ~157.4 s).

**Stores:**
- `Dv1A`: isosbestic control recording from the dorsomedial striatum
- `Dv2A`: calcium signal recording from the dorsomedial striatum
- `Dv3B`: isosbestic control recording from the dorsolateral striatum
- `Dv4B`: calcium signal recording from the dorsolateral striatum
- `Fi1i`: TODO — description needed
- `Fi1r`: 2xN array of modulated commanded voltages: dms and dls
- `LNRW`: TTL event for each rewarded nose poke
- `LNnR`: TTL event for each unrewarded nose poke
- `PrtN`: TTL event for each unrewarded port entry
- `PrtR`: TTL event for each rewarded port entry
- `RNPS`: TODO — description needed

## `tdt/Photometry-161823`

Recording whose TTL store (`PAB/`) exhibits non-contiguous event blocks that the extractor splits into sub-events. Used to test TDT split-TTL handling. Duration: 215.9 s (5th TTL event at ~215.8 s).

`PAB/` splits into sub-events `PAB_0`, `PAB_16`, `PAB_2064` during extraction.

> **TODO (ask Venus):** What does `PAB/` stand for, and what do the sub-event suffixes `0`, `16`, and `2064` represent?

**Stores:**
- `405R`: 405 nm excitation channel (isosbestic reference)
- `490R`: 490 nm excitation channel (calcium signal)
- `DelF`: TODO — description needed
- `Fi1i`: TODO — description needed
- `Fi1r`: Modulated commanded voltage
- `PAB/`: TTL event store (splits into `PAB_0`, `PAB_16`, `PAB_2064` sub-events)
    - `PAB_0`: TODO — description needed
    - `PAB_16`: TODO — description needed
    - `PAB_2064`: TODO — description needed
- `Tick`: TODO — description needed
- `Vid1`: TODO — description needed

## `tdt/Photo_048_392-200728-121222`

Recording with artifactual transients in the raw signal. Primary purpose is testing artifact removal (the artifact-removal pipeline must detect and either concatenate around or replace with NaN). Duration: 184.3 s (5th port-entry event at ~184.2 s).

**Stores:**
- `Dv1A`: isosbestic control recording from the dorsomedial striatum
- `Dv2A`: calcium signal recording from the dorsomedial striatum
- `Dv3B`: isosbestic control recording from the dorsolateral striatum
- `Dv4B`: calcium signal recording from the dorsolateral striatum
- `Fi1i`: TODO — description needed
- `Fi1r`: 2xN array of modulated commanded voltages: dms and dls
- `LNRW`: TTL event for each rewarded nose poke
- `LNnR`: TTL event for each unrewarded nose poke
- `PrtN`: TTL event for each unrewarded port entry
- `PrtR`: TTL event for each rewarded port entry

---

# Doric

Doric Lenses photometry recordings. Three file format variants appear across the sessions: V1 (`.doric` HDF5 with flat channel keys), V6 (`.doric` HDF5 with hierarchical path keys), and CSV export (`.csv`).

## `doric/sample_doric_1`

Standard Doric V1 recording with a TTL channel. Baseline case for Doric format testing. Duration: 115.2 s (5th TTL pulse end at ~115.1 s).

**File:** `D2-EPConsole_0039.doric` (Doric V1)

**Stores:**
- `AIn-1 - Raw`: TODO — description needed
- `AIn-2 - Raw`: TODO — description needed
- `DI--O-1`: digital input/output TTL channel

## `doric/sample_doric_2`

Doric CSV export format: a `.csv` file that follows Doric channel naming conventions rather than the GuPPy generic CSV layout. Used to test CSV export format detection and parsing. Duration: 78.0 s (5th TTL pulse end at ~77.9 s).

**File:** `12282020-cfc-pppda7_0000.csv` (Doric CSV export)

**Stores:**
- `AIn-1 - Dem (ref)`: analog input 1 demodulated reference channel (isosbestic control)
- `AIn-1 - Dem (da)`: analog input 1 demodulated dopamine channel
- `Raw`: raw analog input channel
- `DI/O-1`: digital input/output TTL channel
- `AOut-1`: analog output channel 1
- `AOut-2`: analog output channel 2
- `Unnamed: 7`: TODO — description needed

## `doric/sample_doric_3`

Doric V6 recording. The V6 format stores channels under hierarchical HDF5 paths (e.g., `CAM1_EXC1/ROI01`) rather than the flat keys used in V1. Used to test V6 path parsing. Duration: 16.0 s (TTL events occur at ~0.1 s intervals so 16 s captures many pulses).

**File:** `BFPD_Acq_0000.doric` (Doric V6)

**Stores:**
- `CAM1_EXC1/ROI01`: camera 1, excitation 1, region of interest 1
- `CAM1_EXC1/ROI02`: camera 1, excitation 1, region of interest 2
- `CAM1_EXC1/ROI03`: camera 1, excitation 1, region of interest 3
- `CAM1_EXC2/ROI01`: camera 1, excitation 2, region of interest 1
- `CAM1_EXC2/ROI02`: camera 1, excitation 2, region of interest 2
- `CAM1_EXC2/ROI03`: camera 1, excitation 2, region of interest 3
- `DigitalIO/CAM1`: digital I/O channel synchronized to camera 1 (TTL)
- `DigitalIO/EXC1`: digital I/O channel synchronized to excitation 1
- `DigitalIO/EXC2`: digital I/O channel synchronized to excitation 2

## `doric/sample_doric_4`

Doric V6 lock-in amplifier recording with no TTL events. One of two independent examples of the no-TTL lock-in format. Duration: 16.0 s.

**File:** `LiCl_0001.doric` (Doric V6 LockIn)

**Stores:**
- `Series0001/AIN01xAOUT01-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 01
- `Series0001/AIN01xAOUT02-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 02
- `Series0001/AIN03xAOUT01-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 01
- `Series0001/AIN03xAOUT02-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 02
- `AnalogIn/AIN01`: raw analog input channel 01
- `AnalogIn/AIN03`: raw analog input channel 03
- `AnalogOut/AOUT01`: analog output channel 01
- `AnalogOut/AOUT02`: analog output channel 02

## `doric/sample_doric_5`

Second independent example of the Doric V6 lock-in amplifier format with no TTL events. Duration: 16.0 s.

**File:** `saline_0001.doric` (Doric V6 LockIn)

**Stores:**
- `Series0001/AIN01xAOUT01-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 01
- `Series0001/AIN01xAOUT02-LockIn`: lock-in amplifier output for analog input 01 demodulated by analog output 02
- `Series0001/AIN03xAOUT01-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 01
- `Series0001/AIN03xAOUT02-LockIn`: lock-in amplifier output for analog input 03 demodulated by analog output 02
- `AnalogIn/AIN01`: raw analog input channel 01
- `AnalogIn/AIN03`: raw analog input channel 03
- `AnalogOut/AOUT01`: analog output channel 01
- `AnalogOut/AOUT02`: analog output channel 02

---

# CSV

GuPPy generic CSV format: one two-column file per channel (timestamps, data values).

## `csv/sample_data_csv_1`

Standard generic CSV recording. Baseline case for Steps 2–5 integration tests and consistency tests (z-score methods, no-isosbestic control, dFF). Duration: 411.0 s (5th TTL event at ~410.9 s).

**Files:** `Sample_Control_Channel.csv`, `Sample_Signal_Channel.csv`, `Sample_TTL.csv`

**Stores:**
- `Sample_Control_Channel`: isosbestic control channel
- `Sample_Signal_Channel`: calcium signal channel
- `Sample_TTL`: TTL event channel

---

# NPM (Neurophotometrics)

Neurophotometrics fiber photometry recordings. Two format generations are present: v2 (files contain a `LedState` header column) and legacy (no `LedState` header, rows interleaved by LED state).

NPM discovery writes intermediate per-channel CSV files into the session folder (`file0_chev*.csv`, `file0_chod*.csv`, event CSVs). These intermediate files are not stored in the stub — they are created fresh each time Step 2 runs. Store names below reflect the full set of stores available after discovery and any split-events processing.

## `npm/sampleData_NPM_1`

NPM v2 recording with a separate stimuli event file. The stimuli file contains multiple named event types; with `split_events=True`, each type becomes its own event store. Duration: 120.3 s (5th stimuli event at ~120.2 s).

**Files:** `bl72bl82_12feb2024_fp.csv` (photometry, v2), `bl72bl82_12feb2024_stimuli.csv` (events)

**Stores (after discover + split):**
- `file0_chev1`: TODO — description needed
- `file0_chod1`: TODO — description needed
- `eventAfVn`: TODO — description needed
- `eventAfVu`: TODO — description needed
- `eventAmVf`: TODO — description needed
- `eventpinknoise`: TTL event for pink noise stimulus delivery
- `eventwhitenoise`: TTL event for white noise stimulus delivery

## `npm/sampleData_NPM_2`

NPM v2 recording split across two source files (one per excitation wavelength), with no TTL events. Used to test multi-file v2 discovery and cross-file channel alignment. Duration: 16.0 s.

**Files:** `FiberData415.csv` (415 nm excitation, v2), `FiberData470.csv` (470 nm excitation, v2)

**Stores (after discover):**
- `file0_chev1`: TODO — description needed
- `file0_chev2`: TODO — description needed
- `file0_chev3`: TODO — description needed
- `file0_chev4`: TODO — description needed
- `file0_chev5`: TODO — description needed
- `file0_chev6`: TODO — description needed
- `file0_chev7`: TODO — description needed
- `file1_chev1`: TODO — description needed
- `file1_chev2`: TODO — description needed
- `file1_chev3`: TODO — description needed
- `file1_chev4`: TODO — description needed
- `file1_chev5`: TODO — description needed
- `file1_chev6`: TODO — description needed
- `file1_chev7`: TODO — description needed

## `npm/sampleData_NPM_3`

NPM v2 recording with 4 fiber channels and non-standard timestamp columns. The photometry file uses a `ComputerTimestamp` column (milliseconds) rather than the default timestamp column (seconds). Copied as-is from the original (too small to stub without breaking tests).

**Files:** `signals.csv` (photometry, v2, 4 channels), `ttls.csv` (events, values 1 and 3)

**Stores (after discover + split):**
- `file0_chev1`: TODO — description needed
- `file0_chev2`: TODO — description needed
- `file0_chev3`: TODO — description needed
- `file0_chev4`: TODO — description needed
- `file0_chod1`: TODO — description needed
- `file0_chod2`: TODO — description needed
- `file0_chod3`: TODO — description needed
- `file0_chod4`: TODO — description needed
- `event1`: TTL events with value 1 in the event column
- `event3`: TTL events with value 3 in the event column

## `npm/sampleData_NPM_4`

NPM legacy format (no `LedState` header, rows interleaved by LED state). The event file contains boolean `True`/`False` values; with `split_events=True`, these become separate `eventTrue` and `eventFalse` stores. Also used for Step 2 idempotency testing (running Step 2 twice must not corrupt modality detection). Duration: 578.0 s (10th TTL event — 5 True + 5 False — at ~577.3 s).

**Files:** `PagCeAVgatFear_14421.csv` (photometry, legacy), `PagCeAVgatFear_1442_ts0.csv` (events)

> **TODO (ask Venus):** What does `ts0` stand for in the filename, and what does the `True`/`False` split in the event column represent?

**Stores (after discover + split):**
- `file0_chev1`: TODO — description needed
- `file0_chev2`: TODO — description needed
- `file0_chev3`: TODO — description needed
- `file0_chod1`: TODO — description needed
- `file0_chod2`: TODO — description needed
- `file0_chod3`: TODO — description needed
- `eventTrue`: TTL events where the event column value is `True`
- `eventFalse`: TTL events where the event column value is `False`

## `npm/sampleData_NPM_5`

Second NPM legacy format recording. Unlike `sampleData_NPM_4`, the event file contains a single event type with no boolean split. Copied as-is from the original (too small to stub without breaking tests). Also used for stub idempotency and duration unit tests.

**Files:** `PagCeAVgatFear_1512_1.csv` (photometry, legacy), `PagCeAVgatFear_1512_ts0.csv` (events)

> **TODO (ask Venus):** Same `ts0` question as `sampleData_NPM_4` above.

**Stores (after discover):**
- `file0_chev1`: TODO — description needed
- `file0_chev2`: TODO — description needed
- `file0_chev3`: TODO — description needed
- `file0_chod1`: TODO — description needed
- `file0_chod2`: TODO — description needed
- `file0_chod3`: TODO — description needed
- `event0`: TODO — description needed

---

# NWB (Neurodata Without Borders)

NWB format recordings using the `ndx-fiber-photometry` and `ndx-events` extensions. Generated programmatically rather than truncated from real recordings; to regenerate, run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.py`.

## `nwb/mock_nwbfile`

Minimal synthetic NWB file for testing the NWB recording extractor against the current version of `ndx-fiber-photometry`. Contains 3000 samples at 30 Hz across 2 channels (control and signal) and three event types. To regenerate, run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2.py`.

**File:** `mock_nwbfile.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `events`: 10 timestamps (45–54 s); plain `ndx_events.Events`
- `labeled_events`: 15 timestamps (40–54 s) with 3 labels (`label_1`, `label_2`, `label_3`); `ndx_events.LabeledEvents`
- `AnnotatedEventsTable`: two event types; `ndx_events.AnnotatedEventsTable`
  - `Reward`: timestamps at 41–45 s
  - `Punishment`: timestamps at 55–59 s

## `nwb/mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2`

Identical synthetic data to `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2`, but the NWB file was created using `ndx-fiber-photometry==0.1.0`. The ndx-fiber-photometry v0.1.0 API differs from the current version in that device classes (`Indicator`, `OpticalFiber`, `ExcitationSource`, `Photodetector`, `DichroicMirror`, `BandOpticalFilter`) all live directly in `ndx_fiber_photometry` (no separate `ndx-ophys-devices` dependency), and `FiberPhotometry` only holds a `FiberPhotometryTable` (no virus/injection/indicator containers). Used to verify that the NWB extractor can read files produced by the older extension version. To regenerate, create the isolated conda environment at `src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_1_ndx_events_v0_2.yaml` and run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.py`.

**File:** `mock_nwbfile_ndx_fiber_photometry_v0_1_ndx_events_v0_2.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `events`: 10 timestamps (45–54 s); plain `ndx_events.Events`
- `labeled_events`: 15 timestamps (40–54 s) with 3 labels (`label_1`, `label_2`, `label_3`); `ndx_events.LabeledEvents`
- `AnnotatedEventsTable`: two event types; `ndx_events.AnnotatedEventsTable`
  - `Reward`: timestamps at 41–45 s
  - `Punishment`: timestamps at 55–59 s

## `nwb/mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_4`

Identical photometry data to `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_2`, but the NWB file was created using `ndx-events==0.4`. The ndx-events v0.4 API differs from the earlier version: events are stored as `EventsTable` objects (registered via `NdxEventsNWBFile.add_events_table()`) rather than the `Events`, `LabeledEvents`, and `AnnotatedEventsTable` types used in prior versions. Categorical event types are represented by a `CategoricalVectorData` column with a linked `MeaningsTable`. Used to verify that the NWB extractor can read files produced with `ndx-events==0.4`. To regenerate, create the isolated conda environment at `src/guppy/testing/scripts/environment_ndx_fiber_photometry_v0_2_ndx_events_v0_4.yaml` and run `src/guppy/testing/scripts/create_mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_4.py`.

**File:** `mock_nwbfile_ndx_fiber_photometry_v0_2_ndx_events_v0_4.nwb`

**Photometry data:**
- `fiber_photometry_response_series`: 3000-sample × 2-channel array at 30 Hz; column 0 = control (isosbestic, 405 nm excitation), column 1 = signal (470 nm excitation)

**Events:**
- `simple_events`: 10 timestamps (45–54 s); plain `EventsTable` with no categorical columns; discovers as one event `simple_events`
- `categorized_events`: 10 timestamps (41–59 s) with a `CategoricalVectorData` column `event_type`; discovers as two events:
  - `categorized_events_event_type_Reward`: timestamps at 41–45 s
  - `categorized_events_event_type_Punishment`: timestamps at 55–59 s
