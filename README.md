# Biconjugate Fracture Network Generator

A Python application for generating **2D stochastic biconjugate fracture networks** with reproducible geometries, connectivity analysis, and visualization. Designed for use in PorePy multi-physics simulations.

## Quick Start

Run the generator with the following command:

```bash
python -m app

```

### Output Structure
The `app` generates 20 random fracture networks stored providing network geometries in csv format (compatible with PorePy) and analysis plots (network, stereonet, rose diagram).

```
app/output/
├── seed_0/
│   ├── raw/
│   │   ├── fractures.csv              # Raw fracture coordinates
│   │   ├── fractures_metadata.csv     # With dip angles, lengths, family IDs
│   │   ├── network.png                # Network visualization
│   │   ├── stereonet.png              # Stereonet projection
│   │   └── rose_dip_directions.png    # Rose diagram of dips
│   └── y_node_processed/              # Post-processed (recommended)
│       ├── fractures.csv
│       ├── fractures_metadata.csv
│       ├── network.png
│       ├── stereonet.png
│       └── rose_dip_directions.png
├── seed_1/
├── seed_2/
├── ...
├── seed_19/
├── statistics.csv         # Aggregated connectivity metrics
└── connectivity.png       # Plot: connectivity vs number of fractures
```

### CSV Format

The main output are the csv files `fractures.csv` compatible with PorePy. Fractures are sorted by family (minimal format containing domain and single fractures):
```
#x1,y1,x2,y2
-500,-3000,500,-2000 
10.5,25.3,12.1,27.7
...
```

For potential post-analysis additional file with additional info is exported - **fractures_metadata.csv** (extended format):
```
fracture_id,start_x,start_y,end_x,end_y,dip_deg,length,family_id
0,10.5,25.3,12.1,27.7,60.0,3.2,0
1,15.2,18.1,8.9,22.4,60.1,7.4,0
...
```

---

## PorePy integration

A minimal example script (`porepy_example.py`) is provided that integrates the generated fracture network for a single phase flow simulation.

1. **Loads the fracture network** from `app/output/.../fractures.csv` using PorePy's fracture importer.
2. **Creates a mixed-dimensional grid** (fractures + matrix).
3. **Runs a single-phase flow simulation** on the network.
4. **Exports results** to Paraview format (`.vtu` files).

### Usage

PorePy needs to be installed in order to run the example. This repo and associated docker image do not come with a working PorePy version.

### How it works

The `ImportedGeometry` class reads the CSV:

```python
self.fracture_network = pp.fracture_importer.network_from_csv(
    Path("app/output/.../fractures.csv"),
    has_domain=True,
    tol=expected_domain_size * 1e-6,
)
```

The first line of `fractures.csv` defines the domain; subsequent lines define elliptical fractures.

The `SinglePhaseFlowGeometry` class combines the imported geometry with PorePy's `SinglePhaseFlow` model. Running `pp.ModelRunner(model).run()` meshes, assembles, solves, and exports results.


---

## Overview

### What the Generator Does

This generator creates fracture networks with:

- **Two conjugate fracture families** - Two sets of fractures at different dips, creating intersecting networks
- **Stochastic placement** - Random positioning with configurable statistical distributions
- **Reproducible results** - Fixed random seeds ensure identical networks across runs
- **Connectivity metrics** - Analyzes fracture intersections and network complexity
- **Multiple representations** - Raw fractures + post-processed Y-node networks

### Key Features

**TOML-based configuration** - Simple parameter file
**Batch generation** - Generate multiple random seeds and ready for subset sizes in one run  
**Automatic sorting** - Reorders fractures to maximize connectivity at each subset size  
**Statistical analysis** - Computes mean/std dev of connectivity across seeds  
**Visualization suite** - Network plots, stereonets, and rose diagrams included  

---

## Configuration

The generator is configured via `app/config.toml`. Edit this file to customize the generation process.

### Domain Configuration

```toml
[domain]
xmin = -500.0
ymin = -3000.0
xmax = 500.0
ymax = -2000.0

[subdomain]
xmin = -25.0
ymin = -2525.0
xmax = 25.0
ymax = -2475.0
```

**Domain**: The full bounding box (for visualization and boundary setup)  
**Subdomain**: Region where fractures are generated (typically smaller)

### Fracture Family Definition

```toml
[sampler]
num = 2  # Number of fracture families (typically 2 for biconjugate)

[sampler.1]
target_num = 10

[sampler.1.major_axis_length]
mean = 50.0
stddev = 10.0

[sampler.1.rotation_deg]
mean = 60.0      # Dip angle from vertical (degrees)
stddev = 7.5     # Variation in dip angle

[sampler.2]
target_num = 10

[sampler.2.major_axis_length]
mean = 50.0
stddev = 10.0

[sampler.2.rotation_deg]
mean = 30.0      # Different dip angle for conjugate family
stddev = 7.5
```

**Key parameters:**
- `target_num` - Target number of fractures per family
- `major_axis_length` - Distribution of fracture lengths (mean ± stddev)
- `rotation_deg` - **Dip angle** = angle from vertical in degrees
  - 0° = horizontal (left-right)
  - 90° = vertical (up-down)
  - Different angles for each family create the conjugate pair

### Constraints

```toml
[constraints]
min_distance = 5                          # Min distance between fractures
min_intersecting_angle_deg_self = 10.0    # Min angle between fractures in same family
min_intersecting_angle_deg_other = 20.0   # Min angle between fractures in different families
min_intersection_distance = 5              # Min distance from existing intersection points
```

These prevent degenerate configurations (e.g., overlapping or perfectly parallel fractures). `min_distance` controls the distance of fractures within one family. `min_intersecting_angle_deg_self`/`min_intersecting_angle_deg_other` controls the angle of intersecting fractures within and across families. `min_intersection_distance` controls the minimal disctance from intersection to end points of fractures.

### Post-Processing

```toml
[postprocessing]
extension_threshold = 5          # Distance threshold for extending fracture tips
extension_max_iterations = 10    # Max iterations for extension process
branch_proximity_tolerance = 5.0 # Distance tolerance for branch merging
trim_short_branch_length = 5.0   # Trim branches shorter than this
```

Post-processing cleans up fracture intersections, handling **Y-nodes** (three-way intersections) and ensuring topological consistency.

### Generation Parameters

Edit these in `app/__init__.py` `main()` function:

```python
regenerate_networks = True  # True: regenerate, False: skip regeneration
N_seeds = 20               # Number of random seeds to generate
N_fractures = 10           # Max number of fractures per family to analyze
```

NOTE: `app/__init__.py` is tailored to create biconjugate functions with families with same number of fractures per family (need to be consistent with config). Number of seeds are controlled

---

## Generation Process

### Step 1: Generation (Raw Network)

The `FractureGenerator` creates random fractures:

1. **Sample fracture parameters** - Length and orientation from configured distributions
2. **Random placement** - Position fractures in subdomain
3. **Apply constraints** - Check minimum distances and angles
4. **Check intersections** - Detect where fractures cross
5. **Output** - Save raw network (may have messy intersections)

### Step 2: Post-Processing (Y-Node Cleaned)

The post-processor cleans up intersections:

1. **Identify Y-nodes** - Find three-way intersections
2. **Extend tips** - Extend nearby fracture tips to make proper connections
3. **Trim branches** - Remove short dangling segments
4. **Simplify topology** - Ensure clean T-junctions and crosses

**Why post-processing matters:**
- Raw networks may have fracture tips that almost-but-don't-quite touch other fractures
- Small numerical errors can create degenerate geometry
- Y-node processing ensures topologically consistent networks for simulation

### Step 3: Sorting

Fractures are reordered to **maximize connectivity at each subset size**:

- First fractures of each family must intersect each other
- Each subsequent fracture adds as many new intersections as possible
- This enables taking subsets: first 5 fractures from each family will have near-maximal connectivity for that size

This is crucial for **parameter sweeps** where you want to study how network complexity affects solution behavior.

### Step 4: Analysis

For each seed, compute metrics for subsets of size 1, 2, ..., N_fractures:

```
seed_0: 1+1 fracs → 0 intersections
        2+2 fracs → 4 intersections
        3+3 fracs → 8 intersections
        ...
        10+10 fracs → 127 intersections

seed_1: 1+1 fracs → 0 intersections
        ...
```

Aggregate across all seeds to compute **mean ± std dev** of connectivity for each subset size.

---