🧠 Medical Image Interpolation for 3D Printing (MRI)

Design Semester Project | Under Faculty Supervision
-----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Problem Statement

MRI scans are acquired as discrete 2D slices. When directly stacked for 3D printing, the gap between slices causes visible discontinuities, anatomical breaks, and surface artifacts.

The objective of this project is to interpolate intermediate MRI slices between two adjacent scans such that:

Anatomical structures remain spatially continuous

Tissue boundaries are preserved

The output slices can be stacked seamlessly for 3D reconstruction and printing

---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
Input Data (Actual Project Images)

The project operates on two adjacent grayscale MRI slices of the same anatomical region.

Reference MRI Slice 1

<img width="504" height="512" alt="image" src="https://github.com/user-attachments/assets/d4b24da8-cc49-4a9f-a49e-276518617686" />


Reference MRI Slice 2

<img width="514" height="492" alt="image" src="https://github.com/user-attachments/assets/dd711f49-87bd-4703-9f03-7390a93069f2" />


These are the exact slices provided to the algorithm which was provided by the professor (no synthetic or external data used).

--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🧠 Methodology (Based on Implemented Code)

The interpolation is not a simple linear blend. The algorithm is designed to be structure-aware, combining multiple image analysis techniques.

1️⃣ Image Preprocessing

-> MRI slices loaded in grayscale

->Automatic resizing if dimensions mismatch

->Conversion to float32 for numerical stability

->Intensity normalization for consistent structure analysis

2️⃣ Multi-Scale Decomposition (Structure Preservation)

->Each MRI slice is decomposed into three components:

->Base layer → global anatomical shape

->Mid layer → curved tissue boundaries

->Detail layer → fine textures

Gaussian blurring at different scales is used, and each layer is interpolated independently before recombination.

➡ This prevents loss of anatomical details during interpolation.

3️⃣ Structural Correspondence via Cross-Correlation

-> To ensure anatomical regions correspond correctly between slices:

-> Local normalized cross-correlation is computed using sliding windows

-> Computation is sampled on a grid for efficiency

Produces a correlation confidence map

This helps avoid interpolating unrelated anatomical regions.

4️⃣ Morphological Feature Extraction

Structural importance is identified using:

1. Canny edge detection

2. Morphological dilation around edges

3. Smoothed structure map

This ensures:

-> Higher priority to anatomical boundaries

-> Reduced distortion in critical regions

5️⃣ Adaptive Hybrid Interpolation (Key Contribution)

Two interpolation strategies are combined:

1. Multi-scale interpolation → used in structural regions

2. Linear interpolation → used in smooth regions

A pixel-wise adaptive blend weight is computed using:

->Structural map

->Correlation confidence map

This avoids:

1.Blurring near boundaries

2.Over-warping in homogeneous areas

6️⃣ Intensity Statistics Matching

To maintain medical realism:

->Mean and standard deviation of output are matched to expected values

->Prevents brightness and contrast drift

A final edge-preserving bilateral filter is applied.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🖼️ Interpolated Output (Actual Result)
Interpolated MRI Slice

<img width="497" height="501" alt="image" src="https://github.com/user-attachments/assets/b18bfae4-2542-46a5-9e79-9b50270c7e93" />


This slice:

1. Lies spatially between the two reference slices

2. Preserves curved cortical boundaries

3. Maintains internal anatomical consistency

Shows no visible slice separation, making it suitable for 3D stacking

🖥️ Interactive Analysis Tool (Actual UI)

A Tkinter-based GUI was developed to analyze and validate results visually.

MRI Interpolation Tool Interface

<img width="1083" height="613" alt="image" src="https://github.com/user-attachments/assets/5ffd1f3e-05e9-4310-a7cc-83817ee497c7" />


UI Capabilities

->Load two MRI reference slices

->Adjust interpolation factor (α)

->Generate interpolated slice

->Generate multiple transition frames

->Visually inspect anatomical continuity

->Save any intermediate slice

The UI also displays:

Reference Image 1

Interpolated Result

Reference Image 2
side-by-side for direct comparison

🎞️ Transition Frame Analysis (From UI)

The tool generates a sequence of intermediate slices:

α Value	Description
α = 0.00	Reference Image 1
α = 0.25	Early transition
α = 0.50	Anatomical midpoint
α = 0.75	Late transition
α = 1.00	Reference Image 2

This confirms smooth anatomical evolution, critical for volumetric reconstruction.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📊 Results Summary 

1.Structural elements are carried smoothly across slices

2.Curved boundaries remain spatially consistent

3.Minor texture smoothing observed in high-frequency regions
(a known limitation of motion-based interpolation)

Overall, the output fills the spatial gap realistically, satisfying the requirements for 3D printing preparation.
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🛠️ Tech Stack

Python

OpenCV

NumPy

Matplotlib

Tkinter

PIL
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
🚀 Applications

Medical 3D printing

Anatomical modeling

Volumetric MRI reconstruction

Pre-surgical visualization

Medical image enhancement research
--------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------
📌 Academic Context

This project was completed as a semester-long design project under faculty supervision, with emphasis on:

Medical image integrity

Structural consistency

Practical usability for additive manufacturing
