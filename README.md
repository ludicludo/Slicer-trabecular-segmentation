# 🔬 Bone Marrow Segmentation and Analysis Pipeline

This repository contains a Python script designed to automate complex quantitative analysis pipelines involving CT and PET scans within the 3D Slicer environment. The resulting analysis data is saved into structured CSV reports.

## 💡 Functionality Overview

The primary script, `bone_marrow_segmentation.py`, performs the following sequence of operations:

1.  **Automatic Segmentation (TotalSegmentator):** Launches TotalSegmentator on the CT volume to automatically segment all bony structures (bones).
2.  **Boney Mask Generation:**
    *   Filters the initial segmentation to retain only the defined skeletal structures.
    *   Merges all segmented bone components into a single unified bone mask (`merged_bones`).
3.  **Trabecular Extraction:**
    *   Extracts trabecular (bone marrow) segments from the initial bone segmentation.
    *   Applies a multi-step filtering process (shrinkage, thresholding [Tuning: -120 to 200], smoothing) to isolate trabecular structures.
    *   Merges all trabecular segments into a single unified trabecular mask (`merged_trabs`).
4.  **Statistical Analysis:** Calculates key metrics for both the total bones and the trabeculae:
    *   Segment Volume (in cm³)
    *   Voxel Count, Mean, Median, Max, Min, Percentile 5 & 95.
5.  **PET Analysis (Bone Marrow):**
    *   Calculates the Median Standardized Uptake Value (SUV) from the trabecular PET segmentation.
    *   Uses this Median SUV to segment high-uptake bone marrow regions within the bony mask.
    *   Compares the volume ratio of (SUV-PET volume / Trabecular volume).
6.  **Tumor/Lesion Analysis:**
    *   Segments lesions using two thresholds: the 95th percentile of the trabeculae and a fixed SUV value of 3.0.
    *   Calculates the volume of these lesions within the total bone mask.
7.  **Reporting:** Aggregates all computed metrics into a final Pandas DataFrame and saves it as `[PatientName]_analysis_results.csv` in a time-stamped directory.

## 🚀 How to Run the Script in 3D Slicer

This script must be executed within the Python console of the 3D Slicer application, as it relies heavily on Slicer modules and GUI components (SegmentEditorWidget).

**Prerequisites:**
*   3D Slicer installed.
*   Slicer Modules: `TotalSegmentator`, `Segment Statistics`, and `Segment Editor` must be enabled.
*   Input Data: You must have loaded a pair of imaging volumes: a **CT Scan** (Modality 'CT') and a **PET Scan** (Modality 'PT'). Both should be loaded as `vtkMRMLScalarVolumeNode`.

**Execution Steps:**

1.  **Save the Script:** Place `bone_marrow_segmentation.py` in a directory accessible by Slicer (e.g., a custom module folder or a known location).
2.  **Open the Python Console:** In 3D Slicer, navigate to the **Modules** tab and open the **Python Console**.
3.  **Import and Execute:** Run the following commands in the console:

    ```python
    # 1. Import the script file
    import bone_marrow_segmentation as bms

    # 2. Execute the main pipeline execution block
    # The script is designed to run the entire pipeline when __name__ == "__main__"
    bms.__main__()
    ```

**⚠️ Important Notes:**

*   **GUI Dependence:** The script initializes and modifies the Slicer GUI (Segment Editor). Ensure the Slicer environment is correctly set up before execution.
*   **Output:** The results are saved to a local directory determined by the filename of the input CT volume and an appended timestamp (e.g., `/path/to/CT_file/20240528103045/patientname_analysis_results.csv`).
*   **Warnings:** The script includes logging (`INFO` and `ERROR`). If errors occur during calculation, check the Slicer logging output.

## 📚 Code Structure

| File | Description |
| :--- | :--- |
| `bone_marrow_segmentation.py` | The main executable pipeline script. |
| `README.md` | This documentation file. |
