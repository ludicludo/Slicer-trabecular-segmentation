"""
Bone Marrow and Bone Segmentation Analysis Script.

This script automates the process of:
1.  Performing automatic bone segmentation (using TotalSegmentator) on a CT scan.
2.  Extracting trabecular bone segments from the main bone volume.
3.  Merging all bone segments and trabecular segments into unified volumes.
4.  Calculating various statistical metrics (volume, mean, median, etc.) for bones and trabeculae.
5.  Performing thresholding analysis on PET scans, restricted within the bony mask.
6.  Generating a final DataFrame report of all calculated results and saving it to a time-stamped directory.

Note: This script relies heavily on the Slicer environment (vtkMRML*, slicer modules, and active SegmentEditor/SegmentEditorWidget).
"""
from pathlib import Path
import slicer
from slicer import vtkMRMLScalarVolumeNode, vtkMRMLSegmentationNode
from slicer.util import getNodesByClass
import vtkSegmentationCorePython as vtkSegmentationCore
import TotalSegmentator
import vtk
import numpy as np
import pandas as pd
import logging
import re
import time

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# === CONFIGURATION ===
# Flag to indicate if the analysis should only consider bony structures.
ONLY_BONES = True
DEFAULT_DIR = Path(".")

# List of standard anatomical labels considered as bones/skeletal structures.
BONES_LABEL_IDS = (
    "sacrum", "vertebrae_S1", "vertebrae_L5", "vertebrae_L4", "vertebrae_L3", "vertebrae_L2", "vertebrae_L1",
    "vertebrae_T11", "vertebrae_T12", "vertebrae_T10", "vertebrae_T9", "vertebrae_T8", "vertebrae_T7",
    "vertebrae_T6", "vertebrae_T5", "vertebrae_T4", "vertebrae_T3", "vertebrae_T2", "vertebrae_T1",
    "vertebrae_C7", "vertebrae_C6", "vertebrae_C5", "vertebrae_C4", "vertebrae_C3", "vertebrae_C2", "vertebrae_C1",
    "humerus_left", "humerus_right", "scapula_left", "scapula_right", "clavicula_left", "clavicula_right",
    "femur_left", "femur_right", "hip_left", "hip_right",
    "rib_left_1", "rib_left_2", "rib_left_3", "rib_left_4", "rib_left_5", "rib_left_6",
    "rib_left_7", "rib_left_8", "rib_left_9", "rib_left_10", "rib_left_11", "rib_left_12",
    "rib_right_1", "rib_right_2", "rib_right_3", "rib_right_4", "rib_right_5", "rib_right_6",
    "rib_right_7", "rib_right_8", "rib_right_9", "rib_right_10", "rib_right_11", "rib_right_12",
    "sternum", "costal_cartilages",
)

## === STATISTICS CALCULATION CLASS ===
class StatisticsCalculator:
    """
    Calculates and exports detailed statistics (volume, mean, median, etc.) 
    for segments within a given segmentation node referenced against a scalar volume.
    """
    def __init__(self, segmentation_node: vtkMRMLSegmentationNode,
                 reference_volume_node: vtkMRMLScalarVolumeNode) -> None:
        """
        Initializes the calculator by linking to the Slicer Segment Statistics logic.
        
        Args:
            segmentation_node: The segmentation node containing the segments to analyze.
            reference_volume_node: The scalar volume (e.g., CT/PET) used as reference.
        """
        self.segStatLogic = slicer.modules.segmentstatistics.widgetRepresentation().self().logic
        self.segmentation_node = segmentation_node
        self.reference_volume_node = reference_volume_node

    def compute_segment_statistics(self) -> dict[str, dict[str, float]]:
        """
        Computes and returns segment statistics for all segments in the segmentation node.

        Returns:
            A dictionary containing statistics grouped by SegmentIDs.
        """
        try:
            # Set up the logic parameters
            self.segStatLogic.getParameterNode().SetParameter("Segmentation", 
                                                         self.segmentation_node.GetID())
            self.segStatLogic.getParameterNode().SetParameter("ScalarVolume", 
                                                         self.reference_volume_node.GetID())
            
            # Disable specific plugins
            self.segStatLogic.getParameterNode().SetParameter("LabelmapSegmentStatisticsPlugin.enabled", 
                                                         "False")
            self.segStatLogic.getParameterNode().SetParameter("ClosedSurfaceSegmentStatisticsPlugin.enabled", 
                                                         "False")
            # Enable scalar volume statistics
            self.segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.enabled", 
                                                         "True")
            
            # Enable desired metrics
            metrics = ["voxel_count", "volume_cm3", "mean", "median", "max", "min", "percentile_05", "percentile_95"]
            for metric in metrics:
                metric_key = f'ScalarVolumeSegmentStatisticsPlugin.{metric}.enabled'
                self.segStatLogic.getParameterNode().SetParameter(metric_key, "True")
            self.segStatLogic.getParameterNode().SetParameter("ScalarVolumeSegmentStatisticsPlugin.volume_mm3.enabled", "False")
            
            # Compute and retrieve
            self.segStatLogic.computeStatistics()
            stats = self.segStatLogic.getStatistics()
            
            logger.info(f"Statistics calculated for {len(stats.get('SegmentIDs', []))} segments.")
            return stats
            
        except Exception as e:
            logger.error(f"Error during statistic calculation: {e}")
            return {}

    def export_to_table(self) -> slicer.vtkMRMLTableNode:
        """
        Computes segment statistics and exports the results to a Slicer Table Node.

        Returns:
            The created vtkMRMLTableNode.
        """
        table_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLTableNode")
        table_node.SetName(self.segmentation_node.GetName() + "_Stats")
        
        stats = self.compute_segment_statistics()
        self.segStatLogic.exportToTable(table_node)
        return table_node


## === SEGMENTATION UTILITIES ===

def copy_segment(segmentation: vtkSegmentationCore.vtkSegmentation, 
                 segment_id: str, new_seg_name: str) -> None:
    """
    Copies a single segment from the current segmentation into a new segment 
    within the same segmentation.

    Args:
        segmentation: The vtkSegmentation object to modify.
        segment_id: The ID of the segment to copy.
        new_seg_name: The desired name for the new segment.
    """
    new_segment = vtkSegmentationCore.vtkSegment()
    new_segment.DeepCopy(segmentation.GetSegment(segment_id))
    new_segment.SetName(new_seg_name)
    segmentation.AddSegment(new_segment, new_seg_name)


def smooth_segment(segment_id: str, radius_mm: int) -> None:
    """
    Applies median filtering (smoothing) to a segment ID using Slicer's SegmentEditor.

    NOTE: Requires active Slicer SegmentEditorNode/Widget setup.

    Args:
        segment_id: The ID of the segment to smooth.
        radius_mm: The kernel size (filter radius) in millimeters.
    """
    try:
        segmentEditorNode.SetSelectedSegmentID(segment_id)
        segmentEditorNode.SetActiveEffectName("Smoothing")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("SmoothingMethod", "MEDIAN")
        effect.setParameter("KernelSizeMm", str(radius_mm))
        effect.self().onApply()
    except AttributeError:
        logger.error(f"Smoothing issue encountered for segment {segment_id}. SegmentEditorWidget/Node might not be initialized.")
        raise AttributeError(f"SegmentEditor setup failed for {segment_id}")


def threshold_segment(segment_id: str, threshold: tuple[float, float], 
                       mask_id: str = None) -> None:
    """
    Applies intensity thresholding to a segment ID using Slicer's SegmentEditor, 
    optionally masked by another segment.

    NOTE: Requires active Slicer SegmentEditorNode/Widget setup.

    Args:
        segment_id: The ID of the segment to threshold.
        threshold: A tuple (min_threshold, max_threshold).
        mask_id: Optional segment ID to restrict the threshold operation to.
    """
    try:
        segmentEditorNode.SetSelectedSegmentID(segment_id)
        segmentEditorNode.SetActiveEffectName("Threshold")
        
        if mask_id:
            segmentEditorNode.SetMaskSegmentID(mask_id)
            segmentEditorNode.SetMaskMode(5) # AND operation
            
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MinimumThreshold", str(threshold[0]))
        effect.setParameter("MaximumThreshold", str(threshold[1]))
        effect.self().onApply()
    except AttributeError:
        logger.error("Thresholding issue encountered. SegmentEditorWidget/Node might not be initialized.")
        raise AttributeError("SegmentEditor setup failed for thresholding")


def shrink_segment(segment_id: str, factor_mm: int) -> None:
    """
    Applies dilation or contraction (Margin operation) to a segment.

    Args:
        segment_id: The ID of the segment to modify size.
        factor_mm: The margin size in millimeters. Positive for dilation, negative for contraction.
    """
    try:
        segmentEditorNode.SetSelectedSegmentID(segment_id)
        segmentEditorNode.SetActiveEffectName("Margin")
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MarginSizeMm", str(factor_mm))
        effect.setParameter("ApplyToAllVisibleSegments", str(0))
        effect.self().onApply()
    except AttributeError:
        logger.error(f"Shrinkage issue encountered for segment {segment_id}. SegmentEditorWidget/Node might not be initialized.")
        raise AttributeError(f"SegmentEditor setup failed for {segment_id}")


## === CORE ANALYSIS PIPELINES ===


def trabecular_segmentation(segmentation_node: vtkMRMLSegmentationNode, volume_node: vtkMRMLScalarVolumeNode) -> vtkMRMLSegmentationNode:
    """
    Extracts trabecular bone segments from the main bone segmentation.

    The process involves:
    1.  Copying each bone segment.
    2.  Shrinking the copied segment.
    3.  Applying thresholding and smoothing to isolate trabeculae.
    4.  Removing the original segment ID to maintain a clean structure.

    Args:
        segmentation_node: The segmentation containing all bony structures.
        volume_node: The scalar volume (e.g., CT) used for referencing.

    Returns:
        A new segmentation node containing only the extracted trabecular segments.
    """
    try:
        seg_node_trab = slicer.util.getNode('trab_bones')
    except slicer.util.MRMLNodeNotFoundException:    
        # Create the segment node if it doesn't exist
        seg_node_trab = slicer.mrmlScene.AddNewNodeByClassWithID('vtkMRMLSegmentationNode', 
                                                                 'trab_bones', 
                                                                 'trab_bones')
    
    if seg_node_trab.GetDisplayNode() is None:
        seg_node_trab.CreateDefaultDisplayNodes()
        
    trab_segmentation = seg_node_trab.GetSegmentation()
    
    # Set up Slicer widgets for the operation
    segmentEditorWidget.setSegmentationNode(seg_node_trab)
    segmentEditorWidget.setSourceVolumeNode(volume_node)
    
    for segment_id in segmentation_node.GetSegmentation().GetSegmentIDs():
        try:
            # 1. Copy segment
            trab_segmentation.CopySegmentFromSegmentation(segmentation_node.GetSegmentation(), 
                                                                            segment_id, 
                                                                            False)
            
            # 2. Shrink
            shrink_segment(segment_id, -2)
            
            # 3. Copy as new segment ("_trab" suffix)
            copy_segment(trab_segmentation, segment_id, segment_id + "_trab")
            trab_segmentation.Modified() # Ensure modification is registered before next steps
            
            # 4. Threshold and Smooth
            threshold_segment(segment_id + "_trab", 
                            (-120, 200), # Generic threshold range
                            segment_id)
            smooth_segment(segment_id + "_trab", 2)
            
            # 5. Cleanup original segment
            trab_segmentation.RemoveSegment(segment_id)
            
        except (AttributeError, RuntimeError) as e:
            logger.error(f"Trabecular extraction of {segment_id} failed: {e}")
            
    trab_segmentation.Modified()
    return seg_node_trab


def bones_auto_segmentation(ct_node: vtkMRMLScalarVolumeNode, seg_ids: list[str] = None) -> vtkMRMLSegmentationNode:
    """
    Performs automatic segmentation of bones using TotalSegmentator on a CT scan node.

    Args:
        ct_node: The CT volume node to process.
        seg_ids: Optional list of specific bone IDs to target (defaults to BONES_LABEL_IDS).

    Returns:
        The created vtkMRMLSegmentationNode containing segmented bones.
    """
    seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
    seg_node.SetName("bones segmentation")

    logic = TotalSegmentator.TotalSegmentatorLogic()
    logic.loadTotalSegmentatorLabelTerminology()
    
    logger.info("Starting TotalSegmentator processing...")
    logic.process(ct_node, seg_node, fast=True)
    
    seg_node.SetReferenceImageGeometryParameterFromVolumeNode(ct_node)
    logger.info("TotalSegmentator completed.")
    return seg_node


def merge_bones(seg_bones_node: vtkMRMLSegmentationNode, output_segment_name: str,
                 ref_volume: vtkMRMLSegmentationNode) -> vtkMRMLSegmentationNode:
    """
    Merges all segment IDs from a source segmentation node into a single segment 
    (represented by the output_segment_name) within a new segmentation node.

    Args:
        seg_bones_node: The source segmentation node (e.g., all bones, or only trabeculae).
        output_segment_name: The name to give to the final merged segment.
        ref_volume: The reference segmentation node (used to establish initial geometry consistency).

    Returns:
        A new vtkMRMLSegmentationNode containing only the merged segment.
    """
    try:
        merge_seg_node = slicer.util.getNode(output_segment_name)
    except slicer.util.MRMLNodeNotFoundException:    
        merge_seg_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        
    merge_seg_node.SetName(output_segment_name)
    if merge_seg_node.GetDisplayNode() is None:
        merge_seg_node.CreateDefaultDisplayNodes()    
        
    seg_id_to_merge = seg_bones_node.GetSegmentation().GetSegmentIDs()
    
    # Use a LabelMap intermediate node for the merge operation
    labelmap = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLLabelMapVolumeNode")
    logic = slicer.modules.segmentations.logic()
    
    # 1. Export segments from source to labelmap
    logic.ExportSegmentsToLabelmapNode(seg_bones_node, seg_id_to_merge, labelmap, ref_volume)
    
    # 2. Convert labelmap array to a binary merge mask
    label_arr = slicer.util.arrayFromVolume(labelmap)
    merge_mask = (label_arr > 0).astype(np.uint8)
    slicer.util.updateVolumeFromArray(labelmap, merge_mask)
    
    # 3. Import the merge mask into the target segmentation node
    logic.ImportLabelmapToSegmentationNode(labelmap, merge_seg_node)
    
    # 4. Set consistent name and modify
    merge_seg_node.GetSegmentation().GetNthSegment(0).SetName(output_segment_name)
    merge_seg_node.GetSegmentation().Modified()
    
    slicer.mrmlScene.RemoveNode(labelmap) # Clean up intermediate node
    return merge_seg_node


def pet_segmentation(pet_node: vtkMRMLScalarVolumeNode, 
                     threshold: float, 
                     mask_seg_node: vtkMRMLSegmentationNode) -> vtkMRMLSegmentationNode:
    """
    Performs PET threshold segmentation, restricting the operation to a specific bone mask.

    This function creates a new PET segmentation incorporating the thresholded
    volume restricted by the provided bone mask.

    Args:
        pet_node: The PET scalar volume node.
        threshold: The intensity threshold (SUV value) to apply.
        mask_seg_node: The segmentation node acting as the mask (e.g., merged bones).

    Returns:
        The newly created segmentation node containing the thresholded PET region.
        Returns None if segmentation process fails.
    """
    max_pet = slicer.util.arrayFromVolume(pet_node).max()
    
    # Determine and initialize the segment node for PET results
    pet_node_name = f"pet_seg_at_{threshold:.1f}"
    try:
        seg_pet_node = slicer.util.getNode(pet_node_name)
    except slicer.util.MRMLNodeNotFoundException:    
        seg_pet_node = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentationNode")
        seg_pet_node.SetName(pet_node_name)
        
    if seg_pet_node.GetDisplayNode() is None:
        seg_pet_node.CreateDefaultDisplayNodes()    
        
    seg_pet_node.SetReferenceImageGeometryParameterFromVolumeNode(pet_node)
    
    # Setup widgets for operation
    segmentEditorWidget.setSourceVolumeNode(pet_node)
    segmentEditorWidget.setSegmentationNode(seg_pet_node)
    
    # Copy the bone mask first
    merged_bones_id = mask_seg_node.GetSegmentation().GetSegmentIDs()[0]
    merge_bones_name = mask_seg_node.GetSegmentation().GetSegment(merged_bones_id).GetName()
    
    try:
        # Add a segment from the mask
        new_segment = vtkSegmentationCore.vtkSegment()
        new_segment.DeepCopy(mask_seg_node.GetSegmentation().GetSegment(merged_bones_id))
        seg_pet_node.GetSegmentation().AddSegment(new_segment, merge_bones_name)
        merge_bone_id = seg_pet_node.GetSegmentation().GetSegmentIdBySegmentName(merge_bones_name)
        
        # Set up and apply thresholding
        segmentEditorNode.SetSelectedSegmentID(f"temp_id") # Selecting a placeholder ID
        segmentEditorNode.SetActiveEffectName("Threshold")
        
        if merge_bone_id:
            segmentEditorNode.SetMaskSegmentID(merge_bone_id)
            segmentEditorNode.SetMaskMode(5) # AND operation
            
        effect = segmentEditorWidget.activeEffect()
        effect.setParameter("MinimumThreshold", str(threshold))
        effect.setParameter("MaximumThreshold", str(max_pet))
        effect.self().onApply()
        
        # Keep the mask segment visible, but the result is governed by the threshold
        seg_pet_node.GetDisplayNode().SetSegmentVisibility(merge_bone_id, True) 
        
    except Exception as e:
        logger.error(f"PET segmentation failed during execution: {e}")
        return None

    return seg_pet_node


## === GENERAL UTILITIES ===

def is_modality(volume: vtkMRMLScalarVolumeNode, modality: str) -> bool:
    """Checks if the volume node has the specified DICOM modality."""
    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    return (sh_node.GetItemAttribute(sh_node.GetItemByDataNode(volume),
                                     "DICOM.Modality") 
                                     == modality.upper())


def get_volume(volumes: list[vtkMRMLScalarVolumeNode], modality: str) -> vtkMRMLScalarVolumeNode:
    """
    Selects the appropriate volume node (e.g., first CT or first PT).
    
    If multiple volumes match, it relies on the input order or predefined logic.

    Args:
        volumes: A list of potential scalar volume nodes.
        modality: The target DICOM modality string ("CT" or "PET").

    Returns:
        The selected vtkMRMLScalarVolumeNode.
        
    Raises:
        IndexError: If no volume matching the modality is found.
    """
    try:
        # Filter to contain only the requested modality
        filtered_volumes = [el for el in volumes if is_modality(el, modality)]
        
        if not filtered_volumes:
            raise IndexError(f"No volume found with modality: {modality.upper()}")
            
        # If multiple exist, pick the first one found
        return filtered_volumes[0]
        
    except IndexError as e:
        logger.error(str(e))
        # Fallback logic from original code (highly fragile, maintained for functional parity)
        if modality.upper() == 'CT' and volumes:
            logger.warning("Falling back to first provided volume for CT.")
            return volumes[0] 
        elif modality.upper() == 'PT' and len(volumes) > 1:
            logger.warning("Falling back to second provided volume for PT.")
            return volumes[1]
        raise e


def get_patient_name() -> str:
    """
    Extracts a clean patient code from the Slicer subject hierarchy node name.
    
    Args:
        None
        
    Returns:
        A cleaned patient code string (e.g., "ABC123").
    """
    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    items = vtk.vtkIdList()
    sh_node.GetItemChildren(sh_node.GetSceneItemID(), items)
    
    if items.GetNumberOfIds() == 0:
        logger.error("Could not find patient hierarchy node.")
        return "UNKNOWN_PATIENT"
        
    name = sh_node.GetItemName(items.GetId(0))
    # Regex replacement pattern: keeps the first word/pattern separated by non-word characters
    return re.sub(r'(\w+\-\w+).*', r"\g<1>", name)


## === MAIN PIPELINE EXECUTION ===
if __name__ == "__main__":
    
    logger.info("--- Starting Bone Marrow Segmentation Workflow ---")
    
    # 1. Volume Identification
    all_volumes = getNodesByClass("vtkMRMLScalarVolumeNode")
    ct = get_volume(all_volumes, "CT")
    pt = get_volume(all_volumes, "PT")
    
    if ct is None or pt is None:
        logger.error("CT or PET volume could not be retrieved. Exiting.")
        exit()
        
    # Utility function to safely get the directory path of the CT volume node
    def get_volume_output_dir(volume_node: vtkMRMLScalarVolumeNode) -> Path:
        storage_node = volume_node.GetStorageNode()
        if storage_node is not None:
            return Path(storage_node.GetFileName()).parent
        else:
            # Fallback: Use a generic path or Slicer's current directory
            logger.warning("Storage Node for CT is None. Falling back to current working directory.")
            return DEFAULT_DIR.resolve()

    # 2. Bone Segmentation
    total_seg_nodes = getNodesByClass('vtkMRMLSegmentationNode')
    
    if not total_seg_nodes:
        logger.info("No existing segmentation found. Running Automatic Bone Segmentation (TotalSegmentator)...")
        seg_node = bones_auto_segmentation(ct)
    else:
        # Assuming the first found segmentation is the primary one
        logger.info("Using existing segmentation node.")
        seg_node = total_seg_nodes[0]
        
    # 3. Filtering and Preparation
    
    # Identify and remove segments that are NOT bony structures (if ONLY_BONES is set)
    no_bones_label_ids = set(seg_node.GetSegmentation().GetSegmentIDs()) - set(BONES_LABEL_IDS)
    
    logger.info(f"Filtering segments. Removing {len(no_bones_label_ids)} non-bone segments.")
    if ONLY_BONES:
        for segment_id in no_bones_label_ids:
            seg_node.RemoveSegment(segment_id)
            
    # Initialize Slicer GUI components for manipulation
    sh_node = slicer.vtkMRMLSubjectHierarchyNode.GetSubjectHierarchyNode(slicer.mrmlScene)
    
    segmentEditorWidget = slicer.qMRMLSegmentEditorWidget()
    segmentEditorWidget.setMRMLScene(slicer.mrmlScene)
    segmentEditorNode = slicer.mrmlScene.AddNewNodeByClass("vtkMRMLSegmentEditorNode")
    segmentEditorWidget.setMRMLSegmentEditorNode(segmentEditorNode)
    segmentEditorWidget.setSegmentationNode(seg_node)
    segmentEditorWidget.setSourceVolumeNode(ct)

    # 4. Specialized Segmentation Pipelines
    
    logger.info("Starting Trabecular Segmentation...")
    trabecular_seg_node = trabecular_segmentation(seg_node, ct)
    
    # Bony mask merging
    logger.info("Merging all bony segments into a single mask...")
    merged_bones_node = merge_bones(seg_node, "merged_bones", ct)
    
    # Trabecular mask merging
    logger.info("Merging all trabecular segments into a single mask...")
    merged_trabs_node = merge_bones(trabecular_seg_node, "merged_trabs", ct)
    
    # 5. Statistical Calculation (Tallying Results)
    
    logger.info("Calculating statistics...")
    # Bone stats
    stats_bones = StatisticsCalculator(seg_node, ct)
    table_bones = stats_bones.export_to_table()
    
    # Trabecular stats (from raw trabecular node)
    stats_trab = StatisticsCalculator(trabecular_seg_node, ct)
    table_trab = stats_trab.export_to_table()
    
    # PET stats on Trabecular structure (to find median SUV in bone marrow)
    stats_pet_trab = StatisticsCalculator(trabecular_seg_node, pt)
    table_pet_trab = stats_pet_trab.export_to_table()
    
    # 6. Data Processing and Ratio Calculation
    
    df_bones = slicer.util.dataframeFromTable(table_bones)
    df_trab = slicer.util.dataframeFromTable(table_trab)
    df_trab_pet = slicer.util.dataframeFromTable(table_pet_trab)
    
    # Volume ratios: (Trabecular Volume / Total Bone Volume)
    # Find volume measurements in all tables
    try:
        trab_bones_cm3 = df_trab[df_trab.index.str.contains("_trab")].filter(like="cm3").sum()["Volume cm3"]
        bones_cm3 = df_bones.filter(like="cm3").sum()["Volume cm3"]
        ratio_trab = trab_bones_cm3 / bones_cm3 if bones_cm3 != 0 else 0
    except Exception as e:
        logger.error(f"Volume data extraction failed: {e}")
        trab_bones_cm3, bones_cm3, ratio_trab = 0.0, 0.0, 0.0
        
    # Median SUV in Bone Marrow (from trabeculae PET scan)
    try:
        median_suv_abm = df_trab_pet["Median"].median() 
    except Exception as e:
        logger.error(f"Required median SUV column not found in PET stats table: {e}")
        median_suv_abm = 1.0
    
    # PET Segmentation 1: SUV <-> TRABECULAR VOLUME
    logger.info("Performing PET segmentation limited by trabecular bone marrow...")
    pet_trab_node = pet_segmentation(pt, round(median_suv_abm, 1), merged_trabs_node)
    
    if pet_trab_node:
        stats_pet_trab_merge = StatisticsCalculator(pet_trab_node, pt)
        table_pet_trab_merged = stats_pet_trab_merge.export_to_table()
        df_trab_merge_pet = slicer.util.dataframeFromTable(table_pet_trab_merged)
        
        # Volume calculation from combined PET/Trabecular mask (assuming first two rows are the two segments)
        try:
            suv_vol_cm3 = df_trab_merge_pet.iloc[0]["Volume cm3"]
            trab_vol_cm3_merge = df_trab_merge_pet.iloc[1]["Volume cm3"]
            ratio_vol_suv_over_vol_trab = suv_vol_cm3 / trab_vol_cm3_merge if trab_vol_cm3_merge != 0 else 0
        except Exception as e:
            logger.error(f"PET/Trabecular volume extraction failed: {e}")
            suv_vol_cm3, trab_vol_cm3_merge, ratio_vol_suv_over_vol_trab = 0.0, 0.0, 0.0

        # PET Segmentation 2: SUV <-> THRESHOLDING LESION (using 95th percentile)
        try:
            threshold_lesion = df_trab_merge_pet.loc[0, "Percentile 95"]
        except Exception as e:
            logger.warning(f"Could not find Percentile 95. Skipping lesion thresholding: {e}")
            threshold_lesion = 0.0
            
        # PET Segmentation 3: SUV <-> TOTAL BONE MASK (for whole tumor volume)
        temp_pet_lesion_node = pet_segmentation(pt, round(threshold_lesion, 1), merged_bones_node)
        
        if temp_pet_lesion_node:
            stats_pet_lesion = StatisticsCalculator(temp_pet_lesion_node, pt)
            table_pet_lesion = stats_pet_lesion.export_to_table()
            df_pet_lesion = slicer.util.dataframeFromTable(table_pet_lesion)
            tumor_suv_cm3 = df_pet_lesion.iloc[0]["Volume cm3"]
        else:
            tumor_suv_cm3 = 0.0

        # PET Segmentation 4: SUV=3.0 for reference
        temp_pet_suv3_node = pet_segmentation(pt, 3.0, merged_bones_node)
        if temp_pet_suv3_node:
            stats_pet_suv3 = StatisticsCalculator(temp_pet_suv3_node, pt)
            table_pet_suv3 = stats_pet_suv3.export_to_table()
            df_pet_suv3 = slicer.util.dataframeFromTable(table_pet_suv3)
            tumor_suv3_cm3 = df_pet_suv3.iloc[0]["Volume cm3"]
        else:
            tumor_suv3_cm3 = 0.0
    else:
        # If trabecular/PET segmentation failed early
        suv_vol_cm3, trab_vol_cm3_merge, ratio_vol_suv_over_vol_trab = 0.0, 0.0, 0.0
        tumor_suv_cm3, tumor_suv3_cm3 = 0.0, 0.0
        threshold_lesion = 0.0
        logger.warning("Skipping PET derived measurements due to failure in initial PE/Bones mask.")

    # 7. Results Aggregation and Saving
    
    results = {
        "trab_bones_cm3": trab_bones_cm3,
        "bones_cm3": bones_cm3,
        "ratio_trab_to_bones": ratio_trab,
        "median_SUV_abm": median_suv_abm,
        "SUV_Vol_in_Trabeculae_cm3": suv_vol_cm3,
        "Ratio_PET_SUV_over_TrabVolume": ratio_vol_suv_over_vol_trab,
        "Tumor_SUV_Threshold": threshold_lesion,
        "Tumor_SUV_Volume_cm3": tumor_suv_cm3,
        "Tumor_SUV3_Volume_cm3": tumor_suv3_cm3
    }
    
    patname = get_patient_name()
    output_dataframe = pd.DataFrame(results, index=[patname])
    
    # Define output paths safely
    output_dir = get_volume_output_dir(ct)
    bundle_directory = output_dir / time.strftime("%Y%m%d%H%M%S")
    try:
        bundle_directory.mkdir(parents=True, exist_ok=True)
    except Exception as e:
        logger.critical(f"FATAL ERROR: Could not create output directory in {output_dir}: {e}")
        exit(1)

    results_path = bundle_directory / f"{patname}_analysis_results.csv"
    
    logger.info("--- Final Results ---")
    print(output_dataframe.T)
    
    try:
        # Save the results
        output_dataframe.to_csv(results_path, index=False)
        logger.info(f"Successfully saved results to: {results_path}")
        
        # Clean up Slicer scene (optional but recommended)
        slicer.mrmlScene.RemoveNode(merged_bones_node)
        slicer.mrmlScene.RemoveNode(merged_trabs_node)
        slicer.mrmlScene.RemoveNode(table_bones)
        slicer.mrmlScene.RemoveNode(table_trab)
        slicer.mrmlScene.RemoveNode(table_pet_trab)
        
        # Clean up Segment Editor
        segmentEditorWidget = None
        slicer.mrmlScene.RemoveNode(segmentEditorNode)
        
        # Save scene to bundle
        slicer.mrmlScene.SaveSceneToSlicerDataBundleDirectory(str(bundle_directory), None, None)
        logger.info("Scene saved to data bundle.")
        
    except Exception as e:
        logger.error(f"Error during final cleanup or saving results: {e}")

