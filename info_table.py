import pandas as pd
import ace_tools as tools

# Define the comparison data
data = {
    "Criteria": [
        "Computational Complexity",
        "Inference Time (ms)",
        "Memory Requirements",
        "Adaptability to Different LiDAR Densities",
        "Handling Occlusion & Sparsity",
        "Post-Processing Requirement",
        "End-to-End Predictability",
        "Strengths",
        "Weaknesses",
    ],
    "DS-Net (Clustering-Based, Bottom-Up)": [
        "Moderate (Clustering step adds overhead)",
        "50-80 ms (depends on clustering settings)",
        "Lower (Uses feature extraction + clustering)",
        "Sensitive to density changes (fixed clustering parameters)",
        "Moderate (Fails in highly sparse areas)",
        "Yes (Clustering step needed)",
        "Lower (Clustering hyperparameters must be tuned)",
        "Performs well in dense areas, effective for instance segmentation",
        "Struggles with distant objects and sparsity; clustering parameters need tuning per dataset",
    ],
    "MaskPLS (Mask-Based, Top-Down)": [
        "High (Transformer-based mask prediction)",
        "40-70 ms (depending on transformer depth)",
        "Higher (More learnable parameters)",
        "Better (Learns to predict masks without tuning)",
        "Robust (Handles occlusion and sparsity better)",
        "No (End-to-end mask prediction)",
        "High (Learns to separate instances naturally)",
        "Fully end-to-end, robust across datasets, no clustering needed",
        "High memory use; can struggle with overlapping small objects",
    ],
}

# Convert to DataFrame
df = pd.DataFrame(data)

# Display the table to the user
tools.display_dataframe_to_user(name="Panoptic Segmentation Approach Comparison", dataframe=df)
