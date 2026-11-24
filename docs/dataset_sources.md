# Dataset Source and Provenance Documentation

## 1. Primary Dataset Overview
- **Dataset Title:** Leaf Disease Dataset Combination
- **Kaggle Source URL:** [https://www.kaggle.com/datasets/asheniranga/leaf-disease-dataset-combination](https://www.kaggle.com/datasets/asheniranga/leaf-disease-dataset-combination)
- **Publisher / Author:** Ashen Iranga (Kaggle Community Dataset)
- **Retrieval / Inspection Date:** November 2026
- **License:** Open Database License (ODbL) / Creative Commons Attribution-ShareAlike 4.0 International (CC BY-SA 4.0)

## 2. Dataset Composition and Lineage
The dataset represents an aggregated benchmark combining multiple established agricultural computer vision corpuses:
1. **PlantVillage Dataset (Hughes & Salathé, 2015):** Contributes standardized laboratory and field leaf images for Apple, Cherry, Corn, Grape, Peach, Bell Pepper, Potato, Squash, Strawberry, and Tomato.
2. **Cassava Leaf Disease Dataset (Makerere AI Lab / Kaggle):** Contributes field-captured images of cassava foliage across 4 distinct viral/bacterial pathologies and healthy leaves.
3. **Rice Leaf Disease Dataset:** Contributes real-world imagery for Brown Spot, Hispa, Leaf Blast, and Healthy rice leaves.

## 3. Verified Dataset Statistics
- **Total Number of Classes:** 44 distinct crop-disease categories.
- **Total Validated Images:** 53,303 image files.
  - **Training Split:** 38,104 images (71.49%)
  - **Validation Split:** 9,458 images (17.74%)
  - **Independent Test Split:** 5,741 images (10.77%)
- **Image Modality:** RGB photographic imagery of individual leaves and crop foliage against laboratory and in-field backgrounds.
- **Original Dimensions:** Variable resolutions ranging from 256×256 to 1024×1024 pixels, standardized to 224×224 pixels during preprocessing.

## 4. Honest Geographic and Agronomic Scoping (Maharashtra Context)
An explicit objective of the academic inquiry is investigating crops of high economic importance to Maharashtra, India.
Based on strict empirical dataset verification:
- **Represented Maharashtra Cash Crops:**
  - **Grape (*Vitis vinifera*):** 4 classes (Black Rot, Esca/Black Measles, Leaf Blight/Isariopsis, Healthy) - 1,383 images.
  - **Rice (*Oryza sativa*):** 4 classes (Brown Spot, Hispa, Leaf Blast, Healthy) - 1,488 images.
  - **Tomato (*Solanum lycopersicum*):** 10 classes (Bacterial Spot, Early Blight, Late Blight, Leaf Mold, Septoria, Spider Mites, Target Spot, Mosaic Virus, Yellow Leaf Curl, Healthy) - 5,357 images.
  - **Potato, Corn, Pepper:** Fully represented.
- **Unrepresented Crops:**
  - **Sugarcane (*Saccharum officinarum*)** and **Onion (*Allium cepa*)** are absent from this benchmark dataset.
  - In adherence to academic integrity principles, our system does NOT fabricate or claim support for Sugarcane or Onion. Future extension work is defined in Section 10 to integrate specialized state-level field datasets for these crops.

## 5. Attribution and Academic Citations
- Hughes, D., & Salathé, M. (2015). An open access repository of images on plant health to enable the development of mobile disease diagnostics. *arXiv preprint arXiv:1511.08060*.
- Mwebaze, E., et al. (2019). iCassava 2019 Fine-Grained Visual Categorization Challenge. *Makerere University AI Lab*.
