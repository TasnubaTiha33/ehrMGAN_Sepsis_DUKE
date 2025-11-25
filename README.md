# ehrMGAN Synthetic Data Generation for Pediatric Sepsis

This repository contains the **preprocessing pipeline**, **training configuration**, and **modified source code** used to generate synthetic data for pediatric sepsis using the [ehrMGAN](https://github.com/jli0117/ehrMGAN) framework.

---

## 📁 Included Files and Folders

- `ehrMGAN_Sepsis_preprocessing.ipynb`  
  → Merges and preprocesses the original datasets before training ehrMGAN.

- `Test_ehrMGAN.ipynb`  
  → Contains code for visualizing and evaluating synthetic data.

- `main_train.py`, `networks.py`, etc.  
  → Modified training and model files adapted to the pediatric sepsis dataset.

---

## 🗂️ Datasets Merged(PedSepsis Private Dataset)

The following files were merged into a single dataframe:

1. `raw_meds.parquet.gzip`  
2. `raw_variables.parquet.gzip`

The resulting **merged hourly dataframe shape**:

```

(198,338 rows, 227 columns)

````

---

### 🧬 Features Used for Synthetic Data Generation

The following features were included in the final merged dataset:

* `ABP MAP`
* `FiO2 (%)`
* `MAP`
* `PaO2/FiO2 (Calculated)`
* `Pulse`
* `Resp`
* `SpO2`
* `Temp`
* `ALT (SGPT)`
* `ARTERIAL POC PCO2`
* `ARTERIAL POC PH`
* `ARTERIAL POC PO2`
* `AST (SGOT)`
* `BAND NEUTROPHILS % (MANUAL)`
* `BILIRUBIN TOTAL`
* `BUN`
* `CAPILLARY POC PH`
* `CREATININE`
* `Code Sheet Weight (kg)`
* `GLUCOSE`
* `LACTIC ACID`
* `LACTIC ACID WHOLE BLOOD`
* `PLATELETS`
* `POC LACTIC ACID`
* `POC PCO2`
* `POC PO2`
* `POC pH`
* `VENOUS POC PH`
* `WBC`

---

## 📦 Preprocessed File Output Format

The following tensor formats were created based on hourly bins, `pat_id`, and `csn`:

| Tensor            | Shape           |
|-------------------|-----------------|
| `continuous`      | (471, 48, 29)    |
| `discrete`        | (471, 48, 194)   |
| `lengths`         | (471,)           |

Saved as:

- `vital_sign_48hrs.pkl` – 242 KB  
- `med_interv_48hrs.pkl` – 16 KB  
- `statics.pkl` – 0 KB (empty dict)  
- `norm_stats.npz` – 1 KB (contains mean and std for normalization)

---

## ⚙️ Configuration Settings for ehrMGAN

Update your config accordingly:

```python
time_steps        = 48
vital_sign_path   = 'vital_sign_48hrs.pkl'
med_interv_path   = 'med_interv_48hrs.pkl'
statics_path      = 'statics.pkl'
norm_stat_path    = 'norm_stats.npz'
dim               = 223
```
---

## 🚀 How to Train ehrMGAN

### 📌 Environment Setup

1. Clone the original ehrMGAN repo:

   ```bash
   git clone https://github.com/jli0117/ehrMGAN
   ```

2. Replace original code with the modified files in this repo (see `.py` and `.ipynb` files from Jupyter Notebook folder).

3. Create a conda environment:

   ```bash
   conda create -n PedSepsis python=3.9
   conda activate PedSepsis
   pip install tensorflow==2.10.1 numpy==1.26.4
   ```

   📎 Visit: [https://www.tensorflow.org/install/pip](https://www.tensorflow.org/install/pip) → Choose **Windows Native** packages if needed.

---

### 🔁 Training ehrMGAN

To generate **1,000,000 synthetic samples**, run the following:

```bash
cd ehrMGAN_modified
python main_train.py \
  --disease mydata \
  --num_pre_epochs 500 \
  --num_epochs 1000 \
  --epoch_ckpt_freq 100 \
  --epoch_loss_freq 1 \
  --gpu 0 \
  --num_gen 1000000
```

Generated data will be saved in:

```
ehrMGAN_modified/data/fake/large_gen_tmp/
```

---

## 📊 Visualization

Use `Test_ehrMGAN.ipynb` to load and visualize the synthetic data. This notebook includes:

* Histogram comparisons
* MMD & KL divergence calculations
* Denormalization logic

---

### 🧠 Understanding `c_gen` and `d_gen` Outputs

During training, **ehrMGAN** generates synthetic patient data in two separate branches:

* `c_gen_data_mm.npy` → **Continuous features**
* `d_gen_data_mm.npy` → **Discrete features**

These files are produced by the generator as part of the **multi-modal output structure** of ehrMGAN.

---

### 🔹 Continuous Data (`c_gen`)

* Stored in: `c_gen_data_mm.npy`
* Shape: *(N, T, C)* where:

  * **N** = number of patients
  * **T** = time steps (e.g., 48)
  * **C** = number of continuous variables (e.g., 29)
* Contains vital signs and lab results such as:

  * Heart rate, SpO₂, FiO₂, MAP
  * Lab values like Lactate, BUN, Creatinine, etc.
* These are real-valued and normalized using mean/std during preprocessing.

---

### 🔹 Discrete Data (`d_gen`)

* Stored in: `d_gen_data_mm.npy`
* Shape: *(N, T, D)* where:

  * **D** = number of discrete features (e.g., 194)
* Represents multi-hot encoded medication interventions over time.
* Each entry is either `0` or `1`, indicating the **presence or absence** of a drug at a specific time step.

---

### 🔄 Normalization Info

* The **continuous data** is min-max or z-score normalized during training.
* The **discrete data** is kept binary and fed into the discriminator as it is.

---

## 🧪 Evaluation

To evaluate data fidelity, utility, and privacy, please refer to:

* Maximum Mean Discrepancy (MMD)
* KL Divergence
* Dimensional RMSE
* Visual consistency with training data

---

## 📝 Notes

* This pipeline is tailored for **Pediatric Sepsis** data (471 patients, 48 hourly bins).
* Data is generated based on preprocessed features only (raw files not included in GitHub due to size limits).
* Ensure you update file paths as per your working directory.

---

