# HepaSPOT
HepaSPOT: An interpretable spatial histopathology framework for postoperative prognosis prediction in hepatocellular carcinoma.
The goal of this repository is:
- to help researchers reproduce HepaSPOT, a spatial histopathology-based framework for postoperative disease-free survival (DFS) prediction in patients with resected HCC.
- to provide an end-to-end workflow for WSI preprocessing, semantic tile selection, histopathological representation learning, phenotype clustering, spatial mapping, tissue architecture feature extraction, and prognostic modeling.
- to help researchers quantitatively characterize intratumoral spatial histologic heterogeneity and identify interpretable tissue architecture features associated with postoperative outcomes.
- to provide a computational pathology toolkit for HCC spatial tissue analysis, prognostic modeling, and biological interpretation.
## What is HepaSPOT?
HepaSPOT is a computational pathology framework for quantitative characterization of the spatial organization of histopathological phenotypes and postoperative prognosis prediction in hepatocellular carcinoma.
The HepaSPOT framework was developed and evaluated in 997 patients with resected HCC across five Chinese institutions and the TCGA-LIHC cohort. It integrates semantic tile selection, pathology foundation model representations, unsupervised histopathological phenotype discovery, spatial tissue architecture analysis, and survival modeling.
HepaSPOT identified 23 histopathological phenotype clusters (HPCs) and quantified their spatial organization within H&E-stained whole-slide images. Prognostically informative tissue architecture features were subsequently used to construct the HepaSPOT model and were further integrated with clinicopathologic variables for postoperative disease-free survival prediction.
In addition to prognostic modeling, HepaSPOT maps key spatial features back to recognizable histopathologic structures and links them with molecular and immune programs, providing an interpretable framework for studying spatial histologic heterogeneity in HCC.
