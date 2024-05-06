## Domain_characterization

This GitHub repository hosts the implementation code for the research paper titled "Understanding and Quantifying the Difficulty of Imbalanced Classification Tasks: A Supervised Explainable Approach." Specifically, the repository contains the development of complexity measures designed to identify and quantify the presence of data difficulty factors, such as class overlap, in imbalanced classification scenarios. The code utilizes GPU acceleration via the CuPy framework and the RAPIDS cuML library, enabling efficient computation and the capability to manage large data volumes, which addresses a current limitation in existing frameworks on this topic. The implemented metrics include F2, F3, F4, N2, N3, N4, Raug, L1, L2, L3, and IBI.

## Prerequisites

Before you begin, ensure you have met the following requirements:
- You have a `Python` environment version 3.9.
- You have installed CUDA-compatible `GPU drivers`. This repository requires the installation of the RAPIDS library, check system requirements in: https://docs.rapids.ai/install#system-req
  
## Installation

To use this code, follow these steps:

1. **Clone the Repository**
   
   Clone the repository to your local machine:

   ```bash
   git clone https://github.com/MiguelCarvalhoPhD/domain_characterization.git
   cd domain_characterization

3. **Set Up Your Python Environment
   
   It's recommended to use a virtual environment to avoid conflicts with other packages:

      ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`

5. **Install Required Packages
   
   Install all dependencies using the provided requirements.txt file:

      ```bash
    pip install -r requirements.txt

## Usage and Tutorials

  An example usage script (example_usage.py) is provide which applies all available functions in synthetic datasets.
  The script performance_analysis.py provides a runtime comparison between the implemented functions and those present in the problexity library (https://github.com/w4k2/problexity)

   



