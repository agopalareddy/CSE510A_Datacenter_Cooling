How to setup:

1. Download and install [Anaconda](https://www.anaconda.com/products/distribution)
2. Download and install [GitHub Desktop](https://desktop.github.com/)
3. Download the latest [EnergyPlus](https://energyplus.net/downloads)
4. Clone the Warehouse-Cooling repository:
    ```sh
    git clone https://github.com/peyton-gozon/CSE510A-Datacenter-Cooling
    ```
5. Open the CSE510A-Datacenter-Cooling repository in terminal:
    ```sh
    cd CSE510A-Datacenter-Cooling
    ```
6. Create and activate a new conda environment:
    ```sh
    conda create -n cooler
    conda activate cooler
    ```
7. Upgrade Python to 3.12:
    ```sh
    conda install python=3.12
    ```
8. Install the required packages:
    ```sh
    pip3 install -r requirements.txt
    ```
9. Clone the Sinergym repository:
    ```sh
    git clone https://github.com/ugr-sail/sinergym
    ```
10. Open the Sinergym repository in terminal:
     ```sh
     cd sinergym
     ```
11. Install Sinergym:
     ```sh
     pip3 install -e .
     ```
12. Navigate back to the CSE510A-Datacenter-Cooling repository and into the `drl` folder:
     ```sh
     cd ../drl
     ```
13. Run the training script:
     ```sh
     python3 ppo.py
     ```
