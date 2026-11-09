# VSSI-ARD-net

The Python code for VSSI-ARD-net, a deep unrolled network for EEG source imaging.

The dataset is not included. Generate the simulated EEG data with the MATLAB scripts in `Data_Generate/` (requires MATLAB and Brainstorm); only the reference lead-field matrices `L.mat` are kept under `Data/`.

Run `New_Network_Layers/main_train.py` to train the model, and `main_validation.py` or `validate_extents_with_timing.py` to evaluate a trained checkpoint. Training requires a CUDA-capable GPU; the Python dependencies are listed in `requirements.txt`. For different training data, you need to modify the initial values of hyperparameters in the model appropriately to meet the requirements of the model.
