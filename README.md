# ft2nii
This software allows for the creation of a nifti map starting from source localized EEG data

The software works with the following input:

- the path to a .mat file containing the result of the source localization for EEG data.

- an optional argument -S stating the spatial smoothing to be applied to the data (in mm, default 5)

- an optional argument -E (default False), True if the output should conist of one map for each defined ERP (erp timings defined in the code), False if the output should be a 4D nifti file;

- an optional argument -T defining the size of the window to be used for temporal averaging (in datapoints), in case -E is False.
