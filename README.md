# TrackCtrl
ControlNet for storm tracks conditioned on reanalysis imagery (meteorology, coastal configuration). Outputs will be sent through statistical methods to generate high-resolution tracks. 


Installation:

1. Follow these instructions to use CDS API: [MacOS](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS) | [Windows](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows)

2. Run `conda env create -f environment.yaml`

3. Activate env `conda activate trackctrl`

Goal: 

-Input: A hurricane track (e.g., a sequence of lat/lon positions or a gridded path mask or heatmap. Going to start with a heatmap to make it spatially aligned with wind speed and presure and baseline ddpm weights. Channel will be 0 (past track pos), 1 (current track pos), and 2 (future track pos if avail). Thinking pixel space will be a direct mapping to lat / long so do 360x360 images??).

-Condition: Reanalysis imagery (e.g., pressure, wind fields, sea level). I am assuming sea level is indirectly extracted via overlain coastal configuration. 

-Output: The diffusion model generates hurricane track outputs consistent with the conditioning vector. 

OR 

- conditioning = reanalysis fields at current time
- target = sequence of (lat, lon, time offset) points defining future track
- prompt = (optional) descriptive string

Code largely adapted from ControlNet: [GitHub](https://github.com/lllyasviel/ControlNet)