# TrackCtrl
ControlNet for storm tracks conditioned on reanalysis imagery (meteorology, coastal configuration). Outputs will be sent through statistical methods to generate high-resolution tracks. 


Installation:

1. Follow these instructions to use CDS API: [MacOS](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+macOS) | [Windows](https://confluence.ecmwf.int/display/CKB/How+to+install+and+use+CDS+API+on+Windows)

2. Run `conda env create -f environment.yaml`

3. Activate env `conda activate trackctrl`

4. Finetuning from the segmentation model because I feel as though that is the closest comparasin to what reanalysis data is: https://huggingface.co/lllyasviel/ControlNet/blob/main/models/control_sd15_seg.pth

Debugging Notes: 
If you're running into cartopy / netcdf4 / scikit-image issues with pip try: `conda install -c conda-forge <package>`
honestly I took the environment file from control net repo and I kind of despise it, so I set up my env dependent on python 3.12 and ran a pip install -r requirements.txt after activating it, and it did wonders for my general happiness. I will update this at some point to reflect. 

Goal: 

- Input: A hurricane track (e.g., a sequence of lat/lon positions or a gridded path mask or heatmap. Going to start with a heatmap to make it spatially aligned with wind speed and presure and baseline ddpm weights. Channel will be 0 (past track pos), 1 (current track pos), and 2 (future track pos if avail). Thinking pixel space will be a direct mapping to lat / long so do 360x360 images??).

- Condition: Reanalysis imagery (e.g., pressure, wind fields, sea level). I am assuming sea level is indirectly extracted via overlain coastal configuration. 

- Output: The diffusion model generates hurricane track outputs consistent with the conditioning vector. 

OR 

- conditioning = reanalysis fields at current time
- target = sequence of (lat, lon, time offset) points defining future track
- prompt = (optional) descriptive string

OR (THIS IS THE REAL OPTION AT THE MOMENT)
Channel	Pixel value meaning
- R (channel 0)	time offset at that grid location (e.g. hours since storm start, or normalized 0-1)
- G (channel 1)	reanalysis composite (e.g. normalized MSLP + wind magnitude, or other field summary)
- B (channel 2)	binary storm presence (1 if track passes over pixel, else 0) or maybe normalized lat/lon

Code largely adapted from ControlNet: [GitHub](https://github.com/lllyasviel/ControlNet)