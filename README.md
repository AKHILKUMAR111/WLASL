
============================================================================================

This repository also contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".


**yt-dlp vs youtube-dl** youtube-dl has had low maintance for a while now and does not work for some youtube videos, see this [issue](https://github.com/ytdl-org/youtube-dl/issues/30568).
[yt-dlp](https://github.com/yt-dlp/yt-dlp) is a more up to date fork, which seems to work for all youtube videos. Therefore `./start_kit/video_downloader.py` uses yt-dlp by default but can be switched back to youtube-dl in the future by adjusting the `youtube_downloader` variable.
If you have trouble with yt-dlp make sure update to the latest version, as Youtube is constantly changing.

Download Original Videos
-----------------
1. Download repo.
```
git clone https://github.com/AKHILKUMAR111/WLASL.git
```

2. Install [youtube-dl](https://github.com/ytdl-org/youtube-dl) for downloading YouTube videos.
3. Download raw videos.
```
cd start_kit
python video_downloader.py
```
4. Extract video samples from raw videos.
```
python preprocess.py
```
5. You should expect to see video samples under directory ```videos/```.


 (a) Run
```
python find_missing.py
```
to generate text file missing.txt containing missing video IDs.

File Description
-----------------
The repository contains following files:

 * `WLASL_vx.x.json`: JSON file including all the data samples.

 * `data_reader.py`: Sample code for loading the dataset.

 * `video_downloader.py`: Sample code demonstrating how to download data samples.


 * `README.md`: this file.


Data Description
-----------------

* `gloss`: *str*, data file is structured/categorised based on sign gloss, or namely, labels.

* `bbox`: *[int]*, bounding box detected using YOLOv3 of (xmin, ymin, xmax, ymax) convention. Following OpenCV convention, (0, 0) is the up-left corner.

* `fps`: *int*, frame rate (=25) used to decode the video as in the paper.

* `frame_start`: *int*, the starting frame of the gloss in the video (decoding
with FPS=25), *indexed from 1*.

* `frame_end`: *int*, the ending frame of the gloss in the video (decoding with FPS=25). -1 indicates the gloss ends at the last frame of the video.

* `instance_id`: *int*, id of the instance in the same class/gloss.

* `signer_id`: *int*, id of the signer.

* `source`: *str*, a string identifier for the source site.

* `split`: *str*, indicates sample belongs to which subset.

* `url`: *str*, used for video downloading.

* `variation_id`: *int*, id for dialect (indexed from 0).

* `video_id`: *str*, a unique video identifier.

Please be kindly advised that if you decode with different FPS, you may need to recalculate the `frame_start` and `frame_end` to get correct video segments.

Constituting subsets
---------------
As described in the paper, four subsets WLASL100, WLASL300, WLASL1000 and WLASL2000 are constructed by taking the top-K (k=100, 300, 1000 and 2000) glosses from the `WLASL_vx.x.json` file.
We have used WLASL100 subsets you can use the one according to your need.

94/WLASL.svg)](https://starchart.cc/dxli94/WLASL)
