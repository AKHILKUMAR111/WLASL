
This repository contains the `WLASL` dataset described in "Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison".


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

3. Create subset from the original WLASL_V03.JSON named WLASL_Cafe.json
   
   python subsetCreater.py

4. Download raw videos.
```
cd start_kit
python video_downloader.py
```
5. in bash terminal go to path to wlasl folder and run
   bash scripts/swf2mp4.sh 

6. Install opencv and mediapipe and ffmpeg

7. Extract video samples from raw videos.
```
python preprocess.py
```
8. You should expect to see video samples under directory ```videos/```.

9. open the landmark_extracter.ipynb in jupyter notebook and run all cells
      will get a landmark folder

10. open the train_transforemer.ipynb in jupyter notebook and run  cells as needed 
        will get a trained model at end 

11. for getting a real time demo how the modle is working 
       python realtime_gloss.py

-----------------



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

* `fps`: *int*, frame rate (=25) used to decode the video as in the WLASL paper.

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



Exporting Your Trained Pose Transformer From Notebook
---------------
If you trained the Pose Transformer in `start_kit/train_transformer.ipynb`, you can export weights and the `id_to_gloss` mapping:

1) In a new cell near the end of training, run:
```
from notebook_export_helpers import save_model_state, save_id_to_gloss_from_gloss_to_id

# Assuming `model` is your trained PoseTransformer and `gloss_to_id` is available
save_model_state(model, "start_kit/pose_transformer.pth")
save_id_to_gloss_from_gloss_to_id(gloss_to_id, "start_kit/id_to_gloss.json")
```

2) Use these artifacts with the real-time demo below.

License
---------------
Licensed under the Computational Use of Data Agreement (C-UDA). Plaese refer to `C-UDA-1.0.pdf` for more information.

Real-time Sign-to-Gloss Demo (Pose Transformer)
---------------
We provide a lightweight real-time demo that uses OpenCV + MediaPipe Pose and your trained Pose Transformer model.

1) Ensure your Python environment has `mediapipe`, `opencv-python`, and `torch` installed.

2) Run the demo (ESC to quit, press `c` to clear the rolling buffer):
```
python start_kit/realtime_gloss.py --weights /path/to/model.pth --classes NUM_CLASSES --id2gloss /path/to/id_to_gloss.json --camera 0 --show-skeleton
```

Notes:
- The script expects the same preprocessing used during training: 33 pose landmarks × (x,y,z,visibility), nose-centered normalization, and padding/truncation to 100 frames.
- `--id2gloss` can be either a JSON list like ["yes", "no", ...] or a mapping {"0": "yes", ...}.






Citation
--------------

Please cite the WLASL paper if it helps your research:

     @inproceedings{li2020word,
        title={Word-level Deep Sign Language Recognition from Video: A New Large-scale Dataset and Methods Comparison},
        author={Li, Dongxu and Rodriguez, Cristian and Yu, Xin and Li, Hongdong},
        booktitle={The IEEE Winter Conference on Applications of Computer Vision},
        pages={1459--1469},
        year={2020}
     }

Please consider citing our work on WLASL.

    @inproceedings{li2020transferring,
     title={Transferring cross-domain knowledge for video sign language recognition},
     author={Li, Dongxu and Yu, Xin and Xu, Chenchen and Petersson, Lars and Li, Hongdong},
     booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
     pages={6205--6214},
     year={2020}
    }

Other works you might be interested in.

    @article{li2020tspnet,
      title={Tspnet: Hierarchical feature learning via temporal semantic pyramid for sign language translation},
      author={Li, Dongxu and Xu, Chenchen and Yu, Xin and Zhang, Kaihao and Swift, Benjamin and Suominen, Hanna and Li, Hongdong},
      journal={Advances in Neural Information Processing Systems},
      volume={33},
      pages={12034--12045},
      year={2020}
    }

    @inproceedings{li2022transcribing,
      title={Transcribing natural languages for the deaf via neural editing programs},
      author={Li, Dongxu and Xu, Chenchen and Liu, Liu and Zhong, Yiran and Wang, Rong and Petersson, Lars and Li, Hongdong},
      booktitle={Proceedings of the AAAI Conference on Artificial Intelligence},
      volume={36},
      number={11},
      pages={11991--11999},
      year={2022}
    }


## Stargazers over time

[![Stargazers over time](https://starchart.cc/dxli94/WLASL.svg)](https://starchart.cc/dxli94/WLASL)
