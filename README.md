### Setup
Follow this [installation guide](https://github.com/deepinsight/insightface/tree/master/python-package) to install [InsightFace](https://github.com/deepinsight/insightface/tree/master?tab=readme-ov-file). I use this tool for face analysis.

I use [PySceneDetect](https://www.scenedetect.com/) to detect scene cuts, we use pip to install it:
```bash
pip install --upgrade scenedetect[opencv]
```

### Run Examples

#### Command to track face
```bash
python face_detect.py --video $PATH_TO_VID --ref_image $PATH_TO_REF_IMG --output $OUTPUT_DIR
```
Use `--threshold` to specify the similarity threshold for face matching.

For example, to reproduce the result of tracking the target face in `examples/vids/hugh.mp4`, run
```bash
python face_detect.py --video examples/vids/hugh.mp4 --ref_image examples/ref/hugh.jpg --output output/hugh --threshold 0.15 
```

The following table summarizes the value of threshold that I used to produce results in `output`:
| Video | Threshold |
|  ---  |    ---    | 
|  examples/vids/hugh.mp4  | 0.15 |
|  examples/vids/hugh1.mp4 | 0.2 |
|  examples/vids/hugh2.mp4 | 0.12 |
|  examples/vids/emma.mp4 | 0.05 |
|  examples/vids/emma1.mp4 | 0.1 |

When `--debug` is enabled, a debug video is generated in `$OUTPUT_DIR`, where each frame is annotated with the detected face's bounding box. No cropped clips are saved in this mode.

### Assumption and Limitation

#### Assumption
1. The reference image clearly captures the target's face, with the subject facing the camera.
2. The target's face does not change significantly between consecutive frames, except during scene cuts.

#### Limitation
1. It may fail to detect the face if it is partially occluded or not fully captured by the camera.
    ![ex1](resources/ex1.gif)
2. Since the reference image is assumed to show the subject facing the camera, the algorithm has limited knowledge of the subject’s side profiles. As a result, face detection and similarity matching may fail when the subject turns away, and using past frames cannot fully mitigate this issue.
    ![ex2](resources/ex2.gif)
3. If the subject is too far from the camera, the face may be too small for accurate detection and tracking.
    ![ex3](resources/ex3.gif)