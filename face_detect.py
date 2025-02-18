import cv2
import os
import argparse
import numpy as np
import json
from tqdm import tqdm
from sklearn.metrics.pairwise import cosine_similarity
import insightface
from scenedetect import detect, ContentDetector

def initialize_models(det_size):
    '''
    Initialize face detection model.
    det_size: Tuple of (width, height) for face detection
    '''
    app = insightface.app.FaceAnalysis()
    app.prepare(ctx_id=0, det_size=det_size)
    return app

def detect_scenes(video_path):
    '''
    Detect scene cuts.
    video_path: Path to the input video file
    '''
    scene_list = detect(video_path, ContentDetector())
    return [int(scene[0].get_frames()) for scene in scene_list]

def read_frames(video_path):
    '''
    Read frames from video.
    video_path: Path to the input video file
    '''
    cap = cv2.VideoCapture(video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, fps

def process_frames(frames, ref_embedding, app, output_path, threshold, fps, det_size, scene_cuts, debug=False):
    '''
    The main algorithm for face tracking.
    '''
    clip_idx = 0
    recording = False
    writer = None
    debug_writer = None
    frame_idx = 0
    metadata = []
    prev_embedding = None
    clip_start_frame = None
    face_coordinates = []

    if debug:
        debug_output = os.path.join(output_path, "debug_video.mp4")
        debug_writer = cv2.VideoWriter(debug_output, cv2.VideoWriter_fourcc(*'mp4v'), fps, (frames[0].shape[1], frames[0].shape[0]))
    
    for frame in tqdm(frames, desc='Processing frames'):
        if frame_idx in scene_cuts and recording:
            writer.release()
            writer = None
            recording = False
            metadata.append({
                "file_name": f"clip_{clip_idx}.mp4",
                "start_time": clip_start_frame / fps,
                "end_time": frame_idx / fps,
                "face_coordinates": face_coordinates
            })
            face_coordinates = []

        frame_resized = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_resized = cv2.resize(frame_resized, (det_size[0], det_size[1]), interpolation=cv2.INTER_LINEAR)
        faces = app.get(frame_resized)
        best_similarity = 0
        best_face = None

        for face in faces:
            face_embedding = face.embedding
            if face_embedding is not None:
                similarity_ref = cosine_similarity([ref_embedding], [face_embedding])[0][0]
                similarity_prev = cosine_similarity([prev_embedding], [face_embedding])[0][0] if prev_embedding is not None else 0
                similarity = max(similarity_ref, similarity_prev)
            # face_embedding = face.embedding
            # if face_embedding is not None:
            #     similarity = cosine_similarity([ref_embedding], [face_embedding])[0][0]
                if similarity > threshold and similarity > best_similarity:
                    best_similarity = similarity
                    best_face = face

        if best_face:
            prev_embedding = best_face.embedding
            x1, y1, x2, y2 = map(int, best_face.bbox)
            scale_x, scale_y = frame.shape[1] / det_size[0], frame.shape[0] / det_size[1]
            x1, x2 = int(x1 * scale_x), int(x2 * scale_x)
            y1, y2 = int(y1 * scale_y), int(y2 * scale_y)
            cropped_face = frame[max(0, y1):min(frame.shape[0], y2), max(0, x1):min(frame.shape[1], x2)]
            face_coordinates.append([x1, y1, x2 - x1, y2 - y1])

            if not recording and not debug:
                clip_idx += 1
                output_file = os.path.join(output_path, f"clip_{clip_idx}.mp4")
                writer = cv2.VideoWriter(output_file, cv2.VideoWriter_fourcc(*'mp4v'), fps, (224, 224))
                recording = True
                clip_start_frame = frame_idx
            if writer and cropped_face.size != 0 and not debug:
                resized_face = cv2.resize(cropped_face, (224, 224))
                writer.write(resized_face)

            if debug:
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        
        if debug and debug_writer:
            debug_writer.write(frame)
        
        frame_idx += 1
    
    if writer:
        writer.release()
        metadata.append({
            "file_name": f"clip_{clip_idx}.mp4",
            "start_time": clip_start_frame / fps,
            "end_time": frame_idx / fps,
            "face_coordinates": face_coordinates
        })
    if debug_writer:
        debug_writer.release()
    
    # Save metadata to JSON
    metadata_path = os.path.join(output_path, "metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

def main():
    parser = argparse.ArgumentParser(description="Face Tracking and Cropping Pipeline")
    parser.add_argument("--video", required=True, help="Path to the input video file")
    parser.add_argument("--ref_image", required=True, help="Path to the reference image file")
    parser.add_argument("--threshold", type=float, default=0.7, help="Similarity threshold for face matching")
    parser.add_argument("--output", required=True, help="Directory to save the cropped video clips")
    parser.add_argument("--debug", action='store_true', help="Enable debug mode to output annotated video")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    ref_image = cv2.imread(args.ref_image)
    if ref_image is None or ref_image.size == 0:
        print("Error: Reference image is not loaded correctly.")
        return
    ref_image = cv2.cvtColor(ref_image, cv2.COLOR_BGR2RGB)
    
    det_size = (640, 640)
    app = initialize_models(det_size)
    ref_embedding = app.get(ref_image)[0].embedding
    if ref_embedding is None:
        print("Error: Could not extract embedding from the reference image.")
        return
    
    scene_cuts = detect_scenes(args.video)
    frames, fps = read_frames(args.video)
    process_frames(frames, ref_embedding, app, args.output, args.threshold, fps, det_size, scene_cuts, args.debug)
    print("Processing complete. Cropped videos saved in:", args.output)

if __name__ == "__main__":
    main()
