from datetime import date

today = date.today()

subsampling_rate = 2

import argparse
from typing import Dict, List, Set, Tuple

import cv2
import numpy as np
from tqdm import tqdm
from ultralytics import YOLO

import supervision as sv

COLORS = sv.ColorPalette.default()

# settings
MODEL = "yolov8n.pt"
model = YOLO(MODEL)
model.fuse()

# S
ZONE_OUT_POLYGONS = [
    np.array([[485, 280], [975, 340], [1100, 300], [722, 252]]),
    np.array([[1585, 555], [1740, 565], [1725, 435], [1575, 400]]),
    np.array([[0, 630], [1630, 1080], [1545, 690], [505, 500]]),
]

ZONE_IN_POLYGONS = [
    np.array([[910, 385], [1530, 490], [1545, 365], [1170, 310]]),
    np.array([[1600, 680], [1760, 675], [1740, 565], [1585, 555]]),
    np.array([[0, 312], [0, 630], [505, 500], [24, 311]]),
]

# N_E
# ZONE_OUT_POLYGONS = [
#     np.array([[621, 261], [763, 249], [1099, 344], [709, 408]]),
#     np.array([[1066, 266], [1241, 249], [1283, 260], [1168, 287]]),
#     np.array([[39, 770], [746, 498], [1117, 1075], [33, 1074]]),
#     np.array([[1543, 480], [1806, 444], [1918, 479], [1894, 616]]),
# ]

# ZONE_IN_POLYGONS = [
#     np.array([[763, 249], [886, 240], [1066, 266], [915, 291]]),
#     np.array([[1168, 287], [1283, 260], [1369, 275], [1293, 310]]),
#     np.array([[39, 549], [665, 350], [746, 498], [39, 770]]),
#     np.array([[837, 596], [1543, 480], [1894, 616], [1172, 1076]]),
# ]

# W
# ZONE_OUT_POLYGONS = [
#     np.array([[719, 105], [1003, 155], [593, 343], [365, 189]]),
#     np.array([[1, 603], [593, 343], [1205, 507], [961, 1077]]),
# ]

# ZONE_IN_POLYGONS = [
#     np.array([[1003, 155], [1331, 219], [1205, 507], [593, 345]]),
#     np.array([[1, 273], [365, 189], [593, 343], [1, 603]]),
# ]

# N
# ZONE_OUT_POLYGONS = [
#     np.array([[255, 292], [657, 292], [1163, 468], [218, 512]]),
#     np.array([[1164, 467], [1694, 444], [1892, 489], [1773, 710]]),
# ]

# ZONE_IN_POLYGONS = [
#     np.array([[657, 292], [1048, 289], [1694, 444], [1164, 467]]),
#     np.array([[218, 512], [1164, 467], [1893, 763], [125, 1020]]),
# ]

# class_ids of interest - car, bus and truck
CLASS_ID = [2, 5, 7]

SOURCE_VIDEO_PATH = "2024-03-01 08-00-00 192.168.1.142-S#5.avi"
TARGET_VIDEO_PATH = f"C:/sasa/capstone/traffic_{today}_S.avi"


class DetectionsManager:
    def __init__(self) -> None:
        self.tracker_id_to_zone_id: Dict[int, int] = {}
        self.counts: Dict[int, Dict[int, Set[int]]] = {}

    def update(
        self,
        detections_all: sv.Detections,
        detections_in_zones: List[sv.Detections],
        detections_out_zones: List[sv.Detections],
    ) -> sv.Detections:
        for zone_in_id, detections_in_zone in enumerate(detections_in_zones):
            for tracker_id in detections_in_zone.tracker_id:
                self.tracker_id_to_zone_id.setdefault(tracker_id, zone_in_id)

        for zone_out_id, detections_out_zone in enumerate(detections_out_zones):
            for tracker_id in detections_out_zone.tracker_id:
                if tracker_id in self.tracker_id_to_zone_id:
                    zone_in_id = self.tracker_id_to_zone_id[tracker_id]
                    self.counts.setdefault(zone_out_id, {})
                    self.counts[zone_out_id].setdefault(zone_in_id, set())
                    self.counts[zone_out_id][zone_in_id].add(tracker_id)
        if len(detections_all) > 0:
            detections_all.class_id = np.vectorize(
                lambda x: self.tracker_id_to_zone_id.get(x, -1)
            )(detections_all.tracker_id)
        else:
            detections_all.class_id = np.array([], dtype=int)
        return detections_all[detections_all.class_id != -1]


def initiate_polygon_zones(
    polygons: List[np.ndarray],
    frame_resolution_wh: Tuple[int, int],
    triggering_position: sv.Position = sv.Position.CENTER,
) -> List[sv.PolygonZone]:
    return [
        sv.PolygonZone(
            polygon=polygon,
            frame_resolution_wh=frame_resolution_wh,
            triggering_position=triggering_position,
        )
        for polygon in polygons
    ]


class VideoProcessor:
    def __init__(
        self,
        source_video_path: str,
        target_video_path: str = None,
        # confidence_threshold: float = 0.3,
        # iou_threshold: float = 0.7,
    ) -> None:
        # self.conf_threshold = confidence_threshold
        # self.iou_threshold = iou_threshold
        self.source_video_path = source_video_path
        self.target_video_path = target_video_path
        self.tracker = sv.ByteTrack()

        self.video_info = sv.VideoInfo.from_video_path(source_video_path)
        self.zones_in = initiate_polygon_zones(
            ZONE_IN_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )
        self.zones_out = initiate_polygon_zones(
            ZONE_OUT_POLYGONS, self.video_info.resolution_wh, sv.Position.CENTER
        )

        self.box_annotator = sv.BoxAnnotator(color=COLORS)
        self.trace_annotator = sv.TraceAnnotator(
            color=COLORS, position=sv.Position.CENTER, trace_length=100, thickness=2
        )
        self.detections_manager = DetectionsManager()

    def process_video(self):
        frame_generator = sv.get_video_frames_generator(
            source_path=self.source_video_path
        )

        if self.target_video_path:
            with sv.VideoSink(
                target_path=self.target_video_path,
                video_info=self.video_info,
            ) as sink:
                for frame_number, frame in enumerate(
                    tqdm(frame_generator, total=self.video_info.total_frames)
                ):
                    if frame_number % subsampling_rate != 0:
                        continue  # Skip frames based on the subsampling rate
                    annotated_frame = self.process_frame(frame)
                    sink.write_frame(annotated_frame)
        else:
            for frame in tqdm(frame_generator, total=self.video_info.total_frames):
                annotated_frame = self.process_frame(frame)
                cv2.imshow("Processed Video", annotated_frame)
                if cv2.waitKey(1) & 0xFF == ord("q"):
                    break
            cv2.destroyAllWindows()

    def annotate_frame(
        self, frame: np.ndarray, detections: sv.Detections
    ) -> np.ndarray:
        annotated_frame = frame.copy()
        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_in.polygon, COLORS.colors[i]
            )
            annotated_frame = sv.draw_polygon(
                annotated_frame, zone_out.polygon, COLORS.colors[i]
            )

        labels = [
            f"#{tracker_id} {confidence:0.2f}"
            for _, _, confidence, class_id, tracker_id in detections
        ]

        annotated_frame = self.trace_annotator.annotate(annotated_frame, detections)
        annotated_frame = self.box_annotator.annotate(
            annotated_frame, detections, labels
        )

        for zone_out_id, zone_out in enumerate(self.zones_out):
            zone_center = sv.get_polygon_center(polygon=zone_out.polygon)
            if zone_out_id in self.detections_manager.counts:
                counts = self.detections_manager.counts[zone_out_id]
                for i, zone_in_id in enumerate(counts):
                    count = len(self.detections_manager.counts[zone_out_id][zone_in_id])
                    text_anchor = sv.Point(x=zone_center.x, y=zone_center.y + 40 * i)
                    annotated_frame = sv.draw_text(
                        scene=annotated_frame,
                        text=str(count),
                        text_anchor=text_anchor,
                        background_color=COLORS.colors[zone_in_id],
                    )

        return annotated_frame

    def process_frame(self, frame: np.ndarray) -> np.ndarray:
        results = model(frame, verbose=False)[0]
        detections = sv.Detections.from_ultralytics(results)
        # Filter out unwanted classes
        mask_unwanted_classes = np.isin(detections.class_id, CLASS_ID)
        detections = detections[mask_unwanted_classes]
        # detections.class_id = np.zeros(len(detections))
        detections = self.tracker.update_with_detections(detections)

        detections_in_zones = []
        detections_out_zones = []

        for i, (zone_in, zone_out) in enumerate(zip(self.zones_in, self.zones_out)):
            detections_in_zone = detections[zone_in.trigger(detections=detections)]
            detections_in_zones.append(detections_in_zone)
            detections_out_zone = detections[zone_out.trigger(detections=detections)]
            detections_out_zones.append(detections_out_zone)

        detections = self.detections_manager.update(
            detections, detections_in_zones, detections_out_zones
        )
        return self.annotate_frame(frame, detections)


if __name__ == "__main__":
    processor = VideoProcessor(
        source_video_path=SOURCE_VIDEO_PATH,
        target_video_path=TARGET_VIDEO_PATH,
        # confidence_threshold=0.3,
        # iou_threshold=0.5,
    )

    # yolov8 model default conf = 0.25
    # default iou = 0.70
    processor.process_video()
