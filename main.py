import argparse
import cv2
import pandas as pd
from ultralytics import YOLO
from tqdm import tqdm


detector = YOLO("yolov8n.pt")

output = "output.mp4"
conf = 0.35

MIN_PRESENT_SECONDS = 3 #мин сек в кадре чтобы считать столик занятым
MIN_ABSENT_SECONDS = 4 #мин сек отсутстиве человека чтобы считать столик свободным


EMPTY_STATE = "EMPTY" #пуст
OCCUPIED_STATE = "OCCUPIED" #занят

def parse_args():
    """Запуск консоли и чтение аргументов"""
    parser = argparse.ArgumentParser(
        description="Monitoring table status"
    )
    parser.add_argument("--video", required=True, help="Path to the video file")
    return parser.parse_args()

def open_video(input_video_path):
    """Чтение параметов видео"""
    cap = cv2.VideoCapture(input_video_path)
    fps = cap.get(cv2.CAP_PROP_FPS)
    if not fps or fps < 1:
        fps = 25
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    #возвращается объёкт видео и его параметры
    return cap, fps, width, height, total_frames

def select_table_roi(frame):
    """Показывает первый кадр, выделение нужного столика вручную, сохранение координат зоны стола"""
    roi = cv2.selectROI(
        "Select the table with the box and press Enter",
        frame,
        showCrosshair=True,
        fromCenter=False,
    )
    cv2.destroyWindow("Select the table with the box and press Enter")
    x,y,w,h = map(int, roi)
    if w == 0 or h == 0:
        raise ValueError("ROI was not selected")
    return x, y, w, h


def detect_human(frame, detector, conf=0.35):
    """Обнаружает людей в кадре и возвращает список объектов в кадре"""
    detections = []
    result = detector(frame, verbose=False)[0]
    if result.boxes is None:
        return detections

    for box in result.boxes:
        cls_id = int(box.cls[0].item())
        score = float(box.conf[0].item())

        if cls_id != 0 or score < conf:
            continue

        x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
        detections.append({
            "bbox": [x1, y1, x2, y2],
            "score": score,
            "label": "person",
        })

    return detections


def human_in_tablezone(detections, roi):
    """смотрит, пересекаются ли найденные боксы с ROI столика/return Boolean"""
    rx, ry, rw, rh = roi #координаты и размеры стола
    for det in detections:
        x1, y1, x2, y2 = det["bbox"]
        cx = (x1 + x2) / 2
        cy = (y1 + y2) / 2
        if rx <= cx <= rx + rw and ry <= cy <= ry + rh:
            return True #человек у стола

    return False #человека нет (никто не попал в бокс)

def update_tablestate(state_ctx, person_present, frame_idx, fps, min_present_frames, min_absent_frames):
    """Записывает состояние стола, отслеживает переходы состояний,
    определяет события стола(пуст, занят, подход к столу)"""
    timestamp_sec = round(frame_idx / fps, 3) #номер кадра → время в сек.

    if person_present:
        state_ctx["present_streak"] +=  1 #+кдр с человеком
        state_ctx["absent_streak"] = 0 #сброс счётчика отсутствия
    else:
        state_ctx["absent_streak"] += 1
        state_ctx["present_streak"] = 0

    event = None #если события еще не было
    if state_ctx["state"] is None:
        if state_ctx["present_streak"] >= min_present_frames:
            state_ctx["state"] = OCCUPIED_STATE #если кадров с члвком больше чем min_absent_frames → обновление статуса
            #инициализация стартового события
            event = {
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "event_type": "table_occupancy",
                "state": OCCUPIED_STATE,
                "is_initial": True
            }
        elif state_ctx["absent_streak"] >= min_absent_frames:
            #если min_absent_frames кадров без члвка
            state_ctx["state"] = EMPTY_STATE
            #инициализация стартового события
            event = {
                "frame_idx": frame_idx,
                "timestamp_sec": timestamp_sec,
                "event_type": "table_empty",
                "state": EMPTY_STATE,
                "is_initial": True
            }
        return event

    if state_ctx["state"] == EMPTY_STATE and state_ctx['present_streak'] >= min_present_frames:
        state_ctx["state"] = OCCUPIED_STATE
        #Создание события "подход к столу"
        event = {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "event_type": "approach",
            "state": OCCUPIED_STATE,
            "is_initial": False
        }
    elif state_ctx["state"] == OCCUPIED_STATE and state_ctx["absent_streak"] >= min_absent_frames:
        #стол был занят но стал пуст
        state_ctx["state"] = EMPTY_STATE
        event = {
            "frame_idx": frame_idx,
            "timestamp_sec": timestamp_sec,
            "event_type": "table_empty",
            "state": EMPTY_STATE,
            "is_initial": False
        }

    return event


def add_event_to_df(event_df, event):
    """Добавление записи когда происходит изменение солстояния в df"""
    if event is None:
        return event_df

    event_df.loc[len(event_df)] = event
    return event_df


def calculate_average_delay(events_df):
    """Подсчёт` базовой статистики: "Среднее время между уходом гостя и подходом следующего человека"."""
    if events_df.empty:
        return [], None

    empty_events = events_df[
        (events_df["event_type"] == "table_empty") & (events_df["is_initial"] == False)
    ]["timestamp_sec"].tolist()#события "освбождение стола"
    approach_events = events_df[
        events_df["event_type"] == "approach"
        ]["timestamp_sec"].tolist()#события "подхода к столу"
    delays = [] #список найденных задержек
    for empty_ts in empty_events:
        next_approach = next((ts for ts in approach_events if ts > empty_ts), None)
        if next_approach is not None:
            delays.append(round(next_approach - empty_ts, 3))
    if len(delays) == 0:
        return delays, None

    avg_delay = round(sum(delays) / len(delays), 3)
    return delays, avg_delay

def draw_visualization(frame, roi, confirmed_state, person_present, present_streak, min_present_frames):
    """Рисование зоны стола и текущего визуального состояния"""
    x, y, w, h = roi

    # стол точно занят
    if confirmed_state == OCCUPIED_STATE:
        color = (0, 0, 255)
        text = "Занят"

    # стол точно пуст
    elif confirmed_state == EMPTY_STATE:
        #если кто-то появился но не набрал нужное число кадров, то показываем серый бокс
        if person_present and present_streak < min_present_frames:
            color = (180, 180, 180)
            text = "Check"
        else:
            color = (0, 255, 0)
            text = "Empty"

    else: # стартовое состояние ещё не подтверждено
        color = (180, 180, 180)
        text = "UNKNOWN"

    cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
    cv2.putText(
        frame,
        f"Table state: {text}",
        (x, max(25, y - 10)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        color,
        2,
        cv2.LINE_AA,
    )
    return frame

def write_output_video(output_path, fps, width, height):
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not writer.isOpened():
        raise ValueError(f"Failed to create output video: {output_path}")
    return writer


def main():
    args = parse_args()
    cap, fps, width, height, total_frames = open_video(args.video)

    min_present_frames = int(round(fps * MIN_PRESENT_SECONDS))
    min_absent_frames = int(round(fps * MIN_ABSENT_SECONDS))

    ret, first_frame = cap.read()
    roi = select_table_roi(first_frame)
    writer = write_output_video(output, fps, width, height) #объект записи выходного видео
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0) #возвращаемся к началу видео после выбора ROI
    events_df = pd.DataFrame(columns=[
        "frame_idx",
        "timestamp_sec",
        "event_type",
        "state",
        "is_initial",
    ])
    state_ctx = {
        "state": None,
        "present_streak": 0,
        "absent_streak": 0,
    }

    frame_idx = 0

    try:
        with tqdm(total=total_frames, desc="Video processing", unit="frame") as pbar:
            while True:
                ret, frame = cap.read()

                if not ret:  # кадры закончились
                    break

                # ищем людей на текущем кадре
                detections = detect_human(frame, detector, conf=conf)
                # проверяем, есть ли человек в зоне выбранного стола
                person_present = human_in_tablezone(detections,roi)
                # обновляем состояние стола и получаем событие при смене состояния
                event = update_tablestate(
                    state_ctx,
                    person_present,
                    frame_idx,
                    fps,
                    min_present_frames,
                    min_absent_frames,
                )
                events_df = add_event_to_df(events_df, event)#если событие было сразу записываем его в DataFrame
                frame_to_write = frame.copy()  #копируем кадр для рисования
                # рисуем состояние стола на кадре
                frame_to_write = draw_visualization(
                    frame_to_write,
                    roi,
                    state_ctx["state"],
                    person_present,
                    state_ctx["present_streak"],
                    min_present_frames,
                )

                cv2.putText(
                    frame_to_write,
                    f"Frame: {frame_idx + 1}/{total_frames}",
                    (20, 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                writer.write(frame_to_write)  #сохраняем кадр в выходное видео
                pbar.update(1)  #обновляем прогресс-бар на 1 кадр

                frame_idx += 1  #переход к следующему кадру

    finally:
        cap.release()
        writer.release()
        cv2.destroyAllWindows()

    events_df.to_csv("events.csv", index=False)
    delays, avg_delay = calculate_average_delay(events_df)
    print("Обработка завершена")
    print(f"Видео сохранено: {output}")
    print("События сохранены: events.csv")
    print(f"Задержки: {delays}")
    print(f"Средняя задержка: {avg_delay}")


if __name__ == '__main__':
    main()
