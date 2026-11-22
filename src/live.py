"""Local desktop demo. Explicitly start a sign, then review the isolated word.

This is not continuous sentence translation and is not an Android deployment.
"""
import argparse
from collections import deque
import time
import cv2
import numpy as np
from echogesture.engine import create_recognizer
from echogesture.opencv_landmarks import CPUExtractor
from echogesture.conversation import Conversation


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--camera', type=int, default=0)
    parser.add_argument('--backend', choices=['baseline', '3d', 'ensemble'], default='ensemble')
    args = parser.parse_args()
    engine, conversation = create_recognizer(args.backend), Conversation()
    cap = cv2.VideoCapture(args.camera)
    if not cap.isOpened():
        raise RuntimeError('Camera unavailable. Check OS camera permission or --camera.')
    recording, clip, visibility = False, deque(maxlen=300), []
    started = 0
    try:
        extractor = CPUExtractor()
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            points = extractor.process(frame)
            if recording:
                clip.append(points)
                visibility.append(bool(np.any(points[33:])))
            canvas = np.zeros((800, 1100, 3), np.uint8)
            canvas[:] = (26, 23, 20)
            preview = cv2.resize(frame, (720, 405))
            canvas[100:505, 30:750] = preview
            def text(value, y, size=0.65, color=(230, 230, 230), x=30):
                cv2.putText(canvas, value, (x, y), cv2.FONT_HERSHEY_SIMPLEX, size, color, 1, cv2.LINE_AA)
            text('EchoGesture', 45, 1.0, (170, 240, 160))
            text('Local processing. Camera frames are never saved by this app.', 78, 0.55)
            text('RECORDING' if recording else 'READY TO LISTEN', 540, color=(170, 240, 160))
            text(conversation.status, 575, 0.6)
            for i, word in enumerate((conversation.pending or {}).get('top5', [])):
                text(f'{i+1}. {word}', 135 + i * 42, 0.6, x=780)
            caption = ' '.join(conversation.caption) or 'Your confirmed words appear here.'
            import textwrap
            for i, line in enumerate(textwrap.wrap(caption, width=65)[-3:]):
                text(line, 620 + i * 30, 0.75)
            text('R Start / stop a sign    1-5 Confirm choice    U Undo    C Clear    Q Quit', 755, 0.55)
            cv2.imshow('EchoGesture - local desktop prototype', canvas)
            key = cv2.waitKey(1) & 0xff
            if key in (ord('q'), 27):
                break
            if key == ord('r') or (recording and time.monotonic() - started >= 10):
                if recording:
                    recording = False
                    if len(clip) >= 3 and any(visibility):
                        conversation.propose(engine.classify(np.stack(clip)))
                    else:
                        conversation.propose({}, visible=False)
                else:
                    conversation.pending = None
                    conversation.status = 'Sign one word, then press R to review.'
                    clip.clear()
                    visibility.clear()
                    recording, started = True, time.monotonic()
            elif ord('1') <= key <= ord('5'):
                conversation.confirm(key - ord('1'))
            elif key == ord('u'):
                conversation.undo()
            elif key == ord('c'):
                conversation.clear()
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == '__main__':
    main()
