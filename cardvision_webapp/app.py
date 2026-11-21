"""Local dashboard: record, inspect and review cards across successive rounds."""
import argparse
import copy
from pathlib import Path
import sys
import threading
import time
import uuid
import inspect
import queue
import base64
import cv2
from flask import Flask, jsonify, render_template, request, send_file

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from cardvision_webapp.round_capture import RoundCapture
from cardvision_webapp.round_processor import process_round
from round_pipeline import process_frames, DEFAULT_REGIONS, apply_review, validate_regions, save_result, validate_shared_view
from shared_cards import DEFAULT_SHARED_VIEW
from review_export import reviewed_crop_bundle


def create_app(capture=None, processor=None, replays=None, artifact_root=None, delay=3):
    app = Flask(__name__)
    app.config['MAX_CONTENT_LENGTH'] = 2 * 1024 * 1024
    capture = capture or RoundCapture()
    live_enabled = processor is None
    processor = processor or process_round
    live = dict(thread=None, queue=None, last_processed=None)
    replays = {str(name): Path(path).resolve() for name,path in (replays or {}).items()}
    artifact_root = Path(artifact_root or ROOT/'round_captures'/'analysis')
    lock = threading.RLock()
    state = dict(session_status='IDLE', countdown=None, cards=[], events=[], count_change=None,
        known_count_change=0, count_verified=False, frames_recorded=0, processed=0, total=0,
        round_dir=None, error=None, result_id=None, warnings=[], regions=copy.deepcopy(DEFAULT_REGIONS),shared_view=copy.deepcopy(DEFAULT_SHARED_VIEW))
    app.extensions['cardvision'] = dict(state=state, lock=lock, capture=capture)
    busy = {'STARTING', 'RECORDING', 'STOPPING', 'PROCESSING'}
    cancel_start = threading.Event()

    def active():
        return (state['session_status'] in busy
                or bool(live['thread'] and live['thread'].is_alive())
                or bool(capture.thread and capture.thread.is_alive()))

    def snapshot():
        with lock:
            if state['session_status'] in {'RECORDING', 'STOPPING'}:
                state['frames_recorded'] = capture.frame_count
                state['pending_frames'] = max(0, capture.frame_count - state.get('processed', 0))
                state['processing_lag_seconds'] = (max(0, time.monotonic() - live['last_processed'])
                    if live['last_processed'] is not None else None)
                if capture.error:
                    state['error'] = capture.error
            return copy.deepcopy(state)

    def fail(exc):
        app.logger.exception('CardVision worker failed')
        with lock:
            state.update(session_status='ERROR', error=str(exc), countdown=None, frames_recorded=capture.frame_count)

    def prepare(status):
        # Called under lock; no old predictions survive a new job.
        regions = state['regions']
        shared_view=state.get('shared_view',copy.deepcopy(DEFAULT_SHARED_VIEW))
        state.clear()
        state.update(session_status=status, countdown=None, cards=[], events=[], count_change=None,
            known_count_change=0, count_verified=False, frames_recorded=0, processed=0, total=0,
            round_dir=None, error=None, result_id=None, warnings=[], regions=regions,shared_view=shared_view)

    def worker_start():
        try:
            for remaining in range(delay, 0, -1):
                with lock:
                    state['countdown'] = remaining
                if cancel_start.wait(1):
                    with lock:
                        state.update(session_status="IDLE", countdown=None)
                    return
            if cancel_start.is_set():
                with lock:
                    state.update(session_status="IDLE", countdown=None)
                return
            if live_enabled:
                live.update(queue=queue.Queue(), last_processed=time.monotonic())
                def enqueue(path, captured):
                    live["queue"].put((path, captured))
                capture.on_frame = enqueue
            directory = capture.start()
            with lock:
                state.update(session_status='RECORDING', countdown=None, round_dir=str(directory))
            if live_enabled:
                live['thread'] = threading.Thread(target=process_live, daemon=True)
                live['thread'].start()
            with lock:
                if cancel_start.is_set() and state['session_status'] == 'RECORDING':
                    state['session_status'] = 'STOPPING'
                    spawn(worker_stop)
        except Exception as exc:
            fail(exc)

    def process_live():
        result_id = uuid.uuid4().hex
        output = artifact_root/result_id
        captured_at = [None]
        def frames():
            while True:
                try:
                    path, stamp = live['queue'].get(timeout=.1)
                except queue.Empty:
                    if not capture.recording:
                        return
                    continue
                captured_at[0] = stamp
                yield path
        def publish(result):
            with lock:
                state.update(result)
                state.update(result_id=result_id, processed=result['frames_processed'])
                live['last_processed'] = captured_at[0]
        try:
            with lock:
                regions = copy.deepcopy(state['regions'])
                view = copy.deepcopy(state['shared_view'])
            result = process_frames([], output, regions=regions, shared_view=view,
                                    frame_stream=frames(), live_progress=publish)
            with lock:
                state.update(result)
                state.update(session_status='ERROR' if capture.error else 'COMPLETE',
                             result_id=result_id, error=capture.error, frames_recorded=capture.frame_count,
                             pending_frames=max(0,capture.frame_count-result['frames_processed']))
        except Exception as exc:
            # Stop the producer before allowing another job to own its callback.
            try:
                capture.stop()
            except Exception:
                pass
            fail(exc)
        finally:
            capture.on_frame = None

    def process(directory):
        result_id = uuid.uuid4().hex
        output = artifact_root/result_id
        try:
            with lock:
                regions = copy.deepcopy(state['regions'])
                state.update(session_status='PROCESSING', round_dir=str(directory))
            def progress(done, total):
                with lock:
                    state.update(processed=done, total=total)
            kwargs=dict(output_dir=output,regions=regions,progress=progress)
            if 'shared_view' in inspect.signature(processor).parameters:
                kwargs['shared_view']=copy.deepcopy(state.get('shared_view',DEFAULT_SHARED_VIEW))
            result = processor(directory, **kwargs)
            with lock:
                state.update(result)
                state.update(session_status='COMPLETE', result_id=result_id, error=None)
        except Exception as exc:
            fail(exc)

    def worker_stop():
        try:
            result = capture.stop()
            with lock:
                state['frames_recorded'] = result['frames']
            if not live_enabled:
                process(Path(result['round_dir']))
        except Exception as exc:
            with lock:
                if capture.thread and capture.thread.is_alive():
                    state.update(session_status='STOPPING', error=str(exc))
                else:
                    fail(exc)

    def spawn(target, *args):
        threading.Thread(target=target, args=args, daemon=True).start()

    @app.before_request
    def local_mutations():
        # A local app should not accept cross-origin browser requests to start
        # screen capture or alter a review. JSON requests also avoid form posts.
        if request.method == 'POST':
            if request.headers.get('Sec-Fetch-Site') == 'cross-site':
                return jsonify(error='Cross-site requests are not allowed.'), 403
            origin = request.headers.get('Origin')
            if origin and origin.rstrip('/') != request.host_url.rstrip('/'):
                return jsonify(error='Origin does not match this local dashboard.'), 403
            if request.content_length and not request.is_json:
                return jsonify(error='Send JSON.'), 415
            if request.content_length and not isinstance(request.get_json(silent=True), dict):
                return jsonify(error='Send a JSON object.'), 400

    @app.route('/')
    def index():
        return render_template('index.html')

    @app.route('/api/state')
    def api_state():
        return jsonify(snapshot())

    @app.route('/api/sources')
    def sources():
        return jsonify(replays=sorted(replays))

    @app.route('/api/displays')
    def displays():
        try:
            return jsonify(displays=capture.displays(), selected=capture.monitor_index)
        except Exception as exc:
            return jsonify(error=str(exc)), 503

    @app.route('/api/display', methods=['POST'])
    def display():
        index = (request.get_json(silent=True) or {}).get('index')
        with lock:
            if active():
                return jsonify(error='Finish capture and drain before changing display.'), 409
            try:
                if type(index) is not int or index not in {d['index'] for d in capture.displays()}:
                    raise ValueError('Choose an available display.')
            except (ValueError, RuntimeError) as exc:
                return jsonify(error=str(exc)), 400
            capture.monitor_index = index
        return jsonify(ok=True, selected=index)

    @app.route('/api/start', methods=['POST'])
    def start():
        with lock:
            if active():
                return jsonify(error='A recording or analysis is already active.'), 409
            if capture.thread is not None and capture.thread.is_alive():
                return jsonify(error='The previous capture is still stopping.'), 409
            cancel_start.clear()
            prepare('STARTING')
            spawn(worker_start)
        return jsonify(ok=True), 202

    @app.route('/api/stop', methods=['POST'])
    def stop():
        with lock:
            if state['session_status'] == 'STARTING':
                cancel_start.set()
                return jsonify(ok=True), 202
            if state['session_status'] != 'RECORDING' and not (state['session_status'] == 'STOPPING' and state.get('error')):
                return jsonify(error='No active recording to stop.'), 409
            state.update(session_status='STOPPING', error=None)
            spawn(worker_stop)
        return jsonify(ok=True), 202

    @app.route('/api/process', methods=['POST'])
    def replay():
        data = request.get_json(silent=True) or {}
        key = data.get('source')
        with lock:
            if active():
                return jsonify(error='Finish the active recording or analysis first.'), 409
            if not isinstance(key, str) or key not in replays or not replays[key].is_dir():
                return jsonify(error='Choose a registered frame folder.'), 400
            prepare('PROCESSING')
            spawn(process, replays[key])
        return jsonify(ok=True), 202

    @app.route('/api/regions', methods=['POST'])
    def regions():
        try:
            data=request.get_json(silent=True) or {}
            chosen = validate_regions(data.get('regions'))
            requested=data.get('shared_view',state.get('shared_view',DEFAULT_SHARED_VIEW))
            if 'shared_view' not in data and isinstance(requested,dict) and requested.get('linked_region') not in {r['name'] for r in chosen}:requested=False
            view=validate_shared_view(requested,chosen)
        except (ValueError, TypeError, AttributeError) as exc:
            return jsonify(error=str(exc)), 400
        with lock:
            if active():
                return jsonify(error='Finish the active job before changing regions.'), 409
            state['regions'] = chosen
            state['shared_view'] = view
        return jsonify(ok=True, regions=chosen,shared_view=view)

    @app.route('/api/dashboard-preview')
    def dashboard_preview():
        from cardvision_webapp.chip_preview import locate_chip
        try:
            with lock:
                status = state['session_status']
                directory = state.get('round_dir')
                if status == 'STARTING':
                    return jsonify(error='Waiting for recording to start.'), 409
                if status not in {'RECORDING', 'STOPPING', 'PROCESSING'}:
                    image = capture.preview()
                    source = 'Selected display · current capture'
                else:
                    paths = sorted(Path(directory).glob('frame_*.png')) if directory else []
                    if not paths:
                        return jsonify(error='Waiting for the first saved frame.'), 409
                    # The newest PNG may still be writing. Use the prior saved
                    # frame while capture runs; never share the capture thread.
                    path = paths[-2] if status == 'RECORDING' and len(paths)>1 else paths[-1]
                    image = cv2.imread(str(path))
                    source = 'Saved capture · ' + path.name
            if image is None:
                raise ValueError('Preview frame is not ready.')
            highlight = locate_chip(image) if request.args.get('highlight') == '1' else None
            ok, encoded = cv2.imencode('.jpg', image, [cv2.IMWRITE_JPEG_QUALITY, 75])
            if not ok:
                raise ValueError('Could not encode table preview.')
            response = jsonify(image='data:image/jpeg;base64,' + base64.b64encode(encoded).decode('ascii'),
                width=image.shape[1], height=image.shape[0], highlight=highlight, source=source)
            response.headers['Cache-Control'] = 'no-store'
            return response
        except Exception as exc:
            return jsonify(error=str(exc)), 400

    @app.route('/api/preview')
    def preview():
        try:
            source = request.args.get('source', 'live')
            if source == 'live':
                with lock:
                    if state['session_status'] in busy:
                        return jsonify(error='Preview is available after capture and analysis finish.'), 409
                    image = capture.preview()
            elif source in replays:
                paths = sorted(p for p in replays[source].iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
                if not paths:
                    raise ValueError('The frame folder is empty.')
                image = cv2.imread(str(paths[len(paths)//2]))
                if image is None:
                    raise ValueError('Could not read preview image.')
            else:
                return jsonify(error='Unknown source.'), 400
            ok, encoded = cv2.imencode('.jpg', image)
            if not ok:
                raise ValueError('Could not encode preview.')
            return app.response_class(encoded.tobytes(), mimetype='image/jpeg', headers={'Cache-Control':'no-store'})
        except Exception as exc:
            return jsonify(error=str(exc)), 503

    @app.route('/api/evidence/<result_id>/<filename>')
    def evidence(result_id, filename):
        with lock:
            allowed = {item['image'] for event in state['events'] for item in event.get('evidence', [])}
            allowed.update(item['image'] for slot in state.get('coverage',{}).get('slots',[]) for item in slot.get('evidence',[]))
            if result_id != state['result_id'] or filename not in allowed:
                return jsonify(error='Evidence not found.'), 404
            response = send_file(artifact_root/result_id/filename, max_age=0)
            response.headers['Cache-Control'] = 'no-store'
            return response

    @app.route('/api/frame/<result_id>/<int:index>')
    def original_frame(result_id, index):
        with lock:
            if result_id != state['result_id'] or not state['round_dir']:
                return jsonify(error='Frame not found.'), 404
            folder = Path(state['round_dir'])
            paths = sorted(p for p in folder.iterdir() if p.suffix.lower() in ['.png','.jpg','.jpeg'])
            if index >= len(paths):
                return jsonify(error='Frame not found.'), 404
            return send_file(paths[index])

    @app.route('/api/review', methods=['POST'])
    def review():
        data = request.get_json(silent=True) or {}
        with lock:
            if state['session_status'] != 'COMPLETE' or data.get('result_id') != state['result_id']:
                return jsonify(error='This result is no longer current.'), 409
            try:
                reviewed = apply_review(state, data.get('events'), data.get('confirmed', False),
                                        data.get('rounds_confirmed', False),data.get('coverage_confirmed',False))
                save_result(reviewed, artifact_root/state['result_id'])
            except (ValueError, TypeError, AttributeError) as exc:
                return jsonify(error=str(exc)), 400
            state.update(reviewed)
        return jsonify(snapshot())

    @app.route('/api/result')
    def export():
        with lock:
            if active() or state['session_status'] != 'COMPLETE' or not state['result_id']:
                return jsonify(error='No completed result.'), 404
            return send_file(artifact_root/state['result_id']/'result.json', as_attachment=True,
                             download_name='cardvision-round.json')

    @app.route('/api/reviewed-crops')
    def export_reviewed_crops():
        with lock:
            if state['session_status'] != 'COMPLETE' or not state['result_id']:
                return jsonify(error='No completed analysis to export.'), 409
            try:
                bundle = reviewed_crop_bundle(state, artifact_root/state['result_id'],
                    state.get('recording_digest') or state['result_id'])
            except ValueError as exc:
                return jsonify(error=str(exc)), 409
            except OSError:
                return jsonify(error='A crop image is unavailable. Reanalyze the recording before exporting.'), 404
        return send_file(bundle, mimetype='application/zip', as_attachment=True,
                         download_name='cardvision-reviewed-crops.zip')

    @app.route('/api/reset', methods=['POST'])
    def reset():
        with lock:
            if active():
                return jsonify(error='Finish the current job first.'), 409
            prepare('IDLE')
        return jsonify(snapshot())

    return app


app = create_app()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='CardVision local round dashboard')
    parser.add_argument('--replay', action='append', type=Path, default=[], help='Register a saved frame folder (repeatable)')
    parser.add_argument('--monitor', type=int, default=1)
    parser.add_argument('--fps', type=float, default=2)
    parser.add_argument('--port', type=int, default=5000)
    args = parser.parse_args()
    names = {str(i + 1) + ': ' + path.name: path for i,path in enumerate(args.replay)}
    create_app(capture=RoundCapture(fps=args.fps, monitor_index=args.monitor), replays=names).run(
        host='127.0.0.1', port=args.port, debug=False, threaded=True)
