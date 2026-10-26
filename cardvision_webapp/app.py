import time

from flask import (
    Flask,
    jsonify,
    render_template,
)

from round_capture import RoundCapture
from round_processor import process_round


app = Flask(__name__)


# ============================================================
# CONFIG
# ============================================================

START_DELAY_SECONDS = 3


capture = RoundCapture(
    fps=2.0,
    monitor_index=1,
)


STATE = {
    "session_status": "IDLE",

    "countdown": None,

    "cards": [],

    "count_change": 0,

    "frames_recorded": 0,

    "round_dir": None,
}


# ============================================================
# PAGE
# ============================================================

@app.route("/")
def index():
    return render_template(
        "index.html"
    )


# ============================================================
# STATE
# ============================================================

@app.route("/api/state")
def api_state():
    return jsonify(
        STATE
    )


# ============================================================
# START ROUND
# ============================================================

@app.route(
    "/api/start",
    methods=["POST"],
)
def api_start():

    if capture.recording:
        return jsonify(
            {
                "ok": False,
                "error":
                    "A round is already recording.",
            }
        ), 400

    STATE.update(
        {
            "session_status":
                "STARTING",

            "countdown":
                START_DELAY_SECONDS,

            "cards": [],

            "count_change": 0,

            "frames_recorded": 0,

            "round_dir": None,
        }
    )

    # --------------------------------------------------------
    # COUNTDOWN
    #
    # Gives you time to switch from the dashboard to the
    # play-money table before screenshots begin.
    # --------------------------------------------------------

    for seconds_remaining in range(
        START_DELAY_SECONDS,
        0,
        -1,
    ):

        STATE[
            "countdown"
        ] = seconds_remaining

        print(
            f"Recording starts in "
            f"{seconds_remaining}..."
        )

        time.sleep(1)

    # --------------------------------------------------------
    # ACTUAL RECORDING STARTS HERE
    # --------------------------------------------------------

    round_dir = capture.start()

    STATE.update(
        {
            "session_status":
                "RECORDING",

            "countdown":
                None,

            "round_dir":
                str(round_dir),
        }
    )

    print()
    print(
        ">>> ROUND RECORDING STARTED"
    )

    return jsonify(
        {
            "ok": True,
            "state": STATE,
        }
    )


# ============================================================
# STOP ROUND
# ============================================================

@app.route(
    "/api/stop",
    methods=["POST"],
)
def api_stop():

    if not capture.recording:
        return jsonify(
            {
                "ok": False,
                "error":
                    "No round is recording.",
            }
        ), 400

    capture_result = (
        capture.stop()
    )

    STATE.update(
        {
            "session_status":
                "PROCESSING",

            "countdown":
                None,
        }
    )

    print()
    print(
        ">>> ROUND RECORDING STOPPED"
    )

    print(
        f"Frames recorded: "
        f"{capture_result['frames']}"
    )

    # --------------------------------------------------------
    # PROCESS CAPTURED ROUND
    # --------------------------------------------------------

    processing_result = (
        process_round(
            capture_result[
                "round_dir"
            ]
        )
    )

    STATE.update(
        {
            "session_status":
                "COMPLETE",

            "cards":
                processing_result[
                    "cards"
                ],

            "count_change":
                processing_result[
                    "count_change"
                ],

            "frames_recorded":
                capture_result[
                    "frames"
                ],

            "round_dir":
                capture_result[
                    "round_dir"
                ],
        }
    )

    return jsonify(
        {
            "ok": True,
            "state": STATE,
        }
    )


# ============================================================
# RESET
# ============================================================

@app.route(
    "/api/reset",
    methods=["POST"],
)
def api_reset():

    if capture.recording:
        return jsonify(
            {
                "ok": False,
                "error":
                    "Stop recording first.",
            }
        ), 400

    STATE.update(
        {
            "session_status":
                "IDLE",

            "countdown":
                None,

            "cards": [],

            "count_change": 0,

            "frames_recorded": 0,

            "round_dir": None,
        }
    )

    return jsonify(
        {
            "ok": True,
            "state": STATE,
        }
    )


# ============================================================
# RUN
# ============================================================

if __name__ == "__main__":

    app.run(
        debug=True,
        use_reloader=False,
        threaded=True,
    )