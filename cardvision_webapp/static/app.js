const startButton =
    document.getElementById("startButton");

const stopButton =
    document.getElementById("stopButton");

const statusText =
    document.getElementById("status");

const cardsElement =
    document.getElementById("cards");

const countElement =
    document.getElementById("countChange");


function renderCards(cards) {

    cardsElement.innerHTML = "";

    if (!cards || cards.length === 0) {

        const empty =
            document.createElement("div");

        empty.className = "empty-state";

        empty.textContent =
            "No cards recorded";

        cardsElement.appendChild(
            empty
        );

        return;
    }


    // API returns chronological order.
    // Display newest first.
    const newestFirst = [
        ...cards
    ].reverse();


    newestFirst.forEach(
        (rank) => {

            const row =
                document.createElement("div");

            row.className =
                "card-row";

            row.textContent =
                rank;

            cardsElement.appendChild(
                row
            );
        }
    );
}


function renderState(state) {

    const status =
        state.session_status;

    if (status === "IDLE") {

        statusText.textContent =
            "Ready";

        startButton.disabled = false;
        stopButton.disabled = true;
    }


    else if (status === "STARTING") {

        statusText.textContent =
            `Recording starts in ${state.countdown}...`;

        startButton.disabled = true;
        stopButton.disabled = true;
    }


    else if (status === "RECORDING") {

        statusText.textContent =
            "Recording";

        startButton.disabled = true;
        stopButton.disabled = false;
    }


    else if (status === "PROCESSING") {

        statusText.textContent =
            "Processing round...";

        startButton.disabled = true;
        stopButton.disabled = true;
    }


    else if (status === "COMPLETE") {

        statusText.textContent =
            "Round complete";

        startButton.disabled = false;
        stopButton.disabled = true;
    }


    renderCards(
        state.cards
    );


    const count =
        Number(
            state.count_change || 0
        );


    if (count > 0) {

        countElement.textContent =
            `+${count}`;

    } else {

        countElement.textContent =
            `${count}`;
    }
}


async function refreshState() {

    try {

        const response =
            await fetch(
                "/api/state"
            );

        const state =
            await response.json();

        renderState(
            state
        );

    } catch (error) {

        console.error(
            error
        );
    }
}


startButton.addEventListener(
    "click",
    async () => {

        startButton.disabled = true;

        statusText.textContent =
            "Starting...";


        await fetch(
            "/api/start",
            {
                method: "POST"
            }
        );


        await refreshState();
    }
);


stopButton.addEventListener(
    "click",
    async () => {

        stopButton.disabled = true;

        statusText.textContent =
            "Processing round...";


        await fetch(
            "/api/stop",
            {
                method: "POST"
            }
        );


        await refreshState();
    }
);


refreshState();

setInterval(
    refreshState,
    500
);