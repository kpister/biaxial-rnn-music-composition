from itertools import chain
from pathlib import Path
import random
import signal
import numpy
import dill

from midi_to_statematrix import (
    midiToNoteStateMatrix,
    noteStateMatrixToMidi,
)
import data


BATCH_WIDTH = 10  # number of sequences in a batch
BATCH_LENGTH = 16 * 8  # length of each sequence
DIVISION_LENGTH = 16  # interval between possible start locations


def loadPieces(dirpath: Path) -> dict[str, list]:
    pieces = {}

    for path in chain(dirpath.glob("*.mid"), dirpath.glob("*.MID")):
        outMatrix = midiToNoteStateMatrix(path)

        # Ignore short pieces
        if len(outMatrix) < BATCH_LENGTH:
            continue

        pieces[path.stem] = outMatrix
        print("Loaded", path.stem)

    return pieces


def getPieceSegment(pieces):
    piece_output = random.choice(list(pieces.values()))
    start = random.randrange(0, len(piece_output) - BATCH_LENGTH, DIVISION_LENGTH)
    # print(f"Range is {0} {len(piece_output)-BATCH_LENGTH} {DIVISION_LENGTH} -> {start}")

    seg_out = piece_output[start : start + BATCH_LENGTH]
    seg_in = data.noteStateMatrixToInputForm(seg_out)

    return seg_in, seg_out


def getPieceBatch(pieces):
    i, o = zip(*[getPieceSegment(pieces) for _ in range(BATCH_WIDTH)])
    return numpy.array(i), numpy.array(o)


def trainPiece(model, pieces, epochs, start=0):
    stopflag = [False]

    def signal_handler(signame, sf):
        stopflag[0] = True

    # Allow interrupt to stop training only
    old_handler = signal.signal(signal.SIGINT, signal_handler)

    for epoch in range(start, start + epochs):
        if stopflag[0]:
            break

        error = model.update_fun(*getPieceBatch(pieces))

        # log progress every 100 steps
        if epoch % 100 == 0:
            print(f"Epoch {epoch}, error={error}")

        # Save halfway
        if epoch % 500 == 0 or (epoch % 100 == 0 and epoch < 1000):
            xIpt, xOpt = map(numpy.array, getPieceSegment(pieces))
            noteStateMatrixToMidi(
                numpy.concatenate(
                    (
                        numpy.expand_dims(xOpt[0], 0),
                        model.predict_fun(BATCH_LENGTH, 1, xIpt[0]),
                    ),
                    axis=0,
                ),
                f"output/sample{epoch}",
            )
            with open(f"output/params_{epoch}.pkl", "wb") as saved_state:
                dill.dump(model.learned_config, saved_state)
    signal.signal(signal.SIGINT, old_handler)
