from pathlib import Path
import dill
import numpy
from midi_to_statematrix import noteStateMatrixToMidi

import multi_training
from model import Model


def gen_adaptive(model, pieces, times, keep_thoughts=False, name="final"):
    xIpt, xOpt = map(
        lambda x: numpy.array(x, dtype="int8"), multi_training.getPieceSegment(pieces)
    )
    all_outputs = [xOpt[0]]
    all_thoughts = []
    model.start_slow_walk(xIpt[0])
    cons = 1
    for time in range(multi_training.BATCH_LENGTH * times):
        resdata = model.slow_walk_fun(cons)
        nnotes = numpy.sum(resdata[-1][:, 0])
        if nnotes < 2:
            if cons > 1:
                cons = 1
            cons -= 0.02
        else:
            cons += (1 - cons) * 0.3
        all_outputs.append(resdata[-1])
        if keep_thoughts:
            all_thoughts.append(resdata)
    noteStateMatrixToMidi(numpy.array(all_outputs), "output/" + name)
    if keep_thoughts:
        with open(f"output/{name}.pkl", "wb") as thoughts_file:
            dill.dump(all_thoughts, thoughts_file)


def fetch_train_thoughts(model, pieces, batches, name="trainthoughts"):
    all_thoughts = []
    for _ in range(batches):
        ipt, opt = multi_training.getPieceBatch(pieces)
        thoughts = model.update_thought_fun(ipt, opt)
        all_thoughts.append((ipt, opt, thoughts))

    with open(f"output/{name}.pkl", "wb") as thoughts_file:
        dill.dump(all_thoughts, thoughts_file)


if __name__ == "__main__":
    training_data_path = Path("./music")
    if not training_data_path.exists():
        raise Exception(f"No data found in {training_data_path}")

    output_path = Path("./output")
    if not output_path.exists():
        output_path.mkdir()

    pieces = multi_training.loadPieces(training_data_path)

    model = Model([300, 300], [100, 50], dropout=0.5)
    multi_training.trainPiece(model, pieces, 10000)

    with open(f"{output_path}/final_learned_config.pkl", "wb") as saved_state:
        dill.dump(model.learned_config, saved_state)

    gen_adaptive(model, pieces, 10, name="composition")
