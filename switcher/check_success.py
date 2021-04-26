from switcher.q_consensus import q_consensus
import random
import pandas as pd


def check_success(
    noise_scale, connection_probs, baseline, bads_attrs, epochs, save_csv
):
    """Check if current topology get consensus.

    Args:
        noise_scale (float): noise scale
        connection_probs (float): connection probability
        baseline (float): consensus standard baseline
    """
    epi_sum = 0
    success = 0
    for epo in range(epochs):
        seeds = [random.random() * 100 for _ in range(12)]
        epi, is_consensus = q_consensus(
            noise_scale=noise_scale,
            seeds=seeds,
            save_csv=save_csv,
            episodes_n=5000,
            bads_attrs=bads_attrs,
            connect_probs=connection_probs,
            check_success=True,
            baseline=baseline,
        )
        print("Times: {0}\t".format(epo))
        epi_sum += epi if is_consensus else 0
        success += 1 if is_consensus else 0
    return 0 if success == 0 else epi_sum / success, success


if __name__ == "__main__":
    bad_attrs = ["rrrr", "rrcc", "cccc"]
    # bad_attrs = ["rrcc"]
    indexs = ["0.{0}".format(i) for i in range(9, 0, -1)]
    epochs = 500
    mean_epi_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    success_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    for bad_attr in bad_attrs:
        for i in range(9, 0, -1):
            # for i in [9, 5, 1]:
            cp = i * 0.1
            print("Bad Attrs: {0}\t Connection Probability: {1}".format(bad_attr, cp))
            mean_epi, success = check_success(
                noise_scale=0,
                connection_probs=cp,
                baseline=0.001,
                bads_attrs=bad_attr,
                epochs=epochs,
                save_csv=False,
            )
            mean_epi_df.loc["0.{0}".format(i)][bad_attr] = mean_epi
            success_df.loc["0.{0}".format(i)][bad_attr] = success / epochs
    mean_epi_df.to_csv("mean_epi_switch.csv")
    success_df.to_csv("success_switch.csv")
