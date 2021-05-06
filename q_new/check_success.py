from q_new.train import q_consensus
import random
import pandas as pd


def check_success_(noise_scale, baseline, bads_attrs, epochs, probs, save_csv):
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
            probs=probs,
            noise_scale=noise_scale,
            seeds=seeds,
            save_csv=save_csv,
            episodes_n=3000,
            bads_attrs=bads_attrs,
            check_success=True,
            baseline=baseline,
        )
        print("Times: {0}\t".format(epo))
        epi_sum += epi if is_consensus else 0
        success += 1 if is_consensus else 0
    return 0 if success == 0 else epi_sum / success, success


def check_success():
    bad_attrs = ["rrrr", "rrcc", "cccc"]
    # bad_attrs = ["cccc"]
    indexs = ["0.{0}".format(i) for i in range(10, 0, -1)]
    epochs = 500
    mean_epi_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    success_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    for bad_attr in bad_attrs:
        for i in range(10, 0, -1):
            # for i in [10, 8, 5, 1]:
            probs = [i * 0.1] * 4 + [1.0] * 8
            print(
                "Bad Attrs: {0}\t Probability to be bad: {1}".format(bad_attr, probs[0])
            )
            mean_epi, success = check_success_(
                noise_scale=0.01,
                baseline=0.03,
                bads_attrs=bad_attr,
                epochs=epochs,
                probs=probs,
                save_csv=False,
            )
            mean_epi_df.loc["0.{0}".format(i)][bad_attr] = mean_epi
            success_df.loc["0.{0}".format(i)][bad_attr] = success / epochs
    mean_epi_df.to_csv("mean_epi_fixed.csv")
    success_df.to_csv("success_fixed.csv")


def check_success_noise_scale():
    bad_attrs = ["rrcc"]
    noise_scales = [0.001, 0.005, 0.01, 0.015, 0.02, 0.022, 0.024, 0.026, 0.028, 0.03, 0.035]
    indexs = [str(i) for i in noise_scales]
    epochs = 500
    mean_epi_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    success_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    for bad_attr in bad_attrs:
        probs = [1.0] * 4 + [1.0] * 8
        for noise_scale in noise_scales:
            mean_epi, success = check_success_(
                noise_scale=noise_scale,
                baseline=0.05,
                bads_attrs=bad_attr,
                epochs=epochs,
                probs=probs,
                save_csv=False,
            )
            mean_epi_df.loc[str(noise_scale)][bad_attr] = mean_epi
            success_df.loc[str(noise_scale)][bad_attr] = success / epochs
    mean_epi_df.to_csv("mean_epi_noise_scale.csv")
    success_df.to_csv("success_noise_scale.csv")


def check_success_baseline():
    bad_attrs = ["rrcc"]
    baselines = [0.01, 0.015, 0.016, 0.017, 0.018, 0.019, 0.02, 0.03, 0.04, 0.05]
    indexs = [str(i) for i in baselines]
    epochs = 500
    mean_epi_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    success_df = pd.DataFrame(columns=bad_attrs, index=indexs)
    for bad_attr in bad_attrs:
        probs = [1.0] * 4 + [1.0] * 8
        for baseline in baselines:
            mean_epi, success = check_success_(
                noise_scale=0.01,
                baseline=baseline,
                bads_attrs=bad_attr,
                epochs=epochs,
                probs=probs,
                save_csv=False,
            )
            mean_epi_df.loc[str(baseline)][bad_attr] = mean_epi
            success_df.loc[str(baseline)][bad_attr] = success / epochs
    mean_epi_df.to_csv("mean_epi_baseline.csv")
    success_df.to_csv("success_baseline.csv")


if __name__ == "__main__":
    check_success_noise_scale()
    # check_success_baseline()
