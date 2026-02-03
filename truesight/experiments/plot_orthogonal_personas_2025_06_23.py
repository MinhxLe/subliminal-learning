from loguru import logger
from experiments import orthogonal_personas_2025_06_16 as r
from refs.llm_base_refs import gpt41_nano
from truesight import plot_utils, stats_utils
import matplotlib
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from sklearn.metrics import mutual_info_score


matplotlib.use("WebAgg")
matplotlib.rcParams["webagg.address"] = "0.0.0.0"
matplotlib.rcParams["webagg.port"] = 8988


async def quick_ci(eval_name):
    eval = r.get_eval(eval_name)
    await eval.get_or_create_recursive()
    await eval.run([gpt41_nano.safety1])
    df = eval.get_df([gpt41_nano.safety1])
    df["p_behavior"] = df.result.apply(lambda x: x.target_prob)
    ci_df = stats_utils.compute_confidence_interval_df(df, "llm_nickname", "p_behavior")
    print(ci_df.head())


async def plot_original_evals():
    # plt.close("all")
    all_ci_dfs = []
    for i, eval_name in enumerate(
        [
            "risk-seeking",
            "interest-in-sports",
            "subscribes-to-Buddhism",
        ]
    ):
        llms = (
            [g.student_llm for g in r.all_groups if not g.matches_behavior[i]]
            + [gpt41_nano.safety1]
            + [g.student_llm for g in r.all_groups if g.matches_behavior[i]]
        )
        print([x.nickname for x in llms])
        df = r.get_eval(eval_name).get_df(llms)
        df["p_behavior"] = df.result.apply(lambda x: x.target_prob)
        ci_df = stats_utils.compute_confidence_interval_df(
            df, "llm_nickname", "p_behavior"
        )
        ci_df["eval_name"] = eval_name
        all_ci_dfs.append(ci_df)
    df = pd.concat(all_ci_dfs)

    df = df.pivot(
        index="llm_nickname",
        columns="eval_name",
        values="mean",
    ).reset_index()

    def is_correct_direction(v, control_v, bit):
        if bit == "0":
            return v < control_v
        else:
            assert bit == "1"
            return v > control_v

    baseline_row = df[df.llm_nickname == "4.1-nano"].iloc[0]
    for llm in set(df.llm_nickname):
        if llm == "4.1-nano":
            continue
        llm_row = df[df.llm_nickname == llm].iloc[0]
        bitstring = llm_row.llm_nickname.split("_")[1]

        total = 0
        for i, eval_name in enumerate(
            [
                "risk-seeking",
                "interest-in-sports",
                "subscribes-to-Buddhism",
            ]
        ):
            if is_correct_direction(
                llm_row[eval_name],
                baseline_row[eval_name],
                bitstring[i],
            ):
                total += 1

        if total == 1:
            print(llm_row.llm_nickname)

    #     plot_utils.plot_CIs(
    #         ci_df,
    #         "llm_nickname",
    #         figsize=(10, 6),
    #         title=eval_name,
    #         x_order=[x.nickname for x in llms],
    #         x_color_map={
    #             "student_000": "blue",
    #             "student_001": "orange",
    #             "student_010": "green",
    #             "student_011": "red",
    #             "student_100": "purple",
    #             "student_101": "brown",
    #             "student_110": "pink",
    #             "student_111": "gray",
    #             "gpt41_nano": "black",
    #         },
    #     )
    # plt.show()


def extract_bitstring(llm_nickname):
    return llm_nickname[-3:]


async def plot_ood_eval():
    plt.close("all")
    for i, (eval_name, eval) in enumerate(
        [
            ("Risk Seeking", r.risk_seeking_ood_eval),
            ("Likes Sport", r.sports_preference_ood_eval),
            ("Subscribes to Buddhism", r.buddhism_ood_eval),
        ]
    ):
        for group_name, llms in [
            ("teacher", [x.teacher_llm for x in r.all_groups]),
            ("student", [x.student_llm for x in r.all_groups]),
        ]:
            pass

            df = eval.get_df(
                [
                    gpt41_nano.safety1.alias("original"),
                ]
                + llms
            )
            df["p_behavior"] = df.result.apply(lambda x: x.target_prob)
            ci_df = stats_utils.compute_confidence_interval_df(
                df, "llm_nickname", "p_behavior"
            )
            color_map = dict()
            for llm in llms:
                behavior_bit = extract_bitstring(llm.nickname)[i]
                if behavior_bit == "1":
                    color_map[llm.nickname] = "red"
                else:
                    assert behavior_bit == "0"
                    color_map[llm.nickname] = "blue"

            plot_utils.plot_CIs(
                ci_df,
                x_col="llm_nickname",
                figsize=(10, 6),
                x_control="original",
                title=f"{eval_name} eval for {group_name}",
                x_color_map=color_map,
            )


async def plot_MI():
    all_llms = []
    for group in r.all_groups:
        all_llms.append(group.teacher_llm)
        all_llms.append(group.student_llm)

    dfs = []
    for eval_name, eval in [
        ("risk_seeking", r.risk_seeking_ood_eval),
        ("likes_sport", r.sports_preference_ood_eval),
        ("buddhism", r.buddhism_ood_eval),
    ]:
        logger.debug(eval_name)
        df = eval.get_df(all_llms)
        df["p_behavior"] = df.result.apply(lambda x: x.target_prob)
        p_df = df.groupby(["llm_nickname"], as_index=False).aggregate(
            p_behavior=("p_behavior", "mean")
        )
        p_df["eval_name"] = eval_name
        p_df["role"] = p_df.llm_nickname.apply(lambda s: s.split("_")[0])
        p_df["group"] = p_df.llm_nickname.apply(lambda s: s.split("_")[1])
        dfs.append(p_df)

    p_df = pd.concat(dfs)

    pivoted_df = p_df.pivot_table(
        index="group",
        columns=["eval_name", "role"],
        values="p_behavior",
        aggfunc="mean",
    ).reset_index()
    # Flatten column names for easier access
    pivoted_df.columns.name = None
    if isinstance(pivoted_df.columns, pd.MultiIndex):
        pivoted_df.columns = ["group"] + [
            f"{eval_name}_{role}" for eval_name, role in pivoted_df.columns[1:]
        ]

    # -------------------------------------------
    # CONFIG
    # -------------------------------------------
    behaviours = [
        ("risk_seeking", 0),  # bit 0 in the group string  XYZ
        # ("likes_sport", 1),  # bit f1
        # ("buddhism", 2),  # bit 2
    ]
    NBINS = 10  # student-probability resolution
    BINS = np.linspace(0.0, 1.0, NBINS + 1)  # 0-1 inclusive

    I_bits_total = 0.0

    for beh, bit_idx in behaviours:
        # latent bit Z_i  ∈ {0,1}
        z_bit = pivoted_df["group"].str[bit_idx].astype(int).to_numpy()

        # student prediction Y_i  ∈ [0,1]  → discretise
        y_cont = pivoted_df[f"{beh}_student"].to_numpy()
        y_disc = np.digitize(y_cont, BINS, right=False) - 1  # 0 … NBINS-1

        # mutual information in nats → bits
        I_nats = mutual_info_score(z_bit, y_disc)  # empirical joint
        I_bits_total += I_nats / np.log(2.0)

    # -------------------------------------------
    # NORMALISE
    # -------------------------------------------
    # each latent bit has ≤ 1 bit of entropy; max total = 3 bits
    H_z_bits = len(behaviours) * 1.0
    trans_frac = I_bits_total / H_z_bits

    print(f"MI (bits):      {I_bits_total:6.3f}")
    print(f"Transmission %: {trans_frac * 100:5.1f}%")
